"""Phase 3 feature enrichment for the equity benchmark.

This script extends the existing per-ticker feature store with:
  - lagged cross-asset features downloaded via yfinance
  - overnight / intraday return decomposition
  - volume microstructure features
  - cross-sectional rank features across the 40-ticker panel
  - an XGBoost before/after comparison on baseline vs enriched features

The existing benchmark scripts remain unchanged; this module owns the v2
feature list and the comparison workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipeline import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from pipeline import load_config, make_splits


BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = Path(os.environ.get("EQUITY_BENCH_RESULTS_DIR", BENCH_DIR / "results"))

CROSS_ASSET_CACHE = DATA_DIR / "cross_asset_features.parquet"
XGB_RESULTS_PATH = RESULTS_DIR / "features_v2_xgb_comparison.json"

CROSS_ASSET_SYMBOLS: dict[str, str] = {
    "SPY": "spy",
    "^VIX": "vix",
    "DX-Y.NYB": "dxy",
    "^TNX": "tnx",
    "^IRX": "irx",
    "HYG": "hyg",
    "LQD": "lqd",
    "IEF": "ief",
    "TLT": "tlt",
    "GC=F": "gold",
    "CL=F": "oil",
    "HG=F": "copper",
}

OPTIONAL_CROSS_ASSET_SYMBOLS: dict[str, str] = {
    "^VIX3M": "vix3m",
}

ROLLING_VOL_WINDOW = 21
RANK_BETA_WINDOW = 63
OBV_CHANGE_WINDOW = 10


def _log(msg: str) -> None:
    print(f"[features_v2] {msg}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"not serializable: {type(value)!r}")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _autocorr(values: np.ndarray) -> float:
        if len(values) <= lag:
            return float("nan")
        x = values[:-lag]
        y = values[lag:]
        if np.std(x) == 0 or np.std(y) == 0:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    return series.rolling(window=window, min_periods=window).apply(_autocorr, raw=True)


def _compute_obv(price: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(price.diff()).fillna(0.0)
    signed_volume = volume.fillna(0.0) * direction
    return signed_volume.cumsum()


def _feature_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}_features.parquet"


def get_available_tickers() -> list[str]:
    tickers = sorted(path.stem.replace("_features", "") for path in DATA_DIR.glob("*_features.parquet"))
    if not tickers:
        raise FileNotFoundError(f"no feature parquet files found under {DATA_DIR}")
    return tickers


def load_feature_store(tickers: list[str]) -> dict[str, pd.DataFrame]:
    store: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = _feature_path(ticker)
        if path.exists():
            store[ticker] = pd.read_parquet(path).sort_index()
    return store


def _download_close_series(symbol: str, start_date: str, end_date: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.Series(dtype="float64", name=symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    series = df[close_col].copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.sort_index()
    series.name = symbol
    return series


def build_cross_asset_features(
    start_date: str,
    end_date: str,
    force_download: bool = False,
    allow_partial: bool = False,
) -> pd.DataFrame:
    if CROSS_ASSET_CACHE.exists() and not force_download:
        cached = pd.read_parquet(CROSS_ASSET_CACHE).sort_index()
        _log(f"using cached cross-asset features: {CROSS_ASSET_CACHE}")
        return cached

    prices: dict[str, pd.Series] = {}
    symbols = dict(CROSS_ASSET_SYMBOLS)
    symbols.update(OPTIONAL_CROSS_ASSET_SYMBOLS)

    for symbol, alias in symbols.items():
        try:
            series = _download_close_series(symbol, start_date, end_date)
        except Exception as exc:  # noqa: BLE001
            _log(f"{symbol}: download failed ({exc})")
            continue
        if series.empty:
            if symbol in CROSS_ASSET_SYMBOLS:
                _log(f"{symbol}: no data returned")
            continue
        prices[alias] = series
        _log(f"{symbol}: {len(series)} rows")

    required = {alias for alias in CROSS_ASSET_SYMBOLS.values()}
    missing_required = sorted(required.difference(prices))
    if missing_required:
        if allow_partial:
            _log(f"cross-asset build incomplete; proceeding without required series: {missing_required}")
            return pd.DataFrame()
        raise RuntimeError(f"missing required cross-asset series: {missing_required}")

    close_df = pd.DataFrame(prices).sort_index()
    feats = pd.DataFrame(index=close_df.index)

    for alias in close_df.columns:
        series = close_df[alias]
        feats[f"ca_{alias}_pct_1d"] = series.pct_change()
        feats[f"ca_{alias}_z21"] = _rolling_zscore(series, 21)
        feats[f"ca_{alias}_z63"] = _rolling_zscore(series, 63)

    spread_defs: dict[str, pd.Series] = {
        "yield_curve_slope": close_df["tnx"] - close_df["irx"],
        "hyg_ief_spread": close_df["hyg"] / close_df["ief"] - 1.0,
        "lqd_ief_spread": close_df["lqd"] / close_df["ief"] - 1.0,
    }
    if {"vix", "vix3m"}.issubset(close_df.columns):
        spread_defs["vix_term_structure"] = close_df["vix3m"] - close_df["vix"]

    for name, series in spread_defs.items():
        feats[f"ca_{name}"] = series
        feats[f"ca_{name}_chg_1d"] = series.diff()
        feats[f"ca_{name}_z21"] = _rolling_zscore(series, 21)
        feats[f"ca_{name}_z63"] = _rolling_zscore(series, 63)

    feats = feats.shift(1)
    feats.index.name = "date"
    CROSS_ASSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(CROSS_ASSET_CACHE)
    _log(f"wrote cross-asset cache: {CROSS_ASSET_CACHE}")
    return feats


def compute_intraday_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["r_overnight"] = df["open"] / df["close"].shift(1) - 1.0
    out["r_intraday"] = df["close"] / df["open"].replace(0.0, np.nan) - 1.0
    out["r_overnight_vol_21d"] = out["r_overnight"].rolling(ROLLING_VOL_WINDOW, min_periods=ROLLING_VOL_WINDOW).std()
    out["r_intraday_vol_21d"] = out["r_intraday"].rolling(ROLLING_VOL_WINDOW, min_periods=ROLLING_VOL_WINDOW).std()
    out["r_overnight_autocorr_21d"] = _rolling_autocorr(out["r_overnight"], ROLLING_VOL_WINDOW)
    out["r_intraday_autocorr_21d"] = _rolling_autocorr(out["r_intraday"], ROLLING_VOL_WINDOW)
    out["r_overnight_lag1"] = out["r_overnight"].shift(1)
    out["r_intraday_lag1"] = out["r_intraday"].shift(1)

    out["volume_zscore_21d"] = _rolling_zscore(df["volume"], 21)
    out["dollar_volume"] = df["adj_close"] * df["volume"]
    obv = _compute_obv(df["adj_close"], df["volume"])
    out["obv_10d_change"] = obv.diff(OBV_CHANGE_WINDOW)

    spike = out["volume_zscore_21d"] > 2.0
    out["volume_spike_2sigma"] = spike.where(~out["volume_zscore_21d"].isna()).astype("float64")

    return out


def compute_cross_sectional_rank_frames(store: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if "SPY" not in store:
        raise RuntimeError("SPY_features.parquet is required for beta and residualized momentum ranks")

    metrics: dict[str, dict[str, pd.Series]] = {
        "cs_rank_mom_5d": {},
        "cs_rank_mom_21d": {},
        "cs_rank_mom_63d": {},
        "cs_rank_vol_21d": {},
        "cs_rank_beta_spy_63d": {},
        "cs_rank_resid_mom_21d": {},
    }

    spy_returns = store["SPY"]["adj_close"].pct_change()
    spy_var = spy_returns.rolling(RANK_BETA_WINDOW, min_periods=RANK_BETA_WINDOW).var()
    spy_mom_21 = store["SPY"]["adj_close"].pct_change(21)

    for ticker, df in store.items():
        daily_ret = df["adj_close"].pct_change()
        mom_5 = df["adj_close"].pct_change(5)
        mom_21 = df["adj_close"].pct_change(21)
        mom_63 = df["adj_close"].pct_change(63)
        vol_21 = daily_ret.rolling(21, min_periods=21).std()
        beta_63 = daily_ret.rolling(RANK_BETA_WINDOW, min_periods=RANK_BETA_WINDOW).cov(spy_returns) / spy_var.replace(0.0, np.nan)
        resid_mom_21 = mom_21 - beta_63 * spy_mom_21

        metrics["cs_rank_mom_5d"][ticker] = mom_5
        metrics["cs_rank_mom_21d"][ticker] = mom_21
        metrics["cs_rank_mom_63d"][ticker] = mom_63
        metrics["cs_rank_vol_21d"][ticker] = vol_21
        metrics["cs_rank_beta_spy_63d"][ticker] = beta_63
        metrics["cs_rank_resid_mom_21d"][ticker] = resid_mom_21

    rank_frames: dict[str, pd.DataFrame] = {}
    for feature_name, ticker_series in metrics.items():
        panel = pd.DataFrame(ticker_series).sort_index()
        rank_frames[feature_name] = panel.rank(axis=1, method="average", pct=True)
    return rank_frames


def enrich_feature_store(
    tickers: list[str],
    cross_asset_features: pd.DataFrame,
) -> list[str]:
    store = load_feature_store(tickers)
    rank_frames = compute_cross_sectional_rank_frames(store)
    updated: list[str] = []
    replaceable = set(V2_FEATURE_COLUMNS + OPTIONAL_V2_FEATURE_COLUMNS)

    for ticker, df in store.items():
        drop_cols = [col for col in df.columns if col in replaceable]
        base_df = df.drop(columns=drop_cols, errors="ignore")
        enriched = base_df.join(cross_asset_features, how="left")
        enriched = enriched.join(compute_intraday_volume_features(df), how="left")
        for feature_name, frame in rank_frames.items():
            enriched[feature_name] = frame.get(ticker)
        enriched.to_parquet(_feature_path(ticker))
        updated.append(ticker)
        _log(f"{ticker}: wrote {len(enriched.columns)} columns")

    return updated


V2_FEATURE_COLUMNS: list[str] = [
    *(f"ca_{alias}_pct_1d" for alias in CROSS_ASSET_SYMBOLS.values()),
    *(f"ca_{alias}_z21" for alias in CROSS_ASSET_SYMBOLS.values()),
    *(f"ca_{alias}_z63" for alias in CROSS_ASSET_SYMBOLS.values()),
    "ca_yield_curve_slope",
    "ca_yield_curve_slope_chg_1d",
    "ca_yield_curve_slope_z21",
    "ca_yield_curve_slope_z63",
    "ca_hyg_ief_spread",
    "ca_hyg_ief_spread_chg_1d",
    "ca_hyg_ief_spread_z21",
    "ca_hyg_ief_spread_z63",
    "ca_lqd_ief_spread",
    "ca_lqd_ief_spread_chg_1d",
    "ca_lqd_ief_spread_z21",
    "ca_lqd_ief_spread_z63",
    "r_overnight",
    "r_intraday",
    "r_overnight_vol_21d",
    "r_intraday_vol_21d",
    "r_overnight_autocorr_21d",
    "r_intraday_autocorr_21d",
    "r_overnight_lag1",
    "r_intraday_lag1",
    "volume_zscore_21d",
    "dollar_volume",
    "obv_10d_change",
    "volume_spike_2sigma",
    "cs_rank_mom_5d",
    "cs_rank_mom_21d",
    "cs_rank_mom_63d",
    "cs_rank_vol_21d",
    "cs_rank_beta_spy_63d",
    "cs_rank_resid_mom_21d",
]

OPTIONAL_V2_FEATURE_COLUMNS: list[str] = [
    "ca_vix3m_pct_1d",
    "ca_vix3m_z21",
    "ca_vix3m_z63",
    "ca_vix_term_structure",
    "ca_vix_term_structure_chg_1d",
    "ca_vix_term_structure_z21",
    "ca_vix_term_structure_z63",
]


def get_full_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = list(BASE_FEATURE_COLUMNS)
    for col in V2_FEATURE_COLUMNS + OPTIONAL_V2_FEATURE_COLUMNS:
        if col in df.columns:
            cols.append(col)
    return cols


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"auc": float("nan"), "brier": float("nan"), "n": 0}
    out = {"n": int(len(y_true))}
    if len(np.unique(y_true)) < 2:
        out["auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    return out


def _train_eval_xgb(
    df: pd.DataFrame,
    feature_columns: list[str],
    xgb_cfg: dict[str, Any],
    threshold: int | None,
    train_ratio: float,
    val_ratio: float,
) -> tuple[dict[str, float], dict[str, float]]:
    import xgboost as xgb

    usable = df.dropna(subset=feature_columns + ["direction_5d"]).sort_index()
    splits = make_splits(usable, train_ratio=train_ratio, val_ratio=val_ratio)
    train_df = splits["train"]
    test_df = splits["test"]

    if len(train_df) < 30 or len(test_df) == 0 or len(np.unique(train_df["direction_5d"])) < 2:
        return {"auc": float("nan"), "brier": float("nan"), "n": 0}, {}

    X_train = train_df[feature_columns].astype(np.float32)
    y_train = train_df["direction_5d"].astype(int).to_numpy()
    if threshold is not None and threshold < len(X_train):
        X_train = X_train.iloc[:threshold]
        y_train = y_train[:threshold]

    X_test = test_df[feature_columns].astype(np.float32)
    y_test = test_df["direction_5d"].astype(int).to_numpy()

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg.get("n_estimators", 300)),
        max_depth=int(xgb_cfg.get("max_depth", 6)),
        learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
        subsample=float(xgb_cfg.get("subsample", 0.8)),
        colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=123,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    probs = model.predict_proba(X_test)[:, 1]

    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    importances = {str(name): float(score) for name, score in gains.items()}
    return _compute_metrics(y_test, probs), importances


def run_xgb_feature_comparison(
    tickers: list[str],
    config: dict[str, Any],
    threshold: int | None = 1000,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "meta": {
            "tickers": tickers,
            "threshold": threshold,
            "baseline_feature_count": len(BASE_FEATURE_COLUMNS),
        },
        "per_ticker": {},
    }
    xgb_cfg = config.get("xgboost", {})
    train_ratio = float(config["data"].get("train_ratio", 0.70))
    val_ratio = float(config["data"].get("val_ratio", 0.15))

    base_auc_values: list[float] = []
    full_auc_values: list[float] = []
    weighted_base_auc_num = 0.0
    weighted_full_auc_num = 0.0
    weighted_base_n = 0
    weighted_full_n = 0
    aggregate_importance: dict[str, float] = {}

    for ticker in tickers:
        df = pd.read_parquet(_feature_path(ticker)).sort_index()
        full_features = get_full_feature_columns(df)
        base_metrics, _ = _train_eval_xgb(
            df,
            list(BASE_FEATURE_COLUMNS),
            xgb_cfg,
            threshold,
            train_ratio,
            val_ratio,
        )
        full_metrics, importances = _train_eval_xgb(
            df,
            full_features,
            xgb_cfg,
            threshold,
            train_ratio,
            val_ratio,
        )

        results["per_ticker"][ticker] = {
            "baseline": base_metrics,
            "enriched": full_metrics,
            "delta_auc": (
                float(full_metrics["auc"] - base_metrics["auc"])
                if not (math.isnan(full_metrics["auc"]) or math.isnan(base_metrics["auc"]))
                else float("nan")
            ),
        }

        if not math.isnan(base_metrics["auc"]):
            base_auc_values.append(base_metrics["auc"])
            weighted_base_auc_num += base_metrics["auc"] * base_metrics["n"]
            weighted_base_n += int(base_metrics["n"])
        if not math.isnan(full_metrics["auc"]):
            full_auc_values.append(full_metrics["auc"])
            weighted_full_auc_num += full_metrics["auc"] * full_metrics["n"]
            weighted_full_n += int(full_metrics["n"])

        for name, score in importances.items():
            aggregate_importance[name] = aggregate_importance.get(name, 0.0) + score

    new_feature_importance = {
        name: score
        for name, score in sorted(aggregate_importance.items(), key=lambda item: item[1], reverse=True)
        if name in V2_FEATURE_COLUMNS or name in OPTIONAL_V2_FEATURE_COLUMNS
    }

    results["aggregate"] = {
        "mean_ticker_auc_baseline": float(np.mean(base_auc_values)) if base_auc_values else float("nan"),
        "mean_ticker_auc_enriched": float(np.mean(full_auc_values)) if full_auc_values else float("nan"),
        "weighted_auc_baseline": (
            float(weighted_base_auc_num / weighted_base_n) if weighted_base_n else float("nan")
        ),
        "weighted_auc_enriched": (
            float(weighted_full_auc_num / weighted_full_n) if weighted_full_n else float("nan")
        ),
        "top_new_features_by_gain": dict(list(new_feature_importance.items())[:20]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(XGB_RESULTS_PATH, "w") as handle:
        json.dump(results, handle, indent=2, default=_json_default)
    _log(f"wrote XGBoost comparison: {XGB_RESULTS_PATH}")
    return results


def run_build(force_download: bool = False, allow_partial_cross_asset: bool = False) -> None:
    config = load_config()
    tickers = get_available_tickers()
    cross_asset_features = build_cross_asset_features(
        start_date=str(config["data"]["start_date"]),
        end_date=str(config["data"]["end_date"]),
        force_download=force_download,
        allow_partial=allow_partial_cross_asset,
    )
    updated = enrich_feature_store(tickers, cross_asset_features)
    _log(f"enriched {len(updated)} tickers")


def run_all(
    force_download: bool = False,
    threshold: int | None = 1000,
    allow_partial_cross_asset: bool = False,
) -> dict[str, Any]:
    config = load_config()
    tickers = get_available_tickers()
    cross_asset_features = build_cross_asset_features(
        start_date=str(config["data"]["start_date"]),
        end_date=str(config["data"]["end_date"]),
        force_download=force_download,
        allow_partial=allow_partial_cross_asset,
    )
    enrich_feature_store(tickers, cross_asset_features)
    return run_xgb_feature_comparison(tickers, config, threshold=threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 feature enrichment for the equity benchmark")
    parser.add_argument(
        "--mode",
        choices=["build", "xgb", "all"],
        default="all",
        help="build features only, run XGB comparison only, or do both",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download and rebuild the cross-asset cache",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="training-row cap for the XGBoost comparison",
    )
    parser.add_argument(
        "--allow-partial-cross-asset",
        action="store_true",
        help="continue with local-only features if yfinance downloads are unavailable",
    )
    args = parser.parse_args()

    if args.mode == "build":
        run_build(
            force_download=args.force_download,
            allow_partial_cross_asset=args.allow_partial_cross_asset,
        )
    elif args.mode == "xgb":
        config = load_config()
        tickers = get_available_tickers()
        run_xgb_feature_comparison(tickers, config, threshold=args.threshold)
    else:
        run_all(
            force_download=args.force_download,
            threshold=args.threshold,
            allow_partial_cross_asset=args.allow_partial_cross_asset,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
