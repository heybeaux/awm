"""Phase 4 walk-forward backtest with portfolio construction.

The module combines the Phase 1-3 building blocks into a single pooled,
walk-forward benchmark:
  - calibrate base models on rolling out-of-fold predictions
  - compare two prior-residual fusion heads against a regime-weighted ensemble
  - select and size positions, then apply vol targeting, turnover control,
    and transaction costs
  - aggregate net performance metrics and write a JSON report
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ensemble import REGIME_ORDER, compute_fallback_weights
from features_v2 import get_available_tickers, get_full_feature_columns
from pipeline import load_config
from portfolio import (
    DEFAULT_COST_LEVELS,
    add_risk_estimates,
    apply_turnover_control,
    apply_vol_targeting,
    compute_transaction_costs,
    cost_sensitivity_analysis,
    penalized_sharpe,
    summarize_turnover,
)
from regime_detector import DATA_DIR, DEFAULT_RESULTS_DIR
from selection import add_logit_z_scores, select_top_k
from sizing import compute_edge, size_positions


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RESULTS_NAME = "backtest_v2.json"
DEFAULT_COST_BPS = 5.0
DEFAULT_TARGET_PORTFOLIO_VOL = 0.10
DEFAULT_OUTER_MIN_TRAIN_DAYS = 756
DEFAULT_OUTER_TEST_DAYS = 63
DEFAULT_INNER_MIN_TRAIN_DAYS = 504
DEFAULT_INNER_TEST_DAYS = 63
DEFAULT_K_VALUES: tuple[int, ...] = (3, 5, 7, 10)
DEFAULT_TURNOVER_PENALTY = 0.10
MIN_CALIBRATION_SAMPLES = 20
DEFAULT_ALPHA0 = 1.5
DEFAULT_BETA0 = 1.0


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_days: int
    test_days: int


@dataclass
class HeadSelectionResult:
    winner: str
    selected_k: int
    objective: float
    candidate_results: dict[str, Any]


class IdentityCalibrator:
    def predict(self, values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.clip(arr, 0.0, 1.0)


class ConstantProbModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        n = len(X)
        p = np.full(n, self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class OffsetLogisticModel:
    intercept: float
    coefficients: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca: PCA | None
    base_rate: float


def _log(msg: str) -> None:
    print(f"[backtest_v2] {msg}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        out = value.item()
        if isinstance(out, float) and (math.isnan(out) or math.isinf(out)):
            return None
        return out
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not serializable: {type(value)!r}")


def _save_checkpoint(path: Path, window_results: list[dict[str, Any]]) -> None:
    """Save partial window results to disk for resumption."""
    serializable = []
    for wr in window_results:
        entry = {}
        for k, v in wr.items():
            if isinstance(v, pd.DataFrame):
                entry[k] = v.to_dict(orient="records")
            else:
                entry[k] = v
        serializable.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"window_results": serializable}, f, default=_json_default)


def _safe_sigmoid(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))


def safe_logit(values: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _sharpe(daily_returns: np.ndarray | pd.Series) -> float:
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sigma = float(arr.std(ddof=1))
    if sigma <= 0.0:
        return float("nan")
    return float(arr.mean() / sigma * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    arr = np.asarray(equity_curve, dtype=float)
    if arr.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / np.where(running_max == 0.0, 1.0, running_max) - 1.0
    return float(np.nanmin(drawdown))


def _resolve_results_dir(preferred: Path | None = None) -> Path:
    candidates = [
        Path(preferred) if preferred is not None else DEFAULT_RESULTS_DIR,
        Path(__file__).resolve().parent / "results",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError("unable to resolve a writable results directory")


def _load_hyg_status(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "hyg_audit.json"
    if not path.exists():
        return {"available": False}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"available": True, "parse_error": True}

    return {
        "available": True,
        "all_passed": bool(data.get("summary", {}).get("all_tests_passed", False)),
        "reinstated": bool(data.get("summary", {}).get("reinstate_hyg", data.get("summary", {}).get("reinstated", False))),
    }


def _load_regime_posterior_frame(results_dir: Path) -> pd.DataFrame:
    parquet_path = results_dir / "regime_detector_SPY.parquet"
    if parquet_path.exists():
        posterior = pd.read_parquet(parquet_path).reset_index().rename(columns={"index": "date"})
        posterior["date"] = pd.to_datetime(posterior["date"])
        rename_map = {"regime": "market_regime"}
        return posterior.rename(columns=rename_map)

    from ensemble import build_spy_regime_posterior

    posterior = build_spy_regime_posterior(data_dir=DATA_DIR).posterior_frame.copy()
    posterior = posterior.rename(columns={"regime": "market_regime"})
    return posterior


def load_phase4_panel(
    tickers: Sequence[str] | None = None,
    *,
    results_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load the pooled panel and align embeddings row-wise where available."""
    available = list(tickers) if tickers is not None else get_available_tickers()
    if not available:
        raise FileNotFoundError(f"no feature parquet files found under {DATA_DIR}")

    sample = pd.read_parquet(DATA_DIR / f"{available[0]}_features.parquet").sort_index()
    feature_columns = get_full_feature_columns(sample)

    frames: list[pd.DataFrame] = []
    embedding_chunks: list[np.ndarray] = []
    embedding_offset = 0
    embedding_dim = 64

    for ticker in available:
        feature_path = DATA_DIR / f"{ticker}_features.parquet"
        if not feature_path.exists():
            continue

        df = pd.read_parquet(feature_path).sort_index().copy()
        if df.empty:
            continue
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        df["date"] = pd.to_datetime(df.index)
        df["ticker"] = ticker
        df["ret_5d_fwd"] = df["adj_close"].shift(-5) / df["adj_close"] - 1.0
        df["next_day_return"] = df["daily_return"].shift(-1)
        df["ticker_regime"] = df["regime"] if "regime" in df.columns else pd.NA

        emb_path = DATA_DIR / f"{ticker}_embeddings.npy"
        emb_idx = np.full(len(df), -1, dtype=np.int64)
        if emb_path.exists():
            emb = np.load(emb_path)
            if emb.ndim == 1:
                emb = emb[:, None]
            embedding_dim = int(emb.shape[1])
            usable = min(len(df), len(emb))
            if usable > 0:
                emb = emb[:usable].astype(np.float32)
                emb_idx[:usable] = np.arange(embedding_offset, embedding_offset + usable)
                embedding_chunks.append(emb)
                embedding_offset += usable
        df["_emb_row"] = emb_idx
        frames.append(df.reset_index(drop=True))

    if not frames:
        raise RuntimeError("no feature frames could be loaded")

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    embeddings = (
        np.concatenate(embedding_chunks, axis=0)
        if embedding_chunks
        else np.empty((0, embedding_dim), dtype=np.float32)
    )

    regime_results_dir = results_dir or _resolve_results_dir()
    posterior = _load_regime_posterior_frame(regime_results_dir)
    posterior_cols = ["date", "market_regime"] + [c for c in posterior.columns if c.startswith("pi_")]
    panel = panel.merge(
        posterior.loc[:, posterior_cols].drop_duplicates(subset=["date"]),
        on="date",
        how="left",
        validate="many_to_one",
    )
    if "market_regime" not in panel.columns:
        panel["market_regime"] = panel["ticker_regime"].astype("object")
    for regime in REGIME_ORDER:
        col = f"pi_{regime}"
        if col not in panel.columns:
            panel[col] = 1.0 / len(REGIME_ORDER)
    panel["market_regime"] = panel["market_regime"].fillna("neutral").astype(str)
    return panel, embeddings, feature_columns


def build_walk_forward_windows(
    dates: Sequence[pd.Timestamp],
    *,
    min_train_days: int,
    test_window_days: int,
    step_days: int | None = None,
) -> list[WalkForwardWindow]:
    """Create expanding walk-forward windows on a sorted unique date list."""
    unique_dates = pd.Index(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if len(unique_dates) <= min_train_days:
        return []

    step = int(step_days or test_window_days)
    windows: list[WalkForwardWindow] = []
    start = min_train_days
    while start < len(unique_dates):
        test_slice = unique_dates[start : start + test_window_days]
        if len(test_slice) == 0:
            break
        train_slice = unique_dates[:start]
        windows.append(
            WalkForwardWindow(
                train_start=pd.Timestamp(train_slice[0]),
                train_end=pd.Timestamp(train_slice[-1]),
                test_start=pd.Timestamp(test_slice[0]),
                test_end=pd.Timestamp(test_slice[-1]),
                train_days=int(len(train_slice)),
                test_days=int(len(test_slice)),
            )
        )
        start += step
    return windows


def _window_frame(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (panel["date"] >= start) & (panel["date"] <= end)
    return panel.loc[mask].copy()


def _make_xgb_classifier(
    *,
    monotone_constraints: str | None = None,
    seed: int = 123,
) -> xgb.XGBClassifier:
    kwargs: dict[str, Any] = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if monotone_constraints is not None:
        kwargs["monotone_constraints"] = monotone_constraints
    return xgb.XGBClassifier(**kwargs)


def _fit_awm_beliefs(
    train_df: pd.DataFrame,
    *,
    regime_col: str = "market_regime",
    target_col: str = "direction_5d",
    alpha0: float = DEFAULT_ALPHA0,
    beta0: float = DEFAULT_BETA0,
) -> dict[str, tuple[float, float]]:
    beliefs: dict[str, tuple[float, float]] = {}
    usable = train_df.dropna(subset=[regime_col, target_col])
    for regime, label in zip(usable[regime_col].astype(str), usable[target_col].astype(int), strict=False):
        alpha, beta = beliefs.get(regime, (alpha0, beta0))
        if int(label) == 1:
            alpha += 1.0
        else:
            beta += 1.0
        beliefs[regime] = (alpha, beta)
    return beliefs


def _predict_awm_from_beliefs(
    beliefs: Mapping[str, tuple[float, float]],
    regimes: pd.Series,
    *,
    alpha0: float = DEFAULT_ALPHA0,
    beta0: float = DEFAULT_BETA0,
) -> np.ndarray:
    probs = np.full(len(regimes), alpha0 / (alpha0 + beta0), dtype=float)
    for idx, regime in enumerate(regimes.astype(str).tolist()):
        alpha, beta = beliefs.get(regime, (alpha0, beta0))
        probs[idx] = alpha / (alpha + beta)
    return probs


def _gather_embeddings(indices: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.empty((len(indices), 0), dtype=np.float32)
    dim = embeddings.shape[1]
    out = np.zeros((len(indices), dim), dtype=np.float32)
    valid = indices >= 0
    if valid.any():
        out[valid] = embeddings[indices[valid]]
    return out


def fit_base_models(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Fit pooled XGBoost, pooled embedding logistic, and regime-prior AWM."""
    usable = train_df.dropna(subset=["direction_5d"]).copy()
    if usable.empty:
        raise RuntimeError("training slice has no usable labels")

    base_rate = float(np.clip(usable["direction_5d"].astype(float).mean(), 1e-6, 1.0 - 1e-6))

    xgb_model: Any
    if len(usable) < 100 or usable["direction_5d"].nunique() < 2:
        xgb_model = ConstantProbModel(base_rate)
    else:
        xgb_model = _make_xgb_classifier()
        X_xgb = usable.loc[:, list(feature_columns)].astype(np.float32)
        y = usable["direction_5d"].astype(int).to_numpy()
        xgb_model.fit(X_xgb, y, verbose=False)

    emb_usable = usable.loc[usable["_emb_row"] >= 0].copy()
    if len(emb_usable) < 100 or emb_usable["direction_5d"].nunique() < 2 or embeddings.size == 0:
        lewm_model: Any = ConstantProbModel(base_rate)
    else:
        lewm_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=123,
        )
        emb_matrix = _gather_embeddings(emb_usable["_emb_row"].to_numpy(dtype=int), embeddings)
        lewm_model.fit(emb_matrix, emb_usable["direction_5d"].astype(int).to_numpy())

    beliefs = _fit_awm_beliefs(usable)
    return {
        "xgb_model": xgb_model,
        "lewm_model": lewm_model,
        "awm_beliefs": beliefs,
        "base_rate": base_rate,
    }


def predict_base_models(
    models: Mapping[str, Any],
    df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Predict XGB, Le-WM, and AWM prior probabilities for a frame."""
    pred = df.copy()
    pred["xgb_prob"] = float(models["base_rate"])
    pred["lewm_prob"] = float(models["base_rate"])

    if len(pred):
        X_xgb = pred.loc[:, list(feature_columns)].astype(np.float32)
        pred["xgb_prob"] = np.asarray(models["xgb_model"].predict_proba(X_xgb))[:, 1]

        valid_emb = pred["_emb_row"].to_numpy(dtype=int) >= 0
        if valid_emb.any():
            emb_matrix = _gather_embeddings(pred.loc[valid_emb, "_emb_row"].to_numpy(dtype=int), embeddings)
            pred.loc[valid_emb, "lewm_prob"] = np.asarray(models["lewm_model"].predict_proba(emb_matrix))[:, 1]

    pred["awm_prob"] = _predict_awm_from_beliefs(models["awm_beliefs"], pred["market_regime"])
    return pred


def _fit_probability_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IdentityCalibrator | IsotonicRegression:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if valid.sum() < MIN_CALIBRATION_SAMPLES or np.unique(y[valid]).size < 2:
        return IdentityCalibrator()
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(p[valid], y[valid])
    return calibrator


def fit_base_calibrators(pred_df: pd.DataFrame) -> dict[str, IdentityCalibrator | IsotonicRegression]:
    y = pred_df["direction_5d"].astype(float).to_numpy()
    return {
        "xgb": _fit_probability_calibrator(y, pred_df["xgb_prob"].to_numpy()),
        "lewm": _fit_probability_calibrator(y, pred_df["lewm_prob"].to_numpy()),
        "awm": _fit_probability_calibrator(y, pred_df["awm_prob"].to_numpy()),
    }


def apply_base_calibrators(
    pred_df: pd.DataFrame,
    calibrators: Mapping[str, IdentityCalibrator | IsotonicRegression],
) -> pd.DataFrame:
    calibrated = pred_df.copy()
    for key, col in [("xgb", "xgb_prob"), ("lewm", "lewm_prob"), ("awm", "awm_prob")]:
        out_col = f"{key}_prob_calibrated"
        calibrated[out_col] = np.clip(calibrators[key].predict(calibrated[col].astype(float).to_numpy()), 0.0, 1.0)
        calibrated[f"{key}_logit"] = safe_logit(calibrated[out_col])
    return calibrated


def _build_embedding_pca(train_df: pd.DataFrame, embeddings: np.ndarray, n_components: int = 8) -> PCA | None:
    valid = train_df.loc[train_df["_emb_row"] >= 0, "_emb_row"].to_numpy(dtype=int)
    if embeddings.size == 0 or len(valid) < 20:
        return None
    emb_train = _gather_embeddings(valid, embeddings)
    comp = min(n_components, emb_train.shape[1], len(emb_train))
    if comp < 1:
        return None
    pca = PCA(n_components=comp, random_state=123)
    pca.fit(emb_train)
    return pca


def _build_head_extra_features(
    source_df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    pca: PCA | None,
) -> np.ndarray:
    tech = source_df.loc[:, list(feature_columns)].astype(np.float32).fillna(0.0).to_numpy()
    emb = _gather_embeddings(source_df["_emb_row"].to_numpy(dtype=int), embeddings)
    if pca is not None and emb.size:
        comps = pca.transform(emb)
    else:
        comps = np.zeros((len(source_df), 0), dtype=np.float32)
    return np.column_stack([comps, tech])


def fit_option_a_offset_logistic(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    *,
    offset_col: str = "awm_logit",
    target_col: str = "direction_5d",
    l2_penalty: float = 1e-3,
) -> OffsetLogisticModel | ConstantProbModel:
    usable = train_df.dropna(subset=[target_col, offset_col]).copy()
    if usable.empty or usable[target_col].nunique() < 2:
        base = float(np.clip(usable[target_col].mean() if len(usable) else 0.5, 1e-6, 1.0 - 1e-6))
        return ConstantProbModel(base)

    pca = _build_embedding_pca(usable, embeddings)
    extra = _build_head_extra_features(usable, embeddings, feature_columns, pca)
    scaler = StandardScaler()
    X = scaler.fit_transform(extra) if extra.shape[1] else np.zeros((len(usable), 0), dtype=float)
    offset = usable[offset_col].astype(float).to_numpy()
    y = usable[target_col].astype(float).to_numpy()

    def objective(params: np.ndarray) -> float:
        intercept = params[0]
        beta = params[1:]
        linear = offset + intercept
        if beta.size:
            linear = linear + X @ beta
        loss = np.mean(np.logaddexp(0.0, linear) - y * linear)
        ridge = l2_penalty * float(np.dot(beta, beta))
        return float(loss + ridge)

    init = np.zeros(X.shape[1] + 1, dtype=float)
    result = minimize(objective, init, method="L-BFGS-B")
    if not result.success or np.any(~np.isfinite(result.x)):
        base = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
        return ConstantProbModel(base)

    return OffsetLogisticModel(
        intercept=float(result.x[0]),
        coefficients=np.asarray(result.x[1:], dtype=float),
        scaler_mean=scaler.mean_ if extra.shape[1] else np.zeros(0, dtype=float),
        scaler_scale=scaler.scale_ if extra.shape[1] else np.ones(0, dtype=float),
        pca=pca,
        base_rate=float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6)),
    )


def predict_option_a(
    model: OffsetLogisticModel | ConstantProbModel,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    *,
    offset_col: str = "awm_logit",
) -> np.ndarray:
    if isinstance(model, ConstantProbModel):
        return model.predict_proba(np.zeros((len(df), 1), dtype=float))[:, 1]
    extra = _build_head_extra_features(df, embeddings, feature_columns, model.pca)
    if extra.shape[1]:
        centered = extra - model.scaler_mean
        X = centered / np.where(model.scaler_scale == 0.0, 1.0, model.scaler_scale)
        linear = df[offset_col].astype(float).to_numpy() + model.intercept + X @ model.coefficients
    else:
        linear = df[offset_col].astype(float).to_numpy() + model.intercept
    return _safe_sigmoid(linear)


def fit_option_b_xgboost(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    *,
    offset_col: str = "awm_logit",
    target_col: str = "direction_5d",
) -> dict[str, Any]:
    usable = train_df.dropna(subset=[target_col, offset_col]).copy()
    base_rate = float(np.clip(usable[target_col].mean() if len(usable) else 0.5, 1e-6, 1.0 - 1e-6))
    if usable.empty or usable[target_col].nunique() < 2:
        return {"model": ConstantProbModel(base_rate), "pca": None}

    pca = _build_embedding_pca(usable, embeddings)
    extra = _build_head_extra_features(usable, embeddings, feature_columns, pca)
    X = np.column_stack([usable[offset_col].astype(float).to_numpy(), extra]).astype(np.float32)
    y = usable[target_col].astype(int).to_numpy()
    constraints = "(" + ",".join(["1"] + ["0"] * (X.shape[1] - 1)) + ")"
    model = _make_xgb_classifier(monotone_constraints=constraints)
    model.fit(X, y, verbose=False)
    return {"model": model, "pca": pca}


def predict_option_b(
    bundle: Mapping[str, Any],
    df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    *,
    offset_col: str = "awm_logit",
) -> np.ndarray:
    model = bundle["model"]
    if isinstance(model, ConstantProbModel):
        return model.predict_proba(np.zeros((len(df), 1), dtype=float))[:, 1]
    extra = _build_head_extra_features(df, embeddings, feature_columns, bundle["pca"])
    X = np.column_stack([df[offset_col].astype(float).to_numpy(), extra]).astype(np.float32)
    return np.asarray(model.predict_proba(X))[:, 1]


def _optimize_regime_weights(
    train_df: pd.DataFrame,
    *,
    posterior_cols: Sequence[str],
    base_logit_cols: Sequence[str],
    target_col: str = "direction_5d",
    cap: float = 0.70,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if train_df.empty or train_df[target_col].nunique() < 2:
        return None, {"success": False, "message": "insufficient_samples"}

    posterior = train_df.loc[:, list(posterior_cols)].fillna(1.0 / len(posterior_cols)).to_numpy(dtype=float)
    logits = train_df.loc[:, list(base_logit_cols)].to_numpy(dtype=float)
    y = train_df[target_col].astype(float).to_numpy()
    n_regimes = posterior.shape[1]
    n_models = logits.shape[1]
    design = (posterior[:, :, None] * logits[:, None, :]).reshape(len(train_df), n_regimes * n_models)

    def objective(params: np.ndarray) -> float:
        linear = design @ params
        return float(np.mean(np.logaddexp(0.0, linear) - y * linear))

    bounds = [(0.0, cap)] * (n_regimes * n_models)
    constraints = [
        {
            "type": "ineq",
            "fun": lambda params, row=row: 1.0 - np.sum(params[row * n_models : (row + 1) * n_models]),
        }
        for row in range(n_regimes)
    ]
    init = np.tile(np.full(n_models, 1.0 / n_models), n_regimes)
    result = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success or np.any(~np.isfinite(result.x)):
        return None, {"success": False, "message": result.message}
    return result.x.reshape(n_regimes, n_models), {"success": True, "objective": float(result.fun)}


def fit_regime_ensemble_head(train_df: pd.DataFrame) -> dict[str, Any]:
    posterior_cols = [f"pi_{regime}" for regime in REGIME_ORDER]
    base_prob_cols = ["xgb_prob_calibrated", "lewm_prob_calibrated", "awm_prob_calibrated"]
    base_logit_cols = ["xgb_logit", "lewm_logit", "awm_logit"]

    weights, diagnostics = _optimize_regime_weights(
        train_df.dropna(subset=base_prob_cols + ["direction_5d"]),
        posterior_cols=posterior_cols,
        base_logit_cols=base_logit_cols,
    )

    if weights is None:
        fallback_input = train_df.copy()
        fallback_weights, sharpes = compute_fallback_weights(
            fallback_input,
            model_names=["XGB", "Le-WM", "AWM"],
            date_col="date",
            target_col="direction_5d",
            logit_columns={"XGB": "xgb_logit", "Le-WM": "lewm_logit", "AWM": "awm_logit"},
            model_return_columns=None,
            eta=2.0,
            lookback_days=60,
            cap=0.70,
        )
        weights = np.tile(fallback_weights, (len(REGIME_ORDER), 1))
        diagnostics = {
            "source": "fallback",
            "sharpes": sharpes,
            "message": diagnostics.get("message"),
        }
    else:
        diagnostics = {
            "source": "optimized",
            **diagnostics,
        }

    return {
        "weights": weights,
        "posterior_cols": posterior_cols,
        "base_logit_cols": base_logit_cols,
        "diagnostics": diagnostics,
    }


def predict_regime_ensemble(bundle: Mapping[str, Any], df: pd.DataFrame) -> np.ndarray:
    posterior = df.loc[:, list(bundle["posterior_cols"])].fillna(1.0 / len(bundle["posterior_cols"])).to_numpy(dtype=float)
    logits = df.loc[:, list(bundle["base_logit_cols"])].to_numpy(dtype=float)
    dynamic = posterior @ np.asarray(bundle["weights"], dtype=float)
    linear = np.sum(dynamic * logits, axis=1)
    return _safe_sigmoid(linear)


def generate_oof_predictions(
    train_df: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    *,
    min_train_days: int = DEFAULT_INNER_MIN_TRAIN_DAYS,
    test_window_days: int = DEFAULT_INNER_TEST_DAYS,
) -> pd.DataFrame:
    """Generate expanding-window OOF predictions inside one outer train slice."""
    dates = pd.Index(pd.to_datetime(train_df["date"].dropna().unique())).sort_values()
    windows = build_walk_forward_windows(
        dates,
        min_train_days=min(min_train_days, max(126, len(dates) // 2)),
        test_window_days=min(test_window_days, max(21, len(dates) // 4)),
    )

    if not windows:
        split_idx = max(126, int(len(dates) * 0.8))
        if split_idx >= len(dates):
            return pd.DataFrame()
        windows = [
            WalkForwardWindow(
                train_start=pd.Timestamp(dates[0]),
                train_end=pd.Timestamp(dates[split_idx - 1]),
                test_start=pd.Timestamp(dates[split_idx]),
                test_end=pd.Timestamp(dates[-1]),
                train_days=int(split_idx),
                test_days=int(len(dates) - split_idx),
            )
        ]

    blocks: list[pd.DataFrame] = []
    for window in windows:
        inner_train = _window_frame(train_df, window.train_start, window.train_end)
        inner_test = _window_frame(train_df, window.test_start, window.test_end)
        if inner_train.empty or inner_test.empty:
            continue
        models = fit_base_models(inner_train, embeddings, feature_columns)
        pred = predict_base_models(models, inner_test, embeddings, feature_columns)
        blocks.append(pred)

    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def attach_candidate_probabilities(
    frame: pd.DataFrame,
    *,
    option_a_model: OffsetLogisticModel | ConstantProbModel,
    option_b_model: Mapping[str, Any],
    regime_ensemble: Mapping[str, Any],
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    out = frame.copy()
    out["option_a_prob"] = predict_option_a(option_a_model, out, embeddings, feature_columns)
    out["option_b_prob"] = predict_option_b(option_b_model, out, embeddings, feature_columns)
    out["regime_ensemble_prob"] = predict_regime_ensemble(regime_ensemble, out)
    return out


def run_portfolio_execution(
    frame: pd.DataFrame,
    *,
    prob_col: str,
    k: int,
    cost_bps: float = DEFAULT_COST_BPS,
    target_portfolio_vol: float = DEFAULT_TARGET_PORTFOLIO_VOL,
    cost_levels: Sequence[float] = DEFAULT_COST_LEVELS,
) -> dict[str, Any]:
    """Select, size, vol-target, and execute a daily portfolio for one signal column."""
    working = frame.copy()
    working = add_logit_z_scores(
        working,
        prob_col=prob_col,
        ticker_col="ticker",
        date_col="date",
        window=252,
        min_periods=63,
        out_col="logit_z",
    )
    working = select_top_k(
        working,
        prob_col=prob_col,
        ticker_col="ticker",
        date_col="date",
        k=int(k),
        floor=0.50,
        out_col="selected_topk",
    )
    working = add_risk_estimates(
        working,
        ticker_col="ticker",
        date_col="date",
        daily_return_col="daily_return",
        price_col="adj_close",
    )
    working = size_positions(
        working,
        prob_col=prob_col,
        selection_col="selected_topk",
        ticker_col="ticker",
        date_col="date",
        daily_return_col="daily_return",
        price_col="adj_close",
        vol_col="ewma_vol_63d",
        lambda_fraction=0.25,
        max_weight=0.10,
        target_annual_vol=target_portfolio_vol,
        target_gross_exposure=1.0,
        long_only=True,
    )
    working = apply_vol_targeting(
        working,
        date_col="date",
        weight_col="target_weight",
        ewma_vol_col="ewma_vol_63d",
        realized_vol_col="realized_vol_21d",
        target_annual_vol=target_portfolio_vol,
        out_col="vol_target_weight",
    )
    working = apply_turnover_control(
        working,
        ticker_col="ticker",
        date_col="date",
        desired_weight_col="vol_target_weight",
        signal_z_col="logit_z",
        flip_threshold=0.25,
        out_col="executed_weight",
    )

    default_cost_col = f"transaction_cost_{float(cost_bps):g}"
    costed = compute_transaction_costs(
        working,
        ticker_col="ticker",
        date_col="date",
        executed_weight_col="executed_weight",
        previous_weight_col="previous_weight",
        cost_bps=float(cost_bps),
        out_col=default_cost_col,
    )
    for level in cost_levels:
        col = f"transaction_cost_{float(level):g}"
        if col == default_cost_col:
            continue
        tmp = compute_transaction_costs(
            costed,
            ticker_col="ticker",
            date_col="date",
            executed_weight_col="executed_weight",
            previous_weight_col="previous_weight",
            cost_bps=float(level),
            out_col=col,
        )
        costed[col] = tmp[col]

    costed["edge"] = compute_edge(costed[prob_col].astype(float).to_numpy())
    costed["gross_portfolio_return"] = costed["executed_weight"].astype(float) * costed["next_day_return"].fillna(0.0).astype(float)
    costed["active_position"] = costed["executed_weight"].abs() > 1e-12

    daily = (
        costed.groupby("date", as_index=False)
        .agg(
            gross_return=("gross_portfolio_return", "sum"),
            net_cost=(default_cost_col, "sum"),
            turnover=("turnover", "sum"),
            avg_edge=("edge", lambda s: float(np.nanmean(np.abs(s))) if len(s) else float("nan")),
            active_names=("active_position", "sum"),
            market_regime=("market_regime", "first"),
        )
        .sort_values("date")
    )
    daily["turnover"] = daily["turnover"] / 2.0
    daily["net_return"] = daily["gross_return"] - daily["net_cost"]
    equity = np.cumprod(1.0 + daily["net_return"].to_numpy(dtype=float)) if len(daily) else np.array([1.0])

    trade_mask = costed["is_new_trade"] | costed["is_flip_trade"]
    hit_mask = trade_mask & costed["next_day_return"].notna()
    hit_rate = float(
        (
            np.sign(costed.loc[hit_mask, "executed_weight"].astype(float).to_numpy())
            * costed.loc[hit_mask, "next_day_return"].astype(float).to_numpy()
            > 0.0
        ).mean()
    ) if hit_mask.any() else float("nan")

    metrics = {
        "net_sharpe": _sharpe(daily["net_return"].to_numpy(dtype=float)),
        "compound_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": _max_drawdown(equity),
        "trade_count": int(trade_mask.sum()),
        "hit_rate": hit_rate,
        "avg_edge": float(costed.loc[costed["active_position"], "edge"].abs().mean()) if costed["active_position"].any() else 0.0,
        "avg_daily_turnover": float(daily["turnover"].mean()) if len(daily) else 0.0,
        "turnover_summary": summarize_turnover(costed, date_col="date", turnover_col="turnover"),
    }

    by_regime: dict[str, Any] = {}
    for regime, group in daily.groupby("market_regime", dropna=False):
        curve = np.cumprod(1.0 + group["net_return"].to_numpy(dtype=float))
        by_regime[str(regime)] = {
            "days": int(len(group)),
            "net_sharpe": _sharpe(group["net_return"].to_numpy(dtype=float)),
            "compound_return": float(curve[-1] - 1.0) if len(curve) else 0.0,
        }

    return {
        "metrics": metrics,
        "daily": daily,
        "positioned": costed,
        "by_regime": by_regime,
        "cost_sensitivity": cost_sensitivity_analysis(
            costed,
            gross_return_col="gross_portfolio_return",
            date_col="date",
            cost_levels=cost_levels,
        ),
    }


def choose_candidate_head(
    train_like_df: pd.DataFrame,
    *,
    candidate_prob_cols: Mapping[str, str],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    cost_bps: float = DEFAULT_COST_BPS,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> HeadSelectionResult:
    candidate_results: dict[str, Any] = {}
    best_name = ""
    best_k = int(k_values[0])
    best_objective = -float("inf")

    for name, prob_col in candidate_prob_cols.items():
        best_local: dict[str, Any] | None = None
        for k in k_values:
            execution = run_portfolio_execution(
                train_like_df,
                prob_col=prob_col,
                k=int(k),
                cost_bps=cost_bps,
            )
            daily = execution["daily"]
            score = penalized_sharpe(
                daily["net_return"].to_numpy(dtype=float),
                daily["turnover"].to_numpy(dtype=float),
                turnover_penalty=turnover_penalty,
            )
            record = {
                "k": int(k),
                "objective": score,
                **execution["metrics"],
            }
            if best_local is None or (not math.isnan(score) and score > best_local["objective"]):
                best_local = record
        candidate_results[name] = best_local
        if best_local is not None and best_local["objective"] > best_objective:
            best_name = name
            best_k = int(best_local["k"])
            best_objective = float(best_local["objective"])

    return HeadSelectionResult(
        winner=best_name,
        selected_k=best_k,
        objective=best_objective,
        candidate_results=candidate_results,
    )


def _prepare_head_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["awm_logit"] = safe_logit(out["awm_prob_calibrated"])
    out["xgb_logit"] = safe_logit(out["xgb_prob_calibrated"])
    out["lewm_logit"] = safe_logit(out["lewm_prob_calibrated"])
    return out


def run_single_window(
    panel: pd.DataFrame,
    embeddings: np.ndarray,
    feature_columns: Sequence[str],
    window: WalkForwardWindow,
    *,
    cost_bps: float = DEFAULT_COST_BPS,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> dict[str, Any]:
    train_df = _window_frame(panel, window.train_start, window.train_end)
    test_df = _window_frame(panel, window.test_start, window.test_end)
    if train_df.empty or test_df.empty:
        return {"skipped": True, "reason": "empty_window"}

    oof = generate_oof_predictions(train_df, embeddings, feature_columns)
    if oof.empty:
        return {"skipped": True, "reason": "no_oof"}

    calibrators = fit_base_calibrators(oof)
    oof_cal = _prepare_head_training_frame(apply_base_calibrators(oof, calibrators))

    option_a_oof_model = fit_option_a_offset_logistic(oof_cal, embeddings, feature_columns)
    option_b_oof_model = fit_option_b_xgboost(oof_cal, embeddings, feature_columns)
    regime_oof_head = fit_regime_ensemble_head(oof_cal)
    oof_candidates = attach_candidate_probabilities(
        oof_cal,
        option_a_model=option_a_oof_model,
        option_b_model=option_b_oof_model,
        regime_ensemble=regime_oof_head,
        embeddings=embeddings,
        feature_columns=feature_columns,
    )

    selection = choose_candidate_head(
        oof_candidates,
        candidate_prob_cols={
            "option_a": "option_a_prob",
            "option_b": "option_b_prob",
            "regime_ensemble": "regime_ensemble_prob",
        },
        cost_bps=cost_bps,
        turnover_penalty=turnover_penalty,
    )

    full_models = fit_base_models(train_df, embeddings, feature_columns)
    full_train_pred = _prepare_head_training_frame(
        apply_base_calibrators(
            predict_base_models(full_models, train_df, embeddings, feature_columns),
            calibrators,
        )
    )
    test_pred = _prepare_head_training_frame(
        apply_base_calibrators(
            predict_base_models(full_models, test_df, embeddings, feature_columns),
            calibrators,
        )
    )

    option_a_model = fit_option_a_offset_logistic(full_train_pred, embeddings, feature_columns)
    option_b_model = fit_option_b_xgboost(full_train_pred, embeddings, feature_columns)
    regime_head = fit_regime_ensemble_head(full_train_pred)

    history_dates = pd.Index(pd.to_datetime(full_train_pred["date"].dropna().unique())).sort_values()
    history_tail = history_dates[-252:] if len(history_dates) > 252 else history_dates
    history_frame = full_train_pred.loc[full_train_pred["date"].isin(history_tail)].copy()
    history_candidates = attach_candidate_probabilities(
        history_frame,
        option_a_model=option_a_model,
        option_b_model=option_b_model,
        regime_ensemble=regime_head,
        embeddings=embeddings,
        feature_columns=feature_columns,
    )
    test_candidates = attach_candidate_probabilities(
        test_pred,
        option_a_model=option_a_model,
        option_b_model=option_b_model,
        regime_ensemble=regime_head,
        embeddings=embeddings,
        feature_columns=feature_columns,
    )
    execution_input = (
        pd.concat([history_candidates, test_candidates], ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    candidate_col = {
        "option_a": "option_a_prob",
        "option_b": "option_b_prob",
        "regime_ensemble": "regime_ensemble_prob",
    }[selection.winner]
    execution = run_portfolio_execution(
        execution_input,
        prob_col=candidate_col,
        k=selection.selected_k,
        cost_bps=cost_bps,
    )

    eval_daily = execution["daily"].loc[
        (execution["daily"]["date"] >= window.test_start) & (execution["daily"]["date"] <= window.test_end)
    ].reset_index(drop=True)
    eval_positioned = execution["positioned"].loc[
        (execution["positioned"]["date"] >= window.test_start) & (execution["positioned"]["date"] <= window.test_end)
    ].reset_index(drop=True)

    equity = np.cumprod(1.0 + eval_daily["net_return"].to_numpy(dtype=float)) if len(eval_daily) else np.array([1.0])
    by_regime: dict[str, Any] = {}
    for regime, group in eval_daily.groupby("market_regime", dropna=False):
        curve = np.cumprod(1.0 + group["net_return"].to_numpy(dtype=float))
        by_regime[str(regime)] = {
            "days": int(len(group)),
            "net_sharpe": _sharpe(group["net_return"].to_numpy(dtype=float)),
            "compound_return": float(curve[-1] - 1.0) if len(curve) else 0.0,
        }

    trade_mask = eval_positioned["is_new_trade"] | eval_positioned["is_flip_trade"]
    hit_mask = trade_mask & eval_positioned["next_day_return"].notna()
    hit_rate = float(
        (
            np.sign(eval_positioned.loc[hit_mask, "executed_weight"].astype(float).to_numpy())
            * eval_positioned.loc[hit_mask, "next_day_return"].astype(float).to_numpy()
            > 0.0
        ).mean()
    ) if hit_mask.any() else float("nan")

    cost_levels = {
        float(item["cost_bps"]): item
        for item in cost_sensitivity_analysis(
            eval_positioned,
            gross_return_col="gross_portfolio_return",
            date_col="date",
            cost_levels=DEFAULT_COST_LEVELS,
        )
    }
    window_summary = {
        "window": {
            "train_start": window.train_start,
            "train_end": window.train_end,
            "test_start": window.test_start,
            "test_end": window.test_end,
            "train_days": window.train_days,
            "test_days": window.test_days,
        },
        "winner": selection.winner,
        "selected_k": selection.selected_k,
        "candidate_objectives": selection.candidate_results,
        "ensemble_diagnostics": regime_head["diagnostics"],
        "metrics": {
            "net_sharpe": _sharpe(eval_daily["net_return"].to_numpy(dtype=float)),
            "compound_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
            "max_drawdown": _max_drawdown(equity),
            "trade_count": int(trade_mask.sum()),
            "hit_rate": hit_rate,
            "avg_edge": float(eval_positioned.loc[eval_positioned["active_position"], "edge"].abs().mean()) if eval_positioned["active_position"].any() else 0.0,
            "avg_daily_turnover": float(eval_daily["turnover"].mean()) if len(eval_daily) else 0.0,
        },
        "by_regime": by_regime,
        "cost_sensitivity": [cost_levels[level] for level in sorted(cost_levels)],
        "daily": eval_daily,
    }
    return window_summary


def aggregate_backtest_results(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [result for result in window_results if not result.get("skipped")]
    if not valid:
        return {
            "window_count": 0,
            "mean_window_sharpe": float("nan"),
            "median_window_sharpe": float("nan"),
            "worst_window_drawdown": float("nan"),
            "total_return": 0.0,
            "by_regime": {},
        }

    daily = pd.concat([result["daily"] for result in valid], ignore_index=True).sort_values("date").reset_index(drop=True)
    equity = np.cumprod(1.0 + daily["net_return"].to_numpy(dtype=float)) if len(daily) else np.array([1.0])
    sharpes = [result["metrics"]["net_sharpe"] for result in valid]
    drawdowns = [result["metrics"]["max_drawdown"] for result in valid]

    by_regime: dict[str, Any] = {}
    for regime, group in daily.groupby("market_regime", dropna=False):
        curve = np.cumprod(1.0 + group["net_return"].to_numpy(dtype=float))
        by_regime[str(regime)] = {
            "days": int(len(group)),
            "net_sharpe": _sharpe(group["net_return"].to_numpy(dtype=float)),
            "compound_return": float(curve[-1] - 1.0) if len(curve) else 0.0,
        }

    winner_counts = pd.Series([result["winner"] for result in valid]).value_counts().to_dict()
    return {
        "window_count": int(len(valid)),
        "mean_window_sharpe": float(np.nanmean(sharpes)),
        "median_window_sharpe": float(np.nanmedian(sharpes)),
        "worst_window_drawdown": float(np.nanmin(drawdowns)),
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "by_regime": by_regime,
        "winner_counts": {str(k): int(v) for k, v in winner_counts.items()},
    }


def run_backtest_v2(
    *,
    tickers: Sequence[str] | None = None,
    outer_min_train_days: int = DEFAULT_OUTER_MIN_TRAIN_DAYS,
    outer_test_days: int = DEFAULT_OUTER_TEST_DAYS,
    cost_bps: float = DEFAULT_COST_BPS,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
    output_path: Path | None = None,
) -> dict[str, Any]:
    results_dir = _resolve_results_dir(output_path.parent if output_path is not None else None)
    panel, embeddings, feature_columns = load_phase4_panel(tickers, results_dir=results_dir)
    dates = pd.Index(pd.to_datetime(panel["date"].dropna().unique())).sort_values()
    windows = build_walk_forward_windows(
        dates,
        min_train_days=outer_min_train_days,
        test_window_days=outer_test_days,
    )
    _log(f"loaded {panel['ticker'].nunique()} tickers, {len(panel):,} rows, {len(windows)} outer windows")

    # --- Checkpoint support: resume from partial run ---
    checkpoint_path = (results_dir / "backtest_v2_checkpoint.json")
    window_results: list[dict[str, Any]] = []
    start_idx = 1
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as _cp:
                cp_data = json.load(_cp)
            window_results = cp_data.get("window_results", [])
            start_idx = len(window_results) + 1
            _log(f"RESUMED from checkpoint: {len(window_results)} windows already done, starting at {start_idx}")
        except Exception as e:
            _log(f"checkpoint load failed ({e}), starting fresh")
            window_results = []
            start_idx = 1

    for idx, window in enumerate(windows, start=1):
        if idx < start_idx:
            continue
        _log(f"window {idx}/{len(windows)}: {window.test_start.date()} -> {window.test_end.date()}")
        result = run_single_window(
            panel,
            embeddings,
            feature_columns,
            window,
            cost_bps=cost_bps,
            turnover_penalty=turnover_penalty,
        )
        result["window_index"] = idx
        window_results.append(result)

        # Save checkpoint every 5 windows
        if idx % 5 == 0:
            _save_checkpoint(checkpoint_path, window_results)
            _log(f"checkpoint saved ({idx}/{len(windows)})")

    # Final checkpoint before aggregation
    _save_checkpoint(checkpoint_path, window_results)

    aggregate = aggregate_backtest_results(window_results)
    output = {
        "meta": {
            "tickers": sorted(panel["ticker"].dropna().unique().tolist()),
            "n_tickers": int(panel["ticker"].nunique()),
            "n_rows": int(len(panel)),
            "outer_min_train_days": int(outer_min_train_days),
            "outer_test_days": int(outer_test_days),
            "cost_bps": float(cost_bps),
            "cost_levels": list(DEFAULT_COST_LEVELS),
            "turnover_penalty": float(turnover_penalty),
            "target_portfolio_vol": DEFAULT_TARGET_PORTFOLIO_VOL,
            "hyg_status": _load_hyg_status(results_dir),
        },
        "aggregate": aggregate,
        "windows": [
            {
                **{k: v for k, v in result.items() if k != "daily"},
                "daily": result["daily"].to_dict(orient="records") if isinstance(result.get("daily"), pd.DataFrame) else result.get("daily"),
            }
            for result in window_results
        ],
    }

    final_path = output_path or (results_dir / DEFAULT_RESULTS_NAME)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "w") as handle:
        json.dump(output, handle, indent=2, default=_json_default)
    _log(f"wrote {final_path}")

    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        _log("checkpoint removed (run complete)")

    return output


def generate_daily_signals(
    signal_date: str | pd.Timestamp,
    *,
    tickers: Sequence[str] | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    turnover_penalty: float = DEFAULT_TURNOVER_PENALTY,
) -> dict[str, Any]:
    """Train up to `signal_date` and return a paper-trading signal slate."""
    signal_ts = pd.Timestamp(signal_date).normalize()
    results_dir = _resolve_results_dir()
    panel, embeddings, feature_columns = load_phase4_panel(tickers, results_dir=results_dir)
    if signal_ts not in set(pd.to_datetime(panel["date"])):
        raise ValueError(f"signal date {signal_ts.date()} not present in the feature store")

    train_df = panel.loc[panel["date"] < signal_ts].copy()
    day_df = panel.loc[panel["date"] == signal_ts].copy()
    if train_df.empty or day_df.empty:
        raise RuntimeError("insufficient data for signal generation")

    oof = generate_oof_predictions(train_df, embeddings, feature_columns)
    if oof.empty:
        raise RuntimeError("could not generate OOF predictions for signal selection")

    calibrators = fit_base_calibrators(oof)
    oof_cal = _prepare_head_training_frame(apply_base_calibrators(oof, calibrators))
    selection = choose_candidate_head(
        attach_candidate_probabilities(
            oof_cal,
            option_a_model=fit_option_a_offset_logistic(oof_cal, embeddings, feature_columns),
            option_b_model=fit_option_b_xgboost(oof_cal, embeddings, feature_columns),
            regime_ensemble=fit_regime_ensemble_head(oof_cal),
            embeddings=embeddings,
            feature_columns=feature_columns,
        ),
        candidate_prob_cols={
            "option_a": "option_a_prob",
            "option_b": "option_b_prob",
            "regime_ensemble": "regime_ensemble_prob",
        },
        cost_bps=cost_bps,
        turnover_penalty=turnover_penalty,
    )

    full_models = fit_base_models(train_df, embeddings, feature_columns)
    full_train_pred = _prepare_head_training_frame(
        apply_base_calibrators(
            predict_base_models(full_models, train_df, embeddings, feature_columns),
            calibrators,
        )
    )
    history_dates = pd.Index(pd.to_datetime(full_train_pred["date"].dropna().unique())).sort_values()
    history_tail = history_dates[-252:] if len(history_dates) > 252 else history_dates
    history_frame = full_train_pred.loc[full_train_pred["date"].isin(history_tail)].copy()
    day_pred = _prepare_head_training_frame(
        apply_base_calibrators(
            predict_base_models(full_models, day_df, embeddings, feature_columns),
            calibrators,
        )
    )

    option_a_model = fit_option_a_offset_logistic(full_train_pred, embeddings, feature_columns)
    option_b_model = fit_option_b_xgboost(full_train_pred, embeddings, feature_columns)
    regime_head = fit_regime_ensemble_head(full_train_pred)

    history_candidates = attach_candidate_probabilities(
        history_frame,
        option_a_model=option_a_model,
        option_b_model=option_b_model,
        regime_ensemble=regime_head,
        embeddings=embeddings,
        feature_columns=feature_columns,
    )
    day_candidates = attach_candidate_probabilities(
        day_pred,
        option_a_model=option_a_model,
        option_b_model=option_b_model,
        regime_ensemble=regime_head,
        embeddings=embeddings,
        feature_columns=feature_columns,
    )

    combined = pd.concat([history_candidates, day_candidates], ignore_index=True).sort_values(["date", "ticker"])
    execution = run_portfolio_execution(
        combined,
        prob_col={
            "option_a": "option_a_prob",
            "option_b": "option_b_prob",
            "regime_ensemble": "regime_ensemble_prob",
        }[selection.winner],
        k=selection.selected_k,
        cost_bps=cost_bps,
    )
    signal_rows = execution["positioned"].loc[execution["positioned"]["date"] == signal_ts].copy()
    signal_rows = signal_rows.loc[signal_rows["executed_weight"].abs() > 1e-12].sort_values("executed_weight", ascending=False)

    signal_prob_col = {
        "option_a": "option_a_prob",
        "option_b": "option_b_prob",
        "regime_ensemble": "regime_ensemble_prob",
    }[selection.winner]
    signals = [
        {
            "date": signal_ts.strftime("%Y-%m-%d"),
            "ticker": row.ticker,
            "direction": "long" if float(row.executed_weight) >= 0.0 else "short",
            "confidence": float(getattr(row, signal_prob_col)),
            "position_size": float(abs(row.executed_weight)),
        }
        for row in signal_rows.itertuples()
    ]

    return {
        "meta": {
            "date": signal_ts.strftime("%Y-%m-%d"),
            "winner": selection.winner,
            "selected_k": selection.selected_k,
            "cost_bps": float(cost_bps),
            "turnover_penalty": float(turnover_penalty),
        },
        "signals": signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 walk-forward benchmark with portfolio layer")
    parser.add_argument("--outer-min-train-days", type=int, default=DEFAULT_OUTER_MIN_TRAIN_DAYS)
    parser.add_argument("--outer-test-days", type=int, default=DEFAULT_OUTER_TEST_DAYS)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--turnover-penalty", type=float, default=DEFAULT_TURNOVER_PENALTY)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output).expanduser() if args.output else None
    run_backtest_v2(
        outer_min_train_days=args.outer_min_train_days,
        outer_test_days=args.outer_test_days,
        cost_bps=args.cost_bps,
        turnover_penalty=args.turnover_penalty,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
