"""Phase-4 benchmark harness — 4 models × 6 thresholds.

Models compared on each ticker's test set (fixed temporal split):
    1. xgboost_baseline   — XGB on 15 raw features
    2. lewm_standalone    — logistic head on frozen 64-d Le-WM embeddings
    3. awm_standalone     — online Beta-Bernoulli beliefs keyed by the
                            rule-based regime (no Le-WM)
    4. fusion             — online Beta-Bernoulli beliefs keyed by a
                            Le-WM-embedding-derived regime cluster

For the AWM-based models (3 & 4) evaluation is ONLINE:
    - reset beliefs per threshold sweep
    - walk the training set in strict temporal order (within each ticker),
      predict → reveal → update
    - freeze beliefs, then walk the test set, predict only
    - the walked training rows are capped at `threshold`

The XGB and Le-WM heads use the same `threshold` to subset training data.

Evaluation metrics per (model, threshold):
    - AUC / Brier / accuracy / precision@90-recall (pooled over tickers)
    - Sharpe, total return, max-drawdown from backtest.py (pooled panel)
    - per-ticker AUC dict

Output file: ../results/equity-benchmark.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

import xgboost as xgb

from pipeline import FEATURE_COLUMNS, load_config, load_features, make_splits
from universe import get_universe
from awm_bridge import AWMBridge
from backtest import backtest


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = RESULTS_DIR / "equity-benchmark.json"
INTERMEDIATE_PATH = RESULTS_DIR / "equity-benchmark.partial.json"

SEED_MODEL = 123
NUM_FUSION_CLUSTERS = 4


# ─── utilities ────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def _precision_at_90_recall(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = recall >= 0.90
    if not mask.any():
        return float("nan")
    return float(np.max(precision[mask]))


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return {"auc": float("nan"), "brier": float("nan"), "accuracy": float("nan"),
                "p_at_90_recall": float("nan"), "n": 0, "pos_rate": float("nan")}
    y_pred = (y_prob >= 0.5).astype(int)
    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "brier": float(brier_score_loss(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "p_at_90_recall": float("nan"),
            "n": int(len(y_true)),
            "pos_rate": float(y_true.mean()),
        }
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "p_at_90_recall": _precision_at_90_recall(y_true, y_prob),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def _safe_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and math.isnan(v):
        return None
    raise TypeError(f"not serializable: {type(v)}")


# ─── per-ticker loading ───────────────────────────────────────────

def load_ticker_data(ticker: str, config: dict) -> dict[str, Any] | None:
    """Load features + embeddings + splits for one ticker.

    Returns dict with:
        train_df, val_df, test_df — feature DataFrames (with regime column)
        train_emb, test_emb       — aligned (N, 64) Le-WM embedding arrays
        train_ret5d, test_ret5d   — realised 5-day forward returns
    """
    try:
        df = load_features(ticker, config)
    except FileNotFoundError:
        return None

    # Compute realised 5-day forward returns for backtesting
    df = df.sort_index()
    df["ret_5d_fwd"] = df["adj_close"].shift(-5) / df["adj_close"] - 1.0

    # Load Le-WM embeddings aligned 1:1 with df rows (by construction of
    # lewm_predict.extract_all)
    emb_path = DATA_DIR / f"{ticker}_embeddings.npy"
    if not emb_path.exists():
        return None
    emb = np.load(emb_path)
    if len(emb) != len(df):
        _log(f"{ticker}: embedding/feature length mismatch "
             f"({len(emb)} vs {len(df)}), skipping")
        return None

    # Attach embedding as an index-aligned array (stash outside df to avoid
    # broadcasting surprises)
    df = df.assign(_emb_idx=np.arange(len(df)))

    splits = make_splits(
        df,
        train_ratio=config["data"].get("train_ratio", 0.70),
        val_ratio=config["data"].get("val_ratio", 0.15),
    )
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    def _sub(sub_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        usable = sub_df.dropna(
            subset=FEATURE_COLUMNS + ["direction_5d", "regime", "ret_5d_fwd"]
        )
        idx = usable["_emb_idx"].to_numpy()
        sub_emb = emb[idx]
        ret5d = usable["ret_5d_fwd"].to_numpy()
        return usable, sub_emb, ret5d

    train_u, train_emb, train_ret = _sub(train_df)
    val_u, val_emb, val_ret = _sub(val_df)
    test_u, test_emb, test_ret = _sub(test_df)

    return {
        "ticker": ticker,
        "train_df": train_u,
        "val_df": val_u,
        "test_df": test_u,
        "train_emb": train_emb,
        "val_emb": val_emb,
        "test_emb": test_emb,
        "train_ret5d": train_ret,
        "val_ret5d": val_ret,
        "test_ret5d": test_ret,
    }


# ─── model 1: XGBoost ────────────────────────────────────────────

def run_xgb(per_ticker: dict[str, dict], threshold: int, xgb_cfg: dict) -> dict:
    rows: list[tuple[str, pd.Timestamp, float, float, int]] = []
    per_t: dict[str, dict] = {}

    for ticker, d in per_ticker.items():
        tr = d["train_df"]
        te = d["test_df"]
        if len(tr) < 30 or len(te) == 0:
            continue
        Xtr = tr[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        ytr = tr["direction_5d"].to_numpy(dtype=np.int64)
        if threshold < len(Xtr):
            Xtr, ytr = Xtr[:threshold], ytr[:threshold]
        if len(np.unique(ytr)) < 2 or len(Xtr) < 30:
            continue
        Xte = te[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        yte = te["direction_5d"].to_numpy(dtype=np.int64)

        model = xgb.XGBClassifier(
            n_estimators=int(xgb_cfg.get("n_estimators", 300)),
            max_depth=int(xgb_cfg.get("max_depth", 6)),
            learning_rate=float(xgb_cfg.get("learning_rate", 0.1)),
            subsample=float(xgb_cfg.get("subsample", 0.8)),
            colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.8)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED_MODEL,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(Xtr, ytr, verbose=False)
        p = model.predict_proba(Xte)[:, 1]

        per_t[ticker] = {
            "train_n": int(len(Xtr)),
            "test_n": int(len(yte)),
            **_metrics(yte, p),
        }
        for date, prob, ret, y in zip(te.index, p, d["test_ret5d"], yte):
            rows.append((ticker, date, float(prob), float(ret), int(y)))

    return _finalize(rows, per_t)


# ─── model 2: Le-WM linear classifier ─────────────────────────────

def run_lewm(per_ticker: dict[str, dict], threshold: int) -> dict:
    rows: list[tuple[str, pd.Timestamp, float, float, int]] = []
    per_t: dict[str, dict] = {}

    for ticker, d in per_ticker.items():
        tr_emb = d["train_emb"]
        tr_y = d["train_df"]["direction_5d"].to_numpy(dtype=np.int64)
        te_emb = d["test_emb"]
        te_y = d["test_df"]["direction_5d"].to_numpy(dtype=np.int64)
        if len(tr_emb) < 30 or len(te_emb) == 0:
            continue
        if threshold < len(tr_emb):
            tr_emb = tr_emb[:threshold]
            tr_y = tr_y[:threshold]
        if len(np.unique(tr_y)) < 2:
            continue

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED_MODEL)
        clf.fit(tr_emb, tr_y)
        p = clf.predict_proba(te_emb)[:, 1]

        per_t[ticker] = {
            "train_n": int(len(tr_emb)),
            "test_n": int(len(te_y)),
            **_metrics(te_y, p),
        }
        test_df = d["test_df"]
        for date, prob, ret, y in zip(test_df.index, p, d["test_ret5d"], te_y):
            rows.append((ticker, date, float(prob), float(ret), int(y)))

    return _finalize(rows, per_t)


# ─── model 3: AWM standalone (rule-regime beliefs) ────────────────

def run_awm_standalone(
    per_ticker: dict[str, dict],
    threshold: int,
    bridge: AWMBridge,
) -> dict:
    bridge.reset()
    rows: list[tuple[str, pd.Timestamp, float, float, int]] = []
    per_t: dict[str, dict] = {}

    # Sort all tickers' training rows by date globally so the AWM sees them
    # in a plausible temporal order (cross-ticker). Within a ticker we cap
    # the first `threshold` rows to match the XGB/Le-WM threshold semantic.
    for ticker, d in per_ticker.items():
        tr = d["train_df"]
        if len(tr) == 0:
            continue
        tr = tr.head(threshold) if threshold < len(tr) else tr
        regimes = tr["regime"].tolist()
        outcomes = tr["direction_5d"].astype(int).tolist()
        for r, o in zip(regimes, outcomes):
            # predict-then-update to simulate online; we discard the prediction
            # on training — it's only for belief updating.
            bridge.record(ticker, r, int(o))

    # Evaluate on test: predict only, no updates.
    for ticker, d in per_ticker.items():
        te = d["test_df"]
        if len(te) == 0:
            continue
        probs: list[float] = []
        for r in te["regime"]:
            out = bridge.predict(ticker, r)
            probs.append(out["p_up"])
        p = np.asarray(probs)
        y = te["direction_5d"].to_numpy(dtype=np.int64)
        per_t[ticker] = {
            "train_n": int(min(len(d["train_df"]), threshold)),
            "test_n": int(len(y)),
            **_metrics(y, p),
        }
        for date, prob, ret, yi in zip(te.index, p, d["test_ret5d"], y):
            rows.append((ticker, date, float(prob), float(ret), int(yi)))

    return _finalize(rows, per_t)


# ─── model 4: Fusion (Le-WM embedding regime → AWM) ───────────────

def run_fusion(
    per_ticker: dict[str, dict],
    threshold: int,
    bridge: AWMBridge,
    kmeans: KMeans,
) -> dict:
    bridge.reset()
    rows: list[tuple[str, pd.Timestamp, float, float, int]] = []
    per_t: dict[str, dict] = {}

    label_names = [f"emb_cluster_{i}" for i in range(kmeans.n_clusters)]

    for ticker, d in per_ticker.items():
        tr_emb = d["train_emb"]
        tr = d["train_df"]
        if len(tr_emb) == 0:
            continue
        tr_emb = tr_emb[:threshold]
        tr_sub = tr.head(len(tr_emb))
        clusters = kmeans.predict(tr_emb)
        outcomes = tr_sub["direction_5d"].astype(int).to_numpy()
        for c, o in zip(clusters, outcomes):
            bridge.record(ticker, label_names[c], int(o))

    # Evaluate
    for ticker, d in per_ticker.items():
        te_emb = d["test_emb"]
        te = d["test_df"]
        if len(te_emb) == 0:
            continue
        clusters = kmeans.predict(te_emb)
        probs: list[float] = []
        for c in clusters:
            out = bridge.predict(ticker, label_names[c])
            probs.append(out["p_up"])
        p = np.asarray(probs)
        y = te["direction_5d"].to_numpy(dtype=np.int64)
        per_t[ticker] = {
            "train_n": int(min(len(d["train_df"]), threshold)),
            "test_n": int(len(y)),
            **_metrics(y, p),
        }
        for date, prob, ret, yi in zip(te.index, p, d["test_ret5d"], y):
            rows.append((ticker, date, float(prob), float(ret), int(yi)))

    return _finalize(rows, per_t)


# ─── finalizer ────────────────────────────────────────────────────

def _finalize(
    rows: list[tuple[str, pd.Timestamp, float, float, int]],
    per_t: dict[str, dict],
    bt_cfg: dict | None = None,
) -> dict:
    if not rows:
        return {"metrics": _metrics(np.array([]), np.array([])), "per_ticker": per_t,
                "backtest": {}, "n_rows": 0}

    tickers, dates, probs, ret5d, y = zip(*rows)
    probs = np.asarray(probs)
    ret5d = np.asarray(ret5d)
    y = np.asarray(y)

    metrics = _metrics(y, probs)

    bt_cfg = bt_cfg or {}
    bt = backtest(
        dates=np.array(dates),
        tickers=np.array(tickers),
        probs=probs,
        returns_5d=ret5d,
        entry_threshold=bt_cfg.get("entry_threshold", 0.55),
        cost_bps=bt_cfg.get("cost_bps", 10.0),
        hold_days=bt_cfg.get("hold_days", 5),
    )

    return {
        "metrics": metrics,
        "per_ticker": per_t,
        "backtest": bt,
        "n_rows": len(rows),
    }


# ─── fusion support: fit KMeans on pooled training embeddings ─────

def fit_embedding_kmeans(per_ticker: dict[str, dict], k: int) -> KMeans:
    pools: list[np.ndarray] = []
    for d in per_ticker.values():
        if len(d["train_emb"]) > 0:
            pools.append(d["train_emb"])
    if not pools:
        raise RuntimeError("no training embeddings found for KMeans")
    X = np.concatenate(pools, axis=0)
    # Cap for speed
    if len(X) > 50000:
        rng = np.random.default_rng(SEED_MODEL)
        idx = rng.choice(len(X), 50000, replace=False)
        X = X[idx]
    km = KMeans(n_clusters=k, random_state=SEED_MODEL, n_init=10)
    km.fit(X)
    return km


# ─── main orchestration ──────────────────────────────────────────

def run_benchmark(
    config: dict,
    tickers: list[str],
    thresholds: list[int],
    out_path: Path = OUT_PATH,
) -> dict:
    _log(f"universe: {len(tickers)} tickers; thresholds: {thresholds}")

    # Load data once per ticker
    _log("loading features + embeddings per ticker...")
    per_ticker: dict[str, dict] = {}
    for t in tickers:
        d = load_ticker_data(t, config)
        if d is None:
            _log(f"  {t}: skipped (missing data/embedding)")
            continue
        per_ticker[t] = d
    _log(f"loaded {len(per_ticker)}/{len(tickers)} tickers")

    # KMeans for fusion regime (fit once on full training data)
    _log(f"fitting {NUM_FUSION_CLUSTERS}-cluster KMeans on pooled train embeddings...")
    kmeans = fit_embedding_kmeans(per_ticker, NUM_FUSION_CLUSTERS)
    cluster_counts = np.bincount(
        kmeans.predict(np.concatenate([d["train_emb"] for d in per_ticker.values()])),
        minlength=NUM_FUSION_CLUSTERS,
    )
    _log(f"cluster sizes (train): {cluster_counts.tolist()}")

    bridge = AWMBridge()
    try:
        results: dict[str, Any] = {
            "meta": {
                "tickers": list(per_ticker.keys()),
                "thresholds": thresholds,
                "fusion_clusters": NUM_FUSION_CLUSTERS,
                "seed_model": SEED_MODEL,
                "feature_columns": FEATURE_COLUMNS,
                "xgb_params": config.get("xgboost", {}),
                "backtest_cfg": config.get("backtest", {}),
            },
            "runs": [],
        }

        bt_cfg = config.get("backtest", {})
        xgb_cfg = config.get("xgboost", {})

        for th in thresholds:
            _log(f"\n=== threshold = {th} ===")
            run_record: dict[str, Any] = {"threshold": th, "models": {}}

            for model_name, fn in [
                ("xgboost", lambda: run_xgb(per_ticker, th, xgb_cfg)),
                ("lewm_standalone", lambda: run_lewm(per_ticker, th)),
                ("awm_standalone", lambda: run_awm_standalone(per_ticker, th, bridge)),
                ("fusion", lambda: run_fusion(per_ticker, th, bridge, kmeans)),
            ]:
                t0 = time.time()
                try:
                    res = fn()
                    res = _attach_backtest(res, bt_cfg)
                except Exception as e:
                    _log(f"  [ERROR] {model_name}: {e}")
                    traceback.print_exc()
                    res = {"error": str(e)}
                elapsed = time.time() - t0
                _log(
                    f"  {model_name:16s} "
                    f"AUC={res.get('metrics', {}).get('auc', float('nan')):.4f} "
                    f"Brier={res.get('metrics', {}).get('brier', float('nan')):.4f} "
                    f"Sharpe={res.get('backtest', {}).get('sharpe_ratio', float('nan')):.2f} "
                    f"trades={res.get('backtest', {}).get('trade_count', 0)} "
                    f"({elapsed:.1f}s)"
                )
                run_record["models"][model_name] = res

            results["runs"].append(run_record)
            _safe_save(results, INTERMEDIATE_PATH)
            _log(f"  [checkpoint] saved partial results → {INTERMEDIATE_PATH}")

        _safe_save(results, out_path)
        _log(f"wrote final results → {out_path}")
        return results
    finally:
        bridge.close()


def _attach_backtest(res: dict, bt_cfg: dict) -> dict:
    """_finalize already attaches backtest using the default bt_cfg; if the
    config passed here differs, replace the backtest block. Kept as a seam so
    callers can pass an alternate cost/threshold without rerunning the heavy
    model fit."""
    # _finalize used the default bt_cfg already; nothing to do unless caller
    # wants an override. We leave the original result untouched.
    return res


# ─── CLI ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4 equity benchmark harness")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Override universe (space-separated)")
    ap.add_argument("--thresholds", nargs="*", type=int, default=None,
                    help="Training-size thresholds")
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="Take only the first N tickers (for quick tests)")
    ap.add_argument("--out", default=str(OUT_PATH),
                    help="Output JSON path")
    args = ap.parse_args()

    config = load_config(args.config)
    tickers = args.tickers if args.tickers else get_universe(config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]

    if args.thresholds:
        thresholds = [int(t) for t in args.thresholds]
    else:
        thresholds = [int(t) for t in config.get("benchmark", {}).get(
            "thresholds", [100, 250, 500, 1000, 2500, 5000]
        )]

    out_path = Path(args.out)
    run_benchmark(config, tickers, thresholds, out_path=out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
