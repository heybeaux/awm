"""XGBoost baseline on 15 raw features.

Per-ticker training (default) — each ticker gets its own model. Pooled mode is
also supported via `--pooled`.

The threshold sweep takes only the first N training rows while the validation
and test sets remain fixed per ticker (the published Phase-1 DoD evaluates at
threshold=1000). Results serialise to
`~/awm/bench/results/xgb_baseline.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

try:
    import xgboost as xgb
except ImportError as e:  # pragma: no cover
    raise SystemExit("xgboost not installed — run `pip install -r requirements.txt`") from e

from pipeline import FEATURE_COLUMNS, load_config, load_features, make_splits
from universe import get_universe

SEED_MODEL = 123
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.90) -> float:
    """Precision at the smallest threshold that still achieves `target_recall`."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays ordered by descending threshold
    # (i.e., recall goes from high → low in a specific way). Pair them up.
    mask = recall >= target_recall
    if not mask.any():
        return float("nan")
    return float(np.max(precision[mask]))


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "auc": float("nan"),
            "brier": float(brier_score_loss(y_true, y_prob)) if len(y_true) else float("nan"),
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
            "precision_at_90recall": float("nan"),
            "n": int(len(y_true)),
            "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        }
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_at_90recall": precision_at_recall(y_true, y_prob, 0.90),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    usable = df.dropna(subset=FEATURE_COLUMNS + ["direction_5d"])
    X = usable[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = usable["direction_5d"].to_numpy(dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_model(xgb_cfg: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
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


def train_eval_ticker(
    ticker: str,
    config: dict,
    threshold: int | None = None,
) -> dict[str, Any]:
    """Train per-ticker XGB on first `threshold` training rows (or all),
    evaluate on fixed test set.
    """
    df = load_features(ticker, config)
    splits = make_splits(
        df,
        train_ratio=config["data"].get("train_ratio", 0.70),
        val_ratio=config["data"].get("val_ratio", 0.15),
    )
    if len(splits["train"]) == 0 or len(splits["test"]) == 0:
        return {"ticker": ticker, "error": "empty split"}

    X_train, y_train = _xy(splits["train"])
    X_val, y_val = _xy(splits["val"])
    X_test, y_test = _xy(splits["test"])

    # Subset training data for threshold sweep: first `threshold` rows
    if threshold is not None and threshold < len(X_train):
        X_train = X_train[:threshold]
        y_train = y_train[:threshold]

    if len(X_train) < 30 or len(np.unique(y_train)) < 2:
        return {
            "ticker": ticker,
            "threshold": threshold,
            "train_n": int(len(X_train)),
            "error": "insufficient or single-class training data",
        }

    model = _make_model(config.get("xgboost", {}))
    model.fit(X_train, y_train, verbose=False)

    out: dict[str, Any] = {
        "ticker": ticker,
        "threshold": threshold,
        "train_n": int(len(X_train)),
        "val_n": int(len(X_val)),
        "test_n": int(len(X_test)),
        "train_date_range": [
            str(splits["train"].dropna(subset=FEATURE_COLUMNS + ["direction_5d"]).index.min().date()),
            str(splits["train"].dropna(subset=FEATURE_COLUMNS + ["direction_5d"]).index.max().date()),
        ],
        "test_date_range": [
            str(splits["test"].dropna(subset=FEATURE_COLUMNS + ["direction_5d"]).index.min().date()),
            str(splits["test"].dropna(subset=FEATURE_COLUMNS + ["direction_5d"]).index.max().date()),
        ],
    }

    if len(X_val):
        p_val = model.predict_proba(X_val)[:, 1]
        out["val"] = _compute_metrics(y_val, p_val)
    if len(X_test):
        p_test = model.predict_proba(X_test)[:, 1]
        out["test"] = _compute_metrics(y_test, p_test)

    return out


def train_eval_pooled(
    tickers: list[str],
    config: dict,
    threshold: int | None = None,
) -> dict[str, Any]:
    """Pool all tickers' training rows into a single model; evaluate per-ticker
    on their own test sets.
    """
    X_pool, y_pool = [], []
    per_ticker_splits: dict[str, dict[str, pd.DataFrame]] = {}
    for t in tickers:
        try:
            df = load_features(t, config)
        except FileNotFoundError:
            continue
        splits = make_splits(
            df,
            train_ratio=config["data"].get("train_ratio", 0.70),
            val_ratio=config["data"].get("val_ratio", 0.15),
        )
        per_ticker_splits[t] = splits
        X_tr, y_tr = _xy(splits["train"])
        if threshold is not None and threshold < len(X_tr):
            X_tr = X_tr[:threshold]
            y_tr = y_tr[:threshold]
        X_pool.append(X_tr)
        y_pool.append(y_tr)

    if not X_pool:
        return {"error": "no training data"}

    X_train = np.concatenate(X_pool, axis=0)
    y_train = np.concatenate(y_pool, axis=0)

    if len(X_train) < 30 or len(np.unique(y_train)) < 2:
        return {
            "threshold": threshold,
            "train_n": int(len(X_train)),
            "error": "insufficient pooled training data",
        }

    model = _make_model(config.get("xgboost", {}))
    model.fit(X_train, y_train, verbose=False)

    per_ticker: dict[str, dict[str, float]] = {}
    for t, splits in per_ticker_splits.items():
        X_test, y_test = _xy(splits["test"])
        if len(X_test) == 0:
            continue
        p_test = model.predict_proba(X_test)[:, 1]
        per_ticker[t] = _compute_metrics(y_test, p_test)

    # Aggregate (simple mean over tickers, weighted by n)
    total_n = sum(m["n"] for m in per_ticker.values()) or 1
    agg = {
        "auc": float(np.mean([m["auc"] for m in per_ticker.values() if not np.isnan(m["auc"])])),
        "brier": float(sum(m["brier"] * m["n"] for m in per_ticker.values()) / total_n),
        "accuracy": float(sum(m["accuracy"] * m["n"] for m in per_ticker.values()) / total_n),
    }
    return {
        "threshold": threshold,
        "train_n": int(len(X_train)),
        "tickers_evaluated": len(per_ticker),
        "aggregate_test": agg,
        "per_ticker_test": per_ticker,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run(
    config: dict,
    tickers: list[str],
    thresholds: list[int | None],
    pooled: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "meta": {
            "model": "xgboost",
            "tickers": tickers,
            "thresholds": thresholds,
            "pooled": pooled,
            "feature_columns": FEATURE_COLUMNS,
            "seed_model": SEED_MODEL,
            "xgb_params": config.get("xgboost", {}),
        },
        "runs": [],
    }

    for th in thresholds:
        label = str(th) if th is not None else "all"
        print(f"\n[xgb] threshold={label} {'pooled' if pooled else 'per-ticker'}")
        start = time.time()
        if pooled:
            r = train_eval_pooled(tickers, config, threshold=th)
            results["runs"].append({"threshold": th, "pooled": True, **r})
            if "aggregate_test" in r:
                a = r["aggregate_test"]
                print(f"  pooled train_n={r['train_n']} tickers={r['tickers_evaluated']} "
                      f"AUC={a['auc']:.4f} Brier={a['brier']:.4f} acc={a['accuracy']:.4f}")
            else:
                print(f"  skipped: {r.get('error')}")
        else:
            per_ticker = []
            for t in tickers:
                try:
                    r = train_eval_ticker(t, config, threshold=th)
                except FileNotFoundError:
                    continue
                per_ticker.append(r)
                if "test" in r:
                    m = r["test"]
                    print(f"  {t}: train_n={r['train_n']} test_n={r['test_n']} "
                          f"AUC={m['auc']:.4f} Brier={m['brier']:.4f} "
                          f"acc={m['accuracy']:.4f} P@90R={m['precision_at_90recall']:.4f}")
                else:
                    print(f"  {t}: {r.get('error')}")
            aucs = [r["test"]["auc"] for r in per_ticker if "test" in r and not np.isnan(r["test"]["auc"])]
            agg = {"mean_auc": float(np.mean(aucs)) if aucs else float("nan"), "n_tickers": len(aucs)}
            results["runs"].append({
                "threshold": th,
                "pooled": False,
                "per_ticker": per_ticker,
                "aggregate_test": agg,
            })
            print(f"  → mean AUC across {agg['n_tickers']} tickers: {agg['mean_auc']:.4f}")
        print(f"  (took {time.time() - start:.1f}s)")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="XGBoost baseline for equity benchmark")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Tickers to evaluate (default: universe from config)")
    ap.add_argument("--thresholds", nargs="*", type=int, default=None,
                    help="Training-size thresholds. Use 0 for 'all available'. "
                         "Defaults to config benchmark.thresholds.")
    ap.add_argument("--pooled", action="store_true",
                    help="Train one pooled model across all tickers")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: ~/awm/bench/results/xgb_baseline.json)")
    args = ap.parse_args()

    config = load_config(args.config)
    tickers = args.tickers if args.tickers else get_universe(config)

    if args.thresholds is None:
        thresholds_cfg = config.get("benchmark", {}).get("thresholds", [1000])
        thresholds = [None if t == 0 else int(t) for t in thresholds_cfg]
    else:
        thresholds = [None if t == 0 else t for t in args.thresholds]

    results = run(config, tickers, thresholds, pooled=args.pooled)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "xgb_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[xgb] wrote results → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
