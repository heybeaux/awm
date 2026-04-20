"""HYG quarantine audit for the equity benchmark.

The suspicious outlier in the published benchmark is HYG under the
AWM-standalone regime-belief model. This script reproduces that path and runs
four triage checks:
  1. embargoed walk-forward
  2. label block shuffle
  3. prior removal (Beta(1, 1))
  4. dividend / adjustment audit
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from pipeline import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS
from pipeline import load_config, make_splits
from targets import compute_direction_5d


BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = Path(os.environ.get("EQUITY_BENCH_RESULTS_DIR", BENCH_DIR / "results"))
OUT_PATH = RESULTS_DIR / "hyg_audit.json"

DEFAULT_ALPHA = 1.5
DEFAULT_BETA = 1.0
DEFAULT_THRESHOLD = 500
DEFAULT_BLOCK_SIZE = 30
DEFAULT_EMBARGO_DAYS = 5
DEFAULT_TEST_WINDOW = 63


def _log(msg: str) -> None:
    print(f"[hyg_audit] {msg}", flush=True)


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


def load_hyg_frame() -> pd.DataFrame:
    path = DATA_DIR / "HYG_features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path).sort_index()


def _benchmark_required_columns(label_col: str) -> list[str]:
    return list(BASE_FEATURE_COLUMNS) + ["regime", "ret_5d_fwd", label_col]


def _with_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_index()
    out["ret_5d_fwd"] = out["adj_close"].shift(-5) / out["adj_close"] - 1.0
    return out


def awm_probs_from_regimes(
    train_regimes: pd.Series,
    train_labels: pd.Series,
    test_regimes: pd.Series,
    alpha0: float = DEFAULT_ALPHA,
    beta0: float = DEFAULT_BETA,
) -> np.ndarray:
    beliefs: dict[str, tuple[float, float]] = {}
    for regime, label in zip(train_regimes.astype(str), train_labels.astype(int), strict=False):
        alpha, beta = beliefs.get(regime, (alpha0, beta0))
        if int(label) == 1:
            alpha += 1.0
        else:
            beta += 1.0
        beliefs[regime] = (alpha, beta)

    probs: list[float] = []
    for regime in test_regimes.astype(str):
        alpha, beta = beliefs.get(regime, (alpha0, beta0))
        probs.append(alpha / (alpha + beta))
    return np.asarray(probs, dtype=np.float64)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
    }
    if len(y_true) == 0:
        out["auc"] = float("nan")
        out["brier"] = float("nan")
        return out
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    if len(np.unique(y_true)) < 2:
        out["auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def fixed_split_eval(
    df: pd.DataFrame,
    label_col: str = "direction_5d",
    threshold: int = DEFAULT_THRESHOLD,
    alpha0: float = DEFAULT_ALPHA,
    beta0: float = DEFAULT_BETA,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    prepared = _with_forward_returns(df)
    splits = make_splits(prepared, train_ratio=train_ratio, val_ratio=val_ratio, target_col=label_col)
    required = _benchmark_required_columns(label_col)
    train_df = splits["train"].dropna(subset=required)
    test_df = splits["test"].dropna(subset=required)

    if threshold < len(train_df):
        train_df = train_df.iloc[:threshold]

    probs = awm_probs_from_regimes(train_df["regime"], train_df[label_col], test_df["regime"], alpha0, beta0)
    y_test = test_df[label_col].astype(int).to_numpy()
    return {
        "train_n": int(len(train_df)),
        "test_n": int(len(test_df)),
        **_metrics(y_test, probs),
    }


def walk_forward_eval(
    df: pd.DataFrame,
    label_col: str = "direction_5d",
    alpha0: float = DEFAULT_ALPHA,
    beta0: float = DEFAULT_BETA,
    min_train: int = DEFAULT_THRESHOLD,
    test_window: int = DEFAULT_TEST_WINDOW,
    embargo_days: int = 0,
) -> dict[str, Any]:
    prepared = _with_forward_returns(df)
    if len(prepared) <= min_train + embargo_days:
        return {"auc": float("nan"), "brier": float("nan"), "n": 0, "windows": []}

    all_probs: list[float] = []
    all_labels: list[int] = []
    windows: list[dict[str, Any]] = []
    required = _benchmark_required_columns(label_col)

    test_start = min_train + embargo_days
    while test_start < len(prepared):
        train_end = max(0, test_start - embargo_days)
        test_end = min(test_start + test_window, len(prepared))

        train_df = prepared.iloc[:train_end].dropna(subset=required)
        test_df = prepared.iloc[test_start:test_end].dropna(subset=required)
        if len(train_df) < min_train or len(test_df) == 0:
            test_start = test_end
            continue

        probs = awm_probs_from_regimes(train_df["regime"], train_df[label_col], test_df["regime"], alpha0, beta0)
        labels = test_df[label_col].astype(int).to_numpy()
        window_metrics = _metrics(labels, probs)
        windows.append({
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            **window_metrics,
        })

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        test_start = test_end

    y_true = np.asarray(all_labels, dtype=np.int64)
    y_prob = np.asarray(all_probs, dtype=np.float64)
    return {
        **_metrics(y_true, y_prob),
        "windows": windows,
    }


def block_shuffle_labels(labels: pd.Series, block_size: int, seed: int = 123) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = labels.to_numpy()
    blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
    order = rng.permutation(len(blocks))
    shuffled = np.concatenate([blocks[i] for i in order])
    return pd.Series(shuffled[: len(values)], index=labels.index, name=f"{labels.name}_shuffled")


def dividend_adjustment_audit(
    df: pd.DataFrame,
    direction_threshold: float,
    threshold: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    adjustment_factor = df["adj_close"] / df["close"].replace(0.0, np.nan)
    adjustment_change = adjustment_factor / adjustment_factor.shift(1) - 1.0
    adjustment_event = adjustment_change.abs() > 0.0025
    adjustment_window = adjustment_event | adjustment_event.shift(1, fill_value=False) | adjustment_event.shift(-1, fill_value=False)

    raw_label = compute_direction_5d(df, threshold=direction_threshold, horizon=5, price_col="close")
    adj_label = compute_direction_5d(df, threshold=direction_threshold, horizon=5, price_col="adj_close")

    audit_frame = pd.DataFrame({
        "regime": df["regime"],
        "raw_label": raw_label,
        "adj_label": adj_label,
        "adjustment_window": adjustment_window,
        "adjustment_change": adjustment_change,
    }).dropna(subset=["regime", "raw_label", "adj_label"])

    flips = audit_frame["raw_label"].astype(int) != audit_frame["adj_label"].astype(int)
    near_adjustment = audit_frame["adjustment_window"].astype(bool)

    raw_eval = fixed_split_eval(
        df.assign(direction_5d_raw=raw_label),
        label_col="direction_5d_raw",
        threshold=threshold,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    adj_eval = fixed_split_eval(
        df.assign(direction_5d_adj=adj_label),
        label_col="direction_5d_adj",
        threshold=threshold,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    near_flip_rate = float(flips[near_adjustment].mean()) if near_adjustment.any() else float("nan")
    off_flip_rate = float(flips[~near_adjustment].mean()) if (~near_adjustment).any() else float("nan")

    return {
        "adjustment_events": int(adjustment_event.sum()),
        "max_adjustment_bps": float(adjustment_change.abs().max() * 1e4) if len(adjustment_change.dropna()) else float("nan"),
        "median_adjustment_bps": float(adjustment_change[adjustment_event].abs().median() * 1e4)
        if adjustment_event.any()
        else float("nan"),
        "label_flip_count": int(flips.sum()),
        "label_flip_rate": float(flips.mean()) if len(flips) else float("nan"),
        "label_flip_rate_near_adjustment": near_flip_rate,
        "label_flip_rate_off_adjustment": off_flip_rate,
        "awm_auc_adj_close_labels": float(adj_eval["auc"]),
        "awm_auc_raw_close_labels": float(raw_eval["auc"]),
        "auc_delta_raw_minus_adj": (
            float(raw_eval["auc"] - adj_eval["auc"])
            if not (math.isnan(raw_eval["auc"]) or math.isnan(adj_eval["auc"]))
            else float("nan")
        ),
    }


def run_audit(
    threshold: int = DEFAULT_THRESHOLD,
    block_size: int = DEFAULT_BLOCK_SIZE,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    test_window: int = DEFAULT_TEST_WINDOW,
) -> dict[str, Any]:
    config = load_config()
    df = load_hyg_frame()
    train_ratio = float(config["data"].get("train_ratio", 0.70))
    val_ratio = float(config["data"].get("val_ratio", 0.15))
    direction_threshold = float(config["data"].get("direction_threshold", 0.01))

    baseline = fixed_split_eval(
        df,
        threshold=threshold,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    walk_forward_no_embargo = walk_forward_eval(
        df,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        min_train=threshold,
        test_window=test_window,
        embargo_days=0,
    )
    walk_forward_embargo = walk_forward_eval(
        df,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        min_train=threshold,
        test_window=test_window,
        embargo_days=embargo_days,
    )

    usable = _with_forward_returns(df).dropna(subset=_benchmark_required_columns("direction_5d")).sort_index()
    shuffled_labels = block_shuffle_labels(usable["direction_5d"].astype(int), block_size=block_size)
    shuffled_eval = fixed_split_eval(
        df.assign(direction_5d_shuffled=shuffled_labels.reindex(df.index)),
        label_col="direction_5d_shuffled",
        threshold=threshold,
        alpha0=DEFAULT_ALPHA,
        beta0=DEFAULT_BETA,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    priorless_eval = fixed_split_eval(
        df,
        threshold=threshold,
        alpha0=1.0,
        beta0=1.0,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    adjustment_audit = dividend_adjustment_audit(
        df,
        direction_threshold=direction_threshold,
        threshold=threshold,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    tests = {
        "baseline_fixed_split": baseline,
        "walk_forward_no_embargo": walk_forward_no_embargo,
        "walk_forward_embargo": walk_forward_embargo,
        "label_shuffle": shuffled_eval,
        "prior_removal_beta_1_1": priorless_eval,
        "dividend_adjustment_audit": adjustment_audit,
    }

    passes = {
        "embargoed_walk_forward": bool(
            not math.isnan(walk_forward_embargo["auc"]) and walk_forward_embargo["auc"] > 0.60
        ),
        "label_shuffle_drop": bool(
            not math.isnan(shuffled_eval["auc"]) and abs(shuffled_eval["auc"] - 0.50) <= 0.10
        ),
        "prior_removal": bool(
            not math.isnan(priorless_eval["auc"]) and priorless_eval["auc"] > 0.60
        ),
        "dividend_adjustment": bool(
            not math.isnan(adjustment_audit["awm_auc_raw_close_labels"])
            and adjustment_audit["awm_auc_raw_close_labels"] > 0.60
            and adjustment_audit["label_flip_rate"] < 0.05
        ),
    }

    disposition = "reinstate_tradeable" if all(passes.values()) else "quarantine_feature_only"

    result = {
        "meta": {
            "ticker": "HYG",
            "threshold": threshold,
            "block_size": block_size,
            "embargo_days": embargo_days,
            "test_window": test_window,
        },
        "tests": tests,
        "passes": passes,
        "disposition": disposition,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    _log(f"wrote audit report: {OUT_PATH}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HYG artifact triage tests")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    parser.add_argument("--test-window", type=int, default=DEFAULT_TEST_WINDOW)
    args = parser.parse_args()

    run_audit(
        threshold=args.threshold,
        block_size=args.block_size,
        embargo_days=args.embargo_days,
        test_window=args.test_window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
