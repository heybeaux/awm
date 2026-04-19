"""Bear Market Stress Test — Walk-Forward Across Multiple Regimes.

Tests the AWM+Le-WM fusion model across 20 years of market history,
including 2008 GFC, 2020 COVID crash, and 2022 bear market.

Walk-forward methodology:
  - Expanding training window (min 500 days)
  - 63-day (3-month) non-overlapping test windows
  - Retrain classifier at each step (NOT Le-WM encoder)
  - Reset AWM beliefs at each step
  - Record per-window and per-regime performance

Usage:
  python bear_market_stress_test.py [--device cpu]
"""
from __future__ import annotations

# CRITICAL: Must set before importing torch/xgboost to avoid OpenMP SIGSEGV
# on Apple Silicon (torch 2.11 + xgboost 3.2 conflict)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
import xgboost as xgb
import torch

from pipeline import FEATURE_COLUMNS, load_config, load_features
from universe import get_universe
from backtest import backtest
from lewm_adapter import LeWM

# ─── Config ──────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = RESULTS_DIR / "bear_market_stress_test.json"

MIN_TRAIN_DAYS = 500
TEST_WINDOW_DAYS = 63  # ~3 months
EMB_CHUNK = 512       # Batch size for Le-WM inference (memory safety)

EXCLUDE_TRADING = {"GLD", "HYG", "USO", "UUP", "TLT"}

REGIME_WINDOWS = [
    ("GFC Crash",        "2007-10-09", "2009-03-09"),
    ("GFC Recovery",     "2009-03-10", "2010-12-31"),
    ("2020 COVID Crash", "2020-02-19", "2020-03-23"),
    ("2020 Recovery",    "2020-03-24", "2020-12-31"),
    ("2022 Bear",        "2022-01-03", "2022-10-12"),
    ("2022 Recovery",    "2022-10-13", "2023-12-31"),
]

ENTRY_THRESHOLDS = [0.55, 0.58, 0.60]
COST_BPS = 10


def _log(msg: str) -> None:
    print(f"[stress] {msg}", flush=True)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_safe_json(x) for x in obj.tolist()]
    return obj


def tag_regime(start: pd.Timestamp, end: pd.Timestamp) -> str:
    mid = start + (end - start) / 2
    for name, rs, re in REGIME_WINDOWS:
        if pd.Timestamp(rs) <= mid <= pd.Timestamp(re):
            return name
    return "Normal/Bull"


def safe_auc(y_true, y_prob) -> float:
    try:
        if len(set(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float("nan")


def safe_acc(y_true, y_prob, threshold=0.5) -> float:
    try:
        return accuracy_score(y_true, (y_prob >= threshold).astype(int))
    except Exception:
        return float("nan")


# ─── Le-WM embedding extraction (memory-safe streaming) ─────────

def extract_embeddings_streaming(
    model: LeWM,
    df: pd.DataFrame,
    mask: pd.Series,
    price_cols: list[str],
    lookback: int = 60,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list]:
    """Extract embeddings for rows where mask==True. Returns (embeddings, labels, indices)."""
    subset = df[mask].dropna(subset=["direction_5d"])
    if len(subset) == 0:
        return np.empty((0, 64)), np.empty(0), []

    windows = []
    labels = []
    valid_indices = []
    all_embs = []

    for row_date in subset.index:
        loc = df.index.get_loc(row_date)
        if loc < lookback:
            continue
        w = df.iloc[loc - lookback:loc][price_cols].values.astype(np.float32)
        if w.shape != (lookback, len(price_cols)):
            continue
        # Per-channel z-score
        for c in range(w.shape[1]):
            s = w[:, c].std()
            m = w[:, c].mean()
            if s > 0:
                w[:, c] = (w[:, c] - m) / s
        windows.append(w)
        labels.append(subset.loc[row_date, "direction_5d"])
        valid_indices.append(row_date)

        # Flush chunk
        if len(windows) >= EMB_CHUNK:
            batch = torch.tensor(np.array(windows), dtype=torch.float32, device=device)
            with torch.no_grad():
                emb = model.encoder(batch).cpu().numpy()
            all_embs.append(emb)
            windows = []

    # Flush remainder
    if windows:
        batch = torch.tensor(np.array(windows), dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = model.encoder(batch).cpu().numpy()
        all_embs.append(emb)

    if not all_embs:
        return np.empty((0, 64)), np.empty(0), []

    return np.vstack(all_embs), np.array(labels), valid_indices


# ─── Walk-Forward Engine ────────────────────────────────────────

def run_walk_forward(device: str = "cpu") -> dict[str, Any]:
    cfg = load_config()
    universe = get_universe(cfg)

    _log(f"Loading data for {len(universe)} tickers...")
    all_data: dict[str, pd.DataFrame] = {}
    for ticker in universe:
        try:
            df = load_features(ticker, cfg)
            if len(df) >= MIN_TRAIN_DAYS:
                all_data[ticker] = df
        except Exception as e:
            _log(f"  Skip {ticker}: {e}")

    _log(f"Loaded {len(all_data)} tickers with >= {MIN_TRAIN_DAYS} days")

    # Global date index
    all_dates = sorted(set().union(*(df.index for df in all_data.values())))
    _log(f"Date range: {all_dates[0].strftime('%Y-%m-%d')} → {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)} dates)")

    # Load Le-WM encoder (frozen — we only retrain the classifier)
    _log("Loading Le-WM encoder...")
    ckpt = torch.load(DATA_DIR / "lewm_checkpoint.pt", map_location=device, weights_only=False)
    lewm_config = ckpt.get("config", {"in_channels": 5, "d_model": 64, "ctx_dim": 2})
    lewm_model = LeWM(
        in_channels=lewm_config["in_channels"],
        d_model=lewm_config["d_model"],
        ctx_dim=lewm_config["ctx_dim"],
    )
    lewm_model.load_state_dict(ckpt["state_dict"])
    lewm_model.eval()
    lewm_model.to(device)

    price_cols = ["open", "high", "low", "close", "volume"]

    # Generate walk-forward windows
    start_idx = MIN_TRAIN_DAYS
    windows = []
    while start_idx + TEST_WINDOW_DAYS <= len(all_dates):
        train_end = all_dates[start_idx - 1]
        test_start = all_dates[start_idx]
        test_end_idx = min(start_idx + TEST_WINDOW_DAYS, len(all_dates))
        test_end = all_dates[test_end_idx - 1]
        windows.append((train_end, test_start, test_end))
        start_idx += TEST_WINDOW_DAYS

    _log(f"Generated {len(windows)} walk-forward windows")

    per_window_results = []

    for wi, (train_end, test_start, test_end) in enumerate(windows):
        regime = tag_regime(test_start, test_end)
        _log(f"Window {wi+1}/{len(windows)}: {test_start.strftime('%Y-%m-%d')} → {test_end.strftime('%Y-%m-%d')} [{regime}]")

        # ── Collect XGB features + targets per ticker ──
        all_train_X, all_train_y = [], []
        all_test_X, all_test_y = [], []
        all_test_tickers = []
        all_test_dates = []
        all_test_ret5d = []

        # ── Collect Le-WM embeddings per ticker ──
        all_train_embs, all_train_emb_y = [], []
        all_test_embs, all_test_emb_y = [], []
        test_emb_global_map = []  # maps Le-WM test index → global test index

        # Track which tickers are active + their clean splits
        active_tickers: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
        global_test_idx = 0

        for ticker, df in all_data.items():
            feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            if not feat_cols or "direction_5d" not in df.columns:
                continue

            train_mask = df.index <= train_end
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_clean = df[train_mask].dropna(subset=["direction_5d"] + feat_cols)
            test_clean = df[test_mask].dropna(subset=["direction_5d"] + feat_cols)

            if len(train_clean) < 50 or len(test_clean) < 3:
                continue

            # XGBoost features
            all_train_X.append(train_clean[feat_cols].values)
            all_train_y.append(train_clean["direction_5d"].values)
            all_test_X.append(test_clean[feat_cols].values)
            all_test_y.append(test_clean["direction_5d"].values)
            all_test_tickers.extend([ticker] * len(test_clean))
            all_test_dates.extend(test_clean.index.tolist())

            ret_col = "returns_5d" if "returns_5d" in df.columns else "return_5d"
            if ret_col in test_clean.columns:
                all_test_ret5d.extend(test_clean[ret_col].values.tolist())
            else:
                all_test_ret5d.extend([0.0] * len(test_clean))

            active_tickers.append((ticker, df, train_clean, test_clean))

        if not all_train_X or not all_test_X:
            _log(f"  Skipping — insufficient data")
            continue

        train_X = np.vstack(all_train_X)
        train_y = np.concatenate(all_train_y)
        test_X = np.vstack(all_test_X)
        test_y = np.concatenate(all_test_y)
        test_tickers = np.array(all_test_tickers)
        test_dates = np.array(all_test_dates)
        test_ret5d = np.array(all_test_ret5d)

        n_test = len(test_y)

        # ── XGBoost FIRST (before any PyTorch ops — SIGSEGV workaround) ──
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=123, verbosity=0,
        )
        try:
            xgb_model.fit(train_X, train_y)
            xgb_probs = xgb_model.predict_proba(test_X)[:, 1]
        except Exception:
            xgb_probs = np.full(n_test, 0.5)
        del xgb_model
        gc.collect()

        # ── Le-WM embeddings (AFTER XGBoost) ──
        all_train_embs, all_train_emb_y = [], []
        all_test_embs, all_test_emb_y = [], []
        test_emb_global_map = []
        global_test_idx = 0

        for ticker, df, train_clean, test_clean in active_tickers:
            train_mask = df.index <= train_end
            test_mask = (df.index >= test_start) & (df.index <= test_end)

            train_emb, train_emb_labels, _ = extract_embeddings_streaming(
                lewm_model, df, train_mask, price_cols, device=device
            )
            if train_emb.shape[0] > 0:
                all_train_embs.append(train_emb)
                all_train_emb_y.append(train_emb_labels)

            test_emb, test_emb_labels, test_emb_dates = extract_embeddings_streaming(
                lewm_model, df, test_mask, price_cols, device=device
            )
            if test_emb.shape[0] > 0:
                all_test_embs.append(test_emb)
                all_test_emb_y.append(test_emb_labels)
                for emb_date in test_emb_dates:
                    try:
                        idx_in_test = test_clean.index.get_loc(emb_date) + global_test_idx
                        test_emb_global_map.append(idx_in_test)
                    except KeyError:
                        test_emb_global_map.append(-1)

            global_test_idx += len(test_clean)

        # ── Le-WM classifier ──
        lewm_probs = np.full(n_test, 0.5)

        if all_train_embs and all_test_embs:
            train_embs = np.vstack(all_train_embs)
            train_embs_y = np.concatenate(all_train_emb_y)
            test_embs = np.vstack(all_test_embs)

            try:
                lr = LogisticRegression(max_iter=1000, random_state=123)
                lr.fit(train_embs, train_embs_y)
                lr_probs = lr.predict_proba(test_embs)[:, 1]

                for i, global_idx in enumerate(test_emb_global_map):
                    if 0 <= global_idx < n_test and i < len(lr_probs):
                        lewm_probs[global_idx] = lr_probs[i]
            except Exception:
                pass

        del all_train_embs, all_test_embs, all_train_emb_y, all_test_emb_y
        gc.collect()

        # ── AWM (Bayesian with regime beliefs) ──
        awm_probs = np.full(n_test, 0.5)
        regime_col = "regime" if "regime" in next(iter(all_data.values())).columns else None

        if regime_col:
            regime_beliefs: dict[str, tuple[float, float]] = {}
            for ticker, df, train_clean, test_clean in active_tickers:
                if regime_col not in train_clean.columns:
                    continue
                for _, row in train_clean.iterrows():
                    r = str(row[regime_col])
                    a, b = regime_beliefs.get(r, (1.5, 1.5))
                    if row["direction_5d"] == 1:
                        a += 1
                    else:
                        b += 1
                    regime_beliefs[r] = (a, b)

            idx = 0
            for ticker, df, train_clean, test_clean in active_tickers:
                for _, row in test_clean.iterrows():
                    if idx < n_test and regime_col in row.index:
                        r = str(row[regime_col])
                        a, b = regime_beliefs.get(r, (1.5, 1.5))
                        awm_probs[idx] = a / (a + b)
                    idx += 1

        # ── Fusion ──
        fusion_probs = 0.5 * lewm_probs + 0.5 * awm_probs

        # ── Metrics ──
        fusion_auc = safe_auc(test_y, fusion_probs)
        lewm_auc = safe_auc(test_y, lewm_probs)
        xgb_auc = safe_auc(test_y, xgb_probs)
        awm_auc = safe_auc(test_y, awm_probs)

        # ── Backtests ──
        trading_mask = np.array([t not in EXCLUDE_TRADING for t in test_tickers])
        bt_results = {}

        for thresh in ENTRY_THRESHOLDS:
            for model_name, probs in [("Fusion", fusion_probs), ("Le-WM", lewm_probs), ("XGBoost", xgb_probs)]:
                try:
                    bt = backtest(
                        dates=test_dates[trading_mask],
                        tickers=test_tickers[trading_mask],
                        probs=probs[trading_mask],
                        returns_5d=test_ret5d[trading_mask],
                        entry_threshold=thresh,
                        cost_bps=COST_BPS,
                    )
                    bt_results[f"{model_name}_{thresh}"] = {
                        "sharpe": bt.get("sharpe_ratio", float("nan")),
                        "total_return": bt.get("total_return", 0),
                        "max_drawdown": bt.get("max_drawdown", 0),
                        "trade_count": bt.get("trade_count", 0),
                        "win_rate": bt.get("win_rate", float("nan")),
                    }
                except Exception:
                    bt_results[f"{model_name}_{thresh}"] = {
                        "sharpe": float("nan"), "total_return": 0,
                        "max_drawdown": 0, "trade_count": 0, "win_rate": float("nan"),
                    }

        window_result = {
            "window": wi + 1,
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "regime": regime,
            "n_test_obs": n_test,
            "classification": {
                "fusion_auc": fusion_auc,
                "lewm_auc": lewm_auc,
                "xgb_auc": xgb_auc,
                "awm_auc": awm_auc,
            },
            "backtests": bt_results,
        }
        per_window_results.append(window_result)

        bt_055 = bt_results.get("Fusion_0.55", {})
        _log(f"  AUC: F={fusion_auc:.4f} L={lewm_auc:.4f} X={xgb_auc:.4f} | "
             f"Sharpe@0.55={bt_055.get('sharpe', 0):.2f} Ret={bt_055.get('total_return', 0):.2%}")

        # Memory cleanup
        del train_X, train_y, test_X, test_y, xgb_probs, lewm_probs, awm_probs, fusion_probs
        gc.collect()

    # ── Aggregate by Regime ──
    _log("\n" + "=" * 70)
    _log("REGIME SUMMARY")
    _log("=" * 70)

    regime_groups: dict[str, list] = defaultdict(list)
    for wr in per_window_results:
        regime_groups[wr["regime"]].append(wr)

    regime_summary = {}
    for regime_name in sorted(regime_groups.keys()):
        group = regime_groups[regime_name]
        n_win = len(group)

        f_aucs = [w["classification"]["fusion_auc"] for w in group if not math.isnan(w["classification"]["fusion_auc"])]
        l_aucs = [w["classification"]["lewm_auc"] for w in group if not math.isnan(w["classification"]["lewm_auc"])]
        x_aucs = [w["classification"]["xgb_auc"] for w in group if not math.isnan(w["classification"]["xgb_auc"])]

        regime_bt = {}
        for thresh in ENTRY_THRESHOLDS:
            for model_name in ["Fusion", "Le-WM", "XGBoost"]:
                key = f"{model_name}_{thresh}"
                rets = [w["backtests"].get(key, {}).get("total_return", 0) for w in group]
                sharpes = [w["backtests"].get(key, {}).get("sharpe", 0) for w in group
                           if not math.isnan(w["backtests"].get(key, {}).get("sharpe", 0))]
                dds = [w["backtests"].get(key, {}).get("max_drawdown", 0) for w in group]
                trades = [w["backtests"].get(key, {}).get("trade_count", 0) for w in group]

                compound = 1.0
                for r in rets:
                    compound *= (1 + r)
                compound -= 1

                regime_bt[key] = {
                    "compound_return": compound,
                    "avg_sharpe": np.mean(sharpes) if sharpes else float("nan"),
                    "worst_drawdown": min(dds) if dds else 0,
                    "total_trades": sum(trades),
                }

        regime_summary[regime_name] = {
            "n_windows": n_win,
            "avg_fusion_auc": np.mean(f_aucs) if f_aucs else float("nan"),
            "avg_lewm_auc": np.mean(l_aucs) if l_aucs else float("nan"),
            "avg_xgb_auc": np.mean(x_aucs) if x_aucs else float("nan"),
            "backtests": regime_bt,
        }

        _log(f"\n{regime_name} ({n_win} windows)")
        if f_aucs:
            _log(f"  Avg AUC: F={np.mean(f_aucs):.4f} L={np.mean(l_aucs):.4f} X={np.mean(x_aucs):.4f}")
        for thresh in ENTRY_THRESHOLDS:
            bt = regime_bt.get(f"Fusion_{thresh}", {})
            _log(f"  Fusion@{thresh}: Ret={bt.get('compound_return', 0):.2%} "
                 f"Sharpe={bt.get('avg_sharpe', 0):.2f} "
                 f"DD={bt.get('worst_drawdown', 0):.2%} "
                 f"Trades={bt.get('total_trades', 0)}")

    # ── Overall ──
    all_f = [w["classification"]["fusion_auc"] for w in per_window_results if not math.isnan(w["classification"]["fusion_auc"])]
    all_l = [w["classification"]["lewm_auc"] for w in per_window_results if not math.isnan(w["classification"]["lewm_auc"])]
    all_x = [w["classification"]["xgb_auc"] for w in per_window_results if not math.isnan(w["classification"]["xgb_auc"])]

    _log(f"\nOVERALL ({len(per_window_results)} windows)")
    _log(f"  Avg AUC: Fusion={np.mean(all_f):.4f} Le-WM={np.mean(all_l):.4f} XGB={np.mean(all_x):.4f}")

    for thresh in ENTRY_THRESHOLDS:
        for mn in ["Fusion", "Le-WM", "XGBoost"]:
            rets = [w["backtests"].get(f"{mn}_{thresh}", {}).get("total_return", 0) for w in per_window_results]
            compound = 1.0
            for r in rets:
                compound *= (1 + r)
            compound -= 1
            _log(f"  {mn}@{thresh} compound: {compound:.2%}")

    # ── Save ──
    result = {
        "meta": {
            "type": "bear_market_stress_test",
            "data_range": f"{all_dates[0].strftime('%Y-%m-%d')} to {all_dates[-1].strftime('%Y-%m-%d')}",
            "n_tickers": len(all_data),
            "walk_forward_windows": len(per_window_results),
            "min_train_days": MIN_TRAIN_DAYS,
            "test_window_days": TEST_WINDOW_DAYS,
            "cost_bps": COST_BPS,
        },
        "overall": {
            "avg_fusion_auc": np.mean(all_f) if all_f else None,
            "avg_lewm_auc": np.mean(all_l) if all_l else None,
            "avg_xgb_auc": np.mean(all_x) if all_x else None,
        },
        "regime_summary": regime_summary,
        "per_window": per_window_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(_safe_json(result), f, indent=2, default=str)
    _log(f"\nResults saved to {OUT_PATH}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _log("=" * 70)
    _log("AWM + Le-WM BEAR MARKET STRESS TEST")
    _log("=" * 70)

    t0 = time.time()
    run_walk_forward(device=args.device)
    _log(f"\nCompleted in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
