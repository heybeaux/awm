"""AWM Fusion Benchmark (Phase 4).

Combines Le-WM temporal embeddings with AWM's online Bayesian belief
engine and compares three signal sources:
  1. Le-WM standalone (linear head on frozen embeddings)
  2. AWM standalone (Bayesian posterior, no Le-WM input)
  3. Fusion (adaptive weighted blend of Le-WM + AWM)

For each threshold in [100, 250, 500, 1000, 2500, 5000]:
  - Split data into train/val/test per ticker (temporal, same as lewm_predict.py)
  - Train Le-WM linear head on embeddings
  - Walk through test data chronologically, feeding outcomes to AWM
  - Run backtest on all three signal sets
  - Report AUC, Brier, Sharpe, total return

Usage:
  cd ~/awm/bench/equity
  source .venv/bin/activate
  python fusion_benchmark.py [--threshold 1000] [--no-backtest]
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

from backtest import backtest as run_backtest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR.parent / "results"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path = BENCH_DIR / "config.yaml") -> dict:
    import yaml
    return yaml.safe_load(open(path))


def get_tickers(config: dict) -> list[str]:
    """Get ticker list from config or data directory."""
    uni = config.get("universe", {}).get("tickers", [])
    if uni:
        return uni
    return sorted(
        set(f.stem.replace("_features", "") for f in DATA_DIR.glob("*_features.parquet"))
    )


# ---------------------------------------------------------------------------
# Le-WM model loading (mirrors lewm_predict.py)
# ---------------------------------------------------------------------------

def load_lewm_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load the Le-WM JEPA encoder from checkpoint."""
    from lewm_adapter import LeWM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    in_channels = cfg.get("in_channels", 5)
    d_model = cfg.get("d_model", 64)
    ctx_dim = cfg.get("ctx_dim", 2)

    model = LeWM(in_channels=in_channels, d_model=d_model, ctx_dim=ctx_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def encode_ticker_from_npy(ticker: str) -> np.ndarray | None:
    """Load pre-extracted embeddings from .npy file."""
    path = DATA_DIR / f"{ticker}_embeddings.npy"
    if path.exists():
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Linear classification head (same architecture as lewm_predict.py)
# ---------------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_linear_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    lr: float = 0.001,
    epochs: int = 200,
    patience: int = 20,
    batch_size: int = 512,
) -> LinearHead:
    """Train a linear head on frozen Le-WM embeddings."""
    head = LinearHead(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        head.train()
        for emb_batch, lab_batch in train_loader:
            emb_batch, lab_batch = emb_batch.to(device), lab_batch.to(device)
            optimizer.zero_grad()
            logits = head(emb_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, lab_batch)
            loss.backward()
            optimizer.step()

        # Validation
        head.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for emb_batch, lab_batch in val_loader:
                emb_batch, lab_batch = emb_batch.to(device), lab_batch.to(device)
                logits = head(emb_batch).squeeze(-1)
                val_loss_sum += float(F.binary_cross_entropy_with_logits(logits, lab_batch)) * len(lab_batch)
                val_n += len(lab_batch)

        val_loss = val_loss_sum / max(val_n, 1)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    return head


def predict_with_head(head: LinearHead, embeddings: np.ndarray, device: torch.device) -> np.ndarray:
    """Get P(up) from the linear head."""
    head.eval()
    with torch.no_grad():
        t = torch.tensor(embeddings, dtype=torch.float32).to(device)
        logits = head(t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_features(ticker: str, config: dict) -> pd.DataFrame | None:
    """Load feature parquet for a ticker."""
    path = DATA_DIR / f"{ticker}_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def compute_returns_5d(adj_close: np.ndarray) -> np.ndarray:
    """Compute 5-day forward returns from adjusted close prices."""
    ret = np.full(len(adj_close), np.nan)
    for i in range(len(adj_close) - 5):
        if adj_close[i] != 0 and not np.isnan(adj_close[i]):
            ret[i] = (adj_close[i + 5] - adj_close[i]) / adj_close[i]
    return ret


def prepare_ticker_data(
    ticker: str,
    model: nn.Module,
    config: dict,
    device: torch.device,
    threshold_n: int | None,
) -> dict[str, Any] | None:
    """Load, encode, split, and return all data needed for one ticker."""
    df = load_features(ticker, config)
    if df is None or df.empty:
        return None

    # Use pre-extracted embeddings (much faster)
    emb = encode_ticker_from_npy(ticker)
    if emb is None or len(emb) != len(df):
        print(f"  SKIP {ticker}: no embeddings or shape mismatch")
        return None

    targets = df["direction_5d"].values.astype(np.float32)
    regimes = df["regime"].values  # string array
    adj_close = df["adj_close"].values.astype(np.float64)
    dates = df.index  # DatetimeIndex
    returns_5d = compute_returns_5d(adj_close)

    # Temporal split: 70/15/15
    n = len(df)
    train_ratio = config.get("data", {}).get("train_ratio", 0.70)
    val_ratio = config.get("data", {}).get("val_ratio", 0.15)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))

    # Training embeddings (with optional threshold)
    tr_emb = emb[:tr_end]
    tr_lab = targets[:tr_end]
    if threshold_n is not None and len(tr_emb) > threshold_n:
        tr_emb = tr_emb[:threshold_n]
        tr_lab = tr_lab[:threshold_n]

    # Filter NaN targets
    tr_valid = ~np.isnan(tr_lab)
    va_lab = targets[tr_end:va_end]
    va_emb = emb[tr_end:va_end]
    va_valid = ~np.isnan(va_lab)
    te_lab = targets[va_end:]
    te_emb = emb[va_end:]
    te_valid = ~np.isnan(te_lab)

    # Test metadata (dates, regimes, returns)
    te_dates = dates[va_end:]
    te_regimes = regimes[va_end:]
    te_returns_5d = returns_5d[va_end:]

    return {
        "ticker": ticker,
        "train_emb": tr_emb[tr_valid],
        "train_lab": tr_lab[tr_valid],
        "val_emb": va_emb[va_valid],
        "val_lab": va_lab[va_valid],
        "test_emb": te_emb[te_valid],
        "test_lab": te_lab[te_valid],
        "test_dates": te_dates[te_valid],
        "test_regimes": te_regimes[te_valid],
        "test_returns_5d": te_returns_5d[te_valid],
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute AUC, Brier score, and accuracy."""
    valid = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "brier": 0.25, "accuracy": 0.5}

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    return {
        "auc": round(auc, 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))), 4),
    }


# ---------------------------------------------------------------------------
# Fusion logic
# ---------------------------------------------------------------------------

def adaptive_weight(awm_observations: int) -> float:
    """Le-WM weight decreases as AWM accumulates evidence.

    Starts at 0.7 (trust Le-WM), converges to 0.3 (trust AWM more)
    as observations grow.
    """
    return max(0.3, 0.7 - 0.04 * awm_observations)


def run_fusion_for_threshold(
    tickers: list[str],
    model: nn.Module,
    config: dict,
    device: torch.device,
    threshold_n: int,
    run_backtests: bool = True,
) -> dict[str, Any]:
    """Run the full fusion benchmark for one threshold."""
    from awm_bridge import AWMBridge

    print(f"\n{'='*60}")
    print(f"THRESHOLD = {threshold_n}")
    print(f"{'='*60}")

    # 1. Prepare data for all tickers
    ticker_data = {}
    all_train_emb, all_train_lab = [], []
    all_val_emb, all_val_lab = [], []

    for ticker in tickers:
        td = prepare_ticker_data(ticker, model, config, device, threshold_n)
        if td is None:
            continue
        ticker_data[ticker] = td
        all_train_emb.append(td["train_emb"])
        all_train_lab.append(td["train_lab"])
        all_val_emb.append(td["val_emb"])
        all_val_lab.append(td["val_lab"])

    if not ticker_data:
        return {"error": "No ticker data", "threshold": threshold_n}

    X_train = np.concatenate(all_train_emb)
    y_train = np.concatenate(all_train_lab)
    X_val = np.concatenate(all_val_emb)
    y_val = np.concatenate(all_val_lab)

    print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"  Tickers: {len(ticker_data)}")

    # 2. Train Le-WM linear head
    print("  Training Le-WM linear head...", flush=True)
    head = train_linear_head(X_train, y_train, X_val, y_val, device)

    # 3. Build test observations sorted chronologically across all tickers
    observations = []
    for ticker, td in ticker_data.items():
        probs = predict_with_head(head, td["test_emb"], device)
        for i in range(len(td["test_lab"])):
            regime = str(td["test_regimes"][i]) if td["test_regimes"][i] is not None else "quiet"
            if regime == "nan" or regime == "None":
                regime = "quiet"
            observations.append({
                "date": td["test_dates"][i],
                "ticker": ticker,
                "regime": regime,
                "lewm_p": float(probs[i]),
                "label": float(td["test_lab"][i]),
                "returns_5d": float(td["test_returns_5d"][i]),
            })

    # Sort by date for chronological AWM updates
    observations.sort(key=lambda x: x["date"])
    print(f"  Test observations: {len(observations)} (sorted chronologically)")

    # 4. Walk through test data with AWM
    print("  Running AWM fusion...", flush=True)
    awm_failed = False

    try:
        bridge = AWMBridge(db_path=":memory:", call_timeout_sec=10.0)
    except Exception as e:
        print(f"  WARNING: AWM bridge failed to start: {e}")
        print("  Falling back to Le-WM-only predictions")
        awm_failed = True
        bridge = None

    results_rows = []
    awm_errors = 0

    for obs in observations:
        row = {
            "date": obs["date"],
            "ticker": obs["ticker"],
            "regime": obs["regime"],
            "lewm_p": obs["lewm_p"],
            "label": obs["label"],
            "returns_5d": obs["returns_5d"],
        }

        if awm_failed or bridge is None:
            row["awm_p"] = 0.5
            row["awm_obs"] = 0
            row["fusion_p"] = obs["lewm_p"]
        else:
            try:
                pred = bridge.predict(obs["ticker"], obs["regime"])
                awm_p = pred["p_up"]
                awm_obs = pred["observations"]

                w = adaptive_weight(awm_obs)
                fusion_p = w * obs["lewm_p"] + (1 - w) * awm_p

                row["awm_p"] = float(awm_p)
                row["awm_obs"] = int(awm_obs)
                row["fusion_p"] = float(fusion_p)

                # Record outcome to update AWM posterior
                bridge.record(obs["ticker"], obs["regime"], int(obs["label"]))

            except Exception as e:
                awm_errors += 1
                if awm_errors <= 3:
                    print(f"  AWM error ({awm_errors}): {e}")
                row["awm_p"] = 0.5
                row["awm_obs"] = 0
                row["fusion_p"] = obs["lewm_p"]

        results_rows.append(row)

    if bridge is not None:
        try:
            bridge.close()
        except Exception:
            pass

    if awm_errors > 0:
        print(f"  AWM errors: {awm_errors}/{len(observations)}")

    # 5. Compute metrics
    y_true = np.array([r["label"] for r in results_rows])
    lewm_probs = np.array([r["lewm_p"] for r in results_rows])
    awm_probs = np.array([r["awm_p"] for r in results_rows])
    fusion_probs = np.array([r["fusion_p"] for r in results_rows])

    lewm_metrics = classification_metrics(y_true, lewm_probs)
    awm_metrics = classification_metrics(y_true, awm_probs)
    fusion_metrics = classification_metrics(y_true, fusion_probs)

    print(f"  Le-WM  AUC={lewm_metrics['auc']:.4f}  Brier={lewm_metrics['brier']:.4f}  Acc={lewm_metrics['accuracy']:.4f}")
    print(f"  AWM    AUC={awm_metrics['auc']:.4f}  Brier={awm_metrics['brier']:.4f}  Acc={awm_metrics['accuracy']:.4f}")
    print(f"  Fusion AUC={fusion_metrics['auc']:.4f}  Brier={fusion_metrics['brier']:.4f}  Acc={fusion_metrics['accuracy']:.4f}")

    result: dict[str, Any] = {
        "threshold": threshold_n,
        "n_tickers": len(ticker_data),
        "n_train": len(X_train),
        "n_test": len(results_rows),
        "awm_errors": awm_errors,
        "lewm": {"classification": lewm_metrics},
        "awm_only": {"classification": awm_metrics},
        "fusion": {"classification": fusion_metrics},
    }

    # 6. Backtests
    if run_backtests:
        dates_arr = np.array([r["date"] for r in results_rows])
        tickers_arr = np.array([r["ticker"] for r in results_rows])
        returns_arr = np.array([r["returns_5d"] for r in results_rows])

        # Filter out NaN returns for backtest
        valid_bt = ~np.isnan(returns_arr)
        if valid_bt.sum() > 0:
            bt_dates = dates_arr[valid_bt]
            bt_tickers = tickers_arr[valid_bt]
            bt_returns = returns_arr[valid_bt]

            for name, probs in [("lewm", lewm_probs), ("awm_only", awm_probs), ("fusion", fusion_probs)]:
                bt_probs = probs[valid_bt]
                try:
                    bt = run_backtest(bt_dates, bt_tickers, bt_probs, bt_returns)
                    result[name]["backtest"] = _sanitize(bt)
                    print(f"  {name:8s} backtest: {bt['trade_count']} trades, "
                          f"sharpe={bt['sharpe_ratio']:.2f}, total={bt['total_return']:.2%}")
                except Exception as e:
                    result[name]["backtest"] = {"error": str(e)}
                    print(f"  {name:8s} backtest error: {e}")

    # 7. Per-ticker breakdown
    per_ticker = {}
    for ticker in ticker_data:
        t_rows = [r for r in results_rows if r["ticker"] == ticker]
        if not t_rows:
            continue
        t_true = np.array([r["label"] for r in t_rows])
        t_lewm = np.array([r["lewm_p"] for r in t_rows])
        t_awm = np.array([r["awm_p"] for r in t_rows])
        t_fusion = np.array([r["fusion_p"] for r in t_rows])
        per_ticker[ticker] = {
            "n_test": len(t_rows),
            "lewm_auc": classification_metrics(t_true, t_lewm)["auc"],
            "awm_auc": classification_metrics(t_true, t_awm)["auc"],
            "fusion_auc": classification_metrics(t_true, t_fusion)["auc"],
        }

    result["per_ticker"] = per_ticker
    return result


def _sanitize(obj: Any) -> Any:
    """Replace NaN/inf with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """Print a nice summary table."""
    print("\n" + "=" * 80)
    print("FUSION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Thresh':>8s}  {'Model':>8s}  {'AUC':>7s}  {'Brier':>7s}  {'Acc':>7s}  "
          f"{'Sharpe':>8s}  {'Return':>9s}  {'Trades':>7s}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['threshold']:>8d}  ERROR: {r['error']}")
            continue
        for name, label in [("lewm", "Le-WM"), ("awm_only", "AWM"), ("fusion", "Fusion")]:
            cls = r[name].get("classification", {})
            bt = r[name].get("backtest", {})
            auc = cls.get("auc", "?")
            brier = cls.get("brier", "?")
            acc = cls.get("accuracy", "?")
            sharpe = bt.get("sharpe_ratio", "?")
            total = bt.get("total_return", "?")
            trades = bt.get("trade_count", "?")

            auc_s = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            brier_s = f"{brier:.4f}" if isinstance(brier, float) else str(brier)
            acc_s = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) and sharpe is not None else str(sharpe)
            total_s = f"{total:+.2%}" if isinstance(total, (int, float)) and total is not None else str(total)
            trades_s = str(trades)

            print(f"{r['threshold']:>8d}  {label:>8s}  {auc_s:>7s}  {brier_s:>7s}  {acc_s:>7s}  "
                  f"{sharpe_s:>8s}  {total_s:>9s}  {trades_s:>7s}")
        print("-" * 80)

    # Find best fusion threshold
    best = max(
        [r for r in results if "error" not in r],
        key=lambda r: r["fusion"]["classification"].get("auc", 0),
        default=None,
    )
    if best:
        f_cls = best["fusion"]["classification"]
        f_bt = best["fusion"].get("backtest", {})
        print(f"\nBest fusion: threshold={best['threshold']}, "
              f"AUC={f_cls.get('auc', '?')}, "
              f"Sharpe={f_bt.get('sharpe_ratio', '?')}")

    # Delta vs Le-WM standalone
    print("\nFusion vs Le-WM (AUC delta):")
    for r in results:
        if "error" in r:
            continue
        l_auc = r["lewm"]["classification"].get("auc", 0.5)
        f_auc = r["fusion"]["classification"].get("auc", 0.5)
        delta = f_auc - l_auc
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  threshold={r['threshold']:>5d}: {delta:+.4f} {arrow}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AWM Fusion Benchmark (Phase 4)")
    ap.add_argument("--config", default=str(BENCH_DIR / "config.yaml"))
    ap.add_argument("--threshold", type=int, default=None, help="Run single threshold (default: all)")
    ap.add_argument("--no-backtest", action="store_true", help="Skip backtest (classification only)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)
    tickers = get_tickers(config)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DATA_DIR / "lewm_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Le-WM checkpoint not found at {ckpt_path}")
        return 1

    print(f"[fusion] Loading Le-WM model from {ckpt_path}")
    model = load_lewm_model(ckpt_path, device)
    print(f"[fusion] Device: {device}")
    print(f"[fusion] Tickers: {len(tickers)}")

    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = config.get("benchmark", {}).get("thresholds", [100, 250, 500, 1000, 2500, 5000])

    all_results = []
    t0 = time.time()

    for threshold in thresholds:
        result = run_fusion_for_threshold(
            tickers, model, config, device, threshold,
            run_backtests=not args.no_backtest,
        )
        all_results.append(result)

    elapsed = time.time() - t0
    print(f"\n[fusion] Completed in {elapsed:.1f}s")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "fusion_benchmark.json"
    out_path.write_text(json.dumps(_sanitize(all_results), indent=2))
    print(f"[fusion] Results saved to {out_path}")

    # Summary
    print_summary(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
