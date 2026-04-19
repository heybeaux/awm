"""Le-WM embedding extraction and standalone linear classification.

Two modes:
  --extract   : Load trained checkpoint, encode all windows, save embeddings
  --classify  : Train linear head on frozen embeddings, evaluate, report metrics

Reads features from data/{ticker}_features.parquet (produced by pipeline.py).
Reads checkpoint from data/lewm_checkpoint.pt (produced by lewm_adapter.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from lewm_adapter import LeWM, TemporalEncoder, pick_device, _extract_window, _zscore_window
from pipeline import FEATURE_COLUMNS, load_config, load_features, make_splits
from universe import get_universe

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Embedding Extraction
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> LeWM:
    """Load trained LeWM from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = LeWM(
        in_channels=cfg.get("in_channels", 5),
        d_model=cfg.get("d_model", 64),
        ctx_dim=cfg.get("ctx_dim", 2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model


def encode_ticker(
    model: LeWM,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Encode all windows for one ticker. Returns (N, d_model) ndarray."""
    windows = []
    for _, row in df.iterrows():
        w = _extract_window(row["price_history"])
        if w is not None and w.shape == (60, 5):
            windows.append(w)
        else:
            windows.append(np.zeros((60, 5), dtype=np.float32))

    if not windows:
        return np.zeros((0, 64), dtype=np.float32)

    all_windows = np.stack(windows)  # (N, 60, 5)
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Per-window z-score normalisation (same as training)
            t = torch.tensor(batch, dtype=torch.float32)
            mu = t.mean(dim=1, keepdim=True)
            std = t.std(dim=1, keepdim=True).clamp_min(1e-8)
            t = (t - mu) / std
            t = t.to(device)
            z = model.encoder(t)
            embeddings.append(z.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_all(
    model: LeWM,
    tickers: list[str],
    config: dict,
    device: torch.device,
    data_dir: Path,
) -> None:
    """Extract and save embeddings for all tickers + NDJSON export."""
    ndjson_path = data_dir / "embeddings.ndjson"
    ndjson_f = open(ndjson_path, "w")
    total = 0

    for ticker in tickers:
        try:
            df = load_features(ticker, config)
        except FileNotFoundError:
            print(f"  {ticker}: skipped (no feature file)")
            continue
        if df is None or df.empty:
            print(f"  {ticker}: skipped (no data)")
            continue

        emb = encode_ticker(model, df, device)
        np.save(data_dir / f"{ticker}_embeddings.npy", emb)

        # Write NDJSON
        dates = df.index.values  # date is the index
        for j in range(len(emb)):
            record = {
                "date": str(dates[j])[:10] if j < len(dates) else "",
                "ticker": ticker,
                "embedding": emb[j].tolist(),
            }
            # Include raw features
            feat_dict = {}
            for col in FEATURE_COLUMNS:
                if col in df.columns and j < len(df):
                    val = df.iloc[j][col]
                    if pd.notna(val):
                        feat_dict[col] = float(val)
            record["features"] = feat_dict
            ndjson_f.write(json.dumps(record) + "\n")

        total += len(emb)
        print(f"  {ticker}: {len(emb)} embeddings saved")

    ndjson_f.close()
    print(f"[lewm] extracted {total} embeddings, NDJSON → {ndjson_path}")


# ---------------------------------------------------------------------------
# Linear Classification
# ---------------------------------------------------------------------------

class EmbeddingClassificationDataset(Dataset):
    """Pairs (embedding, direction_5d label) for linear head training."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        mask = ~np.isnan(labels)
        self.embeddings = torch.tensor(embeddings[mask], dtype=torch.float32)
        self.labels = torch.tensor(labels[mask], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.embeddings[idx], self.labels[idx]


def classify_evaluate(
    model: LeWM,
    tickers: list[str],
    config: dict,
    device: torch.device,
    data_dir: Path,
    threshold_n: int | None = None,
) -> dict:
    """Train linear head on frozen embeddings, evaluate on test set."""
    train_embs, train_labels = [], []
    val_embs, val_labels = [], []
    test_embs, test_labels = [], []
    per_ticker_test: dict[str, dict] = {}

    for ticker in tickers:
        try:
            df = load_features(ticker, config)
        except FileNotFoundError:
            continue
        if df is None or df.empty:
            continue

        emb = encode_ticker(model, df, device)
        if len(emb) != len(df):
            print(f"  WARN: {ticker} embedding count mismatch ({len(emb)} vs {len(df)})")
            continue

        targets = df["direction_5d"].values.astype(np.float32)

        # Splits are temporal: 70/15/15 per ticker
        n = len(df)
        train_ratio = config.get("data", {}).get("train_ratio", 0.70)
        val_ratio = config.get("data", {}).get("val_ratio", 0.15)
        tr_end = int(n * train_ratio)
        va_end = int(n * (train_ratio + val_ratio))
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[:tr_end] = True
        va_mask = np.zeros(n, dtype=bool)
        va_mask[tr_end:va_end] = True
        te_mask = np.zeros(n, dtype=bool)
        te_mask[va_end:] = True

        tr_emb = emb[tr_mask]
        tr_lab = targets[tr_mask]

        # Apply threshold (subset from front of training data)
        if threshold_n is not None and len(tr_emb) > threshold_n:
            tr_emb = tr_emb[:threshold_n]
            tr_lab = tr_lab[:threshold_n]

        # Filter NaN targets (last few rows can't compute 5-day forward return)
        tr_valid = ~np.isnan(tr_lab)
        va_lab = targets[va_mask]
        va_emb = emb[va_mask]
        va_valid = ~np.isnan(va_lab)
        te_lab = targets[te_mask]
        te_emb = emb[te_mask]
        te_valid = ~np.isnan(te_lab)

        train_embs.append(tr_emb[tr_valid])
        train_labels.append(tr_lab[tr_valid])
        val_embs.append(va_emb[va_valid])
        val_labels.append(va_lab[va_valid])
        test_embs.append(te_emb[te_valid])
        test_labels.append(te_lab[te_valid])

        # Store per-ticker test data (already NaN-filtered)
        per_ticker_test[ticker] = {
            "embeddings": te_emb[te_valid],
            "labels": te_lab[te_valid],
            "train_n": int(tr_valid.sum()),
        }

    if not train_embs:
        return {"error": "No training data"}

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_val = np.concatenate(val_embs)
    y_val = np.concatenate(val_labels)
    X_test = np.concatenate(test_embs)
    y_test = np.concatenate(test_labels)

    # Train linear head
    d_model = X_train.shape[1]
    head = nn.Sequential(nn.Linear(d_model, 1))
    head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    train_ds = EmbeddingClassificationDataset(X_train, y_train)
    val_ds = EmbeddingClassificationDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(1, 201):
        head.train()
        for emb_batch, lab_batch in train_loader:
            emb_batch = emb_batch.to(device)
            lab_batch = lab_batch.to(device)
            logits = head(emb_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, lab_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Val
        head.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for emb_batch, lab_batch in val_loader:
                emb_batch = emb_batch.to(device)
                lab_batch = lab_batch.to(device)
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

    # Evaluate on test set
    test_ds = EmbeddingClassificationDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    all_probs = []
    all_labels_clean = []

    with torch.no_grad():
        for emb_batch, lab_batch in test_loader:
            emb_batch = emb_batch.to(device)
            logits = head(emb_batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels_clean.extend(lab_batch.numpy())

    y_true = np.array(all_labels_clean)
    y_prob = np.array(all_probs)

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]
    y_pred = (y_prob >= 0.5).astype(int)

    if len(y_true) == 0:
        return {"error": "No valid test samples after NaN filtering"}

    # Metrics
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))

    # Precision at 90% recall
    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
        mask_90 = rec_arr >= 0.90
        p_at_90r = float(prec_arr[mask_90][-1]) if mask_90.any() else 0.0
    except ValueError:
        p_at_90r = 0.0

    # Per-ticker metrics
    ticker_results = {}
    for ticker, td in per_ticker_test.items():
        t_emb = td["embeddings"]
        t_lab = td["labels"]
        if len(t_emb) == 0:
            continue
        mask = ~np.isnan(t_lab)
        t_emb = t_emb[mask]
        t_lab = t_lab[mask]
        if len(t_emb) == 0:
            continue
        with torch.no_grad():
            t_tensor = torch.tensor(t_emb, dtype=torch.float32).to(device)
            t_logits = head(t_tensor).squeeze(-1)
            t_probs = torch.sigmoid(t_logits).cpu().numpy()
        try:
            t_auc = float(roc_auc_score(t_lab, t_probs))
        except ValueError:
            t_auc = 0.5
        t_brier = float(brier_score_loss(t_lab, t_probs))
        t_acc = float(accuracy_score(t_lab, (t_probs >= 0.5).astype(int)))
        try:
            t_prec, t_rec, _ = precision_recall_curve(t_lab, t_probs)
            t_mask = t_rec >= 0.90
            t_p90 = float(t_prec[t_mask][-1]) if t_mask.any() else 0.0
        except ValueError:
            t_p90 = 0.0
        ticker_results[ticker] = {
            "train_n": td["train_n"],
            "test_n": len(t_lab),
            "auc": round(t_auc, 4),
            "brier": round(t_brier, 4),
            "accuracy": round(t_acc, 4),
            "p_at_90_recall": round(t_p90, 4),
        }

    return {
        "model_name": "lewm_standalone",
        "threshold": threshold_n,
        "train_n": len(X_train),
        "val_n": len(X_val),
        "test_n": len(X_test),
        "metrics": {
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "accuracy": round(acc, 4),
            "p_at_90_recall": round(p_at_90r, 4),
        },
        "per_ticker": ticker_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Le-WM embedding extraction & classification")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--extract", action="store_true", help="Extract embeddings for all tickers")
    ap.add_argument("--classify", action="store_true", help="Train linear head and evaluate")
    ap.add_argument("--threshold", type=int, default=None, help="Subset training data to first N days/ticker")
    ap.add_argument("--sweep", action="store_true", help="Run classification at all thresholds from config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: data/lewm_checkpoint.pt)")
    ap.add_argument("--limit-tickers", type=int, default=None)
    args = ap.parse_args()

    if not args.extract and not args.classify and not args.sweep:
        print("Specify --extract, --classify, or --sweep")
        return 1

    config = load_config(args.config)
    prefer = args.device or config.get("benchmark", {}).get("device", "mps")
    device = pick_device(prefer)

    data_dir = Path(config["data"].get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir

    ckpt_path = Path(args.checkpoint) if args.checkpoint else data_dir / "lewm_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[lewm] ERROR: checkpoint not found at {ckpt_path}")
        print("[lewm] Run lewm_adapter.py first to train the encoder")
        return 1

    print(f"[lewm] loading checkpoint from {ckpt_path}")
    model = load_model(ckpt_path, device)
    print(f"[lewm] device={device}")

    tickers = get_universe(config)
    if args.limit_tickers:
        tickers = tickers[: args.limit_tickers]

    if args.extract:
        print(f"[lewm] extracting embeddings for {len(tickers)} tickers...")
        extract_all(model, tickers, config, device, data_dir)

    if args.classify:
        threshold = args.threshold
        print(f"[lewm] classifying (threshold={threshold})...")
        result = classify_evaluate(model, tickers, config, device, data_dir, threshold)
        m = result.get("metrics", {})
        print(f"  → AUC={m.get('auc', '?')} Brier={m.get('brier', '?')} acc={m.get('accuracy', '?')} P@90R={m.get('p_at_90_recall', '?')}")

        # Save result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "lewm_standalone.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        else:
            existing = []
        existing.append(result)
        out_path.write_text(json.dumps(existing, indent=2))
        print(f"  → saved to {out_path}")

    if args.sweep:
        thresholds = config.get("benchmark", {}).get("thresholds", [100, 250, 500, 1000, 2500, 5000])
        all_results = []
        for t in thresholds:
            print(f"\n[lewm] threshold={t}")
            t0 = time.time()
            result = classify_evaluate(model, tickers, config, device, data_dir, t)
            dt = time.time() - t0
            m = result.get("metrics", {})
            print(f"  → AUC={m.get('auc', '?')} Brier={m.get('brier', '?')} acc={m.get('accuracy', '?')} P@90R={m.get('p_at_90_recall', '?')} ({dt:.1f}s)")
            all_results.append(result)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "lewm_standalone.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\n[lewm] all results → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
