"""Le-WM temporal encoder adaptation for the AWM equity benchmark.

Replaces Le-WM's ViT pixel encoder with a 1D conv encoder over 60-day OHLCV
windows. Training objective (JEPA-style): given encoding of window_t plus a
2-dim context vector (day_of_week, volume_regime), predict encoding of
window_{t+1}. Loss = MSE(pred, actual_t+1) + lambda * KL(z || N(0,1)).

Writes a single checkpoint to `data/lewm_checkpoint.pt` (pooled by default).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pipeline import FEATURE_COLUMNS, load_config, load_features, make_splits
from universe import get_universe


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(prefer: str = "mps") -> torch.device:
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """1D conv encoder. Input (B, 60, 5) → (B, 64)."""

    def __init__(self, in_channels: int = 5, d_model: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 60, 5)
        x = x.transpose(1, 2)  # (B, 5, 60)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        return self.norm(x)


class Predictor(nn.Module):
    """Predict next-window embedding from current embedding + 2-dim context."""

    def __init__(self, d_model: int = 64, ctx_dim: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, ctx], dim=-1))


class LeWM(nn.Module):
    def __init__(self, in_channels: int = 5, d_model: int = 64, ctx_dim: int = 2) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(in_channels, d_model)
        self.predictor = Predictor(d_model, ctx_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _zscore_window(w: np.ndarray) -> np.ndarray:
    """Per-channel z-score across the 60-step window. w: (60, 5)."""
    mu = w.mean(axis=0, keepdims=True)
    sd = w.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (w - mu) / sd


def _extract_window(ph) -> np.ndarray | None:
    """Convert a price_history cell to a (60, 5) float32 array or None."""
    if ph is None:
        return None
    # Stored as numpy object array of length 60, each element a list of 5 floats.
    try:
        arr = np.asarray(ph.tolist() if hasattr(ph, "tolist") else ph, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape != (60, 5):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def _volume_regime(volume_ratio: float) -> float:
    """Map volume_ratio to {-1, 0, +1} bucket as a float context feature."""
    if not np.isfinite(volume_ratio):
        return 0.0
    if volume_ratio > 1.5:
        return 1.0
    if volume_ratio < 0.5:
        return -1.0
    return 0.0


@dataclass
class Pair:
    w_t: np.ndarray       # (60, 5)
    w_tp1: np.ndarray     # (60, 5)
    ctx: np.ndarray       # (2,) day_of_week (normalized), volume_regime


class JepaPairDataset(Dataset):
    """(window_t, window_{t+1}) pairs drawn from the 'train' or 'val' split.

    We only use rows where both t and t+1 windows exist (non-null price_history,
    finite values). Windows are z-scored per-channel.
    """

    def __init__(self, pairs: list[Pair]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        p = self.pairs[i]
        return (
            torch.from_numpy(_zscore_window(p.w_t)).float(),
            torch.from_numpy(_zscore_window(p.w_tp1)).float(),
            torch.from_numpy(p.ctx).float(),
        )


def build_pairs_for_split(df_split: pd.DataFrame) -> list[Pair]:
    """From a split DataFrame, build (w_t, w_{t+1}) pairs using consecutive rows.

    Consecutive by index within the split (dates already sorted). Skips rows
    where either window is invalid.
    """
    if df_split.empty or "price_history" not in df_split.columns:
        return []

    df = df_split.sort_index()
    ph = df["price_history"].tolist()
    dow = df["day_of_week"].to_numpy(dtype=np.float32) if "day_of_week" in df.columns else np.zeros(len(df), np.float32)
    vr = df["volume_ratio"].to_numpy(dtype=np.float32) if "volume_ratio" in df.columns else np.zeros(len(df), np.float32)

    pairs: list[Pair] = []
    for i in range(len(df) - 1):
        w_t = _extract_window(ph[i])
        w_tp1 = _extract_window(ph[i + 1])
        if w_t is None or w_tp1 is None:
            continue
        ctx = np.array([
            (dow[i] - 2.0) / 2.0,  # center Mon-Fri around 0
            _volume_regime(float(vr[i])),
        ], dtype=np.float32)
        if not np.isfinite(ctx).all():
            continue
        pairs.append(Pair(w_t=w_t, w_tp1=w_tp1, ctx=ctx))
    return pairs


def build_dataset(
    tickers: list[str],
    config: dict,
    split: str,
) -> JepaPairDataset:
    train_ratio = config["data"].get("train_ratio", 0.70)
    val_ratio = config["data"].get("val_ratio", 0.15)
    all_pairs: list[Pair] = []
    for t in tickers:
        try:
            df = load_features(t, config)
        except FileNotFoundError:
            continue
        splits = make_splits(df, train_ratio=train_ratio, val_ratio=val_ratio)
        sub = splits.get(split)
        if sub is None or sub.empty:
            continue
        all_pairs.extend(build_pairs_for_split(sub))
    return JepaPairDataset(all_pairs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _kl_to_standard_normal(z: torch.Tensor) -> torch.Tensor:
    """Proxy KL(q(z) || N(0,1)) using per-batch moments.

    Encourages the latent batch distribution toward unit-variance, zero-mean.
    """
    mu = z.mean(dim=0)
    var = z.var(dim=0, unbiased=False).clamp_min(1e-6)
    # 0.5 * sum(mu^2 + var - 1 - log var)
    return 0.5 * (mu.pow(2) + var - 1.0 - var.log()).sum()


def train(
    model: LeWM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    reg_lambda: float,
    patience: int,
    ckpt_path: Path,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    best_val = math.inf
    best_state: dict | None = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_mse_sum = 0.0
        tr_kl_sum = 0.0
        tr_n = 0
        for w_t, w_tp1, ctx in train_loader:
            w_t = w_t.to(device, non_blocking=True)
            w_tp1 = w_tp1.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)

            z_t = model.encoder(w_t)
            with torch.no_grad():
                z_tp1 = model.encoder(w_tp1)  # target — we stop-grad into this branch (JEPA-style)
            pred = model.predictor(z_t, ctx)
            mse = F.mse_loss(pred, z_tp1)
            kl = _kl_to_standard_normal(z_t) / max(1, z_t.shape[1])
            loss = mse + reg_lambda * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = w_t.size(0)
            tr_mse_sum += float(mse.detach().cpu()) * bs
            tr_kl_sum += float(kl.detach().cpu()) * bs
            tr_n += bs

        # Validation
        model.eval()
        va_mse_sum = 0.0
        va_n = 0
        with torch.no_grad():
            for w_t, w_tp1, ctx in val_loader:
                w_t = w_t.to(device)
                w_tp1 = w_tp1.to(device)
                ctx = ctx.to(device)
                z_t = model.encoder(w_t)
                z_tp1 = model.encoder(w_tp1)
                pred = model.predictor(z_t, ctx)
                mse = F.mse_loss(pred, z_tp1)
                va_mse_sum += float(mse.detach().cpu()) * w_t.size(0)
                va_n += w_t.size(0)

        tr_loss = tr_mse_sum / max(1, tr_n)
        tr_kl = tr_kl_sum / max(1, tr_n)
        val_loss = va_mse_sum / max(1, va_n) if va_n else float("nan")
        dt = time.time() - t0
        print(
            f"epoch {epoch:3d} | train_mse={tr_loss:.5f} train_kl={tr_kl:.5f} "
            f"val_mse={val_loss:.5f} lr={lr:.1e} ({dt:.1f}s)"
        )
        history.append({"epoch": epoch, "train_mse": tr_loss, "train_kl": tr_kl, "val_mse": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[lewm] early stop at epoch {epoch} (patience {patience})")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "best_val_mse": best_val,
            "history": history,
            "config": {"d_model": 64, "in_channels": 5, "ctx_dim": 2},
        },
        ckpt_path,
    )
    print(f"[lewm] saved best checkpoint (val_mse={best_val:.5f}) → {ckpt_path}")
    return {"best_val_mse": best_val, "history": history, "epochs_ran": len(history)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Le-WM temporal encoder training")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--pooled", action="store_true", help="Train one model on all tickers (default)")
    ap.add_argument("--ticker", default=None, help="Train per-ticker model for this ticker")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None, help="Override device (mps/cuda/cpu)")
    ap.add_argument("--out", default=None, help="Checkpoint path (default: data/lewm_checkpoint.pt)")
    ap.add_argument("--limit-tickers", type=int, default=None, help="Debug: first N tickers only")
    args = ap.parse_args()

    config = load_config(args.config)
    lewm_cfg = config.get("lewm", {})
    seed = int(config.get("benchmark", {}).get("seeds", {}).get("model", 123))
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs if args.epochs is not None else lewm_cfg.get("epochs", 100))
    batch_size = int(args.batch_size if args.batch_size is not None else lewm_cfg.get("batch_size", 256))
    lr = float(args.lr if args.lr is not None else lewm_cfg.get("lr", 1e-3))
    reg_lambda = float(lewm_cfg.get("reg_lambda", 0.01))
    patience = int(lewm_cfg.get("patience", 10))

    prefer = args.device or config.get("benchmark", {}).get("device", "mps")
    device = pick_device(prefer)
    print(f"[lewm] device={device} epochs={epochs} batch={batch_size} lr={lr} lambda={reg_lambda}")

    if args.ticker:
        tickers = [args.ticker]
        print(f"[lewm] per-ticker mode: {args.ticker}")
    else:
        tickers = get_universe(config)
        if args.limit_tickers:
            tickers = tickers[: args.limit_tickers]
        print(f"[lewm] pooled mode: {len(tickers)} tickers")

    print("[lewm] building datasets...")
    train_ds = build_dataset(tickers, config, "train")
    val_ds = build_dataset(tickers, config, "val")
    print(f"[lewm] train pairs={len(train_ds)} val pairs={len(val_ds)}")
    if len(train_ds) < 64 or len(val_ds) < 8:
        print("[lewm] ERROR: not enough pairs to train")
        return 2

    # Pin memory doesn't work with MPS; leave it off.
    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=use_pin,
    )

    model = LeWM(
        in_channels=int(lewm_cfg.get("input_channels", 5)),
        d_model=int(lewm_cfg.get("d_model", 64)),
        ctx_dim=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lewm] model params: {n_params:,}")

    data_dir = Path(config["data"].get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    ckpt_path = Path(args.out) if args.out else data_dir / "lewm_checkpoint.pt"

    summary = train(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, reg_lambda=reg_lambda, patience=patience,
        ckpt_path=ckpt_path,
    )
    print(f"[lewm] done. best val MSE={summary['best_val_mse']:.5f} over {summary['epochs_ran']} epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
