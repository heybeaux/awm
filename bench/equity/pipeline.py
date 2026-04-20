"""Market data pipeline for the AWM equity benchmark.

Responsibilities:
  1. Download daily OHLCV per ticker via yfinance → Parquet cache.
  2. Engineer 10 per-bar features (plus 60-day price_history for Le-WM).
  3. Attach targets (direction_5d, regime).
  4. Produce strict temporal train/val/test splits.
  5. Validate gaps, date coverage, class balance.

Anti-leakage rules enforced throughout:
  - All rolling features use only past data (pandas `.rolling` is backward-
    looking by default, `.shift(1)` is added where the feature needs to avoid
    including the current row's value).
  - Targets are forward-looking by construction; they are the supervised
    signal, not an input feature.
  - Splits are strictly temporal per ticker — no shuffling across time.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from targets import add_targets
from universe import get_universe

SEED_DATA = 42
LOOKBACK = 60


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _load_raw(ticker: str, data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / f"{ticker}_raw.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _save_raw(ticker: str, df: pd.DataFrame, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(data_dir / f"{ticker}_raw.parquet")


def _fetch_one(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch a single ticker. Returns a DataFrame with columns
    [open, high, low, close, adj_close, volume] indexed by date.
    """
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when multiple tickers are requested;
    # flatten to single level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    # Some yfinance versions return auto-adjusted close as 'close' with no
    # separate 'adj_close'. If that happens, alias them.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    needed = ["open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in needed if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    df = df.sort_index()
    return df


def download_data(
    universe: list[str],
    start_date: str,
    end_date: str,
    data_dir: str | Path,
    delay_sec: float = 0.5,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for every ticker. Cached as Parquet under `data_dir`.

    Returns {ticker: raw_df}. Tickers with no data are omitted.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(universe, 1):
        cached = None if force else _load_raw(ticker, data_dir)
        if cached is not None and not cached.empty:
            out[ticker] = cached
            print(f"  [{i}/{len(universe)}] {ticker}: cached ({len(cached)} rows)")
            continue

        try:
            df = _fetch_one(ticker, start_date, end_date)
        except Exception as exc:
            print(f"  [{i}/{len(universe)}] {ticker}: FETCH ERROR {exc}")
            df = pd.DataFrame()

        if df.empty:
            print(f"  [{i}/{len(universe)}] {ticker}: no data")
            time.sleep(delay_sec)
            continue

        _save_raw(ticker, df, data_dir)
        out[ticker] = df
        print(f"  [{i}/{len(universe)}] {ticker}: downloaded ({len(df)} rows, "
              f"{df.index.min().date()} → {df.index.max().date()})")
        time.sleep(delay_sec)

    return out


# ---------------------------------------------------------------------------
# Technical indicators (manual — no ta-lib dependency)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (EMA-based)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd - signal_line


def _bb_pctb(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.Series:
    mid = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = mid + k * std
    lower = mid - k * std
    denom = (upper - lower).replace(0, np.nan)
    return (close - lower) / denom


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "daily_return",
    "log_return",
    "intraday_range",
    "volume_ratio",
    "roll_return_5d",
    "roll_return_10d",
    "roll_return_20d",
    "roll_vol_5d",
    "roll_vol_10d",
    "roll_vol_20d",
]


def engineer_features(ticker: str, raw_df: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Compute 10 scalar features + 60-day price_history array per bar.

    Input: raw OHLCV (index=date, cols=open/high/low/close/adj_close/volume).
    Output: DataFrame indexed by date with:
        - FEATURE_COLUMNS (10 scalars)
        - price_history (list-of-5-element arrays, length=lookback per row, or None)
        - open, high, low, close, adj_close, volume (carried through for targets)
        - ticker
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy().sort_index()
    adj = df["adj_close"]
    vol = df["volume"]

    # Returns
    df["daily_return"] = adj.pct_change()
    df["log_return"] = np.log(adj / adj.shift(1))

    # Intraday range uses raw H/L/C (unadjusted is fine here — it's normalised)
    df["intraday_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # Volume ratio vs. 20-day SMA (past-only: shift(1) so today's volume is
    # not included in the denominator SMA, preserving the "relative to recent
    # history" interpretation without leakage from future bars — though the
    # current-bar volume IS known intraday, so including it is also defensible.
    # We follow the common "volume today vs. past 20-day average" convention.)
    vol_sma20 = vol.rolling(window=20).mean().shift(1)
    df["volume_ratio"] = vol / vol_sma20.replace(0, np.nan)

    # Rolling cumulative returns
    df["roll_return_5d"] = adj / adj.shift(5) - 1.0
    df["roll_return_10d"] = adj / adj.shift(10) - 1.0
    df["roll_return_20d"] = adj / adj.shift(20) - 1.0

    # Rolling volatility of daily returns
    df["roll_vol_5d"] = df["daily_return"].rolling(window=5).std()
    df["roll_vol_10d"] = df["daily_return"].rolling(window=10).std()
    df["roll_vol_20d"] = df["daily_return"].rolling(window=20).std()

    # Price history: array of last `lookback` (O,H,L,C,V) rows.
    # We use adjusted close for C, raw O/H/L/V. Store as list-of-lists so
    # pyarrow can serialise it. Rows without enough history get None.
    channels = np.column_stack([
        df["open"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["adj_close"].to_numpy(),
        df["volume"].to_numpy(),
    ])
    price_history: list[Any] = [None] * len(df)
    for i in range(lookback - 1, len(df)):
        window = channels[i - lookback + 1 : i + 1]  # (lookback, 5)
        if np.isnan(window).any():
            price_history[i] = None
        else:
            price_history[i] = window.tolist()
    df["price_history"] = price_history

    df["ticker"] = ticker
    return df


# ---------------------------------------------------------------------------
# Splits & validation
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    target_col: str = "direction_5d",
) -> dict[str, pd.DataFrame]:
    """Strict temporal split on rows with a defined target.

    Returns {'train': df, 'val': df, 'test': df}.
    """
    if df.empty:
        return {"train": df, "val": df, "test": df}

    usable = df.dropna(subset=[target_col]).sort_index()
    n = len(usable)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = usable.iloc[:n_train]
    val = usable.iloc[n_train : n_train + n_val]
    test = usable.iloc[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def validate_data(df: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Summarise gaps, coverage, class balance. Returns a dict of stats."""
    if df.empty:
        return {"ticker": ticker, "empty": True}

    idx = pd.DatetimeIndex(df.index).sort_values()
    diffs = idx.to_series().diff().dt.days.dropna()
    gaps = int((diffs > 5).sum())  # >5 calendar days = suspicious (weekend=3)

    stats: dict[str, Any] = {
        "ticker": ticker,
        "rows": int(len(df)),
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "gaps_over_5d": gaps,
        "nan_adj_close": int(df["adj_close"].isna().sum()) if "adj_close" in df.columns else None,
    }
    if "direction_5d" in df.columns:
        vc = df["direction_5d"].value_counts(dropna=False).to_dict()
        stats["direction_5d_counts"] = {str(k): int(v) for k, v in vc.items()}
    if "regime" in df.columns:
        vc = df["regime"].value_counts(dropna=False).to_dict()
        stats["regime_counts"] = {str(k): int(v) for k, v in vc.items()}
    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _feature_path(data_dir: Path, ticker: str) -> Path:
    return data_dir / f"{ticker}_features.parquet"


def run_pipeline(
    config: dict,
    universe: list[str] | None = None,
    force_download: bool = False,
) -> dict[str, dict]:
    """End-to-end: download → features → targets → splits → validate.

    Writes per-ticker feature Parquet files to `{data_dir}/{ticker}_features.parquet`.
    Returns {ticker: validation_stats}.
    """
    _set_seeds(SEED_DATA)

    data_cfg = config["data"]
    data_dir = Path(data_cfg.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir

    tickers = universe if universe is not None else get_universe(config)

    print(f"[pipeline] universe: {len(tickers)} tickers → {tickers[:5]}... ")
    print(f"[pipeline] date range: {data_cfg['start_date']} → {data_cfg['end_date']}")
    print(f"[pipeline] data dir: {data_dir}")
    print()

    print("[pipeline] downloading raw OHLCV...")
    raw = download_data(
        tickers,
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        data_dir=data_dir,
        delay_sec=data_cfg.get("fetch_delay_sec", 0.5),
        force=force_download,
    )
    print(f"[pipeline] downloaded {len(raw)}/{len(tickers)} tickers\n")

    print("[pipeline] engineering features + targets...")
    lookback = int(data_cfg.get("lookback_days", LOOKBACK))
    direction_threshold = float(data_cfg.get("direction_threshold", 0.01))
    stats: dict[str, dict] = {}

    for ticker, raw_df in raw.items():
        if raw_df.empty:
            continue
        feats = engineer_features(ticker, raw_df, lookback=lookback)
        feats = add_targets(feats, direction_threshold=direction_threshold)
        feats.to_parquet(_feature_path(data_dir, ticker))

        splits = make_splits(
            feats,
            train_ratio=data_cfg.get("train_ratio", 0.70),
            val_ratio=data_cfg.get("val_ratio", 0.15),
        )
        s = validate_data(feats, ticker)
        s["splits"] = {
            k: {
                "rows": int(len(v)),
                "start": str(v.index.min().date()) if len(v) else None,
                "end": str(v.index.max().date()) if len(v) else None,
                "pos_rate": (float(v["direction_5d"].mean()) if len(v) else None),
            }
            for k, v in splits.items()
        }
        stats[ticker] = s
        print(f"  {ticker}: {s['rows']} rows, {s['start']}→{s['end']}, "
              f"train/val/test = "
              f"{s['splits']['train']['rows']}/"
              f"{s['splits']['val']['rows']}/"
              f"{s['splits']['test']['rows']}")

    print(f"\n[pipeline] wrote feature Parquet for {len(stats)} tickers to {data_dir}")
    return stats


def load_features(ticker: str, config: dict) -> pd.DataFrame:
    data_dir = Path(config["data"].get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent / data_dir
    return pd.read_parquet(_feature_path(data_dir, ticker))


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else Path(__file__).parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AWM equity benchmark data pipeline")
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Override universe with specific tickers (space-separated)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Take only the first N tickers of the universe (for quick tests)")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.tickers:
        universe = args.tickers
    else:
        universe = get_universe(config)
        if args.limit:
            universe = universe[: args.limit]

    stats = run_pipeline(config, universe=universe, force_download=args.force)

    # Report gap flags
    bad = [t for t, s in stats.items() if s.get("gaps_over_5d", 0) > 3]
    if bad:
        print(f"\n[pipeline] WARNING: gaps >5d in: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
