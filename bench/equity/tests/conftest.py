"""Shared synthetic fixtures for the equity benchmark test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline


RNG_SEED = 42
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def make_raw_ohlcv(
    adj_close: np.ndarray | list[float],
    *,
    dates: pd.DatetimeIndex | None = None,
    volume: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV frame from synthetic adjusted closes."""
    adj = np.asarray(adj_close, dtype=float)
    idx = dates if dates is not None else pd.bdate_range("2020-01-01", periods=len(adj))
    vol = np.asarray(volume if volume is not None else np.full(len(adj), 1_000_000.0), dtype=float)

    open_ = adj * 0.998
    close = adj * 1.001
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": adj,
            "volume": vol,
        },
        index=idx,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    """Shared reproducible RNG."""
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def synthetic_prices(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Five 500-day synthetic OHLCV histories."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    out: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(TICKERS):
        daily_return = rng.normal(loc=0.0003 + i * 0.00005, scale=0.012 + i * 0.001, size=len(dates))
        adj_close = 100.0 * np.cumprod(1.0 + daily_return)
        volume = rng.integers(800_000 + i * 100_000, 2_000_000 + i * 100_000, size=len(dates))
        out[ticker] = make_raw_ohlcv(adj_close, dates=dates, volume=volume)

    return out


@pytest.fixture
def synthetic_price_panel(synthetic_prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long-form synthetic OHLCV panel with ticker/date columns."""
    parts: list[pd.DataFrame] = []
    for ticker, frame in synthetic_prices.items():
        tmp = frame.reset_index().rename(columns={"index": "date"})
        tmp["ticker"] = ticker
        tmp["daily_return"] = tmp["adj_close"].pct_change()
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def synthetic_features(synthetic_prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Engineered feature panel from the synthetic OHLCV fixture."""
    parts = [
        pipeline.engineer_features(ticker=ticker, raw_df=frame).reset_index().rename(columns={"index": "date"})
        for ticker, frame in synthetic_prices.items()
    ]
    return pd.concat(parts, ignore_index=True)


@pytest.fixture
def synthetic_predictions(
    synthetic_price_panel: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Synthetic calibrated probabilities aligned to the price panel."""
    frame = synthetic_price_panel.loc[:, ["date", "ticker", "adj_close", "daily_return"]].copy()
    frame["calibrated_prob"] = rng.uniform(0.3, 0.7, len(frame))
    return frame
