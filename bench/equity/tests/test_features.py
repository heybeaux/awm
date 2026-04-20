"""Tests for feature engineering in pipeline.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline
from tests.conftest import make_raw_ohlcv


def test_no_lookahead_rolling() -> None:
    """Rolling features at time T must not change when T+1 data changes."""
    dates = pd.bdate_range("2020-01-01", periods=50)
    prices = 100.0 + np.linspace(0.0, 20.0, len(dates)) + 2.0 * np.sin(np.arange(len(dates)) / 3.0)
    base = make_raw_ohlcv(prices, dates=dates, volume=np.linspace(1_000, 2_000, len(dates)))
    changed = base.copy()
    changed.iloc[41, changed.columns.get_loc("adj_close")] = 10_000.0
    changed.iloc[41, changed.columns.get_loc("close")] = 10_000.0

    features_base = pipeline.engineer_features("AAA", base)
    features_changed = pipeline.engineer_features("AAA", changed)
    t = dates[40]

    for column in ["roll_return_5d", "roll_vol_5d", "roll_vol_10d", "volume_ratio"]:
        assert features_base.loc[t, column] == pytest.approx(features_changed.loc[t, column], rel=0, abs=1e-12)


def test_feature_columns_complete(synthetic_features: pd.DataFrame) -> None:
    """The engineered output should include every declared feature column."""
    assert set(pipeline.FEATURE_COLUMNS).issubset(synthetic_features.columns)


def test_nan_propagation() -> None:
    """Input NaNs should propagate through dependent features instead of becoming zero."""
    dates = pd.bdate_range("2020-01-01", periods=35)
    prices = np.linspace(100.0, 120.0, len(dates))
    raw = make_raw_ohlcv(prices, dates=dates, volume=np.full(len(dates), 1_000_000.0))
    raw.iloc[20, raw.columns.get_loc("adj_close")] = np.nan

    features = pipeline.engineer_features("AAA", raw, lookback=10)

    assert pd.isna(features.iloc[20]["daily_return"])
    assert pd.isna(features.iloc[20]["log_return"])
    assert features.iloc[20]["price_history"] is None
    assert features["roll_return_5d"].isna().sum() > 0


def test_volume_ratio_uses_past() -> None:
    """volume_ratio should divide by a trailing mean of prior volumes, excluding today."""
    dates = pd.bdate_range("2020-01-01", periods=25)
    prices = np.linspace(100.0, 124.0, len(dates))
    volume = np.arange(1, len(dates) + 1, dtype=float)
    raw = make_raw_ohlcv(prices, dates=dates, volume=volume)

    features = pipeline.engineer_features("AAA", raw)
    target_idx = dates[20]
    expected = volume[20] / volume[:20].mean()

    assert features.loc[target_idx, "volume_ratio"] == pytest.approx(expected)


def test_roll_vol_positive(synthetic_features: pd.DataFrame) -> None:
    """Rolling volatility values should always be non-negative where defined."""
    for col in ["roll_vol_5d", "roll_vol_10d", "roll_vol_20d"]:
        finite = synthetic_features[col].dropna()
        assert not finite.empty
        assert (finite >= 0.0).all()


def test_intraday_range_positive() -> None:
    """Intraday range should be non-negative (high >= low by definition)."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    prices = np.linspace(100.0, 120.0, len(dates))
    raw = make_raw_ohlcv(prices, dates=dates, volume=np.full(len(dates), 1_000_000.0))

    features = pipeline.engineer_features("AAA", raw)
    finite = features["intraday_range"].dropna()

    assert (finite >= 0.0).all()


def test_daily_return_calculation() -> None:
    """daily_return should equal adj_close[t] / adj_close[t-1] - 1."""
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = np.array([100.0, 102.0, 101.0, 104.0, 108.0, 110.0])
    raw = make_raw_ohlcv(prices, dates=dates, volume=np.full(len(dates), 1_000_000.0))

    features = pipeline.engineer_features("AAA", raw)

    expected = prices[4] / prices[3] - 1.0
    assert features.iloc[4]["daily_return"] == pytest.approx(expected)
