"""Tests for deterministic market regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import regime_detector


def test_hmm_fit_only_training() -> None:
    """The rule-based regime pipeline must not let future rows alter earlier labels."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    base_prices = 100.0 * np.exp(np.linspace(0.0, 0.25, len(dates)))
    base = pd.DataFrame({"adj_close": base_prices}, index=dates)
    with_future_shock = base.copy()
    with_future_shock.iloc[-1, 0] *= 0.2

    baseline = regime_detector.detect_regimes(base)
    shocked = regime_detector.detect_regimes(with_future_shock)

    cutoff = dates[-2]
    pd.testing.assert_series_equal(
        baseline.loc[:cutoff, "regime"],
        shocked.loc[:cutoff, "regime"],
        check_names=False,
    )


def test_regime_labels_valid() -> None:
    """All emitted labels should belong to the defined five-regime vocabulary."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    returns = rng.normal(0.0002, 0.01, len(dates))
    prices = 100.0 * np.cumprod(1.0 + returns)
    detected = regime_detector.detect_regimes(pd.DataFrame({"adj_close": prices}, index=dates))

    labels = set(detected["regime"].dropna().unique())
    assert labels
    assert labels.issubset(set(regime_detector.REGIME_ORDER))


def test_pure_uptrend_labels() -> None:
    """A smooth persistent uptrend should not be classified as crisis."""
    dates = pd.bdate_range("2020-01-01", periods=400)
    prices = np.linspace(100.0, 200.0, len(dates))
    detected = regime_detector.detect_regimes(pd.DataFrame({"adj_close": prices}, index=dates))

    ready = detected["regime"].dropna()
    assert not ready.empty
    assert (ready != "crisis").all()


def test_regime_deterministic_with_seed() -> None:
    """Given the same seeded synthetic input, detected regimes should be identical."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=450)
    prices_a = 100.0 * np.cumprod(1.0 + rng_a.normal(0.0002, 0.012, len(dates)))
    prices_b = 100.0 * np.cumprod(1.0 + rng_b.normal(0.0002, 0.012, len(dates)))

    first = regime_detector.detect_regimes(pd.DataFrame({"adj_close": prices_a}, index=dates))
    second = regime_detector.detect_regimes(pd.DataFrame({"adj_close": prices_b}, index=dates))

    pd.testing.assert_frame_equal(first, second)
