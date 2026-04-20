"""Adversarial sanity checks for predictive signal quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import sizing


def test_random_predictions_near_zero_sharpe() -> None:
    """Independent random probabilities should not generate material Sharpe."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", periods=2000)
    tickers = [f"T{i:02d}" for i in range(20)]

    parts: list[pd.DataFrame] = []
    for ticker in tickers:
        next_day_return = rng.normal(0.0, 0.01, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "daily_return": next_day_return,
                "adj_close": 100.0 * np.cumprod(1.0 + next_day_return),
                "calibrated_prob": rng.uniform(0.4, 0.6, len(dates)),
            }
        )
        parts.append(frame)

    panel = pd.concat(parts, ignore_index=True)
    sized = sizing.size_positions(
        panel,
        prob_col="calibrated_prob",
        long_only=False,
        max_weight=0.05,
        target_gross_exposure=1.0,
    )
    sized["gross_portfolio_return"] = sized["target_weight"] * sized["daily_return"]
    daily = sized.groupby("date")["gross_portfolio_return"].sum()
    sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252.0)

    assert sharpe == pytest.approx(0.0, abs=0.3)


def test_shuffled_labels_auc_near_random() -> None:
    """Shuffling labels should collapse AUC toward random guessing."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 5000)
    signal = np.clip(0.15 + 0.7 * y_true + rng.normal(0.0, 0.12, len(y_true)), 0.0, 1.0)
    shuffled = rng.permutation(y_true)

    auc = roc_auc_score(shuffled, signal)

    assert auc == pytest.approx(0.5, abs=0.05)


def test_perfect_lookahead_detected() -> None:
    """Injecting tomorrow's return as a feature should create an obviously inflated AUC."""
    rng = np.random.default_rng(42)
    next_day_return = rng.normal(0.0, 0.01, 3000)
    y_true = (next_day_return > 0.0).astype(int)
    lookahead_feature = next_day_return

    auc = roc_auc_score(y_true, lookahead_feature)

    assert auc > 0.9
