"""Tests for execution-layer portfolio utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio


def test_cost_applied_correctly() -> None:
    """A 5 bps new trade should reduce gross PnL by the expected transaction cost."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["AAA"],
            "previous_weight": [0.0],
            "executed_weight": [0.5],
            "next_day_return": [0.01],
        }
    )

    costed = portfolio.compute_transaction_costs(frame, cost_bps=5.0, out_col="transaction_cost_5")
    gross_pnl = costed["executed_weight"] * costed["next_day_return"]
    net_pnl = gross_pnl - costed["transaction_cost_5"]
    expected_cost = 2.0 * 5.0e-4 * 0.5

    assert costed.iloc[0]["transaction_cost_5"] == pytest.approx(expected_cost)
    assert (gross_pnl.iloc[0] - net_pnl.iloc[0]) == pytest.approx(expected_cost)


def test_max_drawdown_calculation() -> None:
    """The drawdown magnitude should match the canonical peak-to-trough formula."""
    equity = np.array([1.0, 1.1, 0.9, 1.0])
    expected_magnitude = (1.1 - 0.9) / 1.1

    result = portfolio._max_drawdown(equity)

    assert result == pytest.approx(-expected_magnitude)


def test_zero_weights_zero_pnl() -> None:
    """Zero executed weights should imply zero gross and net portfolio PnL."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "previous_weight": [0.0, 0.0, 0.0, 0.0],
            "executed_weight": [0.0, 0.0, 0.0, 0.0],
            "next_day_return": [0.02, -0.01, 0.03, -0.02],
        }
    )

    costed = portfolio.compute_transaction_costs(frame, cost_bps=5.0)
    gross = costed["executed_weight"] * costed["next_day_return"]
    net = gross - costed["transaction_cost"]

    assert np.allclose(gross, 0.0)
    assert np.allclose(costed["transaction_cost"], 0.0)
    assert np.allclose(net, 0.0)


def test_turnover_calculation() -> None:
    """Moving from [0.5, 0.5] to [0.3, 0.7] should create 0.4 one-way weight change."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "vol_target_weight": [0.5, 0.3, 0.5, 0.7],
            "logit_z": [1.0, 1.0, 1.0, 1.0],
        }
    )

    executed = portfolio.apply_turnover_control(frame, desired_weight_col="vol_target_weight", signal_z_col="logit_z")
    day_two = executed.loc[executed["date"] == pd.Timestamp("2024-01-03")]

    assert day_two["turnover"].sum() == pytest.approx(0.4)
