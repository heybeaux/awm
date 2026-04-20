"""Tests for Kelly-style edge sizing and vol scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sizing


def test_compute_edge_known_values() -> None:
    """Known probability inputs should map to the expected symmetric edges."""
    probs = np.array([0.75, 0.50, 0.0])
    edges = sizing.compute_edge(probs)

    assert edges.tolist() == pytest.approx([0.5, 0.0, -1.0])


def test_compute_edge_boundary() -> None:
    """Boundary probabilities should map to full negative and positive edge."""
    edges = sizing.compute_edge(np.array([0.0, 1.0]))
    assert edges.tolist() == pytest.approx([-1.0, 1.0])


def test_ewma_vol_positive() -> None:
    """EWMA realized volatility should never be negative where it is defined."""
    returns = pd.Series(np.linspace(-0.02, 0.02, 80))
    vol = sizing.ewma_realized_vol(returns, min_periods=5)

    finite = vol.dropna()
    assert not finite.empty
    assert (finite >= 0.0).all()


def test_size_positions_zero_edge() -> None:
    """Zero edge everywhere should produce zero raw and target weights."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "calibrated_prob": [0.5, 0.5, 0.5, 0.5],
            "ewma_vol_63d": [0.2, 0.2, 0.2, 0.2],
        }
    )

    sized = sizing.size_positions(frame, vol_col="ewma_vol_63d")

    assert np.allclose(sized["raw_weight"], 0.0)
    assert np.allclose(sized["target_weight"], 0.0)


def test_size_positions_max_weight_cap() -> None:
    """Even extreme edge should respect the per-name max weight cap."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "ticker": ["AAA", "BBB", "CCC"],
            "calibrated_prob": [1.0, 1.0, 1.0],
            "ewma_vol_63d": [0.01, 0.01, 0.01],
        }
    )

    sized = sizing.size_positions(frame, vol_col="ewma_vol_63d", max_weight=0.10)

    assert (sized["target_weight"] <= 0.10 + 1e-12).all()
    assert (sized["raw_weight"] <= 0.10 + 1e-12).all()


def test_size_positions_long_only() -> None:
    """Long-only sizing should clip negative edges to zero weight."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "calibrated_prob": [0.0, 0.2, 0.5, 0.9],
            "ewma_vol_63d": [0.15, 0.15, 0.15, 0.15],
        }
    )

    sized = sizing.size_positions(frame, vol_col="ewma_vol_63d", long_only=True)

    assert (sized["target_weight"] >= -1e-12).all()
    assert sized.loc[sized["calibrated_prob"] <= 0.5, "target_weight"].eq(0.0).all()


def test_size_positions_deterministic(synthetic_predictions: pd.DataFrame) -> None:
    """Sizing should be deterministic for identical inputs."""
    first = sizing.size_positions(synthetic_predictions, max_weight=0.08)
    second = sizing.size_positions(synthetic_predictions, max_weight=0.08)

    pd.testing.assert_frame_equal(first, second)
