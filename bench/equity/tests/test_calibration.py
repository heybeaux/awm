"""Tests for isotonic probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import calibration


def _make_window_frame(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    window: str = "w0",
    prob_col: str = "model_prob",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-01", periods=len(y_true)),
            "window": window,
            "y_true": y_true,
            prob_col: y_prob,
        }
    )


def test_identity_calibrator() -> None:
    """IdentityCalibrator should only clip inputs into [0, 1]."""
    calibrator_obj = calibration.IdentityCalibrator()
    values = np.array([-0.2, 0.1, 0.8, 1.4])

    result = calibrator_obj.predict(values)

    assert result.tolist() == pytest.approx([0.0, 0.1, 0.8, 1.0])


def test_isotonic_fit_monotone(tmp_path: pytest.TempPathFactory) -> None:
    """Fitted isotonic outputs must be monotonically non-decreasing."""
    rng = np.random.default_rng(42)
    probs = np.linspace(0.05, 0.95, 80)
    y_true = (probs + rng.normal(0.0, 0.08, len(probs)) > 0.5).astype(int)
    frame = _make_window_frame(y_true, probs)

    calibration.fit_window_calibrators(
        frame,
        model_columns={"Model": "model_prob"},
        output_dir=tmp_path,
    )
    calibrators = calibration.load_saved_calibrators(output_dir=tmp_path)
    fitted = calibrators["w0"]["Model"]

    grid = np.linspace(0.0, 1.0, 101)
    predictions = fitted.predict(grid)

    assert np.all(np.diff(predictions) >= -1e-12)


def test_isotonic_small_sample_fallback(tmp_path: pytest.TempPathFactory) -> None:
    """Small calibration samples should fall back to IdentityCalibrator."""
    probs = np.linspace(0.1, 0.9, 10)
    y_true = np.array([0, 1] * 5)
    frame = _make_window_frame(y_true, probs)

    calibration.fit_window_calibrators(
        frame,
        model_columns={"Model": "model_prob"},
        output_dir=tmp_path,
    )
    calibrators = calibration.load_saved_calibrators(output_dir=tmp_path)

    assert isinstance(calibrators["w0"]["Model"], calibration.IdentityCalibrator)


def test_calibration_no_leakage(tmp_path: pytest.TempPathFactory) -> None:
    """Prediction-frame labels must not affect fitted calibrator outputs."""
    rng = np.random.default_rng(42)
    train_prob = np.linspace(0.05, 0.95, 60)
    train_true = (train_prob > 0.5).astype(int)
    calibration_frame = _make_window_frame(train_true, train_prob, window="train")

    test_prob = rng.uniform(0.1, 0.9, 30)
    prediction_a = _make_window_frame(np.zeros(30, dtype=int), test_prob, window="train")
    prediction_b = _make_window_frame(np.ones(30, dtype=int), test_prob, window="train")

    calibrated_a, artifacts_a = calibration.calibrate_walk_forward_predictions(
        calibration_frame,
        prediction_frame=prediction_a,
        model_columns={"Model": "model_prob"},
        output_dir=tmp_path / "run_a",
    )
    calibrated_b, artifacts_b = calibration.calibrate_walk_forward_predictions(
        calibration_frame,
        prediction_frame=prediction_b,
        model_columns={"Model": "model_prob"},
        output_dir=tmp_path / "run_b",
    )

    assert artifacts_a[0].sample_count == len(calibration_frame)
    assert artifacts_b[0].sample_count == len(calibration_frame)
    assert calibrated_a["model_prob_calibrated"].tolist() == pytest.approx(calibrated_b["model_prob_calibrated"].tolist())


def test_brier_score_improves(tmp_path: pytest.TempPathFactory) -> None:
    """Calibration should not worsen Brier score on the fit/calibration set."""
    rng = np.random.default_rng(42)
    logits = np.linspace(-2.5, 2.5, 100)
    raw_prob = np.clip(1.0 / (1.0 + np.exp(-logits)) + rng.normal(0.0, 0.08, 100), 0.0, 1.0)
    y_true = (logits > 0.25).astype(int)
    frame = _make_window_frame(y_true, raw_prob)

    artifacts = calibration.fit_window_calibrators(
        frame,
        model_columns={"Model": "model_prob"},
        output_dir=tmp_path,
    )

    assert len(artifacts) == 1
    assert artifacts[0].brier_after is not None
    assert artifacts[0].brier_before is not None
    assert artifacts[0].brier_after <= artifacts[0].brier_before + 1e-12
