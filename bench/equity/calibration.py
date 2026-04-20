"""Phase 1 calibration utilities for the equity benchmark v2.

This module fits per-window isotonic calibrators on out-of-fold predictions,
persists them with joblib, and writes reliability diagrams to the shared
results directory.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/equity_bench_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/equity_bench_cache/matplotlib")

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get("EQUITY_BENCH_RESULTS_DIR", BENCH_DIR.parent / "results"))
CALIBRATION_DIR = RESULTS_DIR / "calibration"

DEFAULT_MODEL_COLUMNS: dict[str, str] = {
    "XGB": "xgb_prob",
    "Le-WM": "lewm_prob",
    "Fusion": "fusion_prob",
}


class IdentityCalibrator:
    """Fallback calibrator when isotonic fitting is not feasible."""

    def predict(self, values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.clip(arr, 0.0, 1.0)


@dataclass
class CalibrationArtifact:
    window: str
    model: str
    calibrator_path: str
    reliability_path: str
    sample_count: int
    positive_rate: float | None
    brier_before: float | None
    brier_after: float | None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        out = value.item()
        if isinstance(out, float) and (math.isnan(out) or math.isinf(out)):
            return None
        return out
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not serializable: {type(value)!r}")


def _ensure_date_column(frame: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in frame.columns:
        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.reset_index().rename(columns={frame.index.name or "index": date_col})
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    raise KeyError(f"missing date column {date_col!r}")


def _valid_mask(y_true: pd.Series, y_prob: pd.Series) -> np.ndarray:
    return (~y_true.isna() & ~y_prob.isna()).to_numpy()


def _window_label(window_value: Any) -> str:
    return str(window_value).replace(os.sep, "_").replace(" ", "_")


def _fit_single_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression | IdentityCalibrator:
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return IdentityCalibrator()
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
    return calibrator


def _brier_or_none(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    return float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))


def _plot_reliability(
    y_true: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    output_path: Path,
    title: str,
    n_bins: int = 10,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title)

    for ax, values, label in [
        (axes[0], raw_prob, "Before"),
        (axes[1], calibrated_prob, "After"),
    ]:
        if len(y_true) and len(np.unique(values)) > 1:
            frac_pos, mean_pred = calibration_curve(
                y_true,
                np.clip(values, 0.0, 1.0),
                n_bins=max(2, min(n_bins, len(values))),
                strategy="quantile",
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(label)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_window_calibrators(
    oof_frame: pd.DataFrame,
    *,
    model_columns: Mapping[str, str] = DEFAULT_MODEL_COLUMNS,
    target_col: str = "y_true",
    window_col: str = "window",
    date_col: str = "date",
    output_dir: str | Path = CALIBRATION_DIR,
) -> list[CalibrationArtifact]:
    """Fit and persist isotonic calibrators for each model and walk-forward window."""
    if target_col not in oof_frame.columns:
        raise KeyError(f"missing target column {target_col!r}")
    if window_col not in oof_frame.columns:
        raise KeyError(f"missing window column {window_col!r}")

    df = _ensure_date_column(oof_frame, date_col).sort_values([window_col, date_col])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[CalibrationArtifact] = []
    for window_value, window_df in df.groupby(window_col, sort=True):
        window_key = _window_label(window_value)
        for model_name, prob_col in model_columns.items():
            if prob_col not in window_df.columns:
                continue

            mask = _valid_mask(window_df[target_col], window_df[prob_col])
            if not mask.any():
                continue

            y_true = window_df.loc[mask, target_col].astype(int).to_numpy()
            raw_prob = window_df.loc[mask, prob_col].astype(float).to_numpy()
            calibrator = _fit_single_calibrator(y_true, raw_prob)
            calibrated_prob = np.clip(calibrator.predict(raw_prob), 0.0, 1.0)

            model_dir = output_dir / model_name.lower().replace(" ", "_").replace("-", "_")
            model_dir.mkdir(parents=True, exist_ok=True)
            calibrator_path = model_dir / f"{window_key}.joblib"
            reliability_path = model_dir / f"{window_key}_reliability.png"

            joblib.dump(calibrator, calibrator_path)
            _plot_reliability(
                y_true=y_true,
                raw_prob=raw_prob,
                calibrated_prob=calibrated_prob,
                output_path=reliability_path,
                title=f"{model_name} calibration: {window_key}",
            )

            artifacts.append(
                CalibrationArtifact(
                    window=window_key,
                    model=model_name,
                    calibrator_path=str(calibrator_path),
                    reliability_path=str(reliability_path),
                    sample_count=int(len(y_true)),
                    positive_rate=float(np.mean(y_true)) if len(y_true) else None,
                    brier_before=_brier_or_none(y_true, raw_prob),
                    brier_after=_brier_or_none(y_true, calibrated_prob),
                )
            )

    manifest_path = output_dir / "calibration_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump([asdict(artifact) for artifact in artifacts], handle, indent=2, default=_json_default)
    return artifacts


def load_saved_calibrators(
    output_dir: str | Path = CALIBRATION_DIR,
) -> dict[str, dict[str, IsotonicRegression | IdentityCalibrator]]:
    """Load persisted calibrators keyed as calibrators[window][model]."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "calibration_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing calibration manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    calibrators: dict[str, dict[str, IsotonicRegression | IdentityCalibrator]] = {}
    for artifact in manifest:
        calibrators.setdefault(artifact["window"], {})[artifact["model"]] = joblib.load(
            artifact["calibrator_path"]
        )
    return calibrators


def apply_window_calibrators(
    frame: pd.DataFrame,
    *,
    calibrators: Mapping[str, Mapping[str, IsotonicRegression | IdentityCalibrator]] | None = None,
    model_columns: Mapping[str, str] = DEFAULT_MODEL_COLUMNS,
    window_col: str = "window",
    date_col: str = "date",
    output_dir: str | Path = CALIBRATION_DIR,
) -> pd.DataFrame:
    """Apply saved per-window calibrators to a prediction frame."""
    if window_col not in frame.columns:
        raise KeyError(f"missing window column {window_col!r}")

    calibrator_map = dict(calibrators) if calibrators is not None else load_saved_calibrators(output_dir)
    df = _ensure_date_column(frame, date_col).sort_values([window_col, date_col]).copy()

    for model_name, prob_col in model_columns.items():
        if prob_col not in df.columns:
            continue
        out_col = f"{prob_col}_calibrated"
        df[out_col] = np.nan

        for window_value, idx in df.groupby(window_col, sort=False).groups.items():
            window_key = _window_label(window_value)
            window_calibrator = calibrator_map.get(window_key, {}).get(model_name, IdentityCalibrator())
            values = df.loc[idx, prob_col].astype(float).to_numpy()
            valid = ~np.isnan(values)
            if valid.any():
                calibrated = np.full(len(values), np.nan, dtype=float)
                calibrated[valid] = np.clip(window_calibrator.predict(values[valid]), 0.0, 1.0)
                df.loc[idx, out_col] = calibrated

    return df


def calibrate_walk_forward_predictions(
    calibration_frame: pd.DataFrame,
    *,
    prediction_frame: pd.DataFrame | None = None,
    model_columns: Mapping[str, str] = DEFAULT_MODEL_COLUMNS,
    target_col: str = "y_true",
    window_col: str = "window",
    date_col: str = "date",
    output_dir: str | Path = CALIBRATION_DIR,
) -> tuple[pd.DataFrame, list[CalibrationArtifact]]:
    """Fit per-window calibrators on OOF data and apply them to predictions."""
    artifacts = fit_window_calibrators(
        calibration_frame,
        model_columns=model_columns,
        target_col=target_col,
        window_col=window_col,
        date_col=date_col,
        output_dir=output_dir,
    )
    calibrated_frame = apply_window_calibrators(
        prediction_frame if prediction_frame is not None else calibration_frame,
        model_columns=model_columns,
        window_col=window_col,
        date_col=date_col,
        output_dir=output_dir,
    )
    return calibrated_frame, artifacts


__all__ = [
    "CALIBRATION_DIR",
    "DEFAULT_MODEL_COLUMNS",
    "IdentityCalibrator",
    "CalibrationArtifact",
    "fit_window_calibrators",
    "load_saved_calibrators",
    "apply_window_calibrators",
    "calibrate_walk_forward_predictions",
]
