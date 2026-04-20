"""Phase 2 regime-weighted ensemble utilities for the equity benchmark v2.

Implements:
  1. Softmax regime posterior estimation from SPY indicator states.
  2. Monthly expanding-window fitting of regime-conditional stacking weights.
  3. Rolling-Sharpe fallback weights when the stack is under-sampled.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from regime_detector import (
    DATA_DIR,
    INDICATOR_COLUMNS,
    REGIME_ORDER,
    detect_regimes,
    load_price_history,
)


DEFAULT_MODEL_PROB_CANDIDATES: dict[str, tuple[str, ...]] = {
    "XGB": ("xgb_prob_calibrated", "xgb_prob"),
    "Le-WM": ("lewm_prob_calibrated", "lewm_prob"),
    "Fusion": ("fusion_prob_calibrated", "fusion_prob"),
}
DEFAULT_MODEL_LOGIT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "XGB": ("xgb_logit",),
    "Le-WM": ("lewm_logit",),
    "Fusion": ("fusion_logit",),
}
DEFAULT_MODEL_RETURN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "XGB": ("xgb_return", "xgb_net_return"),
    "Le-WM": ("lewm_return", "lewm_net_return"),
    "Fusion": ("fusion_return", "fusion_net_return"),
}

POSTERIOR_PREFIX = "pi_"
TRADING_DAYS_PER_YEAR = 252.0


@dataclass
class RegimePosteriorResult:
    posterior_frame: pd.DataFrame
    update_summary: pd.DataFrame


@dataclass
class EnsembleFitResult:
    predictions: pd.DataFrame
    posterior: pd.DataFrame
    monthly_weights: pd.DataFrame
    posterior_updates: pd.DataFrame


def _ensure_date_column(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col in frame.columns:
        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out.sort_values(date_col)
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.reset_index().rename(columns={frame.index.name or "index": date_col})
        out[date_col] = pd.to_datetime(out[date_col])
        return out.sort_values(date_col)
    raise KeyError(f"missing date column {date_col!r}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))


def safe_logit(values: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _posterior_columns() -> list[str]:
    return [f"{POSTERIOR_PREFIX}{regime}" for regime in REGIME_ORDER]


def _resolve_candidate_columns(
    frame: pd.DataFrame,
    candidates: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for model_name, choices in candidates.items():
        for candidate in choices:
            if candidate in frame.columns:
                resolved[model_name] = candidate
                break
    return resolved


def _soft_regime_target(regime: str | None, *, peak: float = 0.60) -> np.ndarray:
    base = np.full(len(REGIME_ORDER), (1.0 - peak) / max(len(REGIME_ORDER) - 1, 1), dtype=float)
    if isinstance(regime, str) and regime in REGIME_ORDER:
        base[REGIME_ORDER.index(regime)] = peak
    else:
        base[:] = 1.0 / len(REGIME_ORDER)
    return base


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0.0] = 1.0
    return arr / row_sum


def _month_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("M").dt.to_timestamp()


def build_regime_training_frame(
    ticker: str = "SPY",
    *,
    data_dir: str | Path = DATA_DIR,
) -> pd.DataFrame:
    """Load SPY and produce the trailing indicator + hard-regime frame."""
    prices = load_price_history(ticker=ticker, data_dir=data_dir)
    return detect_regimes(prices).reset_index().rename(columns={"index": "date"})


def fit_regime_posterior(
    regime_frame: pd.DataFrame,
    *,
    date_col: str = "date",
    label_col: str = "regime",
    indicator_cols: Sequence[str] = INDICATOR_COLUMNS,
    min_train_samples: int = 252,
    max_iter: int = 500,
) -> RegimePosteriorResult:
    """Fit an expanding-window softmax posterior over the five hard regimes."""
    df = _ensure_date_column(regime_frame, date_col=date_col).copy()
    required = set(indicator_cols) | {label_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"missing regime columns: {missing}")

    posterior_cols = _posterior_columns()
    for col in posterior_cols:
        df[col] = np.nan
    df["posterior_source"] = pd.NA
    df["update_month"] = _month_start(df[date_col])

    update_rows: list[dict[str, Any]] = []
    unique_months = df["update_month"].drop_duplicates().sort_values()

    for month_start in unique_months:
        current_mask = df["update_month"] == month_start
        current_idx = df.index[current_mask]
        train = df.loc[
            (df[date_col] < month_start)
            & df[label_col].notna()
            & df.loc[:, list(indicator_cols)].notna().all(axis=1)
        ].copy()

        source = "fallback_soft_label"
        probs = np.vstack([_soft_regime_target(value) for value in df.loc[current_idx, label_col]])
        classes_seen = sorted(train[label_col].unique().tolist()) if not train.empty else []

        if len(train) >= min_train_samples and train[label_col].nunique() >= 2:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train.loc[:, list(indicator_cols)].to_numpy(dtype=float))
            y_train = train[label_col].to_numpy(dtype=str)
            model = LogisticRegression(
                solver="lbfgs",
                max_iter=max_iter,
            )
            model.fit(X_train, y_train)

            current_features = df.loc[current_idx, list(indicator_cols)].to_numpy(dtype=float)
            current_valid = ~np.isnan(current_features).any(axis=1)
            probs = np.vstack([_soft_regime_target(value) for value in df.loc[current_idx, label_col]])

            if current_valid.any():
                transformed = scaler.transform(current_features[current_valid])
                model_probs = model.predict_proba(transformed)
                mapped = np.full((model_probs.shape[0], len(REGIME_ORDER)), 1e-9, dtype=float)
                for idx, class_name in enumerate(model.classes_):
                    mapped[:, REGIME_ORDER.index(class_name)] = model_probs[:, idx]
                probs[current_valid] = _normalize_rows(mapped)
            source = "expanding_logit"
            classes_seen = sorted(model.classes_.tolist())

        df.loc[current_idx, posterior_cols] = probs
        df.loc[current_idx, "posterior_source"] = source
        update_rows.append(
            {
                "update_month": month_start,
                "train_samples": int(len(train)),
                "classes_seen": ",".join(classes_seen),
                "source": source,
            }
        )

    posterior_frame = df.loc[:, [date_col, label_col, "posterior_source", "update_month", *indicator_cols, *posterior_cols]]
    return RegimePosteriorResult(
        posterior_frame=posterior_frame,
        update_summary=pd.DataFrame(update_rows),
    )


def build_spy_regime_posterior(
    ticker: str = "SPY",
    *,
    data_dir: str | Path = DATA_DIR,
    min_train_samples: int = 252,
) -> RegimePosteriorResult:
    """Convenience wrapper for fitting the SPY regime posterior end to end."""
    regime_frame = build_regime_training_frame(ticker=ticker, data_dir=data_dir)
    return fit_regime_posterior(regime_frame, min_train_samples=min_train_samples)


def _resolve_model_prob_columns(
    frame: pd.DataFrame,
    model_prob_columns: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if model_prob_columns is not None:
        missing = [model for model, col in model_prob_columns.items() if col not in frame.columns]
        if missing:
            raise KeyError(f"missing probability columns for {missing}")
        return dict(model_prob_columns)

    resolved = _resolve_candidate_columns(frame, DEFAULT_MODEL_PROB_CANDIDATES)
    if not resolved:
        raise KeyError("could not infer any model probability columns")
    return resolved


def _resolve_model_return_columns(
    frame: pd.DataFrame,
    model_return_columns: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if model_return_columns is not None:
        return {name: col for name, col in model_return_columns.items() if col in frame.columns}
    return _resolve_candidate_columns(frame, DEFAULT_MODEL_RETURN_CANDIDATES)


def add_model_logits(
    frame: pd.DataFrame,
    *,
    model_prob_columns: Mapping[str, str] | None = None,
    model_logit_columns: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Attach per-model logit columns, using existing logits where available."""
    df = frame.copy()
    prob_cols = _resolve_model_prob_columns(df, model_prob_columns)
    resolved_logits: dict[str, str] = {}

    for model_name, prob_col in prob_cols.items():
        existing_col = None
        candidate_map = model_logit_columns or {}
        if model_name in candidate_map and candidate_map[model_name] in df.columns:
            existing_col = candidate_map[model_name]
        else:
            for candidate in DEFAULT_MODEL_LOGIT_CANDIDATES.get(model_name, ()):
                if candidate in df.columns:
                    existing_col = candidate
                    break
        if existing_col is None:
            existing_col = f"{prob_col}_logit"
            df[existing_col] = safe_logit(df[prob_col].astype(float).to_numpy())
        resolved_logits[model_name] = existing_col

    return df, prob_cols, resolved_logits


def _project_capped_simplex(
    raw_weights: np.ndarray,
    *,
    cap: float = 0.70,
    target_sum: float = 1.0,
) -> np.ndarray:
    weights = np.clip(np.asarray(raw_weights, dtype=float), 0.0, None)
    n = weights.shape[0]
    if weights.sum() <= 0.0:
        weights = np.ones(n, dtype=float) / n
    else:
        weights = weights / weights.sum()

    output = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    remaining = float(target_sum)

    while active.any():
        active_idx = np.where(active)[0]
        active_weights = weights[active_idx]
        total = active_weights.sum()
        if total <= 0.0:
            output[active_idx] = remaining / active_idx.size
            break
        proposal = active_weights / total * remaining
        over = proposal > cap + 1e-12
        if not over.any():
            output[active_idx] = proposal
            break
        capped_idx = active_idx[over]
        output[capped_idx] = cap
        remaining -= cap * capped_idx.size
        active[capped_idx] = False
        if remaining <= 0.0:
            break

    return np.clip(output, 0.0, cap)


def _binary_log_loss(y_true: np.ndarray, logits: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    l = np.asarray(logits, dtype=float)
    return float(np.mean(np.logaddexp(0.0, l) - y * l))


def _fallback_daily_performance(
    train: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    logit_columns: Mapping[str, str],
    model_return_columns: Mapping[str, str],
) -> pd.DataFrame:
    records = pd.DataFrame({date_col: train[date_col]})
    if model_return_columns:
        for model_name, return_col in model_return_columns.items():
            records[model_name] = train[return_col].astype(float).to_numpy()
    else:
        signed_target = 2.0 * train[target_col].astype(float).to_numpy() - 1.0
        for model_name, logit_col in logit_columns.items():
            confidence = np.tanh(train[logit_col].astype(float).to_numpy() / 2.0)
            records[model_name] = confidence * signed_target
    return records.groupby(date_col, as_index=False).mean(numeric_only=True)


def _rolling_sharpe(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 20:
        return 0.0
    sigma = float(arr.std(ddof=1))
    if sigma <= 0.0:
        return 0.0
    return float(arr.mean() / sigma * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_fallback_weights(
    train: pd.DataFrame,
    *,
    model_names: Sequence[str],
    date_col: str,
    target_col: str,
    logit_columns: Mapping[str, str],
    model_return_columns: Mapping[str, str] | None = None,
    eta: float = 2.0,
    lookback_days: int = 60,
    cap: float = 0.70,
) -> tuple[np.ndarray, dict[str, float]]:
    """Performance-based fallback using exp(eta * rolling_sharpe_60d)."""
    model_return_columns = model_return_columns or {}
    daily_perf = _fallback_daily_performance(
        train,
        date_col=date_col,
        target_col=target_col,
        logit_columns=logit_columns,
        model_return_columns=model_return_columns,
    )
    daily_perf = daily_perf.sort_values(date_col).tail(lookback_days)

    sharpes: dict[str, float] = {}
    score_vector = np.zeros(len(model_names), dtype=float)
    for idx, model_name in enumerate(model_names):
        sharpe = _rolling_sharpe(daily_perf[model_name].to_numpy()) if model_name in daily_perf.columns else 0.0
        sharpes[model_name] = sharpe
        score_vector[idx] = np.exp(eta * sharpe)

    weights = _project_capped_simplex(score_vector, cap=cap, target_sum=1.0)
    return weights, sharpes


def _optimize_regime_weights(
    train: pd.DataFrame,
    *,
    posterior_columns: Sequence[str],
    logit_columns: Sequence[str],
    target_col: str,
    cap: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    n_regimes = len(posterior_columns)
    n_models = len(logit_columns)
    y = train[target_col].astype(float).to_numpy()
    posterior = train.loc[:, list(posterior_columns)].to_numpy(dtype=float)
    logits = train.loc[:, list(logit_columns)].to_numpy(dtype=float)
    design = (posterior[:, :, None] * logits[:, None, :]).reshape(len(train), n_regimes * n_models)

    init_row = np.full(n_models, 1.0 / n_models, dtype=float)
    initial = np.tile(init_row, n_regimes)
    bounds = [(0.0, cap)] * (n_regimes * n_models)
    constraints = [
        {
            "type": "ineq",
            "fun": lambda params, row=row: 1.0 - np.sum(params[row * n_models : (row + 1) * n_models]),
        }
        for row in range(n_regimes)
    ]

    def objective(params: np.ndarray) -> float:
        return _binary_log_loss(y, design @ params)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        return None, {"success": False, "message": result.message}
    weights = result.x.reshape(n_regimes, n_models)
    return weights, {"success": True, "message": result.message, "loss": float(result.fun)}


def apply_regime_weights(
    frame: pd.DataFrame,
    *,
    weights: np.ndarray,
    posterior_columns: Sequence[str],
    logit_columns: Sequence[str],
    out_logit_col: str = "ensemble_logit",
    out_prob_col: str = "ensemble_prob",
) -> pd.DataFrame:
    """Blend model logits using the supplied regime posterior and regime weights."""
    df = frame.copy()
    posterior = df.loc[:, list(posterior_columns)].to_numpy(dtype=float)
    logits = df.loc[:, list(logit_columns)].to_numpy(dtype=float)
    dynamic_weights = posterior @ weights
    combined = np.sum(dynamic_weights * logits, axis=1)
    df[out_logit_col] = combined
    df[out_prob_col] = _sigmoid(combined)
    return df


def fit_regime_weighted_ensemble(
    oof_frame: pd.DataFrame,
    *,
    regime_posterior: pd.DataFrame | None = None,
    regime_ticker: str = "SPY",
    data_dir: str | Path = DATA_DIR,
    date_col: str = "date",
    target_col: str = "y_true",
    model_prob_columns: Mapping[str, str] | None = None,
    model_logit_columns: Mapping[str, str] | None = None,
    model_return_columns: Mapping[str, str] | None = None,
    min_posterior_samples: int = 252,
    min_oof_samples: int = 100,
    eta: float = 2.0,
    fallback_lookback_days: int = 60,
    max_model_weight: float = 0.70,
) -> EnsembleFitResult:
    """Fit the regime posterior and the regime-conditional ensemble on OOF data."""
    if target_col not in oof_frame.columns:
        raise KeyError(f"missing target column {target_col!r}")

    base = _ensure_date_column(oof_frame, date_col=date_col)
    base, resolved_prob_cols, resolved_logit_cols = add_model_logits(
        base,
        model_prob_columns=model_prob_columns,
        model_logit_columns=model_logit_columns,
    )
    resolved_return_cols = _resolve_model_return_columns(base, model_return_columns)

    posterior_result = (
        fit_regime_posterior(regime_posterior, date_col=date_col, min_train_samples=min_posterior_samples)
        if regime_posterior is not None
        else build_spy_regime_posterior(
            ticker=regime_ticker,
            data_dir=data_dir,
            min_train_samples=min_posterior_samples,
        )
    )
    posterior = posterior_result.posterior_frame.copy()
    posterior_cols = _posterior_columns()
    merge_cols = [date_col, "posterior_source", "update_month", *posterior_cols]

    merged = base.merge(posterior.loc[:, merge_cols], on=date_col, how="left", validate="many_to_one")
    merged["update_month"] = _month_start(merged[date_col])

    model_names = list(resolved_logit_cols.keys())
    logit_cols = [resolved_logit_cols[name] for name in model_names]
    monthly_weights_rows: list[dict[str, Any]] = []
    prediction_blocks: list[pd.DataFrame] = []

    for month_start in merged["update_month"].drop_duplicates().sort_values():
        current_mask = merged["update_month"] == month_start
        current = merged.loc[current_mask].copy()
        if current.empty:
            continue

        train = merged.loc[
            (merged[date_col] < month_start)
            & merged[target_col].notna()
            & merged.loc[:, logit_cols].notna().all(axis=1)
            & merged.loc[:, posterior_cols].notna().all(axis=1)
        ].copy()

        use_fallback = len(train) < min_oof_samples or train[target_col].nunique() < 2
        diagnostics: dict[str, Any]
        if use_fallback:
            weights_vec, sharpes = compute_fallback_weights(
                train,
                model_names=model_names,
                date_col=date_col,
                target_col=target_col,
                logit_columns=resolved_logit_cols,
                model_return_columns=resolved_return_cols,
                eta=eta,
                lookback_days=fallback_lookback_days,
                cap=max_model_weight,
            )
            weights = np.tile(weights_vec, (len(REGIME_ORDER), 1))
            diagnostics = {
                "source": "fallback_insufficient_oof",
                "train_samples": int(len(train)),
                "sharpes": sharpes,
            }
        else:
            weights, optimization = _optimize_regime_weights(
                train,
                posterior_columns=posterior_cols,
                logit_columns=logit_cols,
                target_col=target_col,
                cap=max_model_weight,
            )
            if weights is None:
                weights_vec, sharpes = compute_fallback_weights(
                    train,
                    model_names=model_names,
                    date_col=date_col,
                    target_col=target_col,
                    logit_columns=resolved_logit_cols,
                    model_return_columns=resolved_return_cols,
                    eta=eta,
                    lookback_days=fallback_lookback_days,
                    cap=max_model_weight,
                )
                weights = np.tile(weights_vec, (len(REGIME_ORDER), 1))
                diagnostics = {
                    "source": "fallback_optimizer_failed",
                    "train_samples": int(len(train)),
                    "optimizer_message": optimization.get("message"),
                    "sharpes": sharpes,
                }
            else:
                diagnostics = {
                    "source": "stacked_logloss",
                    "train_samples": int(len(train)),
                    "optimizer_message": optimization.get("message"),
                    "objective": optimization.get("loss"),
                }

        current = apply_regime_weights(
            current,
            weights=weights,
            posterior_columns=posterior_cols,
            logit_columns=logit_cols,
        )
        current["ensemble_weight_source"] = diagnostics["source"]
        prediction_blocks.append(current)

        for regime_idx, regime_name in enumerate(REGIME_ORDER):
            row_weights = weights[regime_idx]
            row_record: dict[str, Any] = {
                "update_month": month_start,
                "regime": regime_name,
                "source": diagnostics["source"],
                "train_samples": diagnostics["train_samples"],
                "weight_sum": float(row_weights.sum()),
            }
            for model_idx, model_name in enumerate(model_names):
                row_record[f"weight_{model_name.lower().replace('-', '_').replace(' ', '_')}"] = float(row_weights[model_idx])
            if "objective" in diagnostics:
                row_record["objective"] = diagnostics["objective"]
            if "optimizer_message" in diagnostics:
                row_record["optimizer_message"] = diagnostics["optimizer_message"]
            prediction_meta = diagnostics.get("sharpes")
            if prediction_meta:
                for model_name, sharpe in prediction_meta.items():
                    row_record[f"fallback_sharpe_{model_name.lower().replace('-', '_').replace(' ', '_')}"] = float(sharpe)
            monthly_weights_rows.append(row_record)

    predictions = pd.concat(prediction_blocks, ignore_index=True).sort_values([date_col])
    return EnsembleFitResult(
        predictions=predictions,
        posterior=posterior,
        monthly_weights=pd.DataFrame(monthly_weights_rows),
        posterior_updates=posterior_result.update_summary,
    )
