"""Phase 1 sizing utilities for the equity benchmark v2.

Implements edge-based position sizing from calibrated probabilities with
63-day EWMA vol targeting and daily gross-exposure normalization.
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


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


def compute_edge(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(probabilities, dtype=float)
    return 2.0 * np.clip(arr, 0.0, 1.0) - 1.0


def ewma_realized_vol(
    returns: pd.Series,
    *,
    span: int = 63,
    min_periods: int = 21,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute annualized EWMA realized vol from a daily return series."""
    return returns.ewm(span=span, adjust=False, min_periods=min_periods).std(bias=False) * np.sqrt(annualization)


def add_realized_vol(
    frame: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    daily_return_col: str = "daily_return",
    price_col: str = "adj_close",
    out_col: str = "ewma_vol_63d",
    span: int = 63,
    min_periods: int = 21,
) -> pd.DataFrame:
    """Add 63d EWMA realized vol per ticker."""
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    if daily_return_col not in df.columns:
        if price_col not in df.columns:
            raise KeyError(f"missing daily return column {daily_return_col!r} and price column {price_col!r}")
        df[daily_return_col] = df.groupby(ticker_col, group_keys=False)[price_col].pct_change()

    df[out_col] = df.groupby(ticker_col, group_keys=False)[daily_return_col].transform(
        lambda s: ewma_realized_vol(s, span=span, min_periods=min_periods)
    )
    return df


def _normalize_daily_weights(
    weights: pd.Series,
    *,
    target_gross_exposure: float,
    max_weight: float,
) -> pd.Series:
    if target_gross_exposure <= 0.0:
        return pd.Series(0.0, index=weights.index)

    signed = weights.astype(float)
    abs_weights = signed.abs()
    if abs_weights.sum() <= 0.0:
        return pd.Series(0.0, index=weights.index)

    alloc = pd.Series(0.0, index=weights.index, dtype=float)
    active = abs_weights > 0.0
    remaining = float(target_gross_exposure)

    while active.any() and remaining > 1e-12:
        desired = abs_weights[active]
        desired_sum = float(desired.sum())
        if desired_sum <= 0.0:
            break
        scaled = desired * (remaining / desired_sum)
        hit_cap = scaled >= max_weight - 1e-12

        if not hit_cap.any():
            alloc.loc[desired.index] += scaled
            remaining = 0.0
            break

        capped_index = desired.index[hit_cap]
        alloc.loc[capped_index] = max_weight
        remaining = max(0.0, target_gross_exposure - float(alloc.sum()))
        active.loc[capped_index] = False

    if remaining > 1e-12 and active.any():
        desired = abs_weights[active]
        if float(desired.sum()) > 0.0:
            alloc.loc[desired.index] += desired * (remaining / float(desired.sum()))

    alloc = np.minimum(alloc, max_weight)
    return alloc * np.sign(signed)


def size_positions(
    frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    selection_col: str | None = None,
    ticker_col: str = "ticker",
    date_col: str = "date",
    daily_return_col: str = "daily_return",
    price_col: str = "adj_close",
    vol_col: str | None = None,
    lambda_fraction: float = 0.25,
    max_weight: float = 0.10,
    target_annual_vol: float = 0.15,
    target_gross_exposure: float = 1.0,
    long_only: bool = True,
    vol_span: int = 63,
    vol_min_periods: int = 21,
) -> pd.DataFrame:
    """Size positions from calibrated probabilities.

    The sizing inference used here is:
      raw_weight = lambda_fraction * edge * target_annual_vol / realized_vol

    That scales position size down when realized vol is above the 15% target
    and up when realized vol is below it, subject to the per-name cap.
    """
    if prob_col not in frame.columns:
        raise KeyError(f"missing probability column {prob_col!r}")
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    working_vol_col = vol_col or "ewma_vol_63d"
    if working_vol_col not in df.columns:
        df = add_realized_vol(
            df,
            ticker_col=ticker_col,
            date_col=date_col,
            daily_return_col=daily_return_col,
            price_col=price_col,
            out_col=working_vol_col,
            span=vol_span,
            min_periods=vol_min_periods,
        )

    df["edge"] = compute_edge(df[prob_col])
    if long_only:
        df["edge"] = df["edge"].clip(lower=0.0)

    vol = df[working_vol_col].replace(0.0, np.nan)
    raw_weight = lambda_fraction * df["edge"] * float(target_annual_vol) / vol
    raw_weight = raw_weight.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    lower_clip = 0.0 if long_only else -float(max_weight)
    df["raw_weight"] = raw_weight.clip(lower=lower_clip, upper=float(max_weight))

    if selection_col is not None:
        if selection_col not in df.columns:
            raise KeyError(f"missing selection column {selection_col!r}")
        df["raw_weight"] = df["raw_weight"].where(df[selection_col], 0.0)

    df["target_weight"] = (
        df.groupby(date_col, group_keys=False)["raw_weight"]
        .transform(
            lambda s: _normalize_daily_weights(
                s,
                target_gross_exposure=float(target_gross_exposure),
                max_weight=float(max_weight),
            )
        )
    )
    df["gross_exposure"] = df.groupby(date_col)["target_weight"].transform(lambda s: s.abs().sum())
    df["net_exposure"] = df.groupby(date_col)["target_weight"].transform("sum")
    return df


def summarize_sizing(
    sized_frame: pd.DataFrame,
    *,
    date_col: str = "date",
    weight_col: str = "target_weight",
) -> dict[str, Any]:
    """Summarize daily exposures from a sized portfolio panel."""
    if weight_col not in sized_frame.columns:
        raise KeyError(f"missing weight column {weight_col!r}")

    df = _ensure_date_column(sized_frame, date_col)
    daily = df.groupby(date_col)[weight_col].agg(
        gross_exposure=lambda s: float(s.abs().sum()),
        net_exposure="sum",
        active_names=lambda s: int((s.abs() > 0).sum()),
    )
    return {
        "avg_gross_exposure": float(daily["gross_exposure"].mean()) if len(daily) else 0.0,
        "avg_net_exposure": float(daily["net_exposure"].mean()) if len(daily) else 0.0,
        "avg_active_names": float(daily["active_names"].mean()) if len(daily) else 0.0,
        "max_gross_exposure": float(daily["gross_exposure"].max()) if len(daily) else 0.0,
    }


__all__ = [
    "compute_edge",
    "ewma_realized_vol",
    "add_realized_vol",
    "size_positions",
    "summarize_sizing",
]
