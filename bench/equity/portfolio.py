"""Phase 4 portfolio construction utilities for the equity benchmark v2.

This module adds the execution layer that sits on top of calibrated model
scores:
  - per-ticker 63d EWMA and 21d realized volatility estimates
  - portfolio-level vol targeting with a conservative uncorrelated assumption
  - turnover control via a minimum z-score change to permit sign flips
  - transaction cost accounting with 2/5/10 bps sensitivity support
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_COST_LEVELS: tuple[float, ...] = (2.0, 5.0, 10.0)


@dataclass
class PortfolioCostSummary:
    cost_bps: float
    gross_return: float
    net_return: float
    sharpe_ratio: float
    max_drawdown: float


def _ensure_date_column(frame: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if date_col in frame.columns:
        out = frame.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.reset_index().rename(columns={frame.index.name or "index": date_col})
        out[date_col] = pd.to_datetime(out[date_col])
        return out
    raise KeyError(f"missing date column {date_col!r}")


def _safe_sign(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sign = np.sign(arr)
    sign[np.abs(arr) < 1e-12] = 0.0
    return sign


def _sharpe(daily_returns: np.ndarray) -> float:
    arr = np.asarray(daily_returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    sigma = float(arr.std(ddof=1))
    if sigma <= 0.0:
        return float("nan")
    return float(arr.mean() / sigma * math.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    arr = np.asarray(equity, dtype=float)
    if arr.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(arr)
    drawdown = arr / np.where(running_max == 0.0, 1.0, running_max) - 1.0
    return float(np.nanmin(drawdown))


def compute_ewma_volatility(
    returns: pd.Series,
    *,
    span: int = 63,
    min_periods: int = 21,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute annualized EWMA volatility from a daily return series."""
    return returns.ewm(span=span, adjust=False, min_periods=min_periods).std(bias=False) * math.sqrt(annualization)


def compute_realized_volatility(
    returns: pd.Series,
    *,
    window: int = 21,
    min_periods: int = 10,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Compute annualized rolling realized volatility from a daily return series."""
    return returns.rolling(window=window, min_periods=min_periods).std() * math.sqrt(annualization)


def add_risk_estimates(
    frame: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    daily_return_col: str = "daily_return",
    price_col: str = "adj_close",
    ewma_vol_col: str = "ewma_vol_63d",
    realized_vol_col: str = "realized_vol_21d",
) -> pd.DataFrame:
    """Attach per-ticker 63d EWMA vol and 21d realized vol."""
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    if daily_return_col not in df.columns:
        if price_col not in df.columns:
            raise KeyError(f"missing both {daily_return_col!r} and {price_col!r}")
        df[daily_return_col] = df.groupby(ticker_col, group_keys=False)[price_col].pct_change()

    df[ewma_vol_col] = df.groupby(ticker_col, group_keys=False)[daily_return_col].transform(compute_ewma_volatility)
    df[realized_vol_col] = df.groupby(ticker_col, group_keys=False)[daily_return_col].transform(compute_realized_volatility)
    return df


def portfolio_volatility(
    weights: pd.Series | np.ndarray,
    volatilities: pd.Series | np.ndarray,
) -> float:
    """Portfolio vol under the conservative uncorrelated assumption."""
    w = np.asarray(weights, dtype=float)
    v = np.asarray(volatilities, dtype=float)
    valid = np.isfinite(w) & np.isfinite(v)
    if not valid.any():
        return 0.0
    return float(np.sqrt(np.sum(np.square(w[valid] * v[valid]))))


def apply_vol_targeting(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    weight_col: str = "target_weight",
    ewma_vol_col: str = "ewma_vol_63d",
    realized_vol_col: str = "realized_vol_21d",
    target_annual_vol: float = 0.10,
    max_leverage: float = 1.50,
    out_col: str = "vol_target_weight",
) -> pd.DataFrame:
    """Scale daily weights to a 10% target and cut exposure if realized vol is high."""
    if weight_col not in frame.columns:
        raise KeyError(f"missing weight column {weight_col!r}")
    if ewma_vol_col not in frame.columns or realized_vol_col not in frame.columns:
        raise KeyError(f"missing vol columns {ewma_vol_col!r} / {realized_vol_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([date_col]).copy()
    df[out_col] = 0.0
    df["portfolio_ewma_vol"] = 0.0
    df["portfolio_realized_vol_21d"] = 0.0
    df["vol_scale_factor"] = 1.0
    df["realized_vol_scale_factor"] = 1.0

    for _, idx in df.groupby(date_col, sort=True).groups.items():
        day_slice = df.loc[idx]
        ewma_vol = portfolio_volatility(day_slice[weight_col], day_slice[ewma_vol_col])
        realized_vol = portfolio_volatility(day_slice[weight_col], day_slice[realized_vol_col])

        scale = 1.0 if ewma_vol <= 0.0 else float(target_annual_vol) / ewma_vol
        scale = float(np.clip(scale, 0.0, max_leverage))

        realized_scale = 1.0
        if realized_vol > float(target_annual_vol) and realized_vol > 0.0:
            realized_scale = float(target_annual_vol) / realized_vol
            scale = min(scale, realized_scale)

        df.loc[idx, out_col] = day_slice[weight_col].astype(float) * scale
        df.loc[idx, "portfolio_ewma_vol"] = ewma_vol
        df.loc[idx, "portfolio_realized_vol_21d"] = realized_vol
        df.loc[idx, "vol_scale_factor"] = scale
        df.loc[idx, "realized_vol_scale_factor"] = realized_scale

    return df


def apply_turnover_control(
    frame: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    desired_weight_col: str = "vol_target_weight",
    signal_z_col: str = "logit_z",
    flip_threshold: float = 0.25,
    out_col: str = "executed_weight",
) -> pd.DataFrame:
    """Block sign flips unless the signal z-score moved by more than 0.25."""
    required = {ticker_col, desired_weight_col, signal_z_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing turnover-control columns: {missing}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    df[out_col] = 0.0
    df["previous_weight"] = 0.0
    df["previous_signal_z"] = np.nan
    df["z_change"] = np.nan
    df["flip_blocked"] = False
    df["turnover"] = 0.0

    for ticker, idx in df.groupby(ticker_col, sort=False).groups.items():
        ticker_df = df.loc[idx].sort_values(date_col)
        prev_weight = 0.0
        prev_z = float("nan")

        for row in ticker_df.itertuples():
            desired = float(getattr(row, desired_weight_col))
            signal_z = float(getattr(row, signal_z_col)) if pd.notna(getattr(row, signal_z_col)) else float("nan")
            z_change = abs(signal_z - prev_z) if np.isfinite(signal_z) and np.isfinite(prev_z) else float("inf")

            prev_sign = float(np.sign(prev_weight))
            desired_sign = float(np.sign(desired))
            flip_blocked = False
            executed = desired

            if prev_sign != 0.0 and desired_sign != 0.0 and prev_sign != desired_sign and z_change <= flip_threshold:
                executed = prev_weight
                flip_blocked = True

            turnover = abs(executed - prev_weight)
            row_idx = row.Index
            df.at[row_idx, out_col] = executed
            df.at[row_idx, "previous_weight"] = prev_weight
            df.at[row_idx, "previous_signal_z"] = prev_z
            df.at[row_idx, "z_change"] = z_change if math.isfinite(z_change) else np.nan
            df.at[row_idx, "flip_blocked"] = bool(flip_blocked)
            df.at[row_idx, "turnover"] = turnover

            prev_weight = executed
            prev_z = signal_z

    return df.sort_values([date_col, ticker_col]).reset_index(drop=True)


def summarize_turnover(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    turnover_col: str = "turnover",
) -> dict[str, float]:
    """Summarize turnover from a positioned panel."""
    if turnover_col not in frame.columns:
        raise KeyError(f"missing turnover column {turnover_col!r}")

    df = _ensure_date_column(frame, date_col)
    daily_turnover = df.groupby(date_col)[turnover_col].sum(min_count=1) / 2.0
    return {
        "avg_daily_turnover": float(daily_turnover.mean()) if len(daily_turnover) else 0.0,
        "median_daily_turnover": float(daily_turnover.median()) if len(daily_turnover) else 0.0,
        "max_daily_turnover": float(daily_turnover.max()) if len(daily_turnover) else 0.0,
        "total_turnover": float(daily_turnover.sum()) if len(daily_turnover) else 0.0,
    }


def compute_transaction_costs(
    frame: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "date",
    executed_weight_col: str = "executed_weight",
    previous_weight_col: str = "previous_weight",
    cost_bps: float = 5.0,
    out_col: str = "transaction_cost",
) -> pd.DataFrame:
    """Apply a per-side transaction cost model with round-trip charges on new trades."""
    required = {ticker_col, executed_weight_col, previous_weight_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"missing cost-model columns: {missing}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    side_cost = float(cost_bps) * 1e-4

    prev = df[previous_weight_col].astype(float).to_numpy()
    curr = df[executed_weight_col].astype(float).to_numpy()
    prev_sign = _safe_sign(prev)
    curr_sign = _safe_sign(curr)

    same_sign = (prev_sign == curr_sign) & (prev_sign != 0.0)
    is_new_trade = (np.abs(prev) < 1e-12) & (np.abs(curr) >= 1e-12)
    is_exit_trade = (np.abs(prev) >= 1e-12) & (np.abs(curr) < 1e-12)
    is_flip_trade = (prev_sign != 0.0) & (curr_sign != 0.0) & (prev_sign != curr_sign)
    is_hold_trade = same_sign & np.isclose(prev, curr, atol=1e-12)

    same_sign_added = np.maximum(np.abs(curr) - np.abs(prev), 0.0)
    rebalance_up = same_sign & (same_sign_added > 1e-12)

    costs = np.zeros(len(df), dtype=float)
    costs[is_new_trade] = 2.0 * side_cost * np.abs(curr[is_new_trade])
    costs[is_flip_trade] = 2.0 * side_cost * np.abs(curr[is_flip_trade])
    costs[rebalance_up] = 2.0 * side_cost * same_sign_added[rebalance_up]

    df[out_col] = costs
    df["cost_bps"] = float(cost_bps)
    df["is_new_trade"] = is_new_trade
    df["is_exit_trade"] = is_exit_trade
    df["is_flip_trade"] = is_flip_trade
    df["is_hold_trade"] = is_hold_trade
    return df


def penalized_sharpe(
    daily_returns: pd.Series | np.ndarray,
    daily_turnover: pd.Series | np.ndarray,
    *,
    turnover_penalty: float = 0.10,
) -> float:
    """Sharpe minus a linear turnover penalty used for portfolio selection."""
    sharpe = _sharpe(np.asarray(daily_returns, dtype=float))
    turnover = np.asarray(daily_turnover, dtype=float)
    turnover = turnover[np.isfinite(turnover)]
    mean_turnover = float(turnover.mean()) if turnover.size else 0.0
    if math.isnan(sharpe):
        return float("nan")
    return float(sharpe - turnover_penalty * mean_turnover)


def cost_sensitivity_analysis(
    frame: pd.DataFrame,
    *,
    gross_return_col: str = "gross_portfolio_return",
    date_col: str = "date",
    cost_col_template: str = "transaction_cost_{bps:g}",
    cost_levels: Sequence[float] = DEFAULT_COST_LEVELS,
) -> list[dict[str, Any]]:
    """Summarize gross vs net portfolio performance across 2/5/10 bps cost levels."""
    if gross_return_col not in frame.columns:
        raise KeyError(f"missing gross return column {gross_return_col!r}")

    df = _ensure_date_column(frame, date_col)
    daily = df.groupby(date_col, as_index=False).agg(
        gross_return=(gross_return_col, "sum"),
    )

    summaries: list[dict[str, Any]] = []
    for level in cost_levels:
        col = cost_col_template.format(bps=float(level))
        if col not in df.columns:
            continue
        cost_daily = df.groupby(date_col)[col].sum()
        net_daily = daily["gross_return"].to_numpy(dtype=float) - cost_daily.reindex(daily[date_col], fill_value=0.0).to_numpy(dtype=float)
        gross_curve = np.cumprod(1.0 + daily["gross_return"].to_numpy(dtype=float))
        net_curve = np.cumprod(1.0 + net_daily)
        summaries.append(
            asdict(
                PortfolioCostSummary(
                    cost_bps=float(level),
                    gross_return=float(gross_curve[-1] - 1.0) if len(gross_curve) else 0.0,
                    net_return=float(net_curve[-1] - 1.0) if len(net_curve) else 0.0,
                    sharpe_ratio=_sharpe(net_daily),
                    max_drawdown=_max_drawdown(net_curve) if len(net_curve) else float("nan"),
                )
            )
        )

    return summaries


__all__ = [
    "DEFAULT_COST_LEVELS",
    "TRADING_DAYS_PER_YEAR",
    "PortfolioCostSummary",
    "add_risk_estimates",
    "apply_turnover_control",
    "apply_vol_targeting",
    "compute_ewma_volatility",
    "compute_realized_volatility",
    "compute_transaction_costs",
    "cost_sensitivity_analysis",
    "penalized_sharpe",
    "portfolio_volatility",
    "summarize_turnover",
]
