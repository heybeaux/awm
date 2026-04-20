"""Phase 1 selection utilities for the equity benchmark v2.

Implements daily top-k long selection, logit-z normalization, and simple
training-slice comparison utilities keyed off calibrated probabilities.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_K_VALUES: tuple[int, ...] = (3, 5, 7, 10)
RETURN_COLUMN_CANDIDATES: tuple[str, ...] = (
    "ret_5d_fwd",
    "returns_5d",
    "return_5d",
    "ret5d",
    "forward_return_5d",
)


@dataclass
class SelectionMetrics:
    selection_name: str
    sharpe_ratio: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    trade_count: int
    avg_daily_picks: float


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


def _infer_return_col(frame: pd.DataFrame, return_col: str | None) -> str:
    if return_col is not None:
        if return_col not in frame.columns:
            raise KeyError(f"missing return column {return_col!r}")
        return return_col
    for candidate in RETURN_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"missing return column; tried {RETURN_COLUMN_CANDIDATES}")


def _sharpe(daily_pnl: np.ndarray) -> float:
    if len(daily_pnl) < 2:
        return float("nan")
    mu = float(np.mean(daily_pnl))
    sd = float(np.std(daily_pnl, ddof=1))
    if sd == 0.0:
        return float("nan")
    return mu / sd * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def safe_logit(values: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def add_logit_z_scores(
    frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    ticker_col: str = "ticker",
    date_col: str = "date",
    window: int = 252,
    min_periods: int = 63,
    out_col: str = "logit_z",
) -> pd.DataFrame:
    """Convert probabilities to logits and rolling z-scores per ticker."""
    if prob_col not in frame.columns:
        raise KeyError(f"missing probability column {prob_col!r}")
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([ticker_col, date_col]).copy()
    df["logit_score"] = safe_logit(df[prob_col])

    def _zscore(series: pd.Series) -> pd.Series:
        mean = series.rolling(window=window, min_periods=min_periods).mean()
        std = series.rolling(window=window, min_periods=min_periods).std()
        return (series - mean) / std.replace(0.0, np.nan)

    df[out_col] = df.groupby(ticker_col, group_keys=False)["logit_score"].transform(_zscore)
    df["abs_logit_z"] = df[out_col].abs()
    return df


def select_top_k(
    frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    ticker_col: str = "ticker",
    date_col: str = "date",
    k: int = 5,
    floor: float = 0.50,
    out_col: str = "selected_topk",
) -> pd.DataFrame:
    """Rank tickers daily by calibrated probability and keep the top k longs."""
    if k <= 0:
        raise ValueError("k must be positive")
    if prob_col not in frame.columns:
        raise KeyError(f"missing probability column {prob_col!r}")
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    df = _ensure_date_column(frame, date_col).sort_values([date_col, prob_col, ticker_col], ascending=[True, False, True]).copy()
    df["daily_rank"] = df.groupby(date_col)[prob_col].rank(method="first", ascending=False)
    df[out_col] = (df["daily_rank"] <= float(k)) & (df[prob_col] >= float(floor))
    return df


def select_z_threshold(
    frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    z_col: str = "logit_z",
    date_col: str = "date",
    z_threshold: float = 1.0,
    floor: float = 0.50,
    use_absolute_z: bool = True,
    out_col: str = "selected_z",
) -> pd.DataFrame:
    """Select rows whose rolling logit z-score clears the chosen threshold."""
    if prob_col not in frame.columns:
        raise KeyError(f"missing probability column {prob_col!r}")
    if z_col not in frame.columns:
        raise KeyError(f"missing z-score column {z_col!r}")

    df = _ensure_date_column(frame, date_col).copy()
    z_values = df[z_col].abs() if use_absolute_z else df[z_col]
    df[out_col] = (z_values >= float(z_threshold)) & (df[prob_col] >= float(floor))
    return df


def evaluate_selection(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_col: str | None = None,
    hold_days: int = 5,
    cost_bps: float = 10.0,
    selection_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a selection rule with the repo's 5-day return attribution logic."""
    if signal_col not in frame.columns:
        raise KeyError(f"missing signal column {signal_col!r}")
    if ticker_col not in frame.columns:
        raise KeyError(f"missing ticker column {ticker_col!r}")

    return_col = _infer_return_col(frame, return_col)
    df = _ensure_date_column(frame, date_col).sort_values([date_col, ticker_col]).copy()

    next_entry_date: dict[str, pd.Timestamp] = {}
    trades: list[dict[str, Any]] = []
    daily_pick_counts: list[int] = []
    cost_per_trade = 2.0 * cost_bps * 1e-4

    for trade_date, day_df in df.groupby(date_col, sort=True):
        day_candidates = day_df.loc[
            day_df[signal_col]
            & day_df[return_col].notna()
        ]
        if day_candidates.empty:
            daily_pick_counts.append(0)
            continue

        eligible_rows = []
        for row in day_candidates.itertuples(index=False):
            cutoff = next_entry_date.get(getattr(row, ticker_col))
            if cutoff is not None and trade_date < cutoff:
                continue
            eligible_rows.append(row)

        daily_pick_counts.append(len(eligible_rows))
        if not eligible_rows:
            continue

        weight = 1.0 / len(eligible_rows)
        for row in eligible_rows:
            ticker = getattr(row, ticker_col)
            gross = float(getattr(row, return_col)) * weight
            net = gross - cost_per_trade * weight
            exit_date = trade_date + pd.tseries.offsets.BDay(hold_days)
            next_entry_date[ticker] = exit_date
            trades.append(
                {
                    "entry_date": trade_date,
                    "exit_date": exit_date,
                    "ticker": ticker,
                    "weight": weight,
                    "net_return": net,
                }
            )

    if not trades:
        return asdict(
            SelectionMetrics(
                selection_name=selection_name or signal_col,
                sharpe_ratio=float("nan"),
                total_return=0.0,
                annualized_return=0.0,
                max_drawdown=0.0,
                trade_count=0,
                avg_daily_picks=float(np.mean(daily_pick_counts)) if daily_pick_counts else 0.0,
            )
        )

    pnl_by_day: dict[pd.Timestamp, float] = {}
    for trade in trades:
        span = pd.bdate_range(start=trade["entry_date"], end=trade["exit_date"], freq="B")
        if len(span) <= 1:
            pnl_by_day[trade["entry_date"]] = pnl_by_day.get(trade["entry_date"], 0.0) + trade["net_return"]
            continue
        per_day = trade["net_return"] / max(1, len(span) - 1)
        for day in span[1:]:
            pnl_by_day[day] = pnl_by_day.get(day, 0.0) + per_day

    pnl = pd.Series(pnl_by_day).sort_index()
    equity = (1.0 + pnl).cumprod().to_numpy() if len(pnl) else np.array([1.0])
    total_return = float(equity[-1] - 1.0)
    years = max(len(pnl) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    annualized_return = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else float("nan")
    )

    return asdict(
        SelectionMetrics(
            selection_name=selection_name or signal_col,
            sharpe_ratio=_sharpe(pnl.to_numpy()) if len(pnl) else float("nan"),
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=_max_drawdown(equity),
            trade_count=len(trades),
            avg_daily_picks=float(np.mean(daily_pick_counts)) if daily_pick_counts else 0.0,
        )
    )


def grid_search_top_k(
    training_frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    floor: float = 0.50,
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_col: str | None = None,
    hold_days: int = 5,
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Choose k from {3,5,7,10} by maximizing net Sharpe on the training slice."""
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for k in k_values:
        panel = select_top_k(
            training_frame,
            prob_col=prob_col,
            ticker_col=ticker_col,
            date_col=date_col,
            k=int(k),
            floor=floor,
            out_col="selected_topk",
        )
        metrics = evaluate_selection(
            panel,
            signal_col="selected_topk",
            date_col=date_col,
            ticker_col=ticker_col,
            return_col=return_col,
            hold_days=hold_days,
            cost_bps=cost_bps,
            selection_name=f"top_{k}",
        )
        metrics["k"] = int(k)
        results.append(metrics)

        score = metrics["sharpe_ratio"]
        if best_result is None:
            best_result = metrics
            continue
        best_score = best_result["sharpe_ratio"]
        if math.isnan(best_score) or (not math.isnan(score) and score > best_score):
            best_result = metrics

    return {
        "best_k": None if best_result is None else best_result["k"],
        "best_result": best_result,
        "results": results,
    }


def compare_top_k_vs_z_threshold(
    training_frame: pd.DataFrame,
    *,
    prob_col: str = "calibrated_prob",
    z_threshold: float = 1.0,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    floor: float = 0.50,
    date_col: str = "date",
    ticker_col: str = "ticker",
    return_col: str | None = None,
    hold_days: int = 5,
    cost_bps: float = 10.0,
    z_window: int = 252,
    z_min_periods: int = 63,
    use_absolute_z: bool = True,
) -> dict[str, Any]:
    """Compare the best top-k selector against a logit-z threshold selector."""
    topk_grid = grid_search_top_k(
        training_frame,
        prob_col=prob_col,
        k_values=k_values,
        floor=floor,
        date_col=date_col,
        ticker_col=ticker_col,
        return_col=return_col,
        hold_days=hold_days,
        cost_bps=cost_bps,
    )
    best_k = topk_grid["best_k"] or 5

    topk_panel = select_top_k(
        training_frame,
        prob_col=prob_col,
        ticker_col=ticker_col,
        date_col=date_col,
        k=best_k,
        floor=floor,
        out_col="selected_topk",
    )
    z_panel = add_logit_z_scores(
        training_frame,
        prob_col=prob_col,
        ticker_col=ticker_col,
        date_col=date_col,
        window=z_window,
        min_periods=z_min_periods,
        out_col="logit_z",
    )
    z_panel = select_z_threshold(
        z_panel,
        prob_col=prob_col,
        z_col="logit_z",
        date_col=date_col,
        z_threshold=z_threshold,
        floor=floor,
        use_absolute_z=use_absolute_z,
        out_col="selected_z",
    )

    topk_eval = evaluate_selection(
        topk_panel,
        signal_col="selected_topk",
        date_col=date_col,
        ticker_col=ticker_col,
        return_col=return_col,
        hold_days=hold_days,
        cost_bps=cost_bps,
        selection_name=f"top_{best_k}",
    )
    z_eval = evaluate_selection(
        z_panel,
        signal_col="selected_z",
        date_col=date_col,
        ticker_col=ticker_col,
        return_col=return_col,
        hold_days=hold_days,
        cost_bps=cost_bps,
        selection_name="z_threshold",
    )

    topk_sharpe = topk_eval["sharpe_ratio"]
    z_sharpe = z_eval["sharpe_ratio"]
    preferred = "top_k"
    if math.isnan(topk_sharpe) and not math.isnan(z_sharpe):
        preferred = "z_threshold"
    elif not math.isnan(z_sharpe) and z_sharpe > topk_sharpe:
        preferred = "z_threshold"

    return {
        "best_k": best_k,
        "top_k_grid": topk_grid,
        "top_k": topk_eval,
        "z_threshold": {
            "threshold": z_threshold,
            "use_absolute_z": use_absolute_z,
            **z_eval,
        },
        "preferred": preferred,
    }


__all__ = [
    "DEFAULT_K_VALUES",
    "SelectionMetrics",
    "safe_logit",
    "add_logit_z_scores",
    "select_top_k",
    "select_z_threshold",
    "evaluate_selection",
    "grid_search_top_k",
    "compare_top_k_vs_z_threshold",
]
