"""Simple long/flat backtest simulator.

Strategy
--------
Go long when P(up) > entry_threshold; flat otherwise.
Position size proportional to confidence: size = (P - 0.5) * 2, clipped to [0,1].
Hold for `hold_days` calendar positions (no overlapping positions per ticker —
once entered, a ticker is "in-trade" for hold_days days before it can re-enter).

Returns are the signal's 5-day forward return (`returns_5d`) attributed on the
entry date, minus a round-trip transaction cost of `cost_bps` basis points.

Metrics
-------
total_return       — compound return over the full backtest window
annualized_return  — total_return annualised by trading days elapsed
sharpe_ratio       — annualised Sharpe of daily PnL
max_drawdown       — worst peak-to-trough equity drawdown
win_rate           — fraction of trades with positive net return
trade_count        — number of entries
profit_factor      — gross gains / |gross losses|
avg_trade_return   — mean per-trade net return
buy_and_hold       — naive "always long, equal-weight" benchmark over the
                     same date span for the same tickers
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _safe_div(a: float, b: float, default: float = float("nan")) -> float:
    if b == 0 or np.isnan(b):
        return default
    return a / b


def _sharpe(daily_pnl: np.ndarray) -> float:
    if len(daily_pnl) < 2:
        return float("nan")
    mu = float(np.mean(daily_pnl))
    sd = float(np.std(daily_pnl, ddof=1))
    if sd == 0:
        return float("nan")
    return mu / sd * float(np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def backtest(
    dates: np.ndarray,
    tickers: np.ndarray,
    probs: np.ndarray,
    returns_5d: np.ndarray,
    entry_threshold: float = 0.55,
    cost_bps: float = 10.0,
    hold_days: int = 5,
) -> dict[str, Any]:
    """Run the long/flat backtest.

    Parameters
    ----------
    dates       : array of pd.Timestamp / np.datetime64, one per sample
    tickers     : array of ticker symbols aligned with dates
    probs       : P(up) predictions in [0, 1]
    returns_5d  : actual 5-day forward returns (e.g. 0.012 for +1.2%)
    entry_threshold : enter long when probs > this
    cost_bps    : one-way transaction cost in basis points (applied twice per trade)
    hold_days   : how many trading days a ticker is locked after entry
    """
    n = len(dates)
    if not (len(tickers) == len(probs) == len(returns_5d) == n):
        raise ValueError("all input arrays must be same length")

    if n == 0:
        return _empty_result()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "ticker": tickers,
            "prob": probs,
            "ret5d": returns_5d,
        }
    ).sort_values(["date", "ticker"]).reset_index(drop=True)

    # Greedy per-ticker "no overlap" entry selection.
    trades: list[dict[str, Any]] = []
    next_entry_date: dict[str, pd.Timestamp] = {}

    cost_per_trade = 2.0 * cost_bps * 1e-4  # round-trip

    for row in df.itertuples(index=False):
        if np.isnan(row.prob) or np.isnan(row.ret5d):
            continue
        if row.prob <= entry_threshold:
            continue
        cutoff = next_entry_date.get(row.ticker)
        if cutoff is not None and row.date < cutoff:
            continue
        size = max(0.0, min(1.0, (float(row.prob) - 0.5) * 2.0))
        if size <= 0.0:
            continue
        gross = float(row.ret5d) * size
        net = gross - cost_per_trade * size
        # Entry locks the ticker for `hold_days` business days.
        lock_until = row.date + pd.tseries.offsets.BDay(hold_days)
        next_entry_date[row.ticker] = lock_until
        trades.append(
            {
                "entry_date": row.date,
                "exit_date": lock_until,
                "ticker": row.ticker,
                "prob": float(row.prob),
                "size": size,
                "gross_return": gross,
                "net_return": net,
            }
        )

    if not trades:
        res = _empty_result()
        res["trade_count"] = 0
        res["buy_and_hold"] = _buy_and_hold(df)
        return res

    tdf = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)

    # Build a daily PnL series by attributing each trade's net return across
    # its hold_days trading days (linear attribution). We only use real
    # business days between entry_date and exit_date so Sharpe uses calendar
    # density that reflects real market activity.
    daily: dict[pd.Timestamp, float] = {}
    for t in tdf.itertuples(index=False):
        span = pd.bdate_range(start=t.entry_date, end=t.exit_date, freq="B")
        if len(span) <= 1:
            daily[t.entry_date] = daily.get(t.entry_date, 0.0) + t.net_return
            continue
        # exclude the entry day "open" bar from the span for per-day PnL so
        # we don't double-count; use the len-1 trading days after entry.
        per_day = t.net_return / max(1, len(span) - 1)
        for d in span[1:]:
            daily[d] = daily.get(d, 0.0) + per_day

    if not daily:
        pnl_series = pd.Series(dtype=float)
    else:
        pnl_series = pd.Series(daily).sort_index()

    # Equity curve: starts at 1.0, daily additive PnL (small-return approx).
    if len(pnl_series):
        equity = (1.0 + pnl_series).cumprod().to_numpy()
    else:
        equity = np.array([1.0])

    total_return = float(equity[-1] - 1.0)
    years = max(len(pnl_series) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    annualized_return = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if total_return > -1.0
        else float("nan")
    )
    sharpe = _sharpe(pnl_series.to_numpy()) if len(pnl_series) else float("nan")
    mdd = _max_drawdown(equity)

    gross_gains = float(tdf.loc[tdf.net_return > 0, "net_return"].sum())
    gross_losses = float(-tdf.loc[tdf.net_return < 0, "net_return"].sum())
    profit_factor = _safe_div(gross_gains, gross_losses, default=float("inf"))

    return {
        "trade_count": int(len(tdf)),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "win_rate": float((tdf.net_return > 0).mean()),
        "profit_factor": profit_factor,
        "avg_trade_return": float(tdf.net_return.mean()),
        "avg_position_size": float(tdf["size"].mean()),
        "entry_threshold": entry_threshold,
        "cost_bps": cost_bps,
        "hold_days": hold_days,
        "buy_and_hold": _buy_and_hold(df),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": float("nan"),
        "max_drawdown": 0.0,
        "win_rate": float("nan"),
        "profit_factor": float("nan"),
        "avg_trade_return": float("nan"),
        "avg_position_size": 0.0,
        "entry_threshold": float("nan"),
        "cost_bps": float("nan"),
        "hold_days": 0,
        "buy_and_hold": {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
        },
    }


def _buy_and_hold(df: pd.DataFrame) -> dict[str, float]:
    """Equal-weight, always-long benchmark on the same panel.

    For each date, average the 5-day forward return across all tickers with a
    prediction that day, then compound through the series (treating each 5-day
    bar as a discrete holding period with daily overlap smoothed linearly).
    """
    if df.empty:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
        }

    g = df.groupby("date")["ret5d"].mean().dropna().sort_index()
    if g.empty:
        return {
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe_ratio": float("nan"),
        }

    # approximate daily return ≈ ret5d / 5 (linear attribution)
    daily = g / 5.0
    equity = (1.0 + daily).cumprod().to_numpy()
    total = float(equity[-1] - 1.0)
    years = max(len(daily) / TRADING_DAYS_PER_YEAR, 1 / TRADING_DAYS_PER_YEAR)
    annualized = (
        float((1.0 + total) ** (1.0 / years) - 1.0)
        if total > -1.0
        else float("nan")
    )
    sharpe = _sharpe(daily.to_numpy())
    return {
        "total_return": total,
        "annualized_return": annualized,
        "sharpe_ratio": sharpe,
    }


# ─── smoke test ──────────────────────────────────────────────────

def _smoke() -> int:
    rng = np.random.default_rng(123)
    n = 1000
    dates = pd.bdate_range("2023-01-01", periods=n // 5).repeat(5)[:n]
    tickers = np.array(["AAPL", "MSFT", "NVDA", "GOOGL", "META"] * (n // 5))[:n]
    # Signal that's right 60% of the time
    truth = rng.integers(0, 2, size=n)
    probs = np.where(truth == 1, rng.uniform(0.55, 0.9, n), rng.uniform(0.3, 0.6, n))
    # Returns aligned with truth: +1% if up, -1% if down, plus noise
    returns = np.where(truth == 1, 0.01, -0.01) + rng.normal(0, 0.02, n)
    r = backtest(dates.to_numpy(), tickers, probs, returns)
    print(f"[backtest smoke] {r['trade_count']} trades, "
          f"total={r['total_return']:.2%}, sharpe={r['sharpe_ratio']:.2f}, "
          f"mdd={r['max_drawdown']:.2%}, win={r['win_rate']:.2%}")
    assert r["trade_count"] > 0
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
