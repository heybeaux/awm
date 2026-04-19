"""Target computation for equity benchmark.

- `direction_5d`: binary — 1 if 5-day forward return > threshold (default 1%)
- `regime`: categorical — trending_up / trending_down / mean_reverting / quiet

Features at row `t` use only data up to `t`. Targets use forward data (standard
supervised-learning convention). Rows without a defined target are dropped by
the caller (splits).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_direction_5d(
    df: pd.DataFrame,
    threshold: float = 0.01,
    horizon: int = 5,
    price_col: str = "adj_close",
) -> pd.Series:
    """Binary label: 1 iff close[t+horizon] / close[t] - 1 > threshold.

    Rows near the tail (where t+horizon is out of range) get NaN.
    """
    fwd = df[price_col].shift(-horizon) / df[price_col] - 1.0
    label = (fwd > threshold).astype("float64")
    # Preserve NaN for tail rows where fwd is undefined
    label[fwd.isna()] = np.nan
    return label.rename(f"direction_{horizon}d")


def compute_regime(
    df: pd.DataFrame,
    window: int = 20,
    trend_threshold: float = 0.05,
    price_col: str = "adj_close",
) -> pd.Series:
    """Categorical market regime based on trailing `window` days.

    - trending_up   if ret_window > +trend_threshold
    - trending_down if ret_window < -trend_threshold
    - mean_reverting if vol_window > median(vol_window, expanding)
    - quiet otherwise

    The median comparison uses only *past* volatility (expanding median up to
    t-1) to avoid look-ahead. Early rows (before enough history) are NaN.
    """
    price = df[price_col]
    daily_ret = price.pct_change()

    ret_window = price / price.shift(window) - 1.0
    vol_window = daily_ret.rolling(window=window).std()

    # Past-only expanding median of vol_window (shift by 1 so t uses [0..t-1])
    expanding_median = vol_window.expanding(min_periods=window).median().shift(1)

    out = pd.Series(index=df.index, dtype="object")
    for i in range(len(df)):
        r = ret_window.iloc[i]
        v = vol_window.iloc[i]
        med = expanding_median.iloc[i]
        if pd.isna(r) or pd.isna(v):
            out.iloc[i] = np.nan
            continue
        if r > trend_threshold:
            out.iloc[i] = "trending_up"
        elif r < -trend_threshold:
            out.iloc[i] = "trending_down"
        elif not pd.isna(med) and v > med:
            out.iloc[i] = "mean_reverting"
        else:
            out.iloc[i] = "quiet"
    return out.rename("regime")


def add_targets(
    df: pd.DataFrame,
    direction_threshold: float = 0.01,
    direction_horizon: int = 5,
    regime_window: int = 20,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """Append direction and regime columns to `df`. Returns a new DataFrame."""
    out = df.copy()
    out[f"direction_{direction_horizon}d"] = compute_direction_5d(
        df, threshold=direction_threshold, horizon=direction_horizon, price_col=price_col
    )
    out["regime"] = compute_regime(
        df, window=regime_window, price_col=price_col
    )
    return out


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 300
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    df = pd.DataFrame({"adj_close": prices})
    out = add_targets(df)
    print(out.tail(10))
    print("direction_5d balance:", out["direction_5d"].value_counts(dropna=False).to_dict())
    print("regime counts:", out["regime"].value_counts(dropna=False).to_dict())
