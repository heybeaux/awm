"""Phase 2 regime detection utilities for the equity benchmark v2.

Builds strictly backward-looking SPY indicators, assigns one of five market
regimes, validates them against known stress windows, and checks for
look-ahead leakage with an embargo-style replay test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/equity_bench_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/equity_bench_cache/matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
DEFAULT_RESULTS_DIR = Path(os.environ.get("EQUITY_BENCH_RESULTS_DIR", BENCH_DIR.parent / "results"))

REGIME_ORDER: tuple[str, ...] = ("crisis", "bear", "bull", "mean_revert", "neutral")
REGIME_COLORS: dict[str, str] = {
    "crisis": "#9a031e",
    "bear": "#d00000",
    "bull": "#2a9d8f",
    "mean_revert": "#e9c46a",
    "neutral": "#577590",
}

INDICATOR_COLUMNS: tuple[str, ...] = (
    "trend_sma63_slope_21d",
    "vol_percentile_252d",
    "autocorr_21d",
    "drawdown_63d",
)


@dataclass
class EventValidation:
    name: str
    start: str
    end: str
    expected_regimes: list[str]
    sample_count: int
    dominant_regime: str | None
    dominant_share: float | None
    matched_share: float | None
    passed: bool


@dataclass
class EmbargoValidation:
    embargo_days: int
    sample_step: int
    sample_count: int
    mismatch_count: int
    max_abs_indicator_diff: dict[str, float]
    passed: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        out = value.item()
        if isinstance(out, float) and (math.isnan(out) or math.isinf(out)):
            return None
        return out
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"not serializable: {type(value)!r}")


def _resolve_results_dir(path: str | Path | None = None) -> Path:
    preferred = Path(path or DEFAULT_RESULTS_DIR)
    fallback = BENCH_DIR / "results"
    for candidate in (preferred, fallback):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(f"unable to write results to {preferred} or {fallback}")


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.copy()
        out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        out.index.name = frame.index.name or "date"
        return out
    if "date" in frame.columns:
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").set_index("date")
        return out
    raise KeyError("expected a DatetimeIndex or a 'date' column")


def _resolve_ticker_path(ticker: str, data_dir: str | Path = DATA_DIR) -> Path:
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{ticker}_raw.parquet",
        data_dir / f"{ticker}_features.parquet",
        data_dir / f"{ticker}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no parquet found for ticker {ticker!r} in {data_dir}")


def load_price_history(ticker: str = "SPY", *, data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load a ticker's daily history from the local benchmark cache."""
    path = _resolve_ticker_path(ticker, data_dir=data_dir)
    frame = _ensure_datetime_index(pd.read_parquet(path))
    price_col = "adj_close" if "adj_close" in frame.columns else "close"
    if price_col not in frame.columns:
        raise KeyError(f"{path.name} is missing both 'adj_close' and 'close'")
    return frame.sort_index()


def _rolling_linear_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return float("nan")
    x = np.arange(values.shape[0], dtype=float)
    x_centered = x - x.mean()
    y_centered = values - values.mean()
    denom = float(np.dot(x_centered, x_centered))
    scale = float(abs(values.mean()))
    if denom == 0.0 or scale == 0.0:
        return float("nan")
    slope = float(np.dot(x_centered, y_centered) / denom)
    return slope / scale


def _rolling_percentile_of_last(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.mean(arr <= arr[-1]))


def _rolling_autocorr_lag1(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any() or arr.size < 3:
        return float("nan")
    left = arr[:-1] - arr[:-1].mean()
    right = arr[1:] - arr[1:].mean()
    denom = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def compute_regime_indicators(
    frame: pd.DataFrame,
    *,
    price_col: str | None = None,
) -> pd.DataFrame:
    """Compute strictly trailing SPY indicators used for regime detection."""
    df = _ensure_datetime_index(frame)
    price_field = price_col or ("adj_close" if "adj_close" in df.columns else "close")
    if price_field not in df.columns:
        raise KeyError(f"missing price column {price_field!r}")

    price = df[price_field].astype(float)
    daily_return = price.pct_change()
    sma_63 = price.rolling(window=63, min_periods=63).mean()
    trend = sma_63.rolling(window=21, min_periods=21).apply(_rolling_linear_slope, raw=True)
    ewma_vol_63 = daily_return.ewm(span=63, adjust=False, min_periods=63).std(bias=False) * np.sqrt(252.0)
    vol_percentile = ewma_vol_63.rolling(window=252, min_periods=63).apply(_rolling_percentile_of_last, raw=True)
    autocorr_21 = daily_return.rolling(window=21, min_periods=21).apply(_rolling_autocorr_lag1, raw=True)
    drawdown_63 = price.div(price.rolling(window=63, min_periods=1).max()).sub(1.0)

    out = pd.DataFrame(
        {
            "price": price,
            "daily_return": daily_return,
            "sma_63": sma_63,
            "trend_sma63_slope_21d": trend,
            "ewma_vol_63d": ewma_vol_63,
            "vol_percentile_252d": vol_percentile,
            "autocorr_21d": autocorr_21,
            "drawdown_63d": drawdown_63,
        },
        index=df.index,
    )
    return out


def classify_regimes(indicators: pd.DataFrame) -> pd.DataFrame:
    """Assign a hard regime label from the trailing SPY indicator state."""
    if not set(INDICATOR_COLUMNS).issubset(indicators.columns):
        missing = sorted(set(INDICATOR_COLUMNS) - set(indicators.columns))
        raise KeyError(f"missing indicator columns: {missing}")

    out = indicators.copy()
    ready = out.loc[:, INDICATOR_COLUMNS].notna().all(axis=1)
    regime = pd.Series(pd.NA, index=out.index, dtype="object")

    trend = out["trend_sma63_slope_21d"]
    vol = out["vol_percentile_252d"]
    autocorr = out["autocorr_21d"]
    drawdown = out["drawdown_63d"]

    crisis_mask = ready & (
        ((vol >= 0.95) & (drawdown <= -0.14))
        | ((vol >= 0.985) & (drawdown <= -0.10))
    )
    bear_mask = ready & ~crisis_mask & (trend <= -0.00035) & ((drawdown <= -0.05) | (vol >= 0.70))
    mean_revert_mask = (
        ready
        & ~crisis_mask
        & ~bear_mask
        & (autocorr <= -0.18)
        & (trend.abs() <= 0.0010)
        & (drawdown > -0.15)
    )
    bull_mask = (
        ready
        & ~crisis_mask
        & ~bear_mask
        & ~mean_revert_mask
        & (trend >= 0.00035)
        & (drawdown >= -0.05)
        & (vol <= 0.75)
    )
    neutral_mask = ready & ~(crisis_mask | bear_mask | mean_revert_mask | bull_mask)

    regime.loc[crisis_mask] = "crisis"
    regime.loc[bear_mask] = "bear"
    regime.loc[bull_mask] = "bull"
    regime.loc[mean_revert_mask] = "mean_revert"
    regime.loc[neutral_mask] = "neutral"

    out["regime"] = regime
    return out


def detect_regimes(frame: pd.DataFrame, *, price_col: str | None = None) -> pd.DataFrame:
    """Compute indicators and hard regime labels for a daily price frame."""
    return classify_regimes(compute_regime_indicators(frame, price_col=price_col))


def validate_known_events(regime_frame: pd.DataFrame) -> list[EventValidation]:
    """Check that known stress windows mostly map to the expected regimes."""
    windows = [
        ("GFC 2007-2009", "2007-10-09", "2009-03-09", {"crisis", "bear"}, 0.55),
        ("COVID Mar 2020", "2020-02-19", "2020-03-23", {"crisis"}, 0.50),
        ("2022 Bear", "2022-01-03", "2022-10-12", {"bear", "crisis"}, 0.55),
    ]
    results: list[EventValidation] = []
    regime_series = regime_frame["regime"].dropna()
    for name, start, end, expected_regimes, min_share in windows:
        subset = regime_series.loc[start:end]
        counts = subset.value_counts(normalize=True)
        dominant_regime = counts.index[0] if not counts.empty else None
        dominant_share = float(counts.iloc[0]) if not counts.empty else None
        matched_share = float(counts.reindex(sorted(expected_regimes), fill_value=0.0).sum()) if not counts.empty else None
        results.append(
            EventValidation(
                name=name,
                start=start,
                end=end,
                expected_regimes=sorted(expected_regimes),
                sample_count=int(subset.shape[0]),
                dominant_regime=dominant_regime,
                dominant_share=dominant_share,
                matched_share=matched_share,
                passed=bool((matched_share or 0.0) >= min_share),
            )
        )
    return results


def _select_embargo_dates(index: pd.DatetimeIndex, *, embargo_days: int, sample_step: int) -> list[pd.Timestamp]:
    usable = list(index)
    if len(usable) <= embargo_days + 1:
        return []
    end = len(usable) - embargo_days
    start = max(0, min(315, end))
    return [pd.Timestamp(usable[pos]) for pos in range(start, end, sample_step)]


def verify_no_lookahead(
    frame: pd.DataFrame,
    *,
    embargo_days: int = 5,
    sample_step: int = 21,
    price_col: str | None = None,
    atol: float = 1e-12,
) -> EmbargoValidation:
    """Replay the detector on embargoed cutoffs and confirm past labels are unchanged."""
    df = _ensure_datetime_index(frame)
    checkpoints = _select_embargo_dates(df.index, embargo_days=embargo_days, sample_step=sample_step)
    mismatch_count = 0
    max_abs_diff = {column: 0.0 for column in INDICATOR_COLUMNS}

    for checkpoint in checkpoints:
        checkpoint_loc = df.index.get_loc(checkpoint)
        t_end = df.iloc[: checkpoint_loc + 1]
        t_plus_end = df.iloc[: checkpoint_loc + embargo_days + 1]
        asof_t = detect_regimes(t_end, price_col=price_col).iloc[-1]
        asof_t_plus = detect_regimes(t_plus_end, price_col=price_col).loc[checkpoint]

        row_mismatch = False
        for column in INDICATOR_COLUMNS:
            left = float(asof_t[column])
            right = float(asof_t_plus[column])
            diff = abs(left - right)
            max_abs_diff[column] = max(max_abs_diff[column], diff)
            if not np.isclose(left, right, atol=atol, equal_nan=True):
                row_mismatch = True
        if asof_t["regime"] != asof_t_plus["regime"]:
            row_mismatch = True
        mismatch_count += int(row_mismatch)

    return EmbargoValidation(
        embargo_days=embargo_days,
        sample_step=sample_step,
        sample_count=len(checkpoints),
        mismatch_count=mismatch_count,
        max_abs_indicator_diff=max_abs_diff,
        passed=mismatch_count == 0,
    )


def plot_regimes(
    regime_frame: pd.DataFrame,
    *,
    ticker: str = "SPY",
    output_path: str | Path,
) -> Path:
    """Plot price and trailing indicators with regime shading."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = regime_frame.dropna(subset=["regime"]).copy()
    if frame.empty:
        raise ValueError("no classified regime rows available to plot")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.2]},
    )

    price_ax, trend_ax, stress_ax = axes
    price_ax.plot(frame.index, frame["price"], color="#111827", linewidth=1.3, label=f"{ticker} price")
    price_ax.set_ylabel("Price")
    price_ax.set_title(f"{ticker} online regime classification")

    trend_ax.plot(frame.index, frame["trend_sma63_slope_21d"], color="#2a9d8f", linewidth=1.0, label="Trend slope")
    trend_ax.plot(frame.index, frame["autocorr_21d"], color="#264653", linewidth=0.9, alpha=0.8, label="Autocorr")
    trend_ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    trend_ax.set_ylabel("Trend / ACF")

    stress_ax.plot(frame.index, frame["vol_percentile_252d"], color="#d00000", linewidth=1.0, label="Vol pct")
    stress_ax.plot(frame.index, frame["drawdown_63d"], color="#6d597a", linewidth=1.0, label="Drawdown")
    stress_ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    stress_ax.set_ylabel("Stress")
    stress_ax.set_xlabel("Date")

    current_regime = None
    segment_start = None
    for date, regime in frame["regime"].items():
        if regime != current_regime:
            if current_regime is not None and segment_start is not None:
                for ax in axes:
                    ax.axvspan(segment_start, date, color=REGIME_COLORS[current_regime], alpha=0.12)
            current_regime = regime
            segment_start = date
    if current_regime is not None and segment_start is not None:
        for ax in axes:
            ax.axvspan(segment_start, frame.index[-1], color=REGIME_COLORS[current_regime], alpha=0.12)

    handles = [
        plt.Line2D([0], [0], color=REGIME_COLORS[name], lw=6, alpha=0.7, label=name)
        for name in REGIME_ORDER
    ]
    price_ax.legend(handles=handles, loc="upper left", ncol=5, frameon=False)
    trend_ax.legend(loc="upper left", frameon=False)
    stress_ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_regime_detector(
    ticker: str = "SPY",
    *,
    plot: bool = False,
    data_dir: str | Path = DATA_DIR,
    results_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full detector workflow and persist the regime frame + validation."""
    prices = load_price_history(ticker, data_dir=data_dir)
    regime_frame = detect_regimes(prices)
    event_validation = validate_known_events(regime_frame)
    embargo_validation = verify_no_lookahead(prices)

    results_dir = _resolve_results_dir(results_dir)
    parquet_path = results_dir / f"regime_detector_{ticker}.parquet"
    summary_path = results_dir / f"regime_detector_{ticker}_summary.json"
    regime_frame.to_parquet(parquet_path)

    summary = {
        "ticker": ticker,
        "source_rows": int(len(prices)),
        "classified_rows": int(regime_frame["regime"].notna().sum()),
        "parquet_path": str(parquet_path),
        "event_validation": [asdict(item) for item in event_validation],
        "embargo_validation": asdict(embargo_validation),
        "regime_mix": regime_frame["regime"].value_counts(normalize=True, dropna=True).to_dict(),
    }

    if plot:
        plot_path = results_dir / f"regime_detector_{ticker}.png"
        summary["plot_path"] = str(plot_regimes(regime_frame, ticker=ticker, output_path=plot_path))

    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    summary["summary_path"] = str(summary_path)
    return regime_frame, summary


def _format_event_lines(validations: Iterable[EventValidation]) -> list[str]:
    lines: list[str] = []
    for validation in validations:
        dominant = validation.dominant_regime or "n/a"
        matched_share = "n/a" if validation.matched_share is None else f"{validation.matched_share:.1%}"
        lines.append(
            f"{validation.name}: dominant={dominant}, matched_share={matched_share}, pass={validation.passed}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Online SPY regime classification")
    parser.add_argument("--ticker", default="SPY", help="Ticker to use for the market regime proxy")
    parser.add_argument("--plot", action="store_true", help="Save a regime plot to the results directory")
    args = parser.parse_args()

    _, summary = run_regime_detector(ticker=args.ticker, plot=args.plot)

    print(f"Saved regime frame: {summary['parquet_path']}")
    print(f"Saved summary: {summary['summary_path']}")
    if "plot_path" in summary:
        print(f"Saved plot: {summary['plot_path']}")
    for line in _format_event_lines(EventValidation(**item) for item in summary["event_validation"]):
        print(line)
    embargo = EmbargoValidation(**summary["embargo_validation"])
    print(
        "Embargo check: "
        f"samples={embargo.sample_count}, mismatches={embargo.mismatch_count}, pass={embargo.passed}"
    )


if __name__ == "__main__":
    main()
