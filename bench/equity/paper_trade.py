"""Phase 4 paper-trading signal pipeline.

Daily flow:
  1. download / refresh the local market data cache
  2. rebuild the enriched feature store
  3. train the Phase 4 candidate models through the selected date
  4. emit a JSON signal slate with date / ticker / direction / confidence /
     position_size under results/signals/
"""
from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from backtest_v2 import DEFAULT_COST_BPS, _json_default, _resolve_results_dir, generate_daily_signals
from features_v2 import build_cross_asset_features, enrich_feature_store, get_available_tickers
from pipeline import load_config, run_pipeline


SIGNALS_SUBDIR = "signals"


def _log(msg: str) -> None:
    print(f"[paper_trade] {msg}", flush=True)


def refresh_feature_store(signal_date: pd.Timestamp) -> dict[str, Any]:
    """Refresh raw data and enriched features through the requested date."""
    config = deepcopy(load_config())
    effective_end = (signal_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    config["data"]["end_date"] = effective_end

    tickers = get_available_tickers()
    _log(f"refreshing raw data and baseline features through {effective_end}")
    stats = run_pipeline(config, universe=tickers, force_download=False)

    _log("refreshing cross-asset and enriched feature columns")
    cross_asset = build_cross_asset_features(
        start_date=str(config["data"]["start_date"]),
        end_date=effective_end,
        force_download=False,
        allow_partial=True,
    )
    enrich_feature_store(tickers, cross_asset)
    return {
        "effective_end_date": effective_end,
        "tickers": tickers,
        "pipeline_stats": stats,
    }


def _signal_output_path(signal_date: pd.Timestamp, results_dir: Path | None = None) -> Path:
    base_dir = _resolve_results_dir(results_dir)
    signal_dir = base_dir / SIGNALS_SUBDIR
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir / f"signals_{signal_date.strftime('%Y-%m-%d')}.json"


def run_paper_trade(
    *,
    signal_date: str | pd.Timestamp | None = None,
    refresh_data: bool = True,
    output_path: Path | None = None,
) -> dict[str, Any]:
    signal_ts = pd.Timestamp(signal_date).normalize() if signal_date is not None else pd.Timestamp.today().normalize()
    refresh_summary: dict[str, Any] | None = None

    if refresh_data:
        refresh_summary = refresh_feature_store(signal_ts)

    signal_payload = generate_daily_signals(signal_ts, cost_bps=DEFAULT_COST_BPS)
    output = {
        "date": signal_ts.strftime("%Y-%m-%d"),
        "signals": signal_payload["signals"],
        "meta": {
            **signal_payload["meta"],
            "refresh_data": bool(refresh_data),
            "refresh_summary": refresh_summary,
        },
    }

    final_path = output_path or _signal_output_path(signal_ts)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with open(final_path, "w") as handle:
        json.dump(output, handle, indent=2, default=_json_default)
    _log(f"wrote {final_path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Phase 4 paper-trading signal pipeline")
    parser.add_argument("--date", default=None, help="Signal date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--no-refresh", action="store_true", help="Skip market-data and feature refresh")
    parser.add_argument("--output", default=None, help="Optional explicit JSON output path")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser() if args.output else None
    run_paper_trade(
        signal_date=args.date,
        refresh_data=not args.no_refresh,
        output_path=output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
