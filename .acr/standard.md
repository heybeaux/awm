# AWM (Agent Workflow Model)

**Purpose:** Predictive execution protocol for agent pipelines. The validation domain is daily equity signal generation: walk-forward train + score across ~40 tickers with regime-gated portfolio execution. Surfaces as the *intent* faculty in the Sonder stack — declares what an agent is trying to do, partners with Lattice's policy engine for gating, and feeds the SonderEvent's `intent` field. The signal runner has a real daily cron and is observed in production.
**Repo:** https://github.com/heybeaux/awm
**Status:** active
**Phase:** production — daily signals + Announcer running
**Last verified:** 2026-05-18

## Runtime

- **Local path:** ~/awm (and `~/awm/bench/equity/scripts/daily_signal_logger.py`)
- **Tech:** Python — pandas + scikit-learn + XGBoost
- **Signal runner cron:** `AWM Signal Runner` (id `25be22ce-b6d0-403b-9da1-f7f1a6c7f134`)
  - Schedule: `0 14 * * 1-5` America/Vancouver
  - Cron timeout 2700s (45 min); inner bash `timeout 2400` (40 min)
- **Announcer cron:** id `0d020d84-73d3-4873-b29b-d449d3a0ee80` at 14:45 PT (shifted from 14:30 to give runner room)
- **Outputs:** `~/awm/bench/results/signals/signals_YYYY-MM-DD.json` + regime posterior parquet

## Dependencies

- **Depends on:** OHLCV + cross-asset feature pipeline (`pipeline.py`, `features_v2.py`)
- **Used by:** Sonder intent adapter, Announcer cron, downstream portfolio execution
- **External:** market data sources

## Key contacts

- **Owner:** @beauxwalton
- **Recent contributors:** @beauxwalton

## Quick gotchas

- **0 signals is usually fine.** AWM trades only bull/crisis regimes by design. On bear/neutral/mean_revert days the regime gate zeros position weights — expected behavior, not a bug. Check `meta.regime` before investigating.
- **Runtime is ~18min** because `generate_daily_signals` refits all models from scratch every run (walk-forward by design). If it ever exceeds 30min, don't just bump the timeout — refactor to nightly bake + signal-time inference (drops runtime to <60s).
- **No model persistence.** Each day fits + discards. Don't optimize before the refactor is needed.
- **Observability gap:** signal JSON omits `meta.regime` and `meta.regime_gate_active`. Adding both lets Announcer parse "0 signals" without ambiguity.
- **Stale-cache bug (fixed 2026-04-21).** OHLCV + cross-asset caches used to return without coverage check. If feature-store dates go missing again, check `download_data` (pipeline.py) + `build_cross_asset_features` (features_v2.py).

## Where to learn more

- `deep.md` — model heads, regime ensemble, walk-forward design
- Memory: `awm-signal-runner-runtime.md`, `awm-signal-behavior.md`
