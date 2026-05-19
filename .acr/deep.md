# AWM — Deep

Loaded when designing in or debugging AWM. Token budget ~2500.

## Model architecture

`generate_daily_signals` in `backtest_v2.py` runs:
1. Feature refresh (~30s) — OHLCV + cross-asset
2. Regime detector regen (~15s)
3. OOF predictions across ~40 tickers × ~5000 rows
4. Three model heads:
   - Option A: logistic regression
   - Option B: XGBoost
   - Regime ensemble (combines A+B with regime posterior weighting)

Target: `direction_5d` — boolean (>1% forward return over 5 trading days).

End-to-end ~18min (1078s real, 883s user, 192s sys on 2026-04-28 typical neutral-regime day).

## Regime gate

`DEFAULT_ACTIVE_REGIMES = frozenset({"bull", "crisis"})` lives in `run_portfolio_execution` in `backtest_v2.py`. When SPY regime posterior ∉ active regimes, position weights are zeroed (~line 873). Result: 0 signals on bear/neutral/mean_revert days.

Expected signal frequency: 30-40% of trading days, NOT daily.

## Key decisions / recent incidents

- **2026-04-21** — Stale-cache bug fixed. `download_data` and `build_cross_asset_features` now do target-last-bar coverage check + top-up fetch.
- **2026-04-28** — Runtime characterization: ~18min steady state. Cron timeouts set (2700s outer, 2400s inner). Announcer shifted to 14:45 PT.
- **(earlier)** — bull|crisis active-regime selection intentional. Strategy edge tested out only in those regimes.

## Internal vocabulary

- **Walk-forward** = fit up to `signal_date` exclusive; no lookahead
- **Active regimes** = `{"bull", "crisis"}` — when the gate fires
- **Option A/B/ensemble** = the three model heads
- **direction_5d** = the binary target (>1% in 5 trading days)
- **Regime posterior** = probabilistic regime classification from the detector

## Boundaries

- AWM **does** generate signals, declare intent, partner on gate policy, run the daily cron.
- AWM **does not** execute trades (downstream portfolio logic does), sign events (Sonder), or decide governance (Lattice).
- AWM **is** the intent contributor on Sonder's six-faculty envelope.

## Open questions / parked work

- **Refactor to nightly bake + signal-time inference** when runtime exceeds 30min. Drops to <60s but requires persisting `full_models` + calibrators + 3 heads keyed by (signal_date - 1).
- **Add `meta.regime` + `meta.regime_gate_active` to signal JSON output** for unambiguous downstream parsing.
