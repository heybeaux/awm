# Change Proposal: Equity Benchmark v2 — Calibration, Regime Ensemble & Paper MVP

## Summary

Phase 2 of the equity benchmark: transform the validated fusion system into a paper-tradeable MVP. v1 proved that Fusion outperforms XGBoost in bear markets (AUC 0.598 vs 0.517 in 2022 bear). v2 fixes the thresholding failure that produces zero trades, adds regime-weighted model blending, enriches features, and builds a cost-realistic paper-trading pipeline.

## Motivation

- v1 bear market stress test (77 windows, 40 tickers, 2005–2026) validated the fusion thesis: Fusion wins in crisis, XGBoost wins in calm markets
- But all backtests show zero trades for Fusion/Le-WM at fixed 0.55+ thresholds — the models are well-calibrated near 0.50, so fixed cutoffs starve trades
- XGBoost generates trades but loses money at every threshold (-81.5% compound in Normal/Bull at 0.55)
- Expert panel consultation (3 independent analyses) unanimously recommended: top-k selection, regime-weighted ensemble, cross-asset features, and prior-residual fusion
- HYG ticker shows 0.91 AUC — almost certainly an artifact that could contaminate results
- The system needs cost-realistic portfolio simulation before paper trading
- Canadian TFSA constraints (long-only, no day trading pattern) shape the MVP strategy

## Spec References

- `equity-benchmark` — original spec (v1)
- `equity-benchmark/spec-v2-addendum` — new requirements for calibration, regime ensemble, features, portfolio simulation, and HYG investigation

## Risks

| Risk | Severity | Likelihood |
|------|----------|------------|
| Overfitting thresholds to historical data | High | Medium |
| Regime detection lags real transitions | Medium | High |
| Feature leakage via cross-asset signals | High | Low |
| HYG artifact contaminating priors/thresholds | Medium | High |
| Top-k selection inflating apparent performance | Medium | Medium |
| Insufficient edge after costs (Sharpe < 0.8) | High | Medium |

## Mitigations

- All threshold/k optimization uses only past data (expanding window, never test set)
- Regime detector uses strictly backward-looking indicators with 5-day embargo
- Cross-asset features lagged 1 day to simulate realistic availability
- HYG quarantined: excluded as traded asset, 4 triage tests before reinstatement
- Top-k backtests include transaction costs (2–5 bps) and turnover penalties
- Minimum 20 trade signals required before statistical significance claims
- Paper trading phase before any real capital

## Approach

### Phase 1 — Calibration & Selection Layer (Days 1–2)
Fix the zero-trade problem. Implement isotonic calibration per walk-forward window, top-k selection (daily rank and select top-k longs by ensemble score), logit-z normalization, and edge-based position sizing.

### Phase 2 — Regime-Weighted Ensemble (Days 3–4)
Replace the fixed 50/50 Fusion blend with dynamic regime-conditional weights. Build an online regime detector from backward-looking indicators. Fit stacking weights per regime on out-of-fold data.

### Phase 3 — Feature Engineering & HYG Audit (Days 5–7)
Add free cross-asset signals (VIX, credit spreads, yield curve), overnight/intraday return split, volume features, and cross-sectional ranks. Run the 4 HYG triage tests to determine artifact vs signal.

### Phase 4 — Portfolio Layer & Paper MVP (Days 8–14)
Build cost-realistic portfolio simulation with vol targeting, turnover control, and Kelly-fraction sizing. Run full walk-forward backtest with costs. Build daily signal export pipeline for paper trading.
