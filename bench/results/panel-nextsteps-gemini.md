# Expert Panel: Next Steps — Gemini 2.5 Pro
*Generated: 2026-04-19*

## Executive Summary

This system is not a broad market-beating strategy; it is a **Crisis Alpha** generator. Its primary value is outstanding performance during high-volatility, uncertain, and crisis periods. Its performance in calm, bullish markets is poor. Deploy not to generate primary wealth, but as a powerful diversifier generating uncorrelated returns when other strategies are likely failing. The low median Sharpe (0.33) confirms this: long periods of flat or slightly negative performance punctuated by short bursts of high profitability.

## Paper Trading Setup

### Duration
- Minimum **6 months**. Non-negotiable. Need to cross at least one quarterly earnings cycle and give time for market conditions to change.

### Monitoring Criteria
- **Execution Slippage:** Primary metric. Log price system *thought* it would get vs. paper fill. If average slippage exceeds 2-3 bps, cost model is wrong
- **Regime Drift:** Track live HMM regime classification. Does it align with qualitative market assessment? How frequently does it flip-flop?
- **Performance vs. Backtest:** Benchmark is your own backtest. If market in "Neutral" regime, compare paper Sharpe against backtested "Neutral" stats, not overall average

### Kill Conditions (Pre-defined & Automated)
1. **Drawdown Breach:** Paper drawdown exceeds -18.7% by any amount → kill immediately
2. **Slippage Threshold:** Average transaction costs > 10 bps over 20+ trades → kill
3. **Regime Mismatch:** System in "Crisis" or "Neutral" but losing money consistently for a month (Sharpe < -1.0) → kill

## Additional Pre-Capital Validation

- **Parameter Sensitivity Analysis:** Vary key parameters by ±20% and re-run. If Sharpe collapses, system is brittle and over-optimized. Robust system shows graceful degradation
- **Universe Expansion/Contraction:** Re-run on only 10 most liquid tickers, then on expanded 100 tickers. Tests for universe selection bias. Edge should still be present, even if diluted
- **Cost & Latency Stress Test:** Re-run with 10 bps and 15 bps costs. Simulate 1-second execution latency (signal on bar t, trade at price t+1). Shows how much edge depends on perfect execution

## System Improvements

- **Regime-Specific Off Switch:** Mean-Reversion regime (Sharpe 0.01) — hold 100% cash or risk-free asset. Simplest, most effective improvement
- **Dynamic Asset Universe:** During "Bull" regimes, include higher-beta growth stocks. During "Bear" regimes, shrink to low-volatility sector ETFs and mega-caps
- **Feature Engineering for Bull Markets:** Macro features geared towards detecting volatility/dislocations. Add features for positive momentum and risk-on sentiment (SPY/GLD ratio, bullish options activity)
- **Meta-labeling for Position Sizing:** Keep XGBoost for entry/exit, train secondary model to predict probability that primary signal will be profitable. Refines raw Kelly sizing

## Regime-Specific Strategy

- **Crisis (Sharpe 1.30):** Full allocation. This is where you make money. Ensure execution infrastructure is flawless
- **Neutral (Sharpe 0.94):** Full allocation. Secondary alpha engine
- **Bear (Sharpe 0.40):** Deploy with 50% of normal risk allocation. Edge present but weak
- **Bull (Sharpe 0.28):** **Turn off.** Statistically indistinguishable from noise after costs. Buy and hold SPY instead
- **Mean-Revert (Sharpe 0.01):** **Turn off.** Actively losing money via transaction costs. Move to cash

## TFSA Risk Guardrails

- **Portfolio Allocation:** Max 25-30% of total TFSA. Rest in VEQT/XEQT. System is diversifier, not core investment
- **Concentration Limits:** No single position > 15% of strategy capital, regardless of Kelly
- **Volatility Override:** If entire TFSA portfolio vol > 15%, system deleverages automatically
- **Manual Review:** Any single trade resulting in >5% strategy capital loss triggers alert

## What NOT To Do

- **Don't add more complexity yet.** Harvest existing edge more efficiently (turn off weak regimes) and validate robustness first
- **Don't ignore the Median Sharpe.** Gap between mean (0.81) and median (0.33) is the most important statistic. This is a "grind-grind-grind-jackpot" strategy. If you can't stomach flat periods, you'll abandon at the worst time
- **Don't override the system.** The moment you feel terrified and want to sell is exactly when the system is designed to perform. Trust it or don't run it
- **Don't annualize regime returns.** 1.30 Sharpe in Crisis over 161 days is incredible, but you cannot expect that for a full year
