# Expert Panel: Next Steps — GPT-5
*Generated: 2026-04-19*

## Paper Trading Setup

### Duration and Cadence
- 16 weeks minimum; target 24 weeks to cover at least one regime transition
- Daily close-to-close execution simulation using the same signal cut time you'll have live (e.g., T-15m to close) with Market-on-Close or next-open fills
- Record broker-quoted spreads and realized fills

### Accounts, Automation, and Logging
- Use a single paper account configured to TFSA-like constraints (no shorting/margin, CAD base with FX conversion cost model if buying US ETFs)
- Fix a daily runbook: data pull freeze time, signal generation, sizing, order staging, execution, post-trade PnL and risk attribution
- Log everything: regime state, features snapshot hash, model version, AUM, exposures by sector/ticker, trades, slippage, realized costs, realized beta to SPY/QQQ, realized vol, hit rate, drawdown

### Monitoring Criteria (Weekly Review)
- Live-vs-sim tracking error: < 40 bps/week; if > 80 bps in any week, investigate
- Realized 63d Sharpe vs backtest-inferred: expect ~0.3–0.6 in benign regimes. If 63d realized Sharpe < -0.2 for 8 weeks, flag
- Realized vol vs 10% target: keep 8–12%; outside → rescale
- Turnover and costs: turnover ≤ 120%/month; realized costs ≤ 8 bps/turn
- AUC proxy in live: compute rolling AUC on binary 1–5d forward labels; if < 0.55 for 6 weeks, flag

### Kill/Hold Conditions
- **Hard kill:** live drawdown > 12% OR 2 consecutive months with realized Sharpe < -0.5 OR tracking error to your own sim > 200 bps over a month OR data leak detected
- **Soft hold** (pause size increases): 63d realized Sharpe < 0 for 2 months; AUC < 0.58 for 8 weeks; mean slippage > backtest assumption by 2x for 4+ weeks

## Additional Validation Before Real Capital

### Overfitting/Robustness
- Deflated Sharpe ratio and PBO (Probability of Backtest Overfitting)
- Purged, embargoed K-fold CV on walk-forward residuals; run CSCV on key hyperparams
- Hyperparam/seed sweeps: verify performance plateau; reject "narrow peak" configs

### Data Integrity
- Full survivorship-bias-free universe with point-in-time constituents/delistings
- Corporate-actions audit (splits, dividends, spin-offs). HYG dividend test is good—replicate across equities and ETFs
- Listing inception gates (e.g., COIN post-2021 only); zero-weight before inception

### Execution Realism
- Variable spread/impact model by time-of-day and volatility; stress test costs at 2–3x current assumptions
- Fill model: MOC/OPG vs VWAP; re-run backtest with intended live execution path

### Stability Diagnostics
- Regime labeling stability: HMM state transition matrix consistency through time; NMI of regimes across refits
- Feature attribution stability (SHAP) by regime; drop brittle features
- Sizing audit: with 0.25 Kelly, ensure no single-name tail > 4% daily VaR at 99%

### Out-of-Sample/Orthogonal Checks
- Time-split OOS: 2018–2020 train → 2021–2022 test; 2020 crisis exclusion tests
- Universe OOS: exclude mega-caps; run on sector ETFs only; run on a non-overlapping set
- White's Reality Check/SPA on alternative spec choices

## System Improvements

### Positioning and Risk
- Replace raw 0.25-Kelly with capped fractional Kelly ladder: 0–0.1–0.2–0.25 tied to deflated Sharpe and regime; cap single-name weight at 8%, sector at 25%, crypto proxy at 2%
- Drawdown-aware scaler: reduce gross exposure linearly 100%→50% as DD moves 0%→10%
- Add cross-sectional risk parity overlay to smooth sector concentration

### Signals and Features
- Regime-conditional thresholds: require higher margin in mean-revert; loosen in crisis where you have edge
- Add low-latency market state features: VIX term structure, put/call ratio, breadth (AD line), realized skew; ensure PIT sourcing
- Monotonic XGB is good—add a small ridge/logit ensemble for calibration; isotonic calibration on out-of-fold scores
- Per-regime sample weights in training to de-emphasize regimes with negative edge (mean-revert)

### Turnover/Cost Control
- Hysteresis bands on entry/exit to cut churn; minimum holding period 3–5 trading days unless stop-out
- Stagger rebalances (e.g., 50% trade now, 50% next day) to reduce slippage on crowded names

### Execution
- Use MOC for large ETFs; use participation bands (5–10% ADV) and limit-to-close
- Measure and model queue position; skip orders if predicted slippage > expected edge

## Regime-Specific Recommendations

- **Crisis (best):** Press advantage. Increase gross to defensives/quality. Tighten on cyclicals/COIN. Wider take-profit; let winners run
- **Neutral (strong):** Maintain core. Cross-sectional selection; avoid overtrading
- **Bear (moderate):** Tilt to defensives; reduce beta. Higher entry threshold; smaller sizing; faster de-risk
- **Bull (weak):** Reduce churn: looser exits, longer holding periods. Focus on broad index ETFs
- **Mean-revert (negative):** Default to flat or half-size on strongest signals only; set "no-trade" band

## TFSA Risk Guardrails

- Long-only; no margin, no short, no derivatives
- Max gross: 100%. Cash buffer 5–15% by regime
- Single-name cap 8%; sector cap 25%; crypto proxy 2%
- Hard stop: -15% account DD; soft at -10% triggers 50% gross cut
- Realized vol target 8–12%; outside band auto-rescale
- Max participation: 10% ADV per name/day
- Slippage budget: ≤ 5 bps/trade ETFs, ≤ 10 bps/trade single stocks
- FX costs for US securities in CAD TFSA: 15–25 bps round-trip (Norbert's Gambit)
- Minimize day-trade-like activity; document investment intent

## What NOT To Do

- Don't go live before 16–24 weeks of paper with matched execution
- Don't increase Kelly or remove caps
- Don't trade mean-revert regime like others
- Don't change multiple components at once; one change per cycle with A/B paper tracking
- Don't assume costs; measure live spreads/impact weekly
- Don't rely on COIN as crypto beta proxy—keep tiny or drop in risk-off
- Don't leak future info via index membership, dividends, or point-in-time anomalies
- Don't skip disaster recovery: missed orders, stale data, API failures happen
