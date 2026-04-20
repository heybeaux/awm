# Expert Panel: Adversarial Analysis — Gemini 2.5 Pro
*Generated: 2026-04-19*

## Overall Assessment

The system has numerous red flags characteristic of overfit, fragile, and potentially unprofitable strategies. The combination of AI-generated code with zero tests, survivorship bias, and statistically questionable performance claims suggests the reported edge is likely illusory. Zero confidence in deploying with real capital without addressing these findings.

## 🔴 Critical

### 1. Zero Unit Tests on 180KB of AI-Generated Code
- **Risk:** Entire backtest likely suffering from subtle, critical bug invalidating all results. AI-generated code prone to errors in data handling (look-ahead), transaction costs, position sizing, or metrics. Without tests, no evidence code does what it claims
- **Test:**
  - Line-by-line manual review by senior quant dev
  - Comprehensive unit/integration tests for every function, especially portfolio construction, cost model, performance calculation
  - Benchmark against known-good system: implement simple strategy (e.g., MA crossover) within same framework, verify matches known-good backtesting library
- **Worst case:** Backtest is complete fabrication. Single bug responsible for entire "alpha." Could be -10% or worse live

### 2. Survivorship Bias in Ticker Universe
- **Risk:** Fatal flaw. COIN (2021 only) + mega-caps/sector ETFs selected with modern knowledge. Backtest from 2008 implicitly assumes we knew in 2008 which companies/sectors would win over 18 years. Excludes all failures (Lehman, Bear Stearns, etc.)
- **Test:**
  - Re-run with point-in-time constituent list for each period (e.g., S&P 500 constituents as of Jan 1, 2008, updated periodically)
  - Explicitly include delisted securities
- **Worst case:** Strategy's edge entirely artifact of selecting today's winners. Actual performance on realistic universe could be negative

### 3. Option B Winning 100% of Windows (74/74)
- **Risk:** Statistically impossible in real markets. Immediate sign of bug or leakage. Implies either:
  - Bug in comparison logic (Option A not being calculated correctly)
  - Look-ahead bias where Option B uses test set info
  - Severe overfitting to specific dataset/walk-forward structure
- **Test:**
  - Manually calculate performance for both options for a single random window with debugger
  - Replace Option B predictions with random noise — if still "wins," comparison code is broken
  - Verify no hyperparameter tuning uses test window data
- **Worst case:** Software bug creating illusion of outperformance. No actual edge. All A/B conclusions invalid

## 🟡 Moderate

### 4. Mean Sharpe (0.81) vs. Median Sharpe (0.33)
- **Risk:** Performance driven by small number of extreme outlier periods (fat tails). Strategy experiences long stretches of mediocre performance punctuated by rare massive wins. Characteristic of implicit short-vol strategies or curve-fitting to specific crisis events. Mean of 0.81 is misleading
- **Test:**
  - Histogram/box plot of 74 Sharpe ratios to visualize skewness
  - Deep investigation of highest-Sharpe windows — was there a single lucky trade?
  - Calculate skewness and kurtosis
- **Worst case:** Strategy unreliable; "good" performance depends on correctly predicting rare high-impact events (notoriously difficult). Will likely underperform for years before any "winning" period. -18.7% drawdown likely understates true tail risk

### 5. Crisis Sharpe 1.30 on 161 Days
- **Risk:** Statistically meaningless. Classic data mining. 161 days likely covers one or two specific events (COVID crash). HMM overfit to perfectly label this period; model trained on features that work for that *specific* event. Not generalizable
- **Test:**
  - Identify exact calendar dates; analyze model during other shocks not labeled same way
  - Find other historical crises (dot-com, Eurozone) and test "crisis" model there — will almost certainly fail
  - Check HMM regime stability: small input changes shouldn't cause large retroactive regime changes
- **Worst case:** "Crisis alpha" is mirage. Next crisis (different dynamics) leads to catastrophic losses as model applies rules from completely different event

### 6. Isotonic Calibration on Small Per-Window Samples
- **Risk:** Non-parametric calibration on small validation sets within each 63-day window extremely prone to overfitting. Calibration fitting to noise of that specific period; probabilities and position sizes poorly optimized and fragile
- **Test:**
  - Reliability diagrams for each window — will likely show erratic non-monotonic curves
  - Re-run without isotonic calibration. If performance collapses → calibration was major overfitting source. If similar → calibration useless
- **Worst case:** Adds fragility and complexity with no real benefit. Position sizes erratic and suboptimal live → higher costs and underperformance

## 🟟 Minor

### 7. 4.5% Annualized vs. S&P 500 ~10%
- **Risk:** Strategy likely not viable from business perspective. After considering all risks, complexity, operational costs, and potential flaws, 4.5% too low to justify existence — underperforms simple buy-and-hold SPY
- **Test:**
  - Compare equity curve to SPY, 60/40 portfolio, risk-parity
  - Factor in data/research/infrastructure costs — net return much lower than 4.5%
- **Worst case:** Complex, expensive way to significantly underperform simple cheap index ETF

### 8. Look-ahead Bias Despite Embargo
- **Risk:** Embargo doesn't prevent all look-ahead forms. Nearly identical embargoed (0.734) vs non-embargoed (0.735) AUC is suspicious. Subtle leakage still possible through features using revised economic data, or HMM using full data series for state estimation
- **Test:**
  - Review every input feature for strict point-in-time availability
  - Scrutinize HMM implementation: `fit` only on data available up to that point per window; `predict` for day only uses data available up to that day
- **Worst case:** Subtle look-ahead responsible for small but critical part of edge. Removing it degrades performance enough to make strategy unprofitable after costs
