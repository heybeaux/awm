# Expert Panel: Adversarial Analysis — GPT-5
*Generated: 2026-04-19*

## 🔴 Critical

### 1. Option B "wins" 74/74 windows
- **Risk:** Comparison or evaluation bug. Perfect sweep is exceedingly unlikely under honest OOS evaluation. Possible: same predictions used for both models; leakage in metric; in-sample metrics mistakenly used OOS; alignment/label leakage
- **Test:**
  - Re-run with locked evaluation harness logging per-window predictions, labels, metrics for each option
  - Swap model labels A/B and confirm results change
  - Randomize model outputs (permute predictions), verify comparison no longer shows systematic wins
  - Independent re-implementation of metric/comparator
- **Worst case:** Headline outperformance is artifact; Option B has no real edge; entire selection invalid

### 2. Look-ahead/temporal leakage via features, labels, or regimes
- **Risk:**
  - Macro features/latent embeddings may use revised data not available at time t
  - HMM regimes estimated on full-sample or using future data per window
  - Vol targeting or Kelly inputs using future volatility/drift estimates
- **Test:**
  - Enforce point-in-time data: real-time vintages; publication lags
  - Estimate HMM strictly on training data each window; freeze parameters before scoring test
  - Verify all transformations (scalers, PCA/embeddings, isotonic calibration) fit only on training folds
  - Combinatorially Purged CV with embargo; audit code for df.shift(-1)/merge-asof errors
- **Worst case:** Performance collapses to random or negative once leakage removed

### 3. Zero unit tests on 180KB of financial code
- **Risk:** Silent math/alignment bugs (returns vs prices, compounding, cost application timing, look-ahead through rolling ops, leakage via preprocessing)
- **Test:**
  - Minimal unit tests: shape/NaN checks; alignment assertions; invariants on rolling windows; toy dataset known answers; synthetic label experiments
  - Differential test against second independent codebase
- **Worst case:** Multiple compounding bugs; backtest results not trustworthy

### 4. Isotonic calibration on small per-window samples
- **Risk:** Severe overfitting/calibration leakage inflating AUC/Sharpe with 63-day test windows and low event counts
- **Test:**
  - Nested calibration: fit isotonic only on calibration split within training, not on test
  - Reliability plots and ECE/MCE across windows; compare to Platt or no-calibration
  - Constrain minimum calibration sample size; pool across adjacent windows with strict purging
- **Worst case:** Probabilities are junk; Kelly sizing overbets on noise, Sharpe collapses live

### 5. HMM regime process integrity
- **Risk:** HMM fit uses future data (full sample) or overfits transitions on tiny windows → regime metrics spuriously high
- **Test:**
  - Refit HMM within each training window only; lock parameters; score test
  - Verify regime definitions stable and economically interpretable; run placebo with permuted returns
- **Worst case:** Regime "alpha" entirely look-ahead driven; overall performance overstated

### 6. Universe survivorship/selection bias
- **Risk:** In-sample cherry-picking; missing delisted names; post hoc inclusion of winners. COIN availability changes regime behavior and costs materially
- **Test:**
  - Rebuild universe with point-in-time constituents including delisted/severely underperforming
  - Run sensitivity on universe definitions and inception dates
- **Worst case:** True OOS performance much worse; edge largely selection-driven

## 🟡 Moderate

### 7. Mean vs median Sharpe gap (0.81 vs 0.33)
- **Risk:** Returns rely on few tail wins; unstable conditional alpha; high skew/kurtosis; non-stationarity
- **Test:**
  - Distributional analysis: per-window PnL skew/kurtosis; bootstrapped Sharpe; 5th–95th percentiles
  - Time-decay analysis: early vs late window contributions; jackknife Sharpe
- **Worst case:** Live performance dominated by rare non-repeating events; realized Sharpe near median or lower

### 8. Cost model unrealism (5 bps across all 40 tickers including COIN)
- **Risk:** Understated slippage/fees/borrow; crisis liquidity droughts; ETF spreads widen in stress
- **Test:**
  - Stress cost assumptions: 2–10x slippage during high vol; add borrow fees; venue-specific fees; volume-aware slippage
  - Recompute with intraday bar-level spread models
- **Worst case:** Alpha fully consumed by realistic frictions; returns approach zero or negative

### 9. Kelly sizing (0.25) + target 10% vol with predictive-probability sizing
- **Risk:** Miscalibrated probabilities → overbetting; path-dependency exaggerated; vol targeting may use realized vol with leakage
- **Test:**
  - Replace with fixed risk per trade / capped gross exposure; no Kelly
  - Use strictly ex-ante vol estimates with lags; cap leverage/exposure
  - Compare to 0.1–0.15 Kelly fractions and flat 1/N benchmark
- **Worst case:** Overbet + drawdowns spike; realized Sharpe far lower with safer sizing

### 10. Window size/statistical power (63-day tests, 74 windows)
- **Risk:** Each test period short; variance high; easy to overfit across sequential windows; "won all 74" could be artifact of noisy small tests + bug
- **Test:**
  - Aggregate to quarterly/annual evaluation; block bootstrap; White's reality check/Hansen SPA
- **Worst case:** Performance not statistically distinguishable from chance

### 11. HYG audit signals fragility
- **Risk:**
  - Label shuffle AUC 0.538 > 0.5 may indicate residual structure or improper shuffling/purging
  - Embargoed vs non-embargoed nearly identical (0.734 vs 0.735) suggests embargo not addressing actual leakage channel
- **Test:**
  - Combinatorial purged CV; stratified shuffle preserving time blocks; check class balance/prevalence drift
  - Audit dividend/split adjustments; test with TR vs PR series
- **Worst case:** Edge depends on priors and data treatment quirks; disappears with correct purging

### 12. Option B monotonic constraints interaction with features
- **Risk:** Constraints may encode directional priors that leak target sign via feature engineering; spurious stability
- **Test:**
  - Remove monotonic constraints; compare; randomize feature sign; ablate features for proxy labels
- **Worst case:** Performance hinges on mislabeled or target-proxy feature; collapses after fix

### 13. Metric choice vs objective mismatch (AUC vs PnL)
- **Risk:** High AUC can coexist with poor PnL if calibration only helpful around irrelevant thresholds
- **Test:**
  - Economic metrics: turnover, hit ratio by decile, return per risk bucket, deflated Sharpe
  - Threshold sweeps with transaction costs; cost-aware ROC
- **Worst case:** AUC doesn't translate to tradable alpha

## 🟟 Minor

### 14. Crisis Sharpe 1.30 on 161 days
- **Risk:** Small-sample over-interpretation; multiple comparisons across five regimes
- **Test:** Confidence intervals; Bayesian shrinkage across regimes; SPA test across partitions
- **Worst case:** No true crisis edge; reported Sharpe is noise

### 15. Bull regime Sharpe 0.28, mean-revert near 0
- **Risk:** Strategy fails in dominant regime (bull, 2131 days); overall edge might be timing-luck outside bulls
- **Test:** Compare to simple beta-managed SPY with 10% target vol; evaluate conditional beta/alpha
- **Worst case:** Complexity adds no value compared to simple vol-scaled beta

### 16. Walk-forward design/embargo adequacy
- **Risk:** Embargo too short for signal horizon; overlapping effects via feature lags
- **Test:** Increase embargo; vary purging windows; confirm smooth degradation
- **Worst case:** Hidden leakage persists; true OOS worse

### 17. Data vendor/adjustment hygiene
- **Risk:** Corporate action misadjustments, split/dividend timing, ETF roll artifacts
- **Test:** Cross-validate with second vendor; explicit ex-div dates and adjustment audits
- **Worst case:** Returns overstated; alpha not real

### 18. Execution feasibility and capacity
- **Risk:** Turnover, market impact, borrow constraints ignored; Kelly+vol targeting might force size during illiquidity
- **Test:** Add participation caps; simulate impact; add borrow locate failure probability
- **Worst case:** Strategy not tradable at any reasonable size

## Actionable Audit Plan (Prioritized)

1. Re-run evaluation with locked, independently validated harness to verify 74/74 claim
2. Enforce point-in-time data for all features/regimes; refit HMM/calibration only on training; add combinatorial purging + larger embargo
3. Add unit tests and toy-data validations for returns, costs, alignment, rolling ops, calibration splits
4. Remove isotonic calibration (or nest properly); de-risk Kelly by switching to fixed, capped risk; re-run
5. Replace 5 bps with regime/asset-aware slippage/borrow; add impact; re-run
6. Rebuild universe using point-in-time constituents with delistings; sensitivity analyses
7. Statistical rigor: deflated Sharpe, block bootstrap, White/SPA tests; report medians and robust intervals
