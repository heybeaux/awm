# Equity Prediction Benchmark — v2 Addendum

## Purpose

Extends the v1 equity benchmark spec with requirements for calibration-aware trade selection, regime-weighted ensemble, cross-asset features, cost-realistic portfolio simulation, and HYG artifact investigation. These requirements address the zero-trade problem identified in v1 stress testing and prepare the system for paper trading.

---

## Requirements

### Requirement: Calibration-aware trade selection

The system SHALL use calibrated probabilities and rank-based selection instead of fixed probability thresholds.

#### Scenario: Isotonic calibration per walk-forward window

- GIVEN a walk-forward window with out-of-fold predictions from each model
- WHEN calibration is applied
- THEN isotonic regression is fit on strictly out-of-fold predictions (no test data)
- AND separate calibrators are fit for each model (XGBoost, Le-WM, Fusion)
- AND calibrated probabilities replace raw model outputs for all downstream decisions
- AND reliability diagrams are generated to validate calibration quality

#### Scenario: Top-k daily selection

- GIVEN calibrated predictions for all tickers on a given trading day
- WHEN the selection module runs
- THEN tickers are ranked by calibrated probability (descending for longs)
- AND the top-k tickers are selected (k configurable, default 5)
- AND tickers with calibrated p < 0.50 are excluded even if in top-k
- AND k is optimized on training data by grid search over {3, 5, 7, 10}, maximizing net Sharpe with transaction costs

#### Scenario: Logit-z normalization

- GIVEN calibrated probabilities for a ticker over a rolling window
- WHEN logit-z normalization is applied
- THEN p is converted to logit(p) = log(p / (1-p))
- AND z-score is computed per ticker using a 252-day rolling mean and standard deviation
- AND selection can alternatively use z-score threshold (|z| > z_threshold) instead of top-k
- AND z_threshold is tuned on training data

#### Scenario: Edge-based position sizing

- GIVEN a selected ticker with calibrated probability p
- WHEN position size is computed
- THEN edge = 2p - 1
- AND position = clip(λ × edge / vol_target, max_weight)
- AND λ = 0.25 (quarter-Kelly, configurable)
- AND max_weight = 0.10 (10% of NAV per name)
- AND positions are normalized to sum to target gross exposure

### Requirement: Regime-weighted model ensemble

The system SHALL combine model predictions using dynamic weights conditioned on detected market regime, with no future information leakage.

#### Scenario: Online regime detection

- GIVEN daily market data for SPY up to time t
- WHEN the regime detector runs
- THEN it computes four backward-looking indicators: trend (63d SMA slope), volatility percentile (63d EWMA vol vs 252d history), autocorrelation (21d rolling), and drawdown (vs 63d high)
- AND it classifies the regime into one of: crisis, bear, bull, mean_revert, neutral
- AND no data from time t+1 or later is used in any indicator
- AND regime labels are validated against known events (GFC 2007–2009, COVID 2020, 2022 bear)

#### Scenario: Regime posterior computation

- GIVEN the four regime indicators at time t
- WHEN the posterior is computed
- THEN a probability distribution over 5 regimes is produced via softmax
- AND the softmax weights are fit on pre-sample data (pre-2005) or expanding window
- AND the posterior sums to 1.0

#### Scenario: Dynamic weight blending

- GIVEN model logits from XGBoost, Le-WM, and Fusion, plus regime posterior π_t
- WHEN the ensemble prediction is computed
- THEN L_final = Σ_r π_t(r) × Σ_m w_r_m × L_m
- AND regime-conditional weights w_r_m are fit on out-of-fold predictions
- AND weights satisfy: w ≥ 0, Σw ≤ 1 per regime, max 0.70 per model
- AND weights are updated monthly (not daily)
- AND log-loss on expanding window is the optimization target

#### Scenario: Regime ensemble validation

- GIVEN the regime-weighted ensemble and the v1 fixed 50/50 Fusion
- WHEN compared on the full walk-forward backtest
- THEN overall Sharpe, per-regime Sharpe, and maximum drawdown are reported for both
- AND the ensemble is expected to outperform during regime transitions

### Requirement: Cross-asset feature integration

The system SHALL incorporate free, publicly available cross-asset signals with appropriate lag to prevent look-ahead bias.

#### Scenario: Cross-asset feature download and engineering

- GIVEN daily data for SPY, VIX, DXY, ^TNX (10y), ^IRX (3m), HYG, LQD, IEF, TLT, GC=F, CL=F, HG=F
- WHEN features are computed
- THEN percentage changes, z-scores (21d, 63d), and spreads are calculated
- AND yield curve slope = 10y - 3m
- AND credit spreads = HYG-IEF, LQD-IEF (returns or ratios)
- AND VIX term structure = VIX3M - VIX (if VIX3M available)
- AND all cross-asset features are lagged by 1 trading day to simulate realistic data availability

#### Scenario: Overnight vs intraday return split

- GIVEN daily OHLCV data for each ticker
- WHEN overnight and intraday returns are computed
- THEN R_overnight = open_t / close_{t-1} - 1
- AND R_intraday = close_t / open_t - 1
- AND rolling volatility (21d) is computed for each component
- AND lagged overnight return is included as a predictor for next-day close-to-close

#### Scenario: Cross-sectional rank features

- GIVEN daily features for all 40 tickers
- WHEN cross-sectional ranks are computed
- THEN momentum ranks (5d, 21d, 63d), volatility rank, and beta-to-SPY rank are calculated
- AND ranks are normalized to [0, 1] (rank / N)
- AND residualized momentum (after removing market beta) is also ranked
- AND per-ticker rolling normalization is used to prevent leakage

### Requirement: Cost-realistic portfolio simulation

The system SHALL simulate portfolio execution with transaction costs, volatility targeting, and turnover control.

#### Scenario: Volatility targeting

- GIVEN current portfolio weights and per-ticker volatility estimates
- WHEN the vol targeting module runs
- THEN portfolio volatility is estimated (assuming uncorrelated positions — conservative)
- AND a scaling factor is applied to target 10% annualized volatility
- AND if 21-day realized portfolio vol exceeds the target, positions are scaled down

#### Scenario: Transaction cost modeling

- GIVEN a set of trades to execute
- WHEN costs are applied
- THEN each new trade incurs a configurable round-trip cost (default: 5 bps per side, 10 bps round-trip)
- AND held positions incur no additional cost
- AND sensitivity analysis is performed at 2, 5, and 10 bps per side
- AND all reported Sharpe ratios and returns are net of costs

#### Scenario: Turnover control

- GIVEN current and previous z-scores for each ticker
- WHEN the turnover filter runs
- THEN positions are only flipped when the z-score change exceeds 0.25
- AND monthly portfolio turnover is tracked
- AND target turnover is < 200% per month

### Requirement: HYG artifact investigation

The system SHALL investigate whether HYG's 0.91 AUC is a genuine predictive signal or a statistical artifact, using four independent triage tests.

#### Scenario: Time-shuffle test

- GIVEN HYG feature windows and labels
- WHEN labels are block-shuffled (30-day blocks, preserving temporal structure)
- THEN the full pipeline is re-run on shuffled data
- AND AUC is reported
- AND if AUC remains significantly above 0.50, the signal may be from temporal autocorrelation rather than genuine prediction

#### Scenario: Lag sanity audit

- GIVEN the feature computation pipeline for HYG
- WHEN a lag audit is performed
- THEN every feature at time t is verified to use only data available at or before time t
- AND regime labels, scaler fits, and feature windows are specifically checked
- AND any forward-looking data usage is documented and fixed

#### Scenario: Prior removal test

- GIVEN AWM's Beta distribution priors for HYG
- WHEN the Beta(1.5, 1) prior is replaced with Beta(1, 1) (uninformative)
- THEN the full pipeline is re-run for HYG
- AND AUC with and without the informative prior is compared
- AND if AUC drops significantly (e.g., from 0.91 to < 0.60), the prior was driving the result

#### Scenario: Purged walk-forward test

- GIVEN HYG data split into walk-forward windows
- WHEN a 5-trading-day embargo is applied between training and test sets
- THEN AUC with embargo is reported
- AND compared to AUC without embargo
- AND if AUC drops below 0.60, autocorrelation leakage was inflating the result

#### Scenario: HYG disposition decision

- GIVEN results from all four triage tests
- WHEN a decision is made
- THEN if all four tests show AUC > 0.60: HYG is reinstated as a tradeable asset
- AND if any test fails: HYG is excluded as a traded asset but HYG-derived spreads (HYG-IEF, HYG-SPY) are retained as cross-asset features for other tickers
- AND the decision and supporting evidence are documented in the results report
