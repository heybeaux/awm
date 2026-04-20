# Tasks: Equity Benchmark v2

## Phase 1 — Calibration & Selection (Days 1–2)
_Depends on: v1 benchmark complete (stress test results available)_

- [ ] **1.1** Implement `calibration.py` — Isotonic calibration pipeline
  - Fit isotonic regression on out-of-fold predictions per walk-forward window
  - Separate calibrators per model (XGB, Le-WM, Fusion)
  - Validate: reliability diagrams before/after calibration
  - Store calibrators for reproducibility

- [ ] **1.2** Implement `selection.py` — Top-k trade selection
  - Rank all tickers daily by calibrated probability
  - Select top-k longs (configurable k, default 5)
  - Minimum score floor: skip if calibrated p < 0.50 even in top-k
  - Grid-search k ∈ {3, 5, 7, 10} on training slice, optimizing net Sharpe

- [ ] **1.3** Implement `selection.py` — Logit-z normalization
  - Convert p → logit(p) per ticker
  - Rolling z-score (252d window) per ticker
  - Alternative selection: threshold on |z| > z_threshold
  - Compare top-k vs z-threshold selection on training data

- [ ] **1.4** Implement `sizing.py` — Edge-based position sizing
  - Edge = 2p - 1 for calibrated probabilities
  - Position = clip(λ × Edge / VolTarget, max_weight)
  - λ = 0.25 Kelly fraction (configurable)
  - Max per-name: 10% NAV
  - Normalize portfolio to target gross exposure

- [ ] **1.5** Re-run bear market stress test with top-k selection
  - Replace fixed thresholds (0.55, 0.58, 0.60) with top-k (k=3, 5, 7)
  - Compare Sharpe, return, trade count across k values
  - Validate: no longer zero-trade windows for Fusion/Le-WM

## Phase 2 — Regime Ensemble (Days 3–4)
_Depends on: Phase 1 (calibrated predictions needed)_

- [ ] **2.1** Implement `regime_detector.py` — Online regime classification
  - Backward-looking indicators from SPY:
    - Trend: 63d SMA slope (linear regression over 21d)
    - Vol: 63d EWMA vol as percentile of 252d history
    - Autocorr: 21d rolling autocorrelation of daily returns
    - Drawdown: current vs 63d rolling high
  - 5-regime classification: crisis, bear, bull, mean_revert, neutral
  - Validate: regime labels align with known events (GFC, COVID, 2022)
  - No future information — verify with 5-day embargo test

- [ ] **2.2** Implement `ensemble.py` — Regime posterior
  - Softmax over indicator vector
  - Weights fit on pre-2005 data or expanding window
  - Output: probability distribution over 5 regimes per day

- [ ] **2.3** Implement `ensemble.py` — Dynamic weight blending
  - L_final = Σ_r π(r) × Σ_m w_r_m × L_m
  - Fit regime-conditional stacking weights on OOF predictions
  - Constraints: w ≥ 0, Σw ≤ 1 per regime, max 0.70 per model
  - Minimize log-loss on expanding window
  - Update monthly

- [ ] **2.4** Implement `ensemble.py` — Performance-based weight fallback
  - Exponential weighting: w(model) ∝ exp(η × rolling_sharpe(60d))
  - Fallback when stacking weights are unstable (< 100 OOF samples)

- [ ] **2.5** Benchmark: regime ensemble vs fixed 50/50 Fusion
  - Run full walk-forward with both approaches
  - Compare overall Sharpe, per-regime Sharpe, drawdown
  - Hypothesis: ensemble outperforms in regime transitions

## Phase 3 — Features & HYG Audit (Days 5–7)
_Depends on: Phase 1 (pipeline must accept new features)_

- [ ] **3.1** Implement `features_v2.py` — Cross-asset features
  - Download: SPY, VIX, DXY, ^TNX (10y), ^IRX (3m), HYG, LQD, IEF, TLT, GC=F, CL=F, HG=F
  - Compute: pct changes, z-scores (21d, 63d), spreads (10y-3m, HYG-IEF, LQD-IEF)
  - Lag all cross-asset features by 1 day (availability simulation)
  - Cache to parquet: `data/cross_asset_features.parquet`

- [ ] **3.2** Implement `features_v2.py` — Overnight/intraday split
  - R_overnight = open_t / close_{t-1} - 1
  - R_intraday = close_t / open_t - 1
  - Rolling vol of each component (21d)
  - Lagged O→C as predictor of next-day C→C

- [ ] **3.3** Implement `features_v2.py` — Volume features
  - Volume z-score (21d rolling)
  - Dollar volume (price × volume)
  - OBV (on-balance volume) 10d change
  - Volume spike indicator (> 2σ)

- [ ] **3.4** Implement `features_v2.py` — Cross-sectional ranks
  - Per-day rank across 40 tickers: momentum (5/21/63d), vol, beta-to-SPY
  - Residualized momentum: momentum after removing market beta
  - Output: rank / N (normalized 0–1)

- [ ] **3.5** Re-run XGBoost with enriched features
  - Add all v2 features to XGB training
  - Compare AUC before/after feature enrichment
  - Feature importance analysis: which new features drive lift

- [ ] **3.6** HYG Triage — Test 1: Time shuffle
  - Block-shuffle labels (30d blocks) for HYG
  - Re-run pipeline, measure AUC
  - Expected: AUC → ~0.50 if real signal

- [ ] **3.7** HYG Triage — Test 2: Lag sanity
  - Audit all features at time t for HYG
  - Verify no use of t+1 data in regime labels, scalers, or feature windows
  - Document findings

- [ ] **3.8** HYG Triage — Test 3: Remove priors
  - Replace Beta(1.5, 1) with Beta(1, 1) for HYG
  - Re-run AWM pipeline
  - If AUC drops from 0.91 → ~0.55: confirm prior-driven artifact

- [ ] **3.9** HYG Triage — Test 4: Purged walk-forward
  - 5-day embargo between train and test for HYG
  - Measure AUC with embargo
  - Decision: all 4 pass with AUC > 0.60 → reinstate; otherwise → feature only

## Phase 4 — Portfolio Layer & Paper MVP (Days 8–14)
_Depends on: Phases 1–3 complete_

- [ ] **4.1** Implement `portfolio.py` — Vol targeting
  - 63d EWMA vol per ticker
  - Portfolio vol (assume uncorrelated — conservative)
  - Scale factor to target 10% annualized
  - Reduce positions if 21d realized vol exceeds target

- [ ] **4.2** Implement `portfolio.py` — Turnover control
  - Require z-score change > 0.25 to flip a position
  - Track turnover per window
  - Penalize turnover in Sharpe optimization

- [ ] **4.3** Implement `portfolio.py` — Transaction cost model
  - Configurable: 2, 5, 10 bps per side
  - Round-trip applied on new trades
  - Hold trades exempt
  - Sensitivity analysis across cost levels

- [ ] **4.4** Implement `backtest_v2.py` — Full walk-forward with portfolio layer
  - For each window: calibrate → predict → ensemble → select → size → target → execute
  - Record: net Sharpe, compound return, max drawdown, trade count, hit rate, avg edge
  - Aggregate: mean/median Sharpe, worst-window DD, total return, by-regime breakdown

- [ ] **4.5** Implement `backtest_v2.py` — Prior-residual fusion head
  - Option A: Logistic regression with L_AWM offset + embedding PCA + technicals
  - Option B: XGBoost with monotonic constraints on L_AWM
  - Compare both vs regime-weighted ensemble
  - Select winner for paper MVP

- [ ] **4.6** Results analysis and reporting
  - Markdown report: v2 vs v1 comparison
  - Per-regime Sharpe improvement
  - Feature importance from enriched model
  - HYG investigation results
  - Go/no-go decision for paper trading (Sharpe > 0.8 net of costs)

- [ ] **4.7** Implement `paper_trade.py` — Daily signal pipeline
  - Daily cron: download latest data → compute features → calibrate → ensemble → select → size
  - Output: JSON signal file with {date, ticker, direction, confidence, position_size}
  - Log signals to Engram for audit trail
  - Telegram notification with daily picks

- [ ] **4.8** Paper trading: run 4–6 weeks
  - Minimum 20 trade signals for statistical significance
  - Weekly performance summary
  - Compare paper vs backtest expectations
  - Gate: paper Sharpe within 20% of backtest Sharpe to proceed to live
