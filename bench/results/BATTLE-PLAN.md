# AWM + Le-WM Equity Benchmark — Battle Plan
*Created: 2026-04-19 | Status: ACTIVE*

## Philosophy

We are the dumbest people in the room. We assume nothing works until proven otherwise. We prove it with tests, withheld data, and adversarial validation — not with hope. If this system survives the gauntlet, we back it with cash. If it doesn't, we kill it with zero regret.

---

## Phase 1: Test Coverage (Week 1-2)

The codebase has ZERO tests on 180KB of financial code. This is the #1 priority.

### Unit Tests (per module)

| Module | Priority | Key Invariants to Test |
|--------|----------|----------------------|
| `calibration.py` | 🔴 | Known-answer: hand-computed calibration for 3 tickers, 1 window |
| `selection.py` | 🔴 | Ticker selection logic; point-in-time gating; inception dates |
| `sizing.py` | 🔴 | Kelly formula correctness; cap enforcement; sign errors |
| `regime_detector.py` | 🔴 | HMM trained ONLY on training data; no future leakage |
| `ensemble.py` | 🟡 | Option A vs B comparison logic; metric computation |
| `features_v2.py` | 🔴 | No look-ahead in rolling features; point-in-time data only |
| `hyg_audit.py` | 🟡 | Embargo enforcement; label shuffle correctness |
| `portfolio.py` | 🔴 | Position sizing; drawdown scaler; cost accounting |
| `backtest_v2.py` | 🔴 | End-to-end known-answer on 1 window; return computation |
| `paper_trade.py` | 🟡 | Daily pipeline works with real recent data |

### Integration Tests
- [ ] Known-answer test: manually compute expected PnL for window 1, 3 tickers, verify backtest matches
- [ ] Null model test: feed random predictions → should produce ~0 Sharpe after costs
- [ ] Label swap test: swap Option A/B labels → results should change
- [ ] Cost accounting: verify costs are applied correctly (not double-counted, not missed)
- [ ] Alignment test: features on day T never use data from day T+1 or later

### The "Prove You're Not Stupid" Tests
- [ ] Replace Option B predictions with noise → does it still "win" 74/74? (If yes: COMPARISON BUG)
- [ ] Remove all features, use random predictions → Sharpe should be ≈ 0 after costs
- [ ] Verify HMM is re-fit per window (not using full sample)
- [ ] Check isotonic calibration is fit on training fold, not test

---

## Phase 2: Data Integrity (Week 2-3)

### Survivorship Bias Fix
- [ ] Audit ticker universe: which tickers existed when?
- [ ] COIN: zero-weight before April 2021
- [ ] Add point-in-time inception gates
- [ ] Consider adding delisted securities (Lehman, Bear Stearns era equivalents)

### Point-in-Time Enforcement
- [ ] Audit every feature for revised data usage
- [ ] Verify macro features use real-time vintages with publication lags
- [ ] Corporate actions audit: splits, dividends, spin-offs
- [ ] Cross-validate prices with second data vendor if available

### The Le-WM Advantage: Withholding Data
This is where Le-WM shines. Structure tests to WITHHOLD known data:
- [ ] Train on 2008-2018, test on 2019-2022 (includes COVID — does it see it coming?)
- [ ] Train on 2008-2020, EXCLUDE 2020 crisis from training → test on 2020 crisis alone
- [ ] Train on broad universe, test on tickers NOT in training set
- [ ] Withhold entire regimes from training: train without any "crisis" labels → can it still profit in crisis?

---

## Phase 3: Statistical Rigor (Week 3-4)

### Robustness Tests
- [ ] Parameter sensitivity: vary key XGBoost hyperparams ±20%, verify graceful degradation (not cliff)
- [ ] Seed sensitivity: run with 10 different random seeds, check Sharpe distribution
- [ ] Deflated Sharpe Ratio (accounts for multiple testing)
- [ ] Probability of Backtest Overfitting (PBO)
- [ ] White's Reality Check / Hansen SPA test

### Regime Stability
- [ ] HMM transition matrix consistency through time
- [ ] SHAP feature attribution stability by regime
- [ ] Regime labeling: small input perturbations shouldn't cause regime flips

### Cost Stress Tests
- [ ] Re-run at 10 bps (2x current)
- [ ] Re-run at 15 bps (3x current)
- [ ] Add 1-bar execution lag (signal on bar T, trade at T+1 open)
- [ ] Variable spread model by volatility/time-of-day

---

## Phase 4: Adversarial Validation (Week 4-5)

### Red Team the Results
- [ ] Distribution analysis: histogram of 74 window Sharpes, identify outlier drivers
- [ ] Investigate top 5 windows: what caused the wins? Single trade? Lucky timing? Reproducible?
- [ ] Bottom 5 windows: what went wrong? Predictable failure modes?
- [ ] Remove best 5 windows: what's the Sharpe without the outliers?
- [ ] Mean-revert regime: prove it's genuinely negative (not just costs eating zero-edge)

### Independent Verification
- [ ] Implement simplest possible version in separate codebase (zipline or vectorbt)
- [ ] Compare results for overlapping period
- [ ] If results diverge: investigate why (the bug is in one of them)

---

## Phase 5: Paper Trading (Week 5+, runs 16-24 weeks)

Only enter this phase if Phases 1-4 pass. If any critical finding survives, STOP.

### Setup
- Paper account with TFSA constraints (long-only, no margin, CAD base)
- Daily runbook: data freeze → signal → sizing → order → reconcile
- Log everything to Engram: regime state, confidence, picks, outcomes

### Monitor Weekly
- Realized Sharpe vs backtest Sharpe (target: within 20%)
- Slippage vs assumed (target: < 2x)
- Regime classification accuracy
- Vol vs 10% target (keep 8-12%)

### Kill Conditions
- Drawdown > 12%: hard kill
- Realized Sharpe < -0.5 for 2 consecutive months: hard kill
- Slippage > 2x assumption for 4+ weeks: hard kill
- Regime in "Crisis" but losing money for 30 days: hard kill

---

## Success Criteria

Before deploying real capital, ALL must be true:
- [ ] 100% of unit tests pass (no known bugs)
- [ ] 74/74 sweep explained OR debunked (comparison verified independently)
- [ ] Survivorship bias addressed (point-in-time universe)
- [ ] No look-ahead bias found in any test
- [ ] Deflated Sharpe still > 0.5
- [ ] Paper Sharpe within 20% of backtest after 16+ weeks
- [ ] At least one regime transition captured during paper period
- [ ] Cost stress test at 10 bps still shows positive edge
- [ ] Parameter sensitivity shows plateau, not cliff

---

## Anti-Patterns (DON'T)

- Don't assume the code is correct because it produces results
- Don't cite the crisis Sharpe to anyone (161 days = noise)
- Don't add complexity before proving simplicity works
- Don't go live without ALL success criteria met
- Don't override the system during drawdowns (if validated, trust it)
- Don't trade mean-revert regime (negative edge)
- Don't use more than 25-30% of TFSA for this system

---

*"I don't need a massive edge; these markets are inefficient. I just need to PROVE what we have is real."*
