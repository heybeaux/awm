# AWM + Le-WM Equity Benchmark v3 — Execution Spec
*Created: 2026-04-19 | Owner: Beaux + Rook*

## Overview

Phase v2 proved the system produces 0.73 AUC (embargoed) with public data.
Phase v3 objectives:
1. Prove the code is correct (test suite)
2. Remove noise features, add options/flow data
3. Re-run walk-forward with enriched features
4. Measure Sharpe delta — is options data the step-change we think?

---

## Task 1: Test Suite (CRITICAL PATH)

**Goal:** 100% confidence the backtest code does what it claims.

### 1A: Unit Tests — `tests/test_sizing.py`
```
- Kelly formula: known inputs → verify output matches hand calculation
- Cap enforcement: single-name 8%, sector 25%, crypto 2%
- Sign correctness: positive signal → long position (not short)
- Zero signal → zero position
- Edge cases: NaN handling, division by zero vol
```

### 1B: Unit Tests — `tests/test_calibration.py`
```
- Isotonic calibration fit on training → applied to test (no leakage)
- Known-answer: 10 synthetic predictions → verify calibrated output
- Empty input handling
- Monotonicity constraint preserved
```

### 1C: Unit Tests — `tests/test_features.py`
```
- No look-ahead: feature on day T uses only data ≤ T
- Rolling windows: verify window boundaries (off-by-one check)
- NaN propagation: missing data doesn't silently become 0
- Cross-asset alignment: verify dates match across tickers
```

### 1D: Unit Tests — `tests/test_regime_detector.py`
```
- HMM fit ONLY on training data (not full sample)
- Regime labels stable under small perturbations
- Known regime: feed pure uptrend → should label "bull"
- Verify no future data accessed during predict()
```

### 1E: Unit Tests — `tests/test_portfolio.py`
```
- Cost accounting: 5 bps applied correctly per trade
- Drawdown calculation: known equity curve → verify max DD
- Position sizing respects vol target (10%)
- Turnover calculation matches expected
```

### 1F: Integration Tests — `tests/test_backtest_integration.py`
```
- Known-answer: 1 window, 3 tickers, hand-computed expected PnL
- Null model: random predictions → Sharpe ≈ 0 after costs
- Label swap: permute Option A/B → 74/74 sweep disappears
- No look-ahead: future returns not accessible during feature computation
- Idempotency: same inputs → same outputs (deterministic with fixed seed)
```

### 1G: Adversarial Tests — `tests/test_adversarial.py`
```
- Replace Option B preds with random noise → must NOT win 74/74
- Shuffle all labels → AUC should drop to ~0.5
- Remove all features, predict majority class → Sharpe ≈ 0
- Inject 1-day look-ahead into 1 feature → AUC should spike (proves detection works)
```

---

## Task 2: Feature Cleanup

**Remove (all panels agree these are noise):**
- `rsi_14`
- `macd_hist`
- `bb_pctb`
- `day_of_week`
- `high_52w_pct` (redundant with cross-sectional ranks)

**Keep:** All V2 features (cross-asset, microstructure, ranks) — these are the foundation the options data builds on.

**Implementation:**
- Remove from `FEATURE_COLUMNS` in `pipeline.py`
- Remove from any feature engineering code
- Re-run baseline backtest to confirm Sharpe doesn't drop (proves they were noise)
- If Sharpe drops significantly on removal, put them back and investigate

---

## Task 3: FINRA TRF Off-Exchange Data Pipeline (FREE)

**What:** Daily off-exchange trading share per ticker — shows institutional dark pool activity.
**Why:** All 3 panels flagged this as the best free signal for next-day drift prediction.
**Where:** https://www.finra.org/finra-data/browse-catalog/short-interest/data (and TRF volume data)

### Implementation: `data/finra_trf.py`
```python
# Downloads daily TRF (Trade Reporting Facility) off-exchange volume
# Computes: off_exchange_pct = TRF_volume / total_volume per ticker
# Derives:
#   - off_exchange_pct_z21: 21-day z-score of off-exchange share
#   - off_exchange_pct_delta_5d: 5-day change in off-exchange share
#   - off_exchange_spike: boolean, >2 sigma above 63d mean
# Point-in-time: data published T+1 evening, use with 1-day lag
# Output: parquet file aligned to existing data dates
```

### New features to add:
- `finra_off_exchange_pct`
- `finra_off_exchange_z21`
- `finra_off_exchange_delta_5d`
- `finra_off_exchange_spike`

---

## Task 4: Options Data Pipeline (Polygon.io — $200/mo)

**Pending Beaux's subscription.** Spec now, implement after access confirmed.

### Implementation: `data/options_surface.py`
```python
# Pulls daily options chain snapshots from Polygon.io
# For each ticker, computes:
#   1. IV term structure: IV_30d, IV_60d, IV_90d, term_slope (IV60/IV30 - 1)
#   2. Skew: 25-delta put IV - 25-delta call IV (for 30d expiry)
#   3. Skew change: 1d and 5d delta of skew
#   4. Put/Call volume ratio (daily)
#   5. Unusual activity flag: volume > 3x 20-day average at specific strikes
#   6. GEX proxy: estimated dealer gamma exposure (simplified)
#
# Point-in-time: snapshot at 16:05 ET, no look-ahead
# Output: parquet file, one row per ticker per date
```

### New features:
- `opt_iv30`, `opt_iv60`, `opt_iv90`
- `opt_term_slope`, `opt_term_slope_z21`
- `opt_skew_25d`, `opt_skew_25d_chg_1d`, `opt_skew_25d_chg_5d`
- `opt_pcr_volume`, `opt_pcr_volume_z21`
- `opt_unusual_activity` (boolean)
- `opt_gex_proxy`, `opt_gex_sign`

---

## Task 5: SEC EDGAR Pipeline (FREE)

### Implementation: `data/edgar_signals.py`
```python
# Scrapes/API-pulls from SEC EDGAR:
#   1. Form 4 (insider transactions): clustered buys within 5 days
#   2. 8-K filings: buyback announcements, material agreements
#   3. 13D/G: activist stakes
#
# Filtering rules:
#   - Insider: ignore sales; focus on open-market buys
#   - Cluster score: n_buys * total_dollar_value * c_suite_weight (5d window)
#   - Buyback: flag new/expanded authorizations
#
# Point-in-time: use filing timestamp, not transaction date
# Lag: Form 4 up to T+2; use filing date as feature date
# Output: per-ticker daily signals
```

### New features:
- `edgar_insider_cluster_score`
- `edgar_buyback_flag`
- `edgar_activist_flag`
- `edgar_days_since_insider_buy`

---

## Task 6: Feature Integration & Re-run

Once Tasks 1-5 complete:
1. Merge new features into `features_v2.py` pipeline
2. Update `get_full_feature_columns()` to include new columns
3. Re-run full walk-forward backtest (74 windows)
4. Compare: Sharpe_v3 vs Sharpe_v2
5. SHAP analysis: which new features contribute most?
6. Ablation: run with ONLY new features — do they carry signal alone?

**Success criteria:**
- Embargoed AUC ≥ 0.76 (from 0.73)
- Mean Sharpe ≥ 0.9 (from 0.81)
- Median Sharpe ≥ 0.45 (from 0.33)
- No new look-ahead bias introduced

---

## Task 7: Le-WM Embedding Enrichment

The Le-WM currently compresses macro state into 64-dim embeddings.
Adding options surface data gives it access to FORWARD-LOOKING information.

**New embedding inputs:**
- Options IV term structure (per ticker)
- Skew dynamics
- GEX regime (positive/negative)
- Event clocks (days to earnings, blackout flags)

**Expected improvement:** Le-WM can now learn states like:
- "Complacent rally" (price up, but skew steepening = smart money hedging)
- "Gamma squeeze imminent" (high positive GEX near major strikes)
- "Pre-crisis fragility" (VVIX rising + negative term structure + credit widening)

---

## Execution Order

| Priority | Task | Agent | Est. Time | Depends On |
|----------|------|-------|-----------|------------|
| 🔴 1 | Test suite (Tasks 1A-1G) | Codex | 30-45 min | Nothing |
| 🔴 2 | Feature cleanup | Codex | 10 min | Nothing |
| 🟡 3 | FINRA TRF pipeline | Codex | 20 min | Nothing |
| 🟡 4 | SEC EDGAR pipeline | Codex | 30 min | Nothing |
| 🟡 5 | Options pipeline (spec) | Codex | 20 min | Polygon sub |
| 🟢 6 | Integration + re-run | Codex | 45 min | Tasks 1-5 |
| 🟢 7 | Le-WM enrichment | TBD | 60 min | Task 6 |

**Parallelizable:** Tasks 1, 2, 3, 4 can all run simultaneously.

---

## Data Budget

| Source | Cost | Signal Value | Status |
|--------|------|-------------|--------|
| FINRA TRF | Free | Medium-High | Build now |
| SEC EDGAR | Free | Medium | Build now |
| ETF flows (issuer sites) | Free | Medium | Build now |
| CBOE VIX/VIX1D | Free | Medium | Already have |
| Polygon.io | $200/mo | **Very High** | Pending sub |
| SpotGamma | $75-149/mo | High | Alternative to Polygon |
| ORTEX | $59-99/mo | Medium | Optional |
| Finnhub | $50/mo | Medium | Optional |

**Recommended initial spend: $200/mo (Polygon.io)**

---

## Success Metrics (v3 vs v2)

| Metric | v2 (current) | v3 target | Stretch |
|--------|-------------|-----------|---------|
| Embargoed AUC | 0.734 | 0.76+ | 0.80+ |
| Mean Sharpe | 0.81 | 0.95+ | 1.2+ |
| Median Sharpe | 0.33 | 0.50+ | 0.70+ |
| Crisis Sharpe | 1.30 | 1.50+ | 2.0+ |
| Bull Sharpe | 0.28 | 0.45+ | 0.60+ |
| Test coverage | 0% | 90%+ | 100% |

---

*"We don't need a massive edge. We need to PROVE what we have is real, then feed it better fuel."*
