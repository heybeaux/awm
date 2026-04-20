# Expert Panel Consultation — AWM+Le-WM Equity Benchmark

**Date:** April 19, 2026
**Requested by:** Beaux Walton
**Facilitated by:** Rook

---

## Panel Composition

| Panel | Model | Status |
|-------|-------|--------|
| panel-opus | Claude Opus 4.6 | ✅ Completed (68s) |
| panel-gpt-v2 | GPT-v2 (OpenRouter) | ✅ Completed (47s) |
| panel-gpt5 | GPT-5 (OpenRouter) | ✅ Completed (43s) |
| panel-gemini-v2 | Gemini 3.1 Pro Preview | ❌ Timeout |
| panel-gemini31 | Gemini 3.1 Pro Preview | ❌ Timeout |

**3 of 5 panels delivered.** All three successful panels converged on the same core recommendations.

---

## Stress Test Results (Reference)

77 walk-forward windows × 63 trading days, 40 tickers, 2005–2026.

### Overall AUC

| Model | AUC |
|-------|-----|
| XGBoost | 0.5725 |
| Fusion (Le-WM + AWM) | 0.5583 |
| Le-WM standalone | 0.5343 |

### By Regime

| Regime | Windows | XGBoost | Fusion | Le-WM |
|--------|---------|---------|--------|-------|
| Normal/Bull | 53 | **0.586** | 0.562 | 0.537 |
| 2022 Bear | 5 | 0.517 | **0.598** | 0.580 |
| 2022 Recovery | 6 | **0.571** | 0.557 | 0.531 |
| GFC Crash | 5 | 0.527 | 0.516 | **0.530** |
| GFC Recovery | 5 | **0.548** | 0.537 | 0.506 |
| 2020 Recovery | 3 | **0.587** | 0.556 | 0.504 |

### Backtest Results (All Thresholds)

- XGBoost loses money at all thresholds (0.55, 0.58, 0.60) — deeply negative Sharpe
- Fusion and Le-WM produce zero trades at 0.55+ thresholds — probabilities cluster near 0.50
- This is correct calibration behavior, not a model failure

---

## Consensus Recommendations

### 1. Thresholds — Stop Fixed Probability Cutoffs (Unanimous)

The models are well-calibrated: probabilities cluster near 0.50 because the true signal is thin. Fixed cutoffs at 0.55+ starve trades. This is correct behavior, not a bug.

**Recommended approach:**
- **Top-k selection**: Trade the top 3–5 longs per day by score, regardless of absolute probability
- **Logit-z normalization**: Convert p → logit(p), z-score per ticker over a rolling window, threshold on z
- **Isotonic/Platt calibration**: Fit calibration on strictly out-of-fold predictions per walk-forward window
- **Regime-conditional thresholds**: Maintain separate thresholds per detected regime using past data only
- **Edge-based sizing**: Position = clip(λ × Edge / VolTarget, cap), where Edge = (2p-1), λ = 0.25–0.5 Kelly fraction
- **Optimize for PnL/Sharpe**, not hit rate — grid-search k or abs(z) with transaction costs on training slice

### 2. Regime-Weighted Ensemble — No Hard Switching (Unanimous)

Don't switch to one model per regime. Use dynamic weighted blending:

```
L_final = Σ_r π_t(r) × [w_r_Fusion × L_Fusion + w_r_XGB × L_XGB + w_r_LeWM × L_LeWM]
```

- **π_t(r)**: Regime posterior updated with data up to time t only (no look-ahead)
- **Regime indicators**: SPY 63/126-day SMA slope, realized vol percentile (EWMA/63d), 20–60d autocorrelation, drawdown state
- **Weight constraints**: w ≥ 0, sum ≤ 1 per regime, cap any model at 70%
- **Performance-based weighting**: w_t(model) ∝ exp(η × Sharpe_ew(lookback=60d))
- **Fallback**: In bear/crisis tilt toward Fusion/Le-WM; in bull/low-vol tilt toward XGB

### 3. HYG 0.91 AUC — Quarantine as Artifact (Unanimous)

0.91 AUC on daily next-day direction is not credible. Almost certainly driven by:
- Base-rate inflation from AWM Beta(1.5,1) prior aligning with HYG's persistent upward drift
- ETF pricing quirks (stale NAV, low volatility, serial correlation)
- Possible data leakage via regime labels

**Triage tests (all 4 must pass before reinstating):**
1. **Time-shuffle**: Permute labels within 20–40d blocks → AUC should drop to ~0.5
2. **Lag sanity**: Verify features at t contain no info from t+1 (esp. regime updates)
3. **Remove priors**: Replace Beta(1.5,1) with Beta(1,1) → if AUC collapses, prior was driving it
4. **Purged walk-forward**: 5-day embargo between train and test

**Action**: Exclude HYG as traded asset. Test whether HYG(t-1) spreads help predict other tickers.

### 4. Feature Engineering — Free Daily Signals (Unanimous)

Expected AUC improvement: +0.01–0.03.

**Cross-asset (lag 1 day for availability):**
- VIX, VIX3M-VIX (contango/backwardation)
- DXY, US 10y and 3m yields (FRED), 10y-3m spread
- Credit spreads: HYG-IEF, LQD-IEF ratios
- SPY, QQQ, IWM returns and vol
- Commodities: WTI, copper, gold

**Per-ticker OHLCV:**
- Overnight vs intraday split: R_overnight = open_t/close_{t-1}, R_intraday = close_t/open_t
- Multi-horizon returns: 1, 2, 5, 10, 21, 63d with z-scores
- Volume features: z-scores, OBV changes, turnover spikes
- Range: true range / ATR, close-to-range location, gap vs ATR
- Realized vol variants: EWMA, Parkinson, Garman-Klass

**Cross-sectional:**
- Daily ranks: momentum (5/21/63d), vol, beta-to-SPY
- Residualized returns after market beta

**Seasonality:**
- Day-of-week, month-end proximity (t-5..t+3)

### 5. MVP Paper Strategy (Unanimous)

Designed for TFSA: long-only, daily rebalance, no PDT concern.

| Parameter | Value |
|-----------|-------|
| Universe | 40 tickers (exclude HYG until cleared) |
| Signal | Regime-weighted ensemble logit, isotonic calibrated |
| Selection | Daily top-k longs (k=3–5) |
| Sizing | Equal weight or edge-weighted: w_i ∝ clip(0.25×logit(p_i), 0, cap) |
| Max per name | 10% NAV |
| Vol target | 10% annualized |
| Execution | Next open, 5 bps cost assumption |
| Hold period | 1-day with signal re-eval; 2-day if signal persists |
| Risk | Scale down if realized vol > target over 21d |
| Turnover | Require z change > 0.25 to flip; target < 200%/month |

**Edge bar:** IC ≈ 2×(AUC−0.5). AUC 0.56 → IC 0.12. Cross-sectional selection with 40 names makes this tradeable if Sharpe > 0.8 net of 2–4 bps round-trip.

### 6. Fusion Architecture — Prior-Residual Form (Unanimous)

Replace XGB fusion head with:

```
L_final = L_AWM + f(Le-WM embeddings, technical features)
```

Where L_AWM = logit(p_AWM) is the Bayesian prior, and f is a small model learning residuals.

**Options for f:**
- Logistic regression with L2 and per-ticker intercepts (simplest, best calibration)
- XGBoost with monotonic constraints (prevent inverting strong priors)
- Stacking: softmax meta-learner over [XGB, Le-WM, AWM] with regime features as interaction terms

**Calibration:**
- Temperature scaling per regime
- Post-calibrate final output with isotonic on expanding OOF
- Beta-binomial smoothing of realized hit-rate by regime

---

## Implementation Timeline (2 Weeks)

| Days | Phase | Key Deliverables |
|------|-------|-----------------|
| 1–2 | Calibration & Selection | Isotonic calibration, top-k pipeline, logit-z normalization, edge-based sizing |
| 3–4 | Regime Ensemble | Online regime detector, dynamic weight blending, stacking weights |
| 5–6 | Feature Engineering | Cross-asset features, overnight/intraday split, cross-sectional ranks |
| 7 | HYG Audit | 4 triage tests, include-as-feature vs exclude decision |
| 8–9 | Portfolio Layer | Vol targeting, beta-neutral hedge, turnover penalties, cost-realistic backtest |
| 10–11 | Regime Thresholds | Per-regime threshold optimization, PnL by regime/ticker plots |
| 12–14 | Paper MVP | Paper-trade harness, daily signal export, start live paper |

---

## Key Takeaways

1. **The zero-trade problem is a thresholding failure, not a model failure.** Well-calibrated models near 0.50 are working correctly.
2. **Fusion's bear-market edge is real and exploitable** with regime-conditional weighting.
3. **HYG's 0.91 AUC is fake** until proven otherwise — quarantine immediately.
4. **AUC 0.54–0.56 is tradeable** cross-sectionally with top-k selection and proper sizing.
5. **Replace the XGB fusion head** with a calibrated GLM or prior-residual form for better stability.
