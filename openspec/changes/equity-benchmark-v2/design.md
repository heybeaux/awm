# Design: Equity Benchmark v2

## Overview

This document covers the technical design for transforming the v1 benchmark into a paper-tradeable MVP. Four subsystems: calibration/selection, regime ensemble, feature engineering, and portfolio construction.

---

## 1. Calibration & Selection Pipeline

### Isotonic Calibration

Per walk-forward window, fit isotonic regression on out-of-fold predictions only:

```python
from sklearn.isotonic import IsotonicRegression

# For each walk-forward window:
# 1. Generate OOF predictions on training set (e.g., 5-fold temporal CV)
# 2. Fit isotonic on OOF preds vs actual labels
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(oof_probs, oof_labels)

# 3. Apply to test predictions
calibrated_probs = calibrator.predict(test_probs)
```

Fit separately for each model (XGB, Le-WM, Fusion). Store calibrators per window for reproducibility.

### Top-k Selection

Instead of threshold-based trade entry, rank all tickers daily and select top-k:

```python
def select_top_k(daily_scores: dict[str, float], k: int = 5) -> list[str]:
    """Select top-k tickers by calibrated probability."""
    ranked = sorted(daily_scores.items(), key=lambda x: x[1], reverse=True)
    return [ticker for ticker, score in ranked[:k]]
```

- Default k=5 (tunable on training data)
- Minimum score floor: skip tickers where calibrated p < 0.50 even in top-k
- Grid-search k ∈ {3, 5, 7, 10} on training slice, optimizing net Sharpe with costs

### Logit-z Normalization

Normalize scores across tickers to handle different calibration ranges:

```python
import numpy as np

def logit(p: float) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def logit_z_score(p: float, mu_rolling: float, sigma_rolling: float) -> float:
    """Z-score of logit-transformed probability."""
    z = logit(p)
    return (z - mu_rolling) / max(sigma_rolling, 1e-7)
```

- Rolling window: 252 trading days (1 year)
- Per-ticker normalization
- Threshold on z rather than raw p: trade when |z| > z_threshold
- z_threshold tuned on training data (expected range: 0.3–1.0)

### Edge-Based Position Sizing

```python
def position_size(p: float, vol_target: float, ticker_vol: float,
                  kelly_fraction: float = 0.25, max_weight: float = 0.10) -> float:
    """Kelly-fraction position sizing scaled by vol target."""
    edge = 2 * p - 1  # [-1, 1] range for calibrated probabilities
    raw_size = kelly_fraction * edge / max(ticker_vol, 0.01)
    scaled = raw_size * vol_target
    return np.clip(scaled, 0, max_weight)  # long-only, capped
```

- λ = 0.25 Kelly (conservative — full Kelly is too aggressive for thin edges)
- Vol target: 10% annualized
- Max per-name: 10% NAV
- Normalize across portfolio to sum to target gross exposure

---

## 2. Regime Detection & Ensemble

### Online Regime Detector

Backward-looking only. No future information. Updated daily.

```python
@dataclass
class RegimeState:
    trend: float      # SPY SMA slope (63d or 126d)
    vol_pct: float    # Realized vol percentile (63d EWMA vs 252d history)
    autocorr: float   # 20-60d return autocorrelation
    drawdown: float   # Current drawdown from rolling 63d high
    
    @property
    def regime(self) -> str:
        if self.drawdown < -0.10 and self.vol_pct > 0.80:
            return 'crisis'
        elif self.trend < -0.02 and self.vol_pct > 0.60:
            return 'bear'
        elif self.trend > 0.02 and self.vol_pct < 0.40:
            return 'bull'
        elif abs(self.autocorr) > 0.15:
            return 'mean_revert'
        else:
            return 'neutral'
```

**Indicators (all use SPY as market proxy):**
- `trend`: Slope of 63-day SMA of SPY (linear regression over 21d)
- `vol_pct`: Current 63d EWMA vol as percentile of trailing 252d vol distribution
- `autocorr`: Rolling 21d autocorrelation of SPY daily returns
- `drawdown`: (current_price / 63d_rolling_max) - 1

### Regime Posterior

Softmax over indicator vector:

```python
def regime_posterior(state: RegimeState) -> dict[str, float]:
    """Compute regime probabilities from indicators."""
    # Simple softmax over hand-tuned features
    # Weights fit on pre-sample (pre-2005) or expanding window
    logits = {
        'crisis': -2 * state.trend + 3 * state.vol_pct - 5 * (state.drawdown + 0.1),
        'bear': -3 * state.trend + 2 * state.vol_pct,
        'bull': 3 * state.trend - 2 * state.vol_pct,
        'mean_revert': 3 * abs(state.autocorr) - 1,
        'neutral': 0.0,  # baseline
    }
    # Softmax
    max_l = max(logits.values())
    exps = {k: np.exp(v - max_l) for k, v in logits.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}
```

### Dynamic Weight Blending

```python
def ensemble_predict(model_logits: dict[str, float],
                     regime_weights: dict[str, dict[str, float]],
                     regime_posterior: dict[str, float]) -> float:
    """
    L_final = Σ_r π(r) × Σ_m w_r_m × L_m
    
    model_logits: {'xgb': 0.12, 'lewm': -0.05, 'fusion': 0.08}
    regime_weights: {'crisis': {'xgb': 0.1, 'lewm': 0.4, 'fusion': 0.5}, ...}
    regime_posterior: {'crisis': 0.05, 'bear': 0.10, 'bull': 0.60, ...}
    """
    L_final = 0.0
    for regime, pi in regime_posterior.items():
        w = regime_weights[regime]
        regime_logit = sum(w[m] * model_logits[m] for m in model_logits)
        L_final += pi * regime_logit
    return L_final
```

**Weight fitting:**
- Train regime-conditional stacking weights on out-of-fold predictions
- Constraints: w ≥ 0, sum ≤ 1 per regime, max 0.70 per model
- Optimization: minimize log-loss on expanding window
- Update weights monthly (not daily — too noisy)

---

## 3. Prior-Residual Fusion Architecture

Replace the current 50/50 blend with a structured prior-residual form:

```
L_final = L_AWM + f(Le-WM embeddings, technical features)
```

Where:
- `L_AWM = logit(p_AWM)` — AWM's Bayesian regime belief as prior
- `f` — small model learning the residual (what AWM misses)

### Implementation Options

**Option A — Logistic Regression (recommended for v2):**
```python
from sklearn.linear_model import LogisticRegression

# Features: [logit(p_LeWM), embedding_pca[0:10], vol_z, momentum_z, regime_onehot]
# Target: residual = actual_label - sigmoid(L_AWM)
# But practically: train logistic on full feature set with L_AWM as offset

meta = LogisticRegression(C=1.0, max_iter=1000)
meta.fit(X_meta, y, sample_weight=None)
# Where X_meta columns include logit(p_LeWM), top-10 PCA of embeddings, regime indicators
```

**Option B — XGBoost with monotonic constraints (if Option A underfits):**
```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,  # shallow — prevent overfitting
    learning_rate=0.05,
    monotone_constraints=(1, 0, 0, ...),  # monotonic on L_AWM column
)
```

### Embedding Dimensionality Reduction

Raw 64-dim embeddings are too high-dimensional for the meta-learner at this sample size:
- PCA to 10 dimensions (preserving ~90% variance)
- Or use embedding-derived features: cosine similarity to regime centroids, distance to cluster centers

---

## 4. HYG Quarantine Protocol

### Test 1 — Time Shuffle
```python
# Permute labels within 20-40d blocks
shuffled_labels = block_shuffle(labels, block_size=30)
auc_shuffled = evaluate(model, features, shuffled_labels)
# Expected: AUC → ~0.50 if signal is real
# If AUC stays high: artifact from temporal structure, not true prediction
```

### Test 2 — Lag Sanity
```python
# Verify no feature at time t uses data from t+1
# Check: regime labels, scaler fit ranges, feature computation windows
# Specific: does regime classification use any forward returns?
```

### Test 3 — Remove AWM Priors
```python
# Replace Beta(1.5, 1) with Beta(1, 1) for HYG only
# Re-run full pipeline
# If AUC collapses from 0.91 → ~0.55: prior was driving the result
```

### Test 4 — Purged Walk-Forward
```python
# Embargo 5 trading days between train and test
# Prevents information leakage from autocorrelated features
purged_auc = walk_forward_evaluate(model, features, labels, embargo_days=5)
```

**Decision rule:** All 4 tests must show AUC > 0.60 to reinstate HYG as tradeable. Otherwise, demote to feature-only (HYG-IEF spread as cross-asset signal for other tickers).

---

## 5. Portfolio Construction

### Vol Targeting

```python
def vol_target_scale(portfolio_weights: dict[str, float],
                     ticker_vols: dict[str, float],
                     target_vol: float = 0.10) -> float:
    """Scale portfolio to target annualized volatility."""
    # Simplified: assume uncorrelated (conservative overestimate)
    portfolio_vol = np.sqrt(sum((w * v) ** 2 for w, v in 
                                zip(portfolio_weights.values(), ticker_vols.values())))
    portfolio_vol_ann = portfolio_vol * np.sqrt(252)
    return target_vol / max(portfolio_vol_ann, 0.01)
```

- Target: 10% annualized
- If realized vol exceeds target over 21d, scale positions down
- Use 63d EWMA vol per ticker

### Turnover Control

```python
def should_trade(current_z: float, prev_z: float, 
                 z_change_threshold: float = 0.25) -> bool:
    """Only flip positions when z-score changes meaningfully."""
    return abs(current_z - prev_z) > z_change_threshold
```

- Prevent excessive churn from small score fluctuations
- Target turnover < 200%/month
- Penalize turnover in Sharpe optimization

### Transaction Cost Model

```python
def apply_costs(gross_return: float, is_new_trade: bool,
                cost_bps: float = 5.0) -> float:
    """Apply round-trip transaction costs."""
    if is_new_trade:
        return gross_return - (cost_bps / 10000) * 2  # round-trip
    return gross_return
```

- Default: 5 bps per side (10 bps round-trip) — conservative for liquid US equities
- Sensitivity test at 2, 5, 10 bps
- Include in all Sharpe calculations

---

## 6. Walk-Forward Backtest Protocol

```
For each window w in walk-forward:
  1. Train calibrators on OOF predictions from window w training set
  2. Generate calibrated predictions for all models on test set
  3. Compute regime state from backward-looking indicators
  4. Blend model predictions using regime-weighted ensemble
  5. Select top-k tickers
  6. Size positions using edge-based Kelly sizing
  7. Apply vol targeting and turnover control
  8. Simulate execution with costs
  9. Record: Sharpe, return, drawdown, trades, hit rate
```

**Metrics per window:**
- Net Sharpe (after costs)
- Compound return
- Maximum drawdown
- Trade count
- Hit rate (% profitable trades)
- Average edge per trade

**Aggregate across all windows:**
- Mean/median Sharpe
- Worst-window drawdown
- Total compound return
- By-regime breakdown
