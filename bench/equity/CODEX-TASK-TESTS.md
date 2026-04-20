# Codex Task: Write Comprehensive Test Suite

## Context
You are writing tests for a quantitative trading backtesting system.
The code is in `~/awm/bench/equity/`. Key modules:
- `pipeline.py` — data download, feature engineering, splits
- `sizing.py` — Kelly-based position sizing with vol targeting
- `calibration.py` — isotonic calibration of model probabilities
- `features_v2.py` — cross-asset and microstructure features
- `regime_detector.py` — HMM regime detection
- `portfolio.py` — portfolio construction and PnL
- `backtest_v2.py` — walk-forward backtest orchestrator
- `selection.py` — ticker selection logic
- `ensemble.py` — model comparison (Option A vs B)

## Requirements

### Environment
- Python venv at `~/awm/bench/equity/.venv`
- Use pytest
- `OMP_NUM_THREADS=1` required (already set in modules)
- Install pytest if not present: `pip install pytest`

### Test Files to Create (in `tests/` directory)

#### `tests/test_sizing.py`
Test `sizing.py`:
1. `test_compute_edge_known_values` — prob=0.75 → edge=0.5, prob=0.5 → edge=0.0, prob=0.0 → edge=-1.0 (but clipped to 0 for long-only)
2. `test_compute_edge_boundary` — prob=1.0 → edge=1.0, prob=0.0 → edge=-1.0
3. `test_ewma_vol_positive` — vol should always be non-negative
4. `test_size_positions_zero_edge` — prob=0.5 for all → all weights should be 0 (no edge)
5. `test_size_positions_max_weight_cap` — even with prob=1.0, weight should not exceed max_weight
6. `test_size_positions_long_only` — with long_only=True, no negative weights
7. `test_size_positions_deterministic` — same inputs → same outputs

#### `tests/test_calibration.py`
Test `calibration.py`:
1. `test_identity_calibrator` — IdentityCalibrator returns clipped input
2. `test_isotonic_fit_monotone` — fitted calibrator output is monotonically non-decreasing
3. `test_isotonic_small_sample_fallback` — with <20 samples, should use IdentityCalibrator
4. `test_calibration_no_leakage` — calibrator fit on training data, applied to test data; verify calibrator doesn't see test labels
5. `test_brier_score_improves` — Brier score after calibration ≤ Brier before (on calibration set)

#### `tests/test_features.py`
Test `pipeline.py` feature engineering:
1. `test_no_lookahead_rolling` — for a known series, verify rolling features on day T don't use data from T+1
2. `test_feature_columns_complete` — all FEATURE_COLUMNS present in output
3. `test_nan_propagation` — NaN in price data propagates to features (not silently zeroed)
4. `test_volume_ratio_uses_past` — volume_ratio denominator is rolling mean of PAST data
5. `test_rsi_range` — RSI output is always 0-100
6. `test_bollinger_range` — bb_pctb can be outside 0-1 (confirms not clipped)
7. `test_daily_return_calculation` — verify adj_close[t]/adj_close[t-1] - 1

#### `tests/test_regime_detector.py`
Test `regime_detector.py`:
1. `test_hmm_fit_only_training` — create synthetic data, verify HMM.fit() is called only on training slice
2. `test_regime_labels_valid` — all labels are in {0,1,2,3,4} (5 states)
3. `test_pure_uptrend_labels` — synthetic pure uptrend should NOT be labeled "crisis"
4. `test_regime_deterministic_with_seed` — same input + seed → same regimes

#### `tests/test_portfolio.py`
Test `portfolio.py`:
1. `test_cost_applied_correctly` — with known weights and 5bps cost, verify PnL reduction matches expected
2. `test_max_drawdown_calculation` — known equity curve [1.0, 1.1, 0.9, 1.0] → max DD = (1.1-0.9)/1.1
3. `test_zero_weights_zero_pnl` — all weights=0 → PnL=0 every day
4. `test_turnover_calculation` — going from [0.5, 0.5] to [0.3, 0.7] → turnover = 0.4

#### `tests/test_adversarial.py`
Critical adversarial tests:
1. `test_random_predictions_near_zero_sharpe` — feed uniform random [0.4, 0.6] predictions through full pipeline → Sharpe should be near 0 (within ±0.3)
2. `test_shuffled_labels_auc_near_random` — shuffle y_true labels → AUC ≈ 0.5
3. `test_perfect_lookahead_detected` — if we inject tomorrow's return as a feature, AUC should spike to >0.9 (proves our detection works)

#### `tests/conftest.py`
Shared fixtures:
- `synthetic_prices` — 500 days of random walk for 5 tickers
- `synthetic_features` — 500 days × 5 tickers × 15 features (random but valid ranges)
- `synthetic_predictions` — calibrated probs in [0.3, 0.7]

### Style
- Each test should be self-contained (no external data dependencies)
- Use synthetic/mock data for all tests
- Clear docstrings explaining what each test validates
- Tests should run in <30 seconds total
- Use `np.random.default_rng(42)` for reproducibility

### Run Command
```bash
cd ~/awm/bench/equity && source .venv/bin/activate && python -m pytest tests/ -v
```
