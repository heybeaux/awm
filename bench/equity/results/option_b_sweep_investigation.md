# Option B 74/74 Sweep Investigation

## Scope

I read `backtest_v2.py` and `ensemble.py` and inspected the saved sweep artifact at [backtest_v2.json](/Users/beauxwalton/awm/bench/results/backtest_v2.json).

The saved artifact reports:

- `window_count = 74`
- `winner_counts = {"option_b": 74}`

## 1. How Option A, Option B, and the regime ensemble are compared

In `backtest_v2.py`, the comparison happens inside `run_single_window()`:

1. Base models are trained in inner expanding windows and produce OOF base predictions via `generate_oof_predictions()` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:734)).
2. Those OOF base predictions are calibrated with isotonic calibrators fit on the full OOF set ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:503), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1030)).
3. Three head models are then fit on `oof_cal`:
   - Option A: offset logistic ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:530))
   - Option B: XGBoost head ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:606))
   - Regime ensemble: local regime-weight optimizer ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:682))
4. Those same three fitted heads are immediately scored on that same `oof_cal` frame via `attach_candidate_probabilities()` and `choose_candidate_head()` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1033), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1045), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:968)).
5. `choose_candidate_head()` picks the best `(head, k)` pair by running the full portfolio simulation and maximizing `penalized_sharpe()` on those same rows ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:971), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:978)).
6. Only after that selection step does the code refit the winner on the full outer-train slice and evaluate the chosen winner on the outer test window ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1056), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1103)).

Important detail: the regime ensemble being compared in `backtest_v2.py` is **not** `ensemble.py`'s `fit_regime_weighted_ensemble()` implementation. `backtest_v2.py` only imports `REGIME_ORDER` and `compute_fallback_weights` from `ensemble.py` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:34)) and uses its own local `_optimize_regime_weights()` / `fit_regime_ensemble_head()` path instead ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:645), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:682)).

That means the sweep is **not** testing the monthly expanding regime ensemble defined in [ensemble.py](/Users/beauxwalton/awm/bench/equity/ensemble.py:456).

## 2. Confirmed bug / unfair advantage

### Confirmed bug: in-sample head selection

The main red flag is real.

`run_single_window()` fits all three heads on `oof_cal`, then scores those same fitted heads on `oof_cal`, then chooses the winner from those in-sample scores ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1033), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:1045)).

That is a model-selection leak inside the training pipeline:

- It is **not** outer test leakage. I do **not** see the outer test window being used in the selection step.
- It **is** validation leakage for the head comparison, because the same rows are used to both fit and rank the heads.

Why this strongly favors Option B:

- Option B is the highest-capacity head here: an XGBoost classifier over `awm_logit` plus PCA/technical features ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:619), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:624)).
- Option A is a penalized offset logistic, which is lower-capacity ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:554), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:572)).
- The regime ensemble is even more constrained.

So the winner-selection loop is effectively asking: "Which head best fits the same OOF rows it was trained on?" That setup is exactly where the most flexible model tends to dominate.

### No evidence of a metric mismatch

I do **not** see a bug where different objectives are used for A/B/ensemble. All three are evaluated with the same portfolio simulation and the same `penalized_sharpe()` objective inside `choose_candidate_head()` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:971), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:978)).

### No evidence of direct outer-test leakage

I do **not** see Option B explicitly seeing the outer test labels before selection. The issue is the in-sample comparison on the OOF-calibration frame, not direct use of the outer test window.

## 3. Secondary issue: the regime ensemble is handicapped

Separately from the in-sample selection bug, the compared ensemble head looks structurally weak:

- The local optimizer in `backtest_v2.py` uses nonnegative weights with only a `sum(weights_row) <= 1.0` constraint, not equality ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:667), [backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:668)).
- There is no intercept term in the ensemble linear form; prediction is just `sigmoid(sum(dynamic_weights * logits))` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:726)).
- Portfolio selection then applies `floor=0.50` in `select_top_k()` ([backtest_v2.py](/Users/beauxwalton/awm/bench/equity/backtest_v2.py:816)).

That combination makes it easy for the optimizer to shrink the ensemble toward weak or near-zero logits, which then produces probabilities clustered near `0.5` and very few trades.

The saved results are consistent with that:

- `regime_ensemble` selected `k=3` in all 74 windows
- median `trade_count` for `regime_ensemble` is `22.0`
- median `trade_count` for `option_a` is `1294.0`
- median `trade_count` for `option_b` is `2253.5`

This does not prove a bug by itself, but it does mean the comparison is not "Option B versus a strong implementation of the intended ensemble." The tested ensemble is a simplified local version that appears heavily damped.

The same `<= 1` / no-intercept structure also exists in `ensemble.py`'s optimizer ([ensemble.py](/Users/beauxwalton/awm/bench/equity/ensemble.py:410), [ensemble.py](/Users/beauxwalton/awm/bench/equity/ensemble.py:411), [ensemble.py](/Users/beauxwalton/awm/bench/equity/ensemble.py:436)), but `backtest_v2.py` does not actually call `fit_regime_weighted_ensemble()`; it uses the local duplicate instead.

## 4. Actual stored scores for windows 1, 10, 30, 50, 70

These are the `candidate_objectives` values stored in [backtest_v2.json](/Users/beauxwalton/awm/bench/results/backtest_v2.json). These are the actual scores used to choose the winner in each window.

| Window | Option A | Option B | Regime Ensemble | Winner |
| --- | ---: | ---: | ---: | --- |
| 1 | 2.9383239893 | 6.5708602182 | 0.5105407268 | option_b |
| 10 | 1.0606658036 | 3.8200807623 | 1.0948923860 | option_b |
| 30 | 0.6312888621 | 2.5156364906 | 0.7266983353 | option_b |
| 50 | 0.7811180739 | 2.8348281521 | 0.4263973257 | option_b |
| 70 | 0.7790113654 | 2.2933961788 | 0.5238881342 | option_b |

Selected `k` by window:

| Window | Option A k | Option B k | Regime Ensemble k |
| --- | ---: | ---: | ---: |
| 1 | 10 | 3 | 3 |
| 10 | 10 | 10 | 3 |
| 30 | 3 | 3 | 3 |
| 50 | 7 | 5 | 3 |
| 70 | 5 | 3 | 3 |

## Bottom line

My conclusion:

- `Option B` does **not** appear to be directly using the outer test window.
- But the comparison logic **is flawed**: the head winner is chosen from in-sample scores on the same OOF rows used to fit the heads.
- That flaw strongly favors the highest-capacity head, which is `Option B`.
- The regime ensemble is also not the full `ensemble.py` implementation and appears structurally handicapped, which further tilts the comparison.

## Notes

I did not change code.

I also could not rerun the backtest locally in this shell because `xgboost` is not installed in the available `python3` environment; the score table above was extracted from the saved sweep artifact.
