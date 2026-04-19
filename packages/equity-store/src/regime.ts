/**
 * Market Regime Classifier
 *
 * Maps recent return and volatility statistics to one of four
 * coarse regimes. The label becomes AWM's `stepType`, so beliefs
 * accumulate per-regime per-ticker.
 */

export type Regime = 'trending_up' | 'trending_down' | 'mean_reverting' | 'quiet';

/**
 * Classify the current market regime for a window.
 *
 * - trending_up:    20-day return above +5%
 * - trending_down:  20-day return below -5%
 * - mean_reverting: not trending, with vol above the cross-sectional median
 * - quiet:          everything else
 *
 * Trend dominates volatility — a strongly trending market with high vol is
 * still "trending", not "mean_reverting".
 */
export function classifyRegime(
  ret20d: number,
  vol20d: number,
  medianVol: number,
): Regime {
  if (ret20d > 0.05) return 'trending_up';
  if (ret20d < -0.05) return 'trending_down';
  if (vol20d > medianVol) return 'mean_reverting';
  return 'quiet';
}
