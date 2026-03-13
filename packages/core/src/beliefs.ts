/**
 * Bayesian Belief Engine
 *
 * Maintains Beta distributions per (stepType, profileSlug) pair.
 * Each observation (pass/fail) updates the distribution.
 * Predictions are the posterior mean with confidence from variance.
 */

import type { StepBelief, AWMStore } from './types.js';

/** Default prior: Beta(2, 2) — weakly informative, centered at 50% */
const DEFAULT_ALPHA = 2;
const DEFAULT_BETA = 2;

export class BeliefEngine {
  constructor(private store: AWMStore) {}

  /**
   * Get the current belief for a step type + profile.
   * Returns the global belief if no profile-specific belief exists.
   */
  async getBelief(stepType: string, profileSlug?: string): Promise<StepBelief> {
    // Try profile-specific first
    if (profileSlug) {
      const specific = await this.store.getBelief(stepType, profileSlug);
      if (specific) return specific;
    }

    // Fall back to global
    const global = await this.store.getBelief(stepType, '__global__');
    if (global) return global;

    // No data yet — return uninformative prior
    return {
      stepType,
      profileSlug: profileSlug || '__global__',
      alpha: DEFAULT_ALPHA,
      beta: DEFAULT_BETA,
      observations: 0,
      updatedAt: new Date().toISOString(),
    };
  }

  /**
   * Update belief after observing a step outcome.
   */
  async update(stepType: string, profileSlug: string | undefined, passed: boolean): Promise<StepBelief> {
    const slug = profileSlug || '__global__';
    const belief = await this.getBelief(stepType, slug);

    // Bayesian update: success → alpha++, failure → beta++
    if (passed) {
      belief.alpha += 1;
    } else {
      belief.beta += 1;
    }
    belief.observations += 1;
    belief.profileSlug = slug;
    belief.updatedAt = new Date().toISOString();

    await this.store.setBelief(belief);

    // Also update global belief
    if (slug !== '__global__') {
      await this.update(stepType, undefined, passed);
    }

    return belief;
  }

  /**
   * Get the predicted success probability (posterior mean).
   */
  successProbability(belief: StepBelief): number {
    return belief.alpha / (belief.alpha + belief.beta);
  }

  /**
   * Get the confidence in the prediction (inverse of variance).
   * Higher values = more data = more confident.
   */
  confidence(belief: StepBelief): number {
    const n = belief.alpha + belief.beta;
    // Variance of Beta distribution
    const variance = (belief.alpha * belief.beta) / (n * n * (n + 1));
    // Convert to 0-1 confidence (low variance = high confidence)
    // Max variance for Beta is 0.25 (at alpha=beta=1)
    return Math.max(0, Math.min(1, 1 - (variance / 0.25)));
  }

  /**
   * Predict outcome category based on belief.
   */
  predictOutcome(belief: StepBelief): 'pass' | 'revise' | 'fail' {
    const prob = this.successProbability(belief);
    if (prob > 0.7) return 'pass';
    if (prob > 0.4) return 'revise';
    return 'fail';
  }
}
