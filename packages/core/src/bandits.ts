/**
 * Thompson Sampling Model Router
 *
 * Multi-armed bandit for model selection. Each (model, stepType, profile)
 * combination is an arm with its own Beta reward distribution.
 *
 * At decision time, sample from each arm's distribution and pick the
 * highest sample — naturally balancing exploration and exploitation.
 */

import type { ModelArm, AWMStore } from './types.js';

/** Default prior: Beta(1, 1) — uniform (no preference) */
const DEFAULT_ALPHA = 1;
const DEFAULT_BETA = 1;

/**
 * Simple Beta distribution sampler using the Jöhnk algorithm.
 * No external dependencies.
 */
function sampleBeta(alpha: number, beta: number): number {
  // Special case: both 1 → uniform
  if (alpha === 1 && beta === 1) return Math.random();

  // Use gamma sampling: Beta(a,b) = Ga(a) / (Ga(a) + Ga(b))
  const gammaA = sampleGamma(alpha);
  const gammaB = sampleGamma(beta);
  return gammaA / (gammaA + gammaB);
}

/**
 * Gamma distribution sampler (Marsaglia and Tsang's method).
 */
function sampleGamma(shape: number): number {
  if (shape < 1) {
    return sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let x: number;
    let v: number;

    do {
      x = normalRandom();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.random();

    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/**
 * Standard normal random variable (Box-Muller transform).
 */
function normalRandom(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Exploration decay: after this many total pulls across all arms,
 *  start blending Thompson samples toward the posterior mean. */
const EXPLORATION_DECAY_START = 15;
const EXPLORATION_DECAY_RATE = 0.05; // per pull beyond start

export class ModelRouter {
  constructor(private store: AWMStore) {}

  /**
   * Select the best model for a step using Thompson Sampling
   * with exploration decay.
   *
   * Early on: pure Thompson Sampling (explore freely).
   * After EXPLORATION_DECAY_START pulls: gradually shift toward
   * exploitation (posterior mean) to stabilize routing.
   */
  async selectModel(
    stepType: string,
    profileSlug: string | undefined,
    availableModels: string[],
    costWeight: number = 0.3,
  ): Promise<{ model: string; sampledReward: number; reasoning: string }> {
    const slug = profileSlug || '__global__';
    const arms = await this.store.getArms(stepType, slug);
    const armMap = new Map(arms.map((a) => [a.model, a]));

    // Total pulls across all arms for this step/profile — drives decay
    const totalPulls = arms.reduce((sum, a) => sum + a.pulls, 0);

    // Decay factor: 1.0 (pure sampling) → 0.0 (pure exploitation)
    const explorationFactor = totalPulls <= EXPLORATION_DECAY_START
      ? 1.0
      : Math.max(0.05, 1.0 - (totalPulls - EXPLORATION_DECAY_START) * EXPLORATION_DECAY_RATE);

    let bestModel = availableModels[0];
    let bestReward = -Infinity;
    const samples: Record<string, number> = {};

    for (const model of availableModels) {
      const arm = armMap.get(model) || this.defaultArm(model, stepType, slug);

      // Posterior mean (exploitation value)
      const posteriorMean = arm.alpha / (arm.alpha + arm.beta);

      // Thompson sample (exploration value)
      const thompsonSample = sampleBeta(arm.alpha, arm.beta);

      // Blend: early = mostly sample, late = mostly mean
      const qualitySample = explorationFactor * thompsonSample
        + (1 - explorationFactor) * posteriorMean;

      // Adjust for cost
      const costPenalty = arm.avgCost > 0 ? costWeight * arm.avgCost : 0;
      const adjustedReward = qualitySample - costPenalty;

      samples[model] = adjustedReward;

      if (adjustedReward > bestReward) {
        bestReward = adjustedReward;
        bestModel = model;
      }
    }

    const arm = armMap.get(bestModel);
    const pulls = arm?.pulls || 0;

    return {
      model: bestModel,
      sampledReward: bestReward,
      reasoning:
        pulls > 0
          ? `Selected ${bestModel} (${pulls} pulls, ${((arm!.alpha / (arm!.alpha + arm!.beta)) * 100).toFixed(0)}% success, exploration: ${(explorationFactor * 100).toFixed(0)}%)`
          : `Exploring ${bestModel} (no prior data for this step/profile)`,
    };
  }

  /**
   * Update an arm after observing a step outcome.
   */
  async recordOutcome(
    stepType: string,
    profileSlug: string | undefined,
    model: string,
    passed: boolean,
    cost: number,
    latencyMs: number,
  ): Promise<void> {
    const slug = profileSlug || '__global__';
    const arms = await this.store.getArms(stepType, slug);
    const existing = arms.find((a) => a.model === model);

    const arm: ModelArm = existing || this.defaultArm(model, stepType, slug);

    // Update reward distribution
    if (passed) {
      arm.alpha += 1;
    } else {
      arm.beta += 1;
    }
    arm.pulls += 1;

    // Running average of cost and latency
    arm.avgCost = arm.avgCost + (cost - arm.avgCost) / arm.pulls;
    arm.avgLatencyMs = arm.avgLatencyMs + (latencyMs - arm.avgLatencyMs) / arm.pulls;

    await this.store.setArm(arm);
  }

  private defaultArm(model: string, stepType: string, profileSlug: string): ModelArm {
    return {
      model,
      stepType,
      profileSlug,
      alpha: DEFAULT_ALPHA,
      beta: DEFAULT_BETA,
      pulls: 0,
      avgCost: 0,
      avgLatencyMs: 0,
    };
  }
}
