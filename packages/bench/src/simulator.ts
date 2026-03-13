/**
 * Pipeline Simulator
 *
 * Simulates pipeline execution with deterministic randomness.
 * Produces realistic outcomes based on step configs, profile
 * modifiers, model quality, and constraint effectiveness.
 */

import type {
  PipelineConfig,
  StepConfig,
  ProfileConfig,
  ModelConfig,
  RunResult,
  StepResult,
} from './types.js';
import { SeededRNG } from './rng.js';

export interface SimulatorConfig {
  pipeline: PipelineConfig;
  /** Seed for deterministic randomness */
  seed: number;
  /** Constraint effectiveness rate (0-1): how often do injected constraints prevent revisions */
  constraintEffectiveness?: number;
}

export class PipelineSimulator {
  private rng: SeededRNG;
  private pipeline: PipelineConfig;
  private constraintEffectiveness: number;
  private profileMap: Map<string, ProfileConfig>;
  private modelMap: Map<string, ModelConfig>;

  constructor(config: SimulatorConfig) {
    this.rng = new SeededRNG(config.seed);
    this.pipeline = config.pipeline;
    this.constraintEffectiveness = config.constraintEffectiveness ?? 0.6;
    this.profileMap = new Map(config.pipeline.profiles.map((p) => [p.slug, p]));
    this.modelMap = new Map(config.pipeline.models.map((m) => [m.name, m]));
  }

  /**
   * Simulate a single pipeline run.
   */
  simulateRun(params: {
    runIndex: number;
    profileSlug: string;
    /** Model overrides per step (from AWM routing) */
    modelOverrides?: Record<string, string>;
    /** Constraints injected per step (from AWM) */
    injectedConstraints?: Record<string, string[]>;
  }): RunResult {
    const profile = this.profileMap.get(params.profileSlug);
    const steps: StepResult[] = [];
    let totalCost = 0;
    let totalLatency = 0;
    let revisions = 0;

    for (const stepConfig of this.pipeline.steps) {
      const modelName = params.modelOverrides?.[stepConfig.stepType]
        || this.pipeline.models[0].name;
      const model = this.modelMap.get(modelName) || this.pipeline.models[0];
      const constraints = params.injectedConstraints?.[stepConfig.stepType] || [];

      const result = this.simulateStep(stepConfig, profile, model, constraints);
      steps.push(result);
      totalCost += result.cost;
      totalLatency += result.latencyMs;
      if (result.revised) revisions++;
    }

    return {
      runIndex: params.runIndex,
      runId: `bench-run-${params.runIndex}`,
      profileSlug: params.profileSlug,
      steps,
      totalCost,
      totalLatencyMs: totalLatency,
      revisions,
    };
  }

  /**
   * Simulate a single step execution.
   */
  private simulateStep(
    step: StepConfig,
    profile: ProfileConfig | undefined,
    model: ModelConfig,
    injectedConstraints: string[],
  ): StepResult {
    // Calculate effective pass rate
    let passRate = step.basePassRate;

    // Apply profile modifier
    if (profile) {
      const modifier = profile.stepModifiers[step.stepType] || 0;
      passRate += modifier * step.profileSensitivity;
    }

    // Apply model quality modifier
    passRate *= model.qualityMultiplier;

    // Apply constraint effectiveness
    let constraintPrevented = false;
    if (injectedConstraints.length > 0) {
      // Each relevant constraint reduces revision probability
      const revisionReduction = injectedConstraints.length * this.constraintEffectiveness * 0.15;
      const wouldHaveFailed = this.rng.next() > passRate;
      passRate = Math.min(1.0, passRate + revisionReduction);

      if (wouldHaveFailed && this.rng.next() < revisionReduction) {
        constraintPrevented = true;
      }
    }

    // Clamp pass rate
    passRate = Math.max(0.05, Math.min(0.98, passRate));

    // Determine outcome
    const passed = this.rng.next() < passRate;
    let revised = false;
    let revisionReason: string | undefined;

    if (!passed) {
      // Determine if this was a revision (vs hard failure)
      revised = this.rng.next() < 0.8;

      if (revised) {
        // Pick a revision reason
        revisionReason = this.pickRevisionReason(step, profile);
      }
    }

    // Calculate cost and latency with model characteristics
    const cost = step.baseCost * model.costPerStep;
    const latencyJitter = 1.0 + (this.rng.next() - 0.5) * 0.3;
    const latencyMs = step.baseLatency * (model.latencyMs / 3000) * latencyJitter;

    return {
      stepType: step.stepType,
      model: model.name,
      passed,
      revised,
      revisionReason,
      cost,
      latencyMs,
      constraintsInjected: injectedConstraints,
      constraintPrevented,
    };
  }

  /**
   * Pick a revision reason based on step and profile probabilities.
   */
  private pickRevisionReason(step: StepConfig, profile?: ProfileConfig): string {
    const reasons = [...step.revisionReasons];

    // Add profile-specific reasons
    if (profile?.extraRevisionReasons) {
      for (const r of profile.extraRevisionReasons) {
        if (r.stepType === step.stepType) {
          reasons.push({ reason: r.reason, probability: r.probability });
        }
      }
    }

    // Normalize probabilities
    const total = reasons.reduce((sum, r) => sum + r.probability, 0);
    let roll = this.rng.next() * total;

    for (const r of reasons) {
      roll -= r.probability;
      if (roll <= 0) return r.reason;
    }

    return reasons[reasons.length - 1]?.reason || 'Unknown revision reason';
  }

  /**
   * Get all available model names.
   */
  getModelNames(): string[] {
    return this.pipeline.models.map((m) => m.name);
  }

  /**
   * Get all profile slugs.
   */
  getProfileSlugs(): string[] {
    return this.pipeline.profiles.map((p) => p.slug);
  }

  /**
   * Reset RNG to initial seed for reproducibility.
   */
  reset(seed?: number): void {
    this.rng = new SeededRNG(seed ?? this.rng.getSeed());
  }
}
