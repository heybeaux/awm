/**
 * AWM Oracle
 *
 * The main prediction interface. Combines beliefs, bandits,
 * and constraint patterns to produce step predictions and
 * execution recommendations.
 */

import type {
  AWMStore,
  StepPrediction,
  PredictionContext,
  StepTrace,
} from './types.js';
import { BeliefEngine } from './beliefs.js';
import { ModelRouter } from './bandits.js';
import { ConstraintExtractor } from './constraints.js';
import {
  requireField,
  requireNumber,
  requireRange,
} from './validation.js';

export interface OracleConfig {
  store: AWMStore;
  /** Weight for cost optimization in model routing (0-1). Default: 0.3 */
  costWeight?: number;
  /** Minimum confidence to recommend step skipping. Default: 0.85 */
  skipConfidenceThreshold?: number;
  /** Minimum confidence to use cheap model routing. Default: 0.7 */
  routingConfidenceThreshold?: number;
}

/**
 * Validate OracleConfig inputs at construction time.
 * Throws ConfigValidationError if any field is invalid.
 */
function validateOracleConfig(config: OracleConfig): void {
  requireField(config.store, 'store');

  if (config.costWeight !== undefined) {
    requireNumber(config.costWeight, 'costWeight');
    requireRange(config.costWeight, 'costWeight', -1, 1);
  }

  if (config.skipConfidenceThreshold !== undefined) {
    requireNumber(config.skipConfidenceThreshold, 'skipConfidenceThreshold');
    requireRange(config.skipConfidenceThreshold, 'skipConfidenceThreshold', 0, 1);
  }

  if (config.routingConfidenceThreshold !== undefined) {
    requireNumber(config.routingConfidenceThreshold, 'routingConfidenceThreshold');
    requireRange(config.routingConfidenceThreshold, 'routingConfidenceThreshold', 0, 1);
  }
}

export class Oracle {
  private beliefs: BeliefEngine;
  private router: ModelRouter;
  private constraints: ConstraintExtractor;
  private config: Required<OracleConfig>;

  constructor(config: OracleConfig) {
    validateOracleConfig(config);

    this.config = {
      costWeight: 0.3,
      skipConfidenceThreshold: 0.85,
      routingConfidenceThreshold: 0.7,
      ...config,
    };

    this.beliefs = new BeliefEngine(config.store);
    this.router = new ModelRouter(config.store);
    this.constraints = new ConstraintExtractor(config.store);
  }

  /**
   * Predict the likely outcome of a pipeline step.
   */
  async predict(context: PredictionContext): Promise<StepPrediction> {
    // 1. Get belief about this step type + profile
    const belief = await this.beliefs.getBelief(context.stepType, context.profileSlug);
    const successProb = this.beliefs.successProbability(belief);
    const dataConfidence = this.beliefs.confidence(belief);
    const outcome = this.beliefs.predictOutcome(belief);

    // 2. Get constraint recommendations
    const constraintList = await this.constraints.getConstraints(
      context.stepType,
      context.profileSlug,
    );

    // 3. Get model recommendation — criticality-aware, quality-first routing
    //
    // Three routing modes based on step criticality:
    //   critical:    NEVER downgrade. Always use best available model.
    //   standard:    Quality-first. Cost savings only when proven safe.
    //   exploratory: Cost-first. Safe to experiment with cheaper models.
    //
    // Within 'standard' mode, uses continuous evidence-based scaling
    // with a confidence gate: must have enough data AND high enough
    // success rate before cost optimization unlocks.
    const criticality = context.criticality || 'standard';
    const hasConstraints = constraintList.length > 0;

    let effectiveCostWeight: number;

    if (criticality === 'critical') {
      // Must-pass: always prefer quality, strong negative cost weight
      effectiveCostWeight = -1.0;
    } else if (criticality === 'exploratory') {
      // Safe to experiment: use configured cost weight
      effectiveCostWeight = this.config.costWeight;
    } else {
      // Standard: evidence-based scaling with confidence gate
      const dataStrength = Math.min(1, belief.observations / 20);
      const reliabilityScore = hasConstraints
        ? 0  // known revision problems = never cheap
        : successProb * dataStrength;

      // Confidence gate: need sufficient data AND high confidence
      const isConfident = dataConfidence > 0.6 && belief.observations >= 8;

      if (reliabilityScore > 0.7 && isConfident) {
        // Proven reliable — allow cost optimization proportional to evidence
        effectiveCostWeight = this.config.costWeight * (reliabilityScore - 0.5) * 2;
      } else {
        // Not proven — quality-first with proportional penalty
        effectiveCostWeight = -(0.15 + (1 - successProb) * 0.5);
      }
    }

    const modelRec = await this.router.selectModel(
      context.stepType,
      context.profileSlug,
      context.availableModels,
      effectiveCostWeight,
    );

    // 4. Determine skip recommendation
    const skipRecommendation = dataConfidence >= this.config.skipConfidenceThreshold
      && successProb > 0.95
      && constraintList.length === 0;

    // 5. Use posterior mean as confidence for calibration
    // When we predict 'pass', confidence = P(success)
    // When we predict 'revise' or 'fail', confidence = P(failure)
    const predictionConfidence = outcome === 'pass'
      ? successProb
      : (1 - successProb);

    // 6. Build reasoning
    const reasoning = this.buildReasoning(belief, successProb, dataConfidence, modelRec, constraintList, context);

    return {
      outcome,
      confidence: predictionConfidence,
      suggestedModel: modelRec.model,
      skipRecommendation,
      constraints: constraintList,
      reasoning,
      historicalBasis: belief.observations,
    };
  }

  /**
   * Record a step outcome. Updates beliefs, bandit arms,
   * and triggers constraint analysis if revised.
   */
  async record(trace: StepTrace): Promise<void> {
    // Store the trace
    await this.config.store.storeTrace(trace);

    // Update Bayesian belief
    await this.beliefs.update(trace.stepType, trace.profileSlug, trace.passed);

    // Update bandit arm
    await this.router.recordOutcome(
      trace.stepType,
      trace.profileSlug,
      trace.model,
      trace.passed,
      trace.cost,
      trace.latencyMs,
    );

    // If revised, analyze for constraint patterns
    if (trace.revised && trace.revisionReason) {
      await this.constraints.analyzeRevisions(trace.stepType, trace.profileSlug);
    }
  }

  /**
   * Force constraint pattern extraction across all known step/profile combinations.
   * Call after a warmup phase to ensure patterns are materialized.
   */
  async extractPatterns(stepTypes: string[], profileSlugs: string[]): Promise<void> {
    await this.constraints.analyzeAll(stepTypes, profileSlugs);
  }

  private buildReasoning(
    belief: { observations: number; alpha: number; beta: number },
    successProb: number,
    confidence: number,
    modelRec: { model: string; reasoning: string },
    constraints: string[],
    context: PredictionContext,
  ): string {
    const parts: string[] = [];

    if (belief.observations === 0) {
      parts.push('No historical data for this step/profile combination. Using uninformative prior.');
    } else {
      parts.push(
        `Based on ${belief.observations} prior runs: ${(successProb * 100).toFixed(0)}% success rate ` +
        `(confidence: ${(confidence * 100).toFixed(0)}%).`,
      );
    }

    parts.push(`Model: ${modelRec.reasoning}`);

    if (constraints.length > 0) {
      parts.push(
        `${constraints.length} constraint(s) pre-injected from historical revision patterns.`,
      );
    }

    if (context.acrContext) {
      const coverage = (context.acrContext.coverageRatio * 100).toFixed(0);
      parts.push(`ACR capability coverage: ${coverage}%.`);
      if (context.acrContext.missingCapabilities.length > 0) {
        parts.push(
          `Missing capabilities: ${context.acrContext.missingCapabilities.join(', ')}. ` +
          'This may increase revision risk.',
        );
      }
    }

    return parts.join(' ');
  }
}
