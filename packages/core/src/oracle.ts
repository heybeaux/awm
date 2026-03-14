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

export interface OracleConfig {
  store: AWMStore;
  /** Weight for cost optimization in model routing (0-1). Default: 0.3 */
  costWeight?: number;
  /** Minimum confidence to recommend step skipping. Default: 0.85 */
  skipConfidenceThreshold?: number;
  /** Minimum confidence to use cheap model routing. Default: 0.7 */
  routingConfidenceThreshold?: number;
}

export class Oracle {
  private beliefs: BeliefEngine;
  private router: ModelRouter;
  private constraints: ConstraintExtractor;
  private config: Required<OracleConfig>;

  constructor(config: OracleConfig) {
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

    // 3. Get model recommendation — factor in revision risk
    // Revision cost accounting: a revision means re-running the step,
    // so the effective cost of a cheap-but-risky model is higher than it looks.
    // When constraints exist, this step has a KNOWN revision problem — favor quality.
    const hasRevisionRisk = constraintList.length > 0 || successProb < 0.65;
    const revisionPenalty = hasRevisionRisk
      ? (1 - successProb) // estimated re-run cost fraction
      : 0;

    // Negative cost weight = prioritize quality over cheapness
    const effectiveCostWeight = hasRevisionRisk
      ? -revisionPenalty  // negative flips preference to quality
      : this.config.costWeight;

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
