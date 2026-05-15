/**
 * @heybeaux/awm-mastra
 *
 * Mastra/Forge middleware for AWM. Provides beforeStep/afterStep hooks
 * that integrate AWM predictions into pipeline execution.
 *
 * Modes:
 * - 'shadow': Log predictions but don't modify execution (safe for testing)
 * - 'advisory': Include predictions in step context but let agent decide
 * - 'active': AWM modifies model routing and injects constraints directly
 */

import type { StepPrediction, StepTrace, ACRContext } from '../../core/src/types.js';
import type { Oracle } from '../../core/src/oracle.js';
import {
  ConfigValidationError,
  requireField,
  requireOneOf,
  requireNonEmptyArray,
  requireStringArray,
  requireFunction,
} from '../../core/src/validation.js';

export { wrapStep } from './step-wrapper.js';
export type { WrapStepConfig, StepDefinition, AWMStepData } from './step-wrapper.js';
export { MetricsEngine } from './metrics.js';
export type {
  AWMDashboardMetrics,
  StepMetrics,
  ProfileMetrics,
  ModelMetrics,
  ConstraintMetrics,
  DailyMetrics,
} from './metrics.js';

// ─── Configuration ───

export type AWMMode = 'shadow' | 'advisory' | 'active';

export interface AWMMiddlewareConfig {
  /** AWM Oracle instance */
  oracle: Oracle;
  /** Execution mode */
  mode: AWMMode;
  /** Callback when AWM makes a prediction */
  onPrediction?: (prediction: StepPrediction, stepType: string) => void;
  /** Callback when AWM records an outcome */
  onOutcome?: (trace: StepTrace) => void;
  /** Default models available for routing */
  defaultModels?: string[];
  /** Enable step skipping (only in active mode) */
  enableSkipping?: boolean;
  /** Custom fingerprint function for inputs */
  fingerprintFn?: (input: unknown) => string;
}

// ─── Step Context (what AWM adds to each step) ───

export interface AWMStepContext {
  /** AWM's prediction for this step */
  prediction: StepPrediction;
  /** Constraints to inject into the prompt (empty in shadow mode) */
  constraints: string[];
  /** Recommended model (original model in shadow mode) */
  model: string;
  /** Should this step be skipped? */
  skip: boolean;
  /** AWM mode for this run */
  mode: AWMMode;
}

// ─── Validation ───

const VALID_MODES: readonly AWMMode[] = ['shadow', 'advisory', 'active'];

/**
 * Validate AWMMiddlewareConfig inputs at construction time.
 * Throws ConfigValidationError if any field is invalid.
 */
function validateMiddlewareConfig(config: AWMMiddlewareConfig): void {
  requireField(config.oracle, 'oracle');
  requireField(config.mode, 'mode');
  requireOneOf(config.mode, 'mode', VALID_MODES);

  if (config.defaultModels !== undefined) {
    requireNonEmptyArray(config.defaultModels, 'defaultModels');
    requireStringArray(config.defaultModels, 'defaultModels');
  }

  if (config.onPrediction !== undefined) {
    requireFunction(config.onPrediction, 'onPrediction');
  }

  if (config.onOutcome !== undefined) {
    requireFunction(config.onOutcome, 'onOutcome');
  }

  if (config.fingerprintFn !== undefined) {
    requireFunction(config.fingerprintFn, 'fingerprintFn');
  }

  if (config.enableSkipping !== undefined && typeof config.enableSkipping !== 'boolean') {
    throw new ConfigValidationError(
      `"enableSkipping" must be a boolean, got ${typeof config.enableSkipping}.`,
    );
  }
}

// ─── Middleware ───

export class AWMMiddleware {
  private oracle: Oracle;
  private config: Required<AWMMiddlewareConfig>;
  private runTraces: Map<string, StepTrace[]> = new Map();

  constructor(config: AWMMiddlewareConfig) {
    validateMiddlewareConfig(config);

    this.oracle = config.oracle;
    this.config = {
      onPrediction: () => {},
      onOutcome: () => {},
      defaultModels: ['claude-sonnet-4', 'gemini-2.5-flash', 'claude-opus-4'],
      enableSkipping: false,
      fingerprintFn: defaultFingerprint,
      ...config,
    };
  }

  /**
   * Call before a pipeline step executes.
   * Returns AWM context that should be merged into the step's execution.
   */
  async beforeStep(params: {
    runId: string;
    workflowType: string;
    stepType: string;
    stepIndex: number;
    profileSlug?: string;
    sector?: string;
    currentModel: string;
    input: unknown;
    availableModels?: string[];
    acrContext?: ACRContext;
    costBudget?: number;
  }): Promise<AWMStepContext> {
    const availableModels = params.availableModels || this.config.defaultModels;

    const prediction = await this.oracle.predict({
      stepType: params.stepType,
      profileSlug: params.profileSlug,
      sector: params.sector,
      availableModels,
      inputFingerprint: this.config.fingerprintFn(params.input),
      acrContext: params.acrContext,
      costBudget: params.costBudget,
    });

    this.config.onPrediction(prediction, params.stepType);

    // Build context based on mode
    const context: AWMStepContext = {
      prediction,
      constraints: [],
      model: params.currentModel,
      skip: false,
      mode: this.config.mode,
    };

    if (this.config.mode === 'shadow') {
      // Shadow: log everything, change nothing
      return context;
    }

    if (this.config.mode === 'advisory' || this.config.mode === 'active') {
      // Advisory + Active: provide constraints
      context.constraints = prediction.constraints;
    }

    if (this.config.mode === 'active') {
      // Active: modify model routing
      context.model = prediction.suggestedModel;

      // Active: enable skipping if configured and confident
      if (this.config.enableSkipping && prediction.skipRecommendation) {
        context.skip = true;
      }
    }

    return context;
  }

  /**
   * Call after a pipeline step completes.
   * Records the outcome for future predictions.
   */
  async afterStep(params: {
    runId: string;
    workflowType: string;
    stepType: string;
    stepIndex: number;
    profileSlug?: string;
    sector?: string;
    model: string;
    input: unknown;
    passed: boolean;
    revised?: boolean;
    revisionReason?: string;
    approvalResult?: 'approved' | 'rejected' | 'revised' | 'auto-approved';
    tokensIn: number;
    tokensOut: number;
    cost: number;
    latencyMs: number;
    acrContext?: ACRContext;
    awmPrediction?: StepPrediction;
  }): Promise<void> {
    const trace: StepTrace = {
      traceId: `${params.runId}-${params.stepIndex}-${Date.now()}`,
      runId: params.runId,
      workflowType: params.workflowType,
      stepType: params.stepType,
      stepIndex: params.stepIndex,
      profileSlug: params.profileSlug,
      sector: params.sector,
      model: params.model,
      inputFingerprint: this.config.fingerprintFn(params.input),
      passed: params.passed,
      revised: params.revised || false,
      revisionReason: params.revisionReason,
      approvalResult: params.approvalResult,
      tokensIn: params.tokensIn,
      tokensOut: params.tokensOut,
      cost: params.cost,
      latencyMs: params.latencyMs,
      acrCapabilities: params.acrContext?.loadedCapabilities,
      acrCoverageRatio: params.acrContext?.coverageRatio,
      acrLodLevel: params.acrContext?.lodLevel,
      acrMissingCapabilities: params.acrContext?.missingCapabilities,
      awmPrediction: params.awmPrediction,
      awmWasCorrect: params.awmPrediction
        ? (params.awmPrediction.outcome === 'pass') === params.passed
        : undefined,
      timestamp: new Date().toISOString(),
    };

    await this.oracle.record(trace);
    this.config.onOutcome(trace);

    // Track traces per run for metrics
    const runTraces = this.runTraces.get(params.runId) || [];
    runTraces.push(trace);
    this.runTraces.set(params.runId, runTraces);
  }

  /**
   * Get metrics for a completed run.
   */
  getRunMetrics(runId: string): RunMetrics | null {
    const traces = this.runTraces.get(runId);
    if (!traces || traces.length === 0) return null;

    const totalCost = traces.reduce((sum, t) => sum + t.cost, 0);
    const totalLatency = traces.reduce((sum, t) => sum + t.latencyMs, 0);
    const revisions = traces.filter((t) => t.revised).length;
    const failures = traces.filter((t) => !t.passed).length;

    const predictionsCorrect = traces.filter((t) => t.awmWasCorrect === true).length;
    const predictionsTotal = traces.filter((t) => t.awmWasCorrect !== undefined).length;

    return {
      runId,
      steps: traces.length,
      totalCost,
      totalLatencyMs: totalLatency,
      revisions,
      failures,
      predictionAccuracy: predictionsTotal > 0 ? predictionsCorrect / predictionsTotal : undefined,
      traces,
    };
  }

  /**
   * Build a constraint injection string for a step's system prompt.
   * Formats AWM constraints as a clear section the LLM can follow.
   */
  static formatConstraints(constraints: string[]): string {
    if (constraints.length === 0) return '';

    return [
      '',
      '## ⚠️ AWM Constraints (from historical analysis)',
      'The following constraints are based on patterns from previous pipeline runs.',
      'Following these will reduce the likelihood of revision:',
      '',
      ...constraints.map((c, i) => `${i + 1}. ${c}`),
      '',
    ].join('\n');
  }
}

// ─── Types ───

export interface RunMetrics {
  runId: string;
  steps: number;
  totalCost: number;
  totalLatencyMs: number;
  revisions: number;
  failures: number;
  predictionAccuracy?: number;
  traces: StepTrace[];
}

// ─── Utilities ───

/**
 * Default input fingerprinting. Produces a simple hash
 * for detecting similar inputs across runs.
 */
function defaultFingerprint(input: unknown): string {
  const str = typeof input === 'string' ? input : JSON.stringify(input);
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return `fp_${Math.abs(hash).toString(36)}`;
}

// ─── Factory ───

/**
 * Create an AWM middleware instance.
 * Convenience function for quick setup.
 */
export function createAWMMiddleware(config: AWMMiddlewareConfig): AWMMiddleware {
  return new AWMMiddleware(config);
}
