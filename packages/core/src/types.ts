/**
 * AWM Core Types
 *
 * Trace schema, prediction interfaces, and belief types for the
 * Agent Workflow Model predictive execution protocol.
 */

// ─── Trace Schema ─────────────────────────────────────────────

/**
 * A recorded outcome from a single pipeline step execution.
 * This is the fundamental unit of AWM's learning data.
 */
export interface StepTrace {
  /** Unique trace identifier */
  traceId: string;
  /** Pipeline run this trace belongs to */
  runId: string;
  /** Type of workflow (e.g., 'creative-campaign', 'paid-media') */
  workflowType: string;
  /** Type of step (e.g., 'creative-director', 'red-team') */
  stepType: string;
  /** Position of this step in the pipeline */
  stepIndex: number;

  // ─── Context ───
  /** Client/profile identifier */
  profileSlug?: string;
  /** Client sector (e.g., 'nonprofit', 'tech-consulting') */
  sector?: string;
  /** Model used for this step */
  model: string;
  /** Hash of the input to detect similar runs */
  inputFingerprint: string;

  // ─── ACR Context (optional) ───
  /** Capabilities loaded for this step */
  acrCapabilities?: string[];
  /** Ratio of needed capabilities that were mounted */
  acrCoverageRatio?: number;
  /** LOD level used for primary capability */
  acrLodLevel?: 'index' | 'summary' | 'standard' | 'deep';
  /** Capabilities that were needed but not available */
  acrMissingCapabilities?: string[];

  // ─── Outcome ───
  /** Did the step complete successfully? */
  passed: boolean;
  /** Was the output revised after initial generation? */
  revised: boolean;
  /** Reason for revision (from approval gate feedback) */
  revisionReason?: string;
  /** Approval gate result */
  approvalResult?: ApprovalResult;
  /** Hash of the output for delta detection */
  outputFingerprint?: string;
  /** Cosine similarity to most similar prior output (0-1) */
  outputSimilarityToPrior?: number;

  // ─── Cost ───
  /** Input tokens consumed */
  tokensIn: number;
  /** Output tokens generated */
  tokensOut: number;
  /** Dollar cost of this step */
  cost: number;
  /** Wall-clock latency in milliseconds */
  latencyMs: number;

  // ─── AWM Metadata ───
  /** If AWM was active, what did it predict? */
  awmPrediction?: StepPrediction;
  /** Was AWM's prediction correct? */
  awmWasCorrect?: boolean;

  /** ISO 8601 timestamp */
  timestamp: string;
}

export type ApprovalResult = 'approved' | 'rejected' | 'revised' | 'auto-approved';

// ─── Prediction ───────────────────────────────────────────────

/**
 * AWM's prediction for a pipeline step before execution.
 */
export interface StepPrediction {
  /** Predicted outcome category */
  outcome: 'pass' | 'revise' | 'fail';
  /** Confidence in this prediction (0-1) */
  confidence: number;
  /** Recommended model for cost optimization */
  suggestedModel: string;
  /** Should this step be skipped (output expected unchanged)? */
  skipRecommendation: boolean;
  /** Constraints to pre-inject into the step prompt */
  constraints: string[];
  /** Human-readable explanation of the prediction */
  reasoning: string;
  /** Number of historical traces informing this prediction */
  historicalBasis: number;
  /** Cost estimates for different model choices */
  costEstimate?: Record<string, number>;
}

/**
 * Context provided to AWM for making a prediction.
 */
export interface PredictionContext {
  /** Type of step about to execute */
  stepType: string;
  /** Client/profile identifier */
  profileSlug?: string;
  /** Client sector */
  sector?: string;
  /** Models available for this step */
  availableModels: string[];
  /** Hash of the current input */
  inputFingerprint?: string;
  /** ACR capability context (if available) */
  acrContext?: ACRContext;
  /** Cost budget remaining for this run */
  costBudget?: number;
  /** Previous step outcomes in this run */
  priorSteps?: StepOutcomeSummary[];
  /** Step criticality classification */
  criticality?: StepCriticality;
}

/**
 * Step criticality classification.
 * Drives how aggressively AWM optimizes cost vs quality.
 */
export type StepCriticality =
  | 'critical'      // must-pass — never downgrade model (approval gates, compliance, red-team)
  | 'standard'      // default — quality-first, cost savings when proven safe
  | 'exploratory';  // safe to experiment — cheap retries acceptable (drafts, brainstorming)

/**
 * ACR capability context provided to AWM.
 */
export interface ACRContext {
  /** Capabilities loaded for this step */
  loadedCapabilities: string[];
  /** Ratio of needed capabilities that are mounted (0-1) */
  coverageRatio: number;
  /** LOD level for primary capability */
  lodLevel: 'index' | 'summary' | 'standard' | 'deep';
  /** Capabilities needed but not available */
  missingCapabilities: string[];
}

/**
 * Summary of a completed step's outcome (for feeding into predictions).
 */
export interface StepOutcomeSummary {
  stepType: string;
  passed: boolean;
  revised: boolean;
  model: string;
  cost: number;
}

// ─── Beliefs ──────────────────────────────────────────────────

/**
 * Bayesian belief state for a (step_type, profile) pair.
 * Uses Beta distribution: Beta(alpha, beta).
 */
export interface StepBelief {
  /** Step type this belief tracks */
  stepType: string;
  /** Profile this belief is scoped to (or '__global__') */
  profileSlug: string;
  /** Successes + prior (Beta distribution alpha) */
  alpha: number;
  /** Failures + prior (Beta distribution beta) */
  beta: number;
  /** Total observations */
  observations: number;
  /** Last updated timestamp */
  updatedAt: string;
}

// ─── Bandit Arms ──────────────────────────────────────────────

/**
 * A bandit arm representing a model choice for a step type.
 * Maintains reward distribution for Thompson Sampling.
 */
export interface ModelArm {
  /** Model identifier */
  model: string;
  /** Step type this arm is scoped to */
  stepType: string;
  /** Profile scope (or '__global__') */
  profileSlug: string;
  /** Reward successes (Beta alpha) */
  alpha: number;
  /** Reward failures (Beta beta) */
  beta: number;
  /** Total times this arm was pulled */
  pulls: number;
  /** Average cost when using this model */
  avgCost: number;
  /** Average latency when using this model */
  avgLatencyMs: number;
}

// ─── Constraint Patterns ──────────────────────────────────────

/**
 * A pattern extracted from approval gate revisions.
 * When matched, the constraint is pre-injected into the step prompt.
 */
export interface ConstraintPattern {
  /** Unique pattern identifier */
  patternId: string;
  /** Step type where this pattern applies */
  stepType: string;
  /** Profile scope (or '__global__') */
  profileSlug: string;
  /** The constraint text to inject */
  constraint: string;
  /** How many times this revision reason appeared */
  occurrences: number;
  /** Frequency relative to total runs (0-1) */
  frequency: number;
  /** Did injecting this constraint reduce revisions? */
  effectivenessScore?: number;
  /** Source revision reasons that generated this pattern */
  sourceReasons: string[];
}

// ─── Store Interface ──────────────────────────────────────────

/**
 * Storage backend interface. AWM Core uses this to persist
 * traces, beliefs, arms, and patterns. Implement this to
 * bring your own storage (Engram, SQLite, Postgres, etc.)
 */
export interface AWMStore {
  // Traces
  storeTrace(trace: StepTrace): Promise<void>;
  queryTraces(query: TraceQuery): Promise<StepTrace[]>;

  // Beliefs
  getBelief(stepType: string, profileSlug: string): Promise<StepBelief | null>;
  setBelief(belief: StepBelief): Promise<void>;

  // Bandit arms
  getArms(stepType: string, profileSlug: string): Promise<ModelArm[]>;
  setArm(arm: ModelArm): Promise<void>;

  // Constraint patterns
  getPatterns(stepType: string, profileSlug?: string): Promise<ConstraintPattern[]>;
  setPattern(pattern: ConstraintPattern): Promise<void>;
}

/**
 * Query parameters for retrieving historical traces.
 */
export interface TraceQuery {
  stepType?: string;
  profileSlug?: string;
  sector?: string;
  model?: string;
  passed?: boolean;
  revised?: boolean;
  /** Maximum number of traces to return */
  limit?: number;
  /** Return traces similar to this input fingerprint */
  similarTo?: string;
  /** Minimum trust score (if backed by Engram) */
  minTrust?: number;
}
