/**
 * AWM Benchmark Types
 *
 * Defines the benchmark framework: scenarios, metrics, standards,
 * and report formats for reproducible AWM evaluation.
 */

// ─── Scenario Definition ──────────────────────────────────

/**
 * A benchmark scenario defines a simulated pipeline environment
 * with known characteristics. AWM must learn these characteristics
 * and demonstrate measurable improvement over baseline.
 */
export interface BenchScenario {
  /** Unique scenario identifier */
  id: string;
  /** Human-readable name */
  name: string;
  /** What this scenario tests */
  description: string;
  /** Difficulty: how hard is this for AWM to learn? */
  difficulty: 'easy' | 'medium' | 'hard' | 'adversarial';
  /** Scenario category */
  category: ScenarioCategory;
  /** Pipeline configuration */
  pipeline: PipelineConfig;
  /** Number of runs in warm-up phase (AWM learns, not scored) */
  warmupRuns: number;
  /** Number of runs in evaluation phase (scored) */
  evalRuns: number;
  /** Random seed for reproducibility */
  seed: number;
  /** Standards this scenario evaluates */
  standards: StandardId[];
}

export type ScenarioCategory =
  | 'prediction_accuracy'    // Can AWM predict outcomes correctly?
  | 'cost_optimization'      // Does AWM reduce cost without quality loss?
  | 'constraint_learning'    // Does AWM learn from revision patterns?
  | 'model_routing'          // Does Thompson Sampling find efficient models?
  | 'cold_start'             // How fast does AWM converge from zero data?
  | 'adversarial'            // Can AWM handle distribution shifts and noise?
  | 'multi_profile'          // Does AWM maintain separate beliefs per profile?
  | 'end_to_end';            // Full pipeline lifecycle evaluation

/**
 * Simulated pipeline configuration within a scenario.
 */
export interface PipelineConfig {
  /** Pipeline type identifier */
  workflowType: string;
  /** Steps in the pipeline */
  steps: StepConfig[];
  /** Client profiles to simulate */
  profiles: ProfileConfig[];
  /** Available models with cost/quality characteristics */
  models: ModelConfig[];
}

/**
 * A simulated pipeline step with known behavior characteristics.
 */
export interface StepConfig {
  /** Step type identifier */
  stepType: string;
  /** Base pass rate for this step (0-1) */
  basePassRate: number;
  /** Revision reasons with their probabilities */
  revisionReasons: { reason: string; probability: number }[];
  /** How much profile context affects pass rate */
  profileSensitivity: number;
  /** How much model choice affects pass rate */
  modelSensitivity: number;
  /** Base cost multiplier (1.0 = standard) */
  baseCost: number;
  /** Base latency in ms */
  baseLatency: number;
}

/**
 * A simulated client profile with behavioral modifiers.
 */
export interface ProfileConfig {
  /** Profile slug */
  slug: string;
  /** Sector */
  sector: string;
  /** Per-step pass rate modifiers (-0.5 to +0.5) */
  stepModifiers: Record<string, number>;
  /** Profile-specific revision reasons */
  extraRevisionReasons?: { stepType: string; reason: string; probability: number }[];
}

/**
 * A simulated model with cost/quality characteristics.
 */
export interface ModelConfig {
  /** Model name */
  name: string;
  /** Cost per step (dollars) */
  costPerStep: number;
  /** Latency in ms */
  latencyMs: number;
  /** Quality multiplier affecting pass rate (0.5 - 1.5) */
  qualityMultiplier: number;
}

// ─── Standards ────────────────────────────────────────────

/**
 * AWM Standards — the benchmarks AWM must meet.
 * Each standard has a threshold that defines "passing."
 */
export type StandardId =
  | 'S1_PREDICTION_ACCURACY'
  | 'S2_COST_REDUCTION'
  | 'S3_REVISION_REDUCTION'
  | 'S4_CONVERGENCE_SPEED'
  | 'S5_MODEL_ROUTING_EFFICIENCY'
  | 'S6_CONSTRAINT_EFFECTIVENESS'
  | 'S7_COLD_START_RECOVERY'
  | 'S8_PROFILE_ISOLATION'
  | 'S9_ADVERSARIAL_ROBUSTNESS'
  | 'S10_CALIBRATION';

export interface Standard {
  id: StandardId;
  name: string;
  description: string;
  /** Minimum threshold to pass */
  threshold: number;
  /** Unit of measurement */
  unit: string;
  /** Higher is better? */
  higherIsBetter: boolean;
}

export const STANDARDS: Record<StandardId, Standard> = {
  S1_PREDICTION_ACCURACY: {
    id: 'S1_PREDICTION_ACCURACY',
    name: 'Prediction Accuracy',
    description: 'Percentage of step outcomes correctly predicted (pass/revise/fail)',
    threshold: 0.65,
    unit: '%',
    higherIsBetter: true,
  },
  S2_COST_REDUCTION: {
    id: 'S2_COST_REDUCTION',
    name: 'Cost Reduction',
    description: 'Cost savings vs baseline (uniform model, no optimization)',
    threshold: 0.10,
    unit: '%',
    higherIsBetter: true,
  },
  S3_REVISION_REDUCTION: {
    id: 'S3_REVISION_REDUCTION',
    name: 'Revision Rate Reduction',
    description: 'Reduction in approval gate rejections via constraint pre-injection',
    threshold: 0.15,
    unit: '%',
    higherIsBetter: true,
  },
  S4_CONVERGENCE_SPEED: {
    id: 'S4_CONVERGENCE_SPEED',
    name: 'Convergence Speed',
    description: 'Number of runs until prediction accuracy exceeds 60%',
    threshold: 25,
    unit: 'runs',
    higherIsBetter: false,
  },
  S5_MODEL_ROUTING_EFFICIENCY: {
    id: 'S5_MODEL_ROUTING_EFFICIENCY',
    name: 'Model Routing Efficiency',
    description: 'Cost savings from model routing vs always using most expensive model',
    threshold: 0.20,
    unit: '%',
    higherIsBetter: true,
  },
  S6_CONSTRAINT_EFFECTIVENESS: {
    id: 'S6_CONSTRAINT_EFFECTIVENESS',
    name: 'Constraint Effectiveness',
    description: 'Pass rate improvement when AWM-injected constraints are applied',
    threshold: 0.10,
    unit: '%',
    higherIsBetter: true,
  },
  S7_COLD_START_RECOVERY: {
    id: 'S7_COLD_START_RECOVERY',
    name: 'Cold Start Recovery',
    description: 'Runs until AWM outperforms random baseline on a new profile',
    threshold: 15,
    unit: 'runs',
    higherIsBetter: false,
  },
  S8_PROFILE_ISOLATION: {
    id: 'S8_PROFILE_ISOLATION',
    name: 'Profile Isolation',
    description: 'Independence of predictions between different client profiles',
    threshold: 0.85,
    unit: 'score',
    higherIsBetter: true,
  },
  S9_ADVERSARIAL_ROBUSTNESS: {
    id: 'S9_ADVERSARIAL_ROBUSTNESS',
    name: 'Adversarial Robustness',
    description: 'Prediction accuracy maintained after distribution shift (sudden behavior change)',
    threshold: 0.50,
    unit: '%',
    higherIsBetter: true,
  },
  S10_CALIBRATION: {
    id: 'S10_CALIBRATION',
    name: 'Prediction Calibration',
    description: 'How well confidence scores correlate with actual accuracy (Brier score)',
    threshold: 0.25,
    unit: 'brier',
    higherIsBetter: false,
  },
};

// ─── Results ──────────────────────────────────────────────

/**
 * Result of evaluating a single standard.
 */
export interface StandardResult {
  standard: Standard;
  value: number;
  passed: boolean;
  detail: string;
}

/**
 * Result of running a single scenario.
 */
export interface ScenarioResult {
  scenario: BenchScenario;
  standards: StandardResult[];
  passed: boolean;
  /** Raw run data for analysis */
  warmupRuns: RunResult[];
  evalRuns: RunResult[];
  durationMs: number;
}

/**
 * Result of a single simulated pipeline run.
 */
export interface RunResult {
  runIndex: number;
  runId: string;
  profileSlug: string;
  steps: StepResult[];
  totalCost: number;
  totalLatencyMs: number;
  revisions: number;
}

/**
 * Result of a single simulated step.
 */
export interface StepResult {
  stepType: string;
  model: string;
  passed: boolean;
  revised: boolean;
  revisionReason?: string;
  cost: number;
  latencyMs: number;
  /** AWM's prediction (if active) */
  predictedOutcome?: 'pass' | 'revise' | 'fail';
  predictedConfidence?: number;
  predictionCorrect?: boolean;
  /** Did AWM inject constraints? */
  constraintsInjected: string[];
  /** Did constraints prevent a revision? */
  constraintPrevented: boolean;
}

/**
 * Full benchmark report across all scenarios.
 */
export interface BenchmarkReport {
  version: string;
  timestamp: string;
  awmVersion: string;
  totalScenarios: number;
  passed: number;
  failed: number;
  standardsSummary: Record<StandardId, { passed: number; failed: number; avgValue: number }>;
  scenarios: ScenarioResult[];
  overallGrade: 'A' | 'B' | 'C' | 'D' | 'F';
  durationMs: number;
}
