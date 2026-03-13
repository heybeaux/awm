/**
 * @heybeaux/awm-core
 *
 * Agent Workflow Model — Predictive execution protocol for agent pipelines.
 * Zero dependencies. Bring your own storage.
 */

export { Oracle } from './oracle.js';
export type { OracleConfig } from './oracle.js';
export { BeliefEngine } from './beliefs.js';
export { ModelRouter } from './bandits.js';
export { ConstraintExtractor } from './constraints.js';
export { InMemoryStore } from './store.js';

export type {
  // Trace schema
  StepTrace,
  ApprovalResult,

  // Predictions
  StepPrediction,
  PredictionContext,
  ACRContext,
  StepOutcomeSummary,

  // Beliefs
  StepBelief,

  // Bandits
  ModelArm,

  // Constraints
  ConstraintPattern,

  // Store
  AWMStore,
  TraceQuery,
} from './types.js';
