/**
 * @heybeaux/awm-bench
 *
 * AWM Benchmark Suite — reproducible evaluation of
 * predictive pipeline orchestration.
 */

export { BenchmarkRunner } from './runner.js';
export type { RunnerConfig } from './runner.js';
export { PipelineSimulator } from './simulator.js';
export type { SimulatorConfig } from './simulator.js';
export { SCENARIOS, getScenario, getScenariosByCategory } from './scenarios/index.js';
export { formatReport, formatJSON } from './report.js';
export { SeededRNG } from './rng.js';

export { STANDARDS } from './types.js';
export type {
  BenchScenario,
  BenchmarkReport,
  ScenarioResult,
  RunResult,
  StepResult,
  StandardResult,
  StandardId,
  Standard,
} from './types.js';
