/**
 * Benchmark Runner
 *
 * Executes scenarios against an AWM Oracle instance,
 * evaluates results against standards, and produces reports.
 */

import { Oracle, InMemoryStore } from '../../core/src/index.js';
import type { StepTrace, StepPrediction } from '../../core/src/types.js';
import { PipelineSimulator } from './simulator.js';
import type {
  BenchScenario,
  ScenarioResult,
  RunResult,
  StepResult,
  StandardResult,
  StandardId,
  BenchmarkReport,
  Standard,
} from './types.js';
import { STANDARDS } from './types.js';
import { SeededRNG } from './rng.js';

export interface RunnerConfig {
  /** Scenarios to run (default: all) */
  scenarios?: BenchScenario[];
  /** Print progress to stdout */
  verbose?: boolean;
}

export class BenchmarkRunner {
  private verbose: boolean;

  constructor(config?: RunnerConfig) {
    this.verbose = config?.verbose ?? false;
  }

  /**
   * Run all provided scenarios and produce a full report.
   */
  async runAll(scenarios: BenchScenario[]): Promise<BenchmarkReport> {
    const start = Date.now();
    const results: ScenarioResult[] = [];

    for (const scenario of scenarios) {
      if (this.verbose) {
        process.stdout.write(`\n  Running: ${scenario.name} [${scenario.difficulty}]...`);
      }
      const result = await this.runScenario(scenario);
      results.push(result);
      if (this.verbose) {
        const status = result.passed ? '✅' : '❌';
        process.stdout.write(` ${status} (${result.durationMs}ms)\n`);
      }
    }

    // Aggregate standards summary
    const standardsSummary: Record<string, { passed: number; failed: number; avgValue: number }> = {};
    for (const sid of Object.keys(STANDARDS)) {
      const matching = results.flatMap((r) => r.standards.filter((s) => s.standard.id === sid));
      if (matching.length > 0) {
        standardsSummary[sid] = {
          passed: matching.filter((m) => m.passed).length,
          failed: matching.filter((m) => !m.passed).length,
          avgValue: matching.reduce((sum, m) => sum + m.value, 0) / matching.length,
        };
      }
    }

    const totalPassed = results.filter((r) => r.passed).length;
    const totalFailed = results.filter((r) => !r.passed).length;

    return {
      version: '0.1.0',
      timestamp: new Date().toISOString(),
      awmVersion: '0.1.0',
      totalScenarios: results.length,
      passed: totalPassed,
      failed: totalFailed,
      standardsSummary: standardsSummary as BenchmarkReport['standardsSummary'],
      scenarios: results,
      overallGrade: this.computeGrade(totalPassed, results.length),
      durationMs: Date.now() - start,
    };
  }

  /**
   * Run a single scenario.
   */
  async runScenario(scenario: BenchScenario): Promise<ScenarioResult> {
    const start = Date.now();
    const store = new InMemoryStore();
    const oracle = new Oracle({ store });
    const simulator = new PipelineSimulator({
      pipeline: scenario.pipeline,
      seed: scenario.seed,
    });

    const rng = new SeededRNG(scenario.seed + 1);
    const profiles = simulator.getProfileSlugs();
    const models = simulator.getModelNames();

    // ─── Warmup Phase ───
    const warmupResults: RunResult[] = [];
    for (let i = 0; i < scenario.warmupRuns; i++) {
      const profileSlug = profiles[i % profiles.length];
      const result = await this.executeRun(oracle, simulator, {
        runIndex: i,
        profileSlug,
        models,
        awmActive: true,
        rng,
      });
      warmupResults.push(result);
    }

    // ─── Extract patterns after warmup ───
    // In production, patterns accumulate organically. In benchmarks,
    // we force extraction to simulate a system that's been running.
    const stepTypes = scenario.pipeline.steps.map((s) => s.stepType);
    await oracle.extractPatterns(stepTypes, profiles);

    // ─── Distribution shift for adversarial scenario ───
    let shiftedSimulator = simulator;
    if (scenario.id === 'distribution-shift') {
      // After warmup, the strategist pass rate drops dramatically
      const shiftedPipeline = JSON.parse(JSON.stringify(scenario.pipeline));
      shiftedPipeline.steps[0].basePassRate = 0.35; // was 0.90
      shiftedPipeline.steps[1].basePassRate = 0.45; // was 0.72
      shiftedSimulator = new PipelineSimulator({
        pipeline: shiftedPipeline,
        seed: scenario.seed + 10000,
      });
    }

    // ─── Evaluation Phase ───
    const evalResults: RunResult[] = [];
    for (let i = 0; i < scenario.evalRuns; i++) {
      const profileSlug = profiles[i % profiles.length];

      // For adversarial: switch simulator halfway through eval
      const activeSimulator = scenario.id === 'distribution-shift' && i >= scenario.evalRuns / 2
        ? shiftedSimulator
        : (scenario.id === 'distribution-shift' ? simulator : shiftedSimulator || simulator);

      const result = await this.executeRun(oracle, activeSimulator, {
        runIndex: scenario.warmupRuns + i,
        profileSlug,
        models,
        awmActive: true,
        rng,
      });
      evalResults.push(result);
    }

    // ─── Baseline (no AWM) for comparison ───
    const baselineSimulator = new PipelineSimulator({
      pipeline: scenario.pipeline,
      seed: scenario.seed, // same seed = same "reality"
    });
    const baselineResults: RunResult[] = [];
    for (let i = 0; i < scenario.evalRuns; i++) {
      const profileSlug = profiles[i % profiles.length];
      const result = baselineSimulator.simulateRun({
        runIndex: i,
        profileSlug,
        modelOverrides: Object.fromEntries(
          scenario.pipeline.steps.map((s) => [s.stepType, scenario.pipeline.models[0].name]),
        ),
      });
      baselineResults.push(result);
    }

    // ─── Evaluate Standards ───
    const standardResults = scenario.standards.map((sid) =>
      this.evaluateStandard(sid, evalResults, baselineResults, warmupResults, scenario),
    );

    return {
      scenario,
      standards: standardResults,
      passed: standardResults.every((s) => s.passed),
      warmupRuns: warmupResults,
      evalRuns: evalResults,
      durationMs: Date.now() - start,
    };
  }

  /**
   * Execute a single pipeline run with AWM integration.
   */
  private async executeRun(
    oracle: Oracle,
    simulator: PipelineSimulator,
    params: {
      runIndex: number;
      profileSlug: string;
      models: string[];
      awmActive: boolean;
      rng: SeededRNG;
    },
  ): Promise<RunResult> {
    const modelOverrides: Record<string, string> = {};
    const injectedConstraints: Record<string, string[]> = {};
    const predictions: Record<string, StepPrediction> = {};

    // Get AWM predictions for each step
    if (params.awmActive) {
      for (const step of simulator['pipeline'].steps) {
        const prediction = await oracle.predict({
          stepType: step.stepType,
          profileSlug: params.profileSlug,
          availableModels: params.models,
          criticality: step.criticality,
        });
        predictions[step.stepType] = prediction;
        modelOverrides[step.stepType] = prediction.suggestedModel;
        injectedConstraints[step.stepType] = prediction.constraints;
      }
    }

    // Simulate the run
    const result = simulator.simulateRun({
      runIndex: params.runIndex,
      profileSlug: params.profileSlug,
      modelOverrides,
      injectedConstraints,
    });

    // Attach prediction data to step results
    for (const step of result.steps) {
      const pred = predictions[step.stepType];
      if (pred) {
        step.predictedOutcome = pred.outcome;
        step.predictedConfidence = pred.confidence;
        step.predictionCorrect = (pred.outcome === 'pass') === step.passed;
      }
    }

    // Record outcomes back to AWM
    for (let i = 0; i < result.steps.length; i++) {
      const step = result.steps[i];
      const trace: StepTrace = {
        traceId: `bench-${params.runIndex}-${i}`,
        runId: result.runId,
        workflowType: simulator['pipeline'].workflowType,
        stepType: step.stepType,
        stepIndex: i,
        profileSlug: params.profileSlug,
        model: step.model,
        inputFingerprint: `input-${params.runIndex}-${i}`,
        passed: step.passed,
        revised: step.revised,
        revisionReason: step.revisionReason,
        tokensIn: 1500,
        tokensOut: 2000,
        cost: step.cost,
        latencyMs: step.latencyMs,
        timestamp: new Date().toISOString(),
        awmPrediction: predictions[step.stepType],
        awmWasCorrect: step.predictionCorrect,
      };
      await oracle.record(trace);
    }

    return result;
  }

  /**
   * Evaluate a single standard against results.
   */
  private evaluateStandard(
    standardId: StandardId,
    evalResults: RunResult[],
    baselineResults: RunResult[],
    warmupResults: RunResult[],
    scenario: BenchScenario,
  ): StandardResult {
    const standard = STANDARDS[standardId];

    switch (standardId) {
      case 'S1_PREDICTION_ACCURACY':
        return this.evalPredictionAccuracy(standard, evalResults);
      case 'S2_COST_REDUCTION':
        return this.evalCostReduction(standard, evalResults, baselineResults);
      case 'S3_REVISION_REDUCTION':
        return this.evalRevisionReduction(standard, evalResults, baselineResults);
      case 'S4_CONVERGENCE_SPEED':
        return this.evalConvergenceSpeed(standard, [...warmupResults, ...evalResults]);
      case 'S5_MODEL_ROUTING_EFFICIENCY':
        return this.evalModelRouting(standard, evalResults, baselineResults);
      case 'S6_CONSTRAINT_EFFECTIVENESS':
        return this.evalConstraintEffectiveness(standard, evalResults);
      case 'S7_COLD_START_RECOVERY':
        return this.evalColdStart(standard, evalResults);
      case 'S8_PROFILE_ISOLATION':
        return this.evalProfileIsolation(standard, evalResults);
      case 'S9_ADVERSARIAL_ROBUSTNESS':
        return this.evalAdversarialRobustness(standard, evalResults);
      case 'S10_CALIBRATION':
        return this.evalCalibration(standard, evalResults);
      default:
        return { standard, value: 0, passed: false, detail: 'Unknown standard' };
    }
  }

  // ─── Standard Evaluators ────────────────────────────────

  private evalPredictionAccuracy(standard: Standard, results: RunResult[]): StandardResult {
    const steps = results.flatMap((r) => r.steps);
    const withPredictions = steps.filter((s) => s.predictionCorrect !== undefined);
    if (withPredictions.length === 0) {
      return { standard, value: 0, passed: false, detail: 'No predictions recorded' };
    }
    const correct = withPredictions.filter((s) => s.predictionCorrect).length;
    const accuracy = correct / withPredictions.length;
    return {
      standard,
      value: accuracy,
      passed: accuracy >= standard.threshold,
      detail: `${(accuracy * 100).toFixed(1)}% accuracy (${correct}/${withPredictions.length} correct). Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalCostReduction(standard: Standard, awmResults: RunResult[], baseline: RunResult[]): StandardResult {
    const awmCost = awmResults.reduce((sum, r) => sum + r.totalCost, 0) / awmResults.length;
    const baseCost = baseline.reduce((sum, r) => sum + r.totalCost, 0) / baseline.length;
    if (baseCost === 0) return { standard, value: 0, passed: false, detail: 'Baseline cost is zero' };
    const reduction = (baseCost - awmCost) / baseCost;
    return {
      standard,
      value: reduction,
      passed: reduction >= standard.threshold,
      detail: `${(reduction * 100).toFixed(1)}% cost reduction ($${awmCost.toFixed(3)} vs $${baseCost.toFixed(3)} baseline). Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalRevisionReduction(standard: Standard, awmResults: RunResult[], baseline: RunResult[]): StandardResult {
    const awmRevisions = awmResults.reduce((sum, r) => sum + r.revisions, 0) / awmResults.length;
    const baseRevisions = baseline.reduce((sum, r) => sum + r.revisions, 0) / baseline.length;
    if (baseRevisions === 0) return { standard, value: 1.0, passed: true, detail: 'No baseline revisions' };
    const reduction = (baseRevisions - awmRevisions) / baseRevisions;
    return {
      standard,
      value: reduction,
      passed: reduction >= standard.threshold,
      detail: `${(reduction * 100).toFixed(1)}% revision reduction (${awmRevisions.toFixed(2)} vs ${baseRevisions.toFixed(2)} baseline). Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalConvergenceSpeed(standard: Standard, allResults: RunResult[]): StandardResult {
    // Find the first window of 10 consecutive runs with >60% prediction accuracy
    const windowSize = 10;
    for (let i = 0; i <= allResults.length - windowSize; i++) {
      const window = allResults.slice(i, i + windowSize);
      const steps = window.flatMap((r) => r.steps);
      const withPred = steps.filter((s) => s.predictionCorrect !== undefined);
      if (withPred.length === 0) continue;
      const accuracy = withPred.filter((s) => s.predictionCorrect).length / withPred.length;
      if (accuracy >= 0.6) {
        return {
          standard,
          value: i,
          passed: i <= standard.threshold,
          detail: `Converged at run ${i} (accuracy ${(accuracy * 100).toFixed(1)}%). Threshold: ≤${standard.threshold} runs`,
        };
      }
    }
    return {
      standard,
      value: allResults.length,
      passed: false,
      detail: `Did not converge within ${allResults.length} runs. Threshold: ≤${standard.threshold} runs`,
    };
  }

  private evalModelRouting(standard: Standard, awmResults: RunResult[], baseline: RunResult[]): StandardResult {
    // Compare AWM cost vs always-expensive baseline
    const awmCost = awmResults.reduce((sum, r) => sum + r.totalCost, 0);
    const expensiveCost = baseline.reduce((sum, r) => {
      // Recalculate with most expensive model
      return sum + r.steps.length * 0.45; // opus cost
    }, 0);
    if (expensiveCost === 0) return { standard, value: 0, passed: false, detail: 'Zero baseline' };
    const savings = (expensiveCost - awmCost) / expensiveCost;
    return {
      standard,
      value: savings,
      passed: savings >= standard.threshold,
      detail: `${(savings * 100).toFixed(1)}% savings vs always-expensive ($${awmCost.toFixed(2)} vs $${expensiveCost.toFixed(2)}). Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalConstraintEffectiveness(standard: Standard, results: RunResult[]): StandardResult {
    const steps = results.flatMap((r) => r.steps);

    const constrainedSteps = steps.filter((s) => s.constraintsInjected.length > 0);
    if (constrainedSteps.length === 0) {
      return { standard, value: 0, passed: false, detail: 'No constraints were injected during evaluation' };
    }

    // Constraint effectiveness = what fraction of would-be failures did constraints prevent?
    // A "would-be failure" is either a rescue (constraint prevented failure) or an actual failure.
    const rescues = constrainedSteps.filter((s) => s.constraintPrevented).length;
    const failures = constrainedSteps.filter((s) => !s.passed).length;
    const wouldBeFailures = rescues + failures;

    if (wouldBeFailures === 0) {
      return { standard, value: 1.0, passed: true, detail: 'All constrained steps passed (no failure opportunities)' };
    }

    // Rescue rate: of all would-be failures, how many did constraints save?
    const rescueRate = rescues / wouldBeFailures;

    return {
      standard,
      value: rescueRate,
      passed: rescueRate >= standard.threshold,
      detail: `${(rescueRate * 100).toFixed(1)}% of would-be failures rescued by constraints (${rescues}/${wouldBeFailures}). ${constrainedSteps.length} constrained steps total. Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalColdStart(standard: Standard, results: RunResult[]): StandardResult {
    // Random baseline accuracy: ~50% for binary pass/fail
    const randomBaseline = 0.5;
    const windowSize = 5;
    for (let i = 0; i <= results.length - windowSize; i++) {
      const window = results.slice(i, i + windowSize);
      const steps = window.flatMap((r) => r.steps);
      const withPred = steps.filter((s) => s.predictionCorrect !== undefined);
      if (withPred.length === 0) continue;
      const accuracy = withPred.filter((s) => s.predictionCorrect).length / withPred.length;
      if (accuracy > randomBaseline + 0.05) {
        return {
          standard,
          value: i,
          passed: i <= standard.threshold,
          detail: `Beat random baseline at run ${i} (${(accuracy * 100).toFixed(1)}% vs ${(randomBaseline * 100).toFixed(0)}% random). Threshold: ≤${standard.threshold} runs`,
        };
      }
    }
    return {
      standard,
      value: results.length,
      passed: false,
      detail: `Did not beat random baseline within ${results.length} runs`,
    };
  }

  private evalProfileIsolation(standard: Standard, results: RunResult[]): StandardResult {
    // Group results by profile
    const byProfile = new Map<string, RunResult[]>();
    for (const r of results) {
      const arr = byProfile.get(r.profileSlug) || [];
      arr.push(r);
      byProfile.set(r.profileSlug, arr);
    }

    if (byProfile.size < 2) {
      return { standard, value: 0, passed: false, detail: 'Need at least 2 profiles' };
    }

    // Measure per-profile prediction accuracy — if beliefs are isolated,
    // each profile should have good accuracy for its own behavior pattern
    const profileAccuracies: Record<string, number> = {};
    let totalAccuracy = 0;
    let profileCount = 0;

    for (const [slug, runs] of byProfile) {
      const steps = runs.flatMap((r) => r.steps);
      const withPred = steps.filter((s) => s.predictionCorrect !== undefined);
      if (withPred.length < 5) continue; // need sufficient data

      const accuracy = withPred.filter((s) => s.predictionCorrect).length / withPred.length;
      profileAccuracies[slug] = accuracy;
      totalAccuracy += accuracy;
      profileCount++;
    }

    if (profileCount < 2) {
      return { standard, value: 0, passed: false, detail: 'Insufficient prediction data per profile' };
    }

    // Isolation score: average per-profile accuracy (higher = beliefs are well-separated)
    // If beliefs bleed, profile with different behavior will have low accuracy
    const avgAccuracy = totalAccuracy / profileCount;

    // Also check variance: if all profiles have similar accuracy, beliefs are isolated
    const accuracies = Object.values(profileAccuracies);
    const variance = accuracies.reduce((sum, a) => sum + (a - avgAccuracy) ** 2, 0) / profileCount;
    const consistency = 1 - Math.min(1, variance * 4); // low variance = high consistency

    // Combined score: accuracy matters more, consistency is a bonus
    const score = avgAccuracy * 0.7 + consistency * 0.3;

    const profileDetail = Object.entries(profileAccuracies)
      .map(([slug, acc]) => `${slug}: ${(acc * 100).toFixed(0)}%`)
      .join(', ');

    return {
      standard,
      value: score,
      passed: score >= standard.threshold,
      detail: `Isolation score: ${score.toFixed(2)}. Per-profile accuracy: ${profileDetail}. Consistency: ${consistency.toFixed(2)}. Threshold: ${standard.threshold}`,
    };
  }

  private evalAdversarialRobustness(standard: Standard, results: RunResult[]): StandardResult {
    // Split results: first half (pre-shift) vs second half (post-shift)
    const midpoint = Math.floor(results.length / 2);
    const postShift = results.slice(midpoint);

    // Check prediction accuracy in the recovery window (last 25% of post-shift)
    const recoveryStart = Math.floor(postShift.length * 0.75);
    const recovery = postShift.slice(recoveryStart);
    const steps = recovery.flatMap((r) => r.steps);
    const withPred = steps.filter((s) => s.predictionCorrect !== undefined);
    if (withPred.length === 0) {
      return { standard, value: 0, passed: false, detail: 'No predictions in recovery window' };
    }
    const accuracy = withPred.filter((s) => s.predictionCorrect).length / withPred.length;
    return {
      standard,
      value: accuracy,
      passed: accuracy >= standard.threshold,
      detail: `Recovery accuracy: ${(accuracy * 100).toFixed(1)}% (last 25% of post-shift runs). Threshold: ${(standard.threshold * 100).toFixed(0)}%`,
    };
  }

  private evalCalibration(standard: Standard, results: RunResult[]): StandardResult {
    const steps = results.flatMap((r) => r.steps);
    const withConf = steps.filter((s) => s.predictedConfidence !== undefined && s.predictionCorrect !== undefined);
    if (withConf.length === 0) {
      return { standard, value: 1.0, passed: false, detail: 'No confidence data' };
    }

    // Brier score: mean squared error between confidence and actual outcome
    let brierSum = 0;
    for (const step of withConf) {
      const predicted = step.predictedConfidence!;
      const actual = step.predictionCorrect ? 1.0 : 0.0;
      brierSum += (predicted - actual) ** 2;
    }
    const brierScore = brierSum / withConf.length;

    return {
      standard,
      value: brierScore,
      passed: brierScore <= standard.threshold,
      detail: `Brier score: ${brierScore.toFixed(4)} (lower = better calibrated). Threshold: ≤${standard.threshold}`,
    };
  }

  // ─── Grading ────────────────────────────────────────────

  private computeGrade(passed: number, total: number): 'A' | 'B' | 'C' | 'D' | 'F' {
    const ratio = passed / total;
    if (ratio >= 0.90) return 'A';
    if (ratio >= 0.75) return 'B';
    if (ratio >= 0.60) return 'C';
    if (ratio >= 0.40) return 'D';
    return 'F';
  }
}
