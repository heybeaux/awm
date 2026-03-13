/**
 * AWM Metrics API
 *
 * Provides aggregated metrics for the Forge dashboard.
 * Shows prediction accuracy, cost savings, revision rates,
 * and per-step/per-profile breakdowns.
 */

import type { StepTrace, AWMStore, StepBelief, ModelArm, ConstraintPattern } from '../../core/src/types.js';

export interface AWMDashboardMetrics {
  /** Overall summary */
  summary: {
    totalTraces: number;
    totalRuns: number;
    overallPassRate: number;
    overallRevisionRate: number;
    predictionAccuracy: number | null;
    totalCost: number;
    avgCostPerRun: number;
    avgLatencyPerStep: number;
  };

  /** Per-step breakdown */
  steps: StepMetrics[];

  /** Per-profile breakdown */
  profiles: ProfileMetrics[];

  /** Model routing insights */
  models: ModelMetrics[];

  /** Active constraint patterns */
  constraints: ConstraintMetrics[];

  /** Trend over time (last 30 days) */
  trend: DailyMetrics[];
}

export interface StepMetrics {
  stepType: string;
  totalRuns: number;
  passRate: number;
  revisionRate: number;
  avgCost: number;
  avgLatencyMs: number;
  predictionAccuracy: number | null;
  topRevisionReasons: { reason: string; count: number }[];
}

export interface ProfileMetrics {
  profileSlug: string;
  totalRuns: number;
  passRate: number;
  revisionRate: number;
  avgCost: number;
  avgRunCost: number;
}

export interface ModelMetrics {
  model: string;
  totalUses: number;
  passRate: number;
  avgCost: number;
  avgLatencyMs: number;
  /** Steps where this model is the current Thompson Sampling leader */
  leadingSteps: string[];
}

export interface ConstraintMetrics {
  patternId: string;
  stepType: string;
  profileSlug: string;
  constraint: string;
  occurrences: number;
  frequency: number;
  effectivenessScore?: number;
}

export interface DailyMetrics {
  date: string;
  traces: number;
  passRate: number;
  revisionRate: number;
  totalCost: number;
  predictionAccuracy: number | null;
}

/**
 * Compute dashboard metrics from stored traces.
 */
export class MetricsEngine {
  constructor(private store: AWMStore) {}

  async computeMetrics(daysBack: number = 30): Promise<AWMDashboardMetrics> {
    // Fetch all traces within window
    const traces = await this.store.queryTraces({ limit: 10000 });
    const cutoff = Date.now() - daysBack * 86400000;
    const recentTraces = traces.filter((t) => new Date(t.timestamp).getTime() > cutoff);

    return {
      summary: this.computeSummary(recentTraces),
      steps: this.computeStepMetrics(recentTraces),
      profiles: this.computeProfileMetrics(recentTraces),
      models: await this.computeModelMetrics(recentTraces),
      constraints: await this.computeConstraintMetrics(recentTraces),
      trend: this.computeTrend(recentTraces),
    };
  }

  private computeSummary(traces: StepTrace[]): AWMDashboardMetrics['summary'] {
    if (traces.length === 0) {
      return {
        totalTraces: 0, totalRuns: 0, overallPassRate: 0,
        overallRevisionRate: 0, predictionAccuracy: null,
        totalCost: 0, avgCostPerRun: 0, avgLatencyPerStep: 0,
      };
    }

    const uniqueRuns = new Set(traces.map((t) => t.runId));
    const passed = traces.filter((t) => t.passed).length;
    const revised = traces.filter((t) => t.revised).length;
    const totalCost = traces.reduce((sum, t) => sum + t.cost, 0);
    const avgLatency = traces.reduce((sum, t) => sum + t.latencyMs, 0) / traces.length;

    const withPredictions = traces.filter((t) => t.awmWasCorrect !== undefined);
    const correctPredictions = withPredictions.filter((t) => t.awmWasCorrect).length;

    return {
      totalTraces: traces.length,
      totalRuns: uniqueRuns.size,
      overallPassRate: passed / traces.length,
      overallRevisionRate: revised / traces.length,
      predictionAccuracy: withPredictions.length > 0
        ? correctPredictions / withPredictions.length
        : null,
      totalCost,
      avgCostPerRun: totalCost / uniqueRuns.size,
      avgLatencyPerStep: avgLatency,
    };
  }

  private computeStepMetrics(traces: StepTrace[]): StepMetrics[] {
    const byStep = new Map<string, StepTrace[]>();
    for (const t of traces) {
      const arr = byStep.get(t.stepType) || [];
      arr.push(t);
      byStep.set(t.stepType, arr);
    }

    return Array.from(byStep.entries()).map(([stepType, stepTraces]) => {
      const passed = stepTraces.filter((t) => t.passed).length;
      const revised = stepTraces.filter((t) => t.revised).length;
      const totalCost = stepTraces.reduce((sum, t) => sum + t.cost, 0);
      const avgLatency = stepTraces.reduce((sum, t) => sum + t.latencyMs, 0) / stepTraces.length;

      const withPredictions = stepTraces.filter((t) => t.awmWasCorrect !== undefined);
      const correctPredictions = withPredictions.filter((t) => t.awmWasCorrect).length;

      // Top revision reasons
      const reasonCounts = new Map<string, number>();
      for (const t of stepTraces.filter((t) => t.revisionReason)) {
        const count = reasonCounts.get(t.revisionReason!) || 0;
        reasonCounts.set(t.revisionReason!, count + 1);
      }
      const topReasons = Array.from(reasonCounts.entries())
        .map(([reason, count]) => ({ reason, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 5);

      return {
        stepType,
        totalRuns: stepTraces.length,
        passRate: passed / stepTraces.length,
        revisionRate: revised / stepTraces.length,
        avgCost: totalCost / stepTraces.length,
        avgLatencyMs: avgLatency,
        predictionAccuracy: withPredictions.length > 0
          ? correctPredictions / withPredictions.length
          : null,
        topRevisionReasons: topReasons,
      };
    });
  }

  private computeProfileMetrics(traces: StepTrace[]): ProfileMetrics[] {
    const byProfile = new Map<string, StepTrace[]>();
    for (const t of traces) {
      const slug = t.profileSlug || '__global__';
      const arr = byProfile.get(slug) || [];
      arr.push(t);
      byProfile.set(slug, arr);
    }

    return Array.from(byProfile.entries()).map(([profileSlug, profileTraces]) => {
      const passed = profileTraces.filter((t) => t.passed).length;
      const revised = profileTraces.filter((t) => t.revised).length;
      const totalCost = profileTraces.reduce((sum, t) => sum + t.cost, 0);
      const uniqueRuns = new Set(profileTraces.map((t) => t.runId));

      return {
        profileSlug,
        totalRuns: profileTraces.length,
        passRate: passed / profileTraces.length,
        revisionRate: revised / profileTraces.length,
        avgCost: totalCost / profileTraces.length,
        avgRunCost: totalCost / uniqueRuns.size,
      };
    });
  }

  private async computeModelMetrics(traces: StepTrace[]): Promise<ModelMetrics[]> {
    const byModel = new Map<string, StepTrace[]>();
    for (const t of traces) {
      const arr = byModel.get(t.model) || [];
      arr.push(t);
      byModel.set(t.model, arr);
    }

    return Array.from(byModel.entries()).map(([model, modelTraces]) => {
      const passed = modelTraces.filter((t) => t.passed).length;
      const totalCost = modelTraces.reduce((sum, t) => sum + t.cost, 0);
      const avgLatency = modelTraces.reduce((sum, t) => sum + t.latencyMs, 0) / modelTraces.length;

      return {
        model,
        totalUses: modelTraces.length,
        passRate: passed / modelTraces.length,
        avgCost: totalCost / modelTraces.length,
        avgLatencyMs: avgLatency,
        leadingSteps: [], // populated when we have bandit data
      };
    });
  }

  private async computeConstraintMetrics(traces: StepTrace[]): Promise<ConstraintMetrics[]> {
    // Get all unique step types
    const stepTypes = new Set(traces.map((t) => t.stepType));
    const allPatterns: ConstraintMetrics[] = [];

    for (const stepType of stepTypes) {
      const patterns = await this.store.getPatterns(stepType);
      for (const p of patterns) {
        allPatterns.push({
          patternId: p.patternId,
          stepType: p.stepType,
          profileSlug: p.profileSlug,
          constraint: p.constraint,
          occurrences: p.occurrences,
          frequency: p.frequency,
          effectivenessScore: p.effectivenessScore,
        });
      }
    }

    return allPatterns;
  }

  private computeTrend(traces: StepTrace[]): DailyMetrics[] {
    const byDay = new Map<string, StepTrace[]>();

    for (const t of traces) {
      const date = t.timestamp.slice(0, 10); // YYYY-MM-DD
      const arr = byDay.get(date) || [];
      arr.push(t);
      byDay.set(date, arr);
    }

    return Array.from(byDay.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, dayTraces]) => {
        const passed = dayTraces.filter((t) => t.passed).length;
        const revised = dayTraces.filter((t) => t.revised).length;
        const totalCost = dayTraces.reduce((sum, t) => sum + t.cost, 0);

        const withPredictions = dayTraces.filter((t) => t.awmWasCorrect !== undefined);
        const correctPredictions = withPredictions.filter((t) => t.awmWasCorrect).length;

        return {
          date,
          traces: dayTraces.length,
          passRate: passed / dayTraces.length,
          revisionRate: revised / dayTraces.length,
          totalCost,
          predictionAccuracy: withPredictions.length > 0
            ? correctPredictions / withPredictions.length
            : null,
        };
      });
  }
}
