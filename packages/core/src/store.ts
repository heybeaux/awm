/**
 * In-Memory Store
 *
 * Default AWMStore implementation for testing and standalone use.
 * Stores everything in memory — data is lost on restart.
 * For persistence, use @heybeaux/awm-engram or implement AWMStore.
 */

import type {
  AWMStore,
  StepTrace,
  StepBelief,
  ModelArm,
  ConstraintPattern,
  TraceQuery,
} from './types.js';

export class InMemoryStore implements AWMStore {
  private traces: StepTrace[] = [];
  private beliefs: Map<string, StepBelief> = new Map();
  private arms: Map<string, ModelArm[]> = new Map();
  private patterns: Map<string, ConstraintPattern> = new Map();

  // ─── Traces ───

  async storeTrace(trace: StepTrace): Promise<void> {
    this.traces.push(trace);
  }

  async queryTraces(query: TraceQuery): Promise<StepTrace[]> {
    let results = [...this.traces];

    if (query.stepType) {
      results = results.filter((t) => t.stepType === query.stepType);
    }
    if (query.profileSlug) {
      results = results.filter((t) => t.profileSlug === query.profileSlug);
    }
    if (query.sector) {
      results = results.filter((t) => t.sector === query.sector);
    }
    if (query.model) {
      results = results.filter((t) => t.model === query.model);
    }
    if (query.passed !== undefined) {
      results = results.filter((t) => t.passed === query.passed);
    }
    if (query.revised !== undefined) {
      results = results.filter((t) => t.revised === query.revised);
    }

    // Sort by recency (newest first)
    results.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    if (query.limit) {
      results = results.slice(0, query.limit);
    }

    return results;
  }

  // ─── Beliefs ───

  private beliefKey(stepType: string, profileSlug: string): string {
    return `${stepType}::${profileSlug}`;
  }

  async getBelief(stepType: string, profileSlug: string): Promise<StepBelief | null> {
    return this.beliefs.get(this.beliefKey(stepType, profileSlug)) || null;
  }

  async setBelief(belief: StepBelief): Promise<void> {
    this.beliefs.set(this.beliefKey(belief.stepType, belief.profileSlug), belief);
  }

  // ─── Bandit Arms ───

  private armKey(stepType: string, profileSlug: string): string {
    return `${stepType}::${profileSlug}`;
  }

  async getArms(stepType: string, profileSlug: string): Promise<ModelArm[]> {
    return this.arms.get(this.armKey(stepType, profileSlug)) || [];
  }

  async setArm(arm: ModelArm): Promise<void> {
    const key = this.armKey(arm.stepType, arm.profileSlug);
    const existing = this.arms.get(key) || [];
    const idx = existing.findIndex((a) => a.model === arm.model);
    if (idx >= 0) {
      existing[idx] = arm;
    } else {
      existing.push(arm);
    }
    this.arms.set(key, existing);
  }

  // ─── Constraint Patterns ───

  async getPatterns(stepType: string, profileSlug?: string): Promise<ConstraintPattern[]> {
    const all = Array.from(this.patterns.values());
    return all.filter(
      (p) =>
        p.stepType === stepType &&
        (profileSlug === undefined || p.profileSlug === profileSlug || p.profileSlug === '__global__'),
    );
  }

  async setPattern(pattern: ConstraintPattern): Promise<void> {
    this.patterns.set(pattern.patternId, pattern);
  }
}
