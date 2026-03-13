import { describe, it, expect, beforeEach } from 'vitest';
import { InMemoryStore } from '../store.js';
import type { StepTrace, StepBelief, ModelArm, ConstraintPattern } from '../types.js';

describe('InMemoryStore', () => {
  let store: InMemoryStore;

  beforeEach(() => {
    store = new InMemoryStore();
  });

  // ─── Traces ───

  describe('traces', () => {
    const trace: StepTrace = {
      traceId: 'trace-1',
      runId: 'run-1',
      workflowType: 'creative-campaign',
      stepType: 'creative-director',
      stepIndex: 1,
      profileSlug: 'acme',
      sector: 'nonprofit',
      model: 'claude-sonnet-4',
      inputFingerprint: 'abc123',
      passed: true,
      revised: false,
      tokensIn: 1500,
      tokensOut: 2000,
      cost: 0.12,
      latencyMs: 3500,
      timestamp: '2026-03-13T12:00:00Z',
    };

    it('stores and retrieves traces', async () => {
      await store.storeTrace(trace);
      const results = await store.queryTraces({});
      expect(results).toHaveLength(1);
      expect(results[0].traceId).toBe('trace-1');
    });

    it('filters by stepType', async () => {
      await store.storeTrace(trace);
      await store.storeTrace({ ...trace, traceId: 'trace-2', stepType: 'red-team' });

      const results = await store.queryTraces({ stepType: 'creative-director' });
      expect(results).toHaveLength(1);
    });

    it('filters by profileSlug', async () => {
      await store.storeTrace(trace);
      await store.storeTrace({ ...trace, traceId: 'trace-2', profileSlug: 'globex' });

      const results = await store.queryTraces({ profileSlug: 'acme' });
      expect(results).toHaveLength(1);
    });

    it('filters by passed status', async () => {
      await store.storeTrace(trace);
      await store.storeTrace({ ...trace, traceId: 'trace-2', passed: false });

      const results = await store.queryTraces({ passed: true });
      expect(results).toHaveLength(1);
    });

    it('filters by model', async () => {
      await store.storeTrace(trace);
      await store.storeTrace({ ...trace, traceId: 'trace-2', model: 'gemini-flash' });

      const results = await store.queryTraces({ model: 'claude-sonnet-4' });
      expect(results).toHaveLength(1);
    });

    it('respects limit', async () => {
      for (let i = 0; i < 10; i++) {
        await store.storeTrace({ ...trace, traceId: `trace-${i}` });
      }

      const results = await store.queryTraces({ limit: 3 });
      expect(results).toHaveLength(3);
    });

    it('sorts by recency (newest first)', async () => {
      await store.storeTrace({ ...trace, traceId: 'old', timestamp: '2026-03-01T12:00:00Z' });
      await store.storeTrace({ ...trace, traceId: 'new', timestamp: '2026-03-13T12:00:00Z' });

      const results = await store.queryTraces({});
      expect(results[0].traceId).toBe('new');
    });

    it('combines multiple filters', async () => {
      await store.storeTrace(trace);
      await store.storeTrace({ ...trace, traceId: 't2', stepType: 'red-team' });
      await store.storeTrace({ ...trace, traceId: 't3', passed: false });
      await store.storeTrace({ ...trace, traceId: 't4', profileSlug: 'other' });

      const results = await store.queryTraces({
        stepType: 'creative-director',
        profileSlug: 'acme',
        passed: true,
      });
      expect(results).toHaveLength(1);
      expect(results[0].traceId).toBe('trace-1');
    });
  });

  // ─── Beliefs ───

  describe('beliefs', () => {
    const belief: StepBelief = {
      stepType: 'creative-director',
      profileSlug: 'acme',
      alpha: 10,
      beta: 3,
      observations: 11,
      updatedAt: new Date().toISOString(),
    };

    it('stores and retrieves beliefs', async () => {
      await store.setBelief(belief);
      const result = await store.getBelief('creative-director', 'acme');
      expect(result?.alpha).toBe(10);
      expect(result?.beta).toBe(3);
    });

    it('returns null for unknown beliefs', async () => {
      const result = await store.getBelief('unknown', 'unknown');
      expect(result).toBeNull();
    });

    it('overwrites existing beliefs', async () => {
      await store.setBelief(belief);
      await store.setBelief({ ...belief, alpha: 20 });
      const result = await store.getBelief('creative-director', 'acme');
      expect(result?.alpha).toBe(20);
    });
  });

  // ─── Arms ───

  describe('arms', () => {
    const arm: ModelArm = {
      model: 'claude-sonnet-4',
      stepType: 'creative-director',
      profileSlug: 'acme',
      alpha: 5,
      beta: 2,
      pulls: 7,
      avgCost: 0.15,
      avgLatencyMs: 3500,
    };

    it('stores and retrieves arms', async () => {
      await store.setArm(arm);
      const results = await store.getArms('creative-director', 'acme');
      expect(results).toHaveLength(1);
      expect(results[0].model).toBe('claude-sonnet-4');
    });

    it('returns empty array for unknown arms', async () => {
      const results = await store.getArms('unknown', 'unknown');
      expect(results).toHaveLength(0);
    });

    it('updates existing arm by model', async () => {
      await store.setArm(arm);
      await store.setArm({ ...arm, alpha: 10, pulls: 12 });

      const results = await store.getArms('creative-director', 'acme');
      expect(results).toHaveLength(1); // not duplicated
      expect(results[0].alpha).toBe(10);
      expect(results[0].pulls).toBe(12);
    });

    it('stores multiple models for same step/profile', async () => {
      await store.setArm(arm);
      await store.setArm({ ...arm, model: 'gemini-flash' });

      const results = await store.getArms('creative-director', 'acme');
      expect(results).toHaveLength(2);
    });
  });

  // ─── Patterns ───

  describe('patterns', () => {
    const pattern: ConstraintPattern = {
      patternId: 'cp_123',
      stepType: 'creative-director',
      profileSlug: 'acme',
      constraint: 'Include impact statistics in the output',
      occurrences: 5,
      frequency: 0.35,
      sourceReasons: ['Missing impact statistics'],
    };

    it('stores and retrieves patterns', async () => {
      await store.setPattern(pattern);
      const results = await store.getPatterns('creative-director', 'acme');
      expect(results).toHaveLength(1);
      expect(results[0].constraint).toContain('impact statistics');
    });

    it('returns patterns for global scope', async () => {
      await store.setPattern({ ...pattern, profileSlug: '__global__' });
      const results = await store.getPatterns('creative-director', 'acme');
      expect(results).toHaveLength(1); // includes __global__
    });

    it('returns empty array for unknown patterns', async () => {
      const results = await store.getPatterns('unknown');
      expect(results).toHaveLength(0);
    });
  });
});
