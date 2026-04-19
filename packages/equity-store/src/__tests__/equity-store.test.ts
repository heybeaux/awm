import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { EquityStore } from '../equity-store.js';
import type { StepTrace, StepBelief, ModelArm, ConstraintPattern } from '@heybeaux/awm-core';

function makeTrace(overrides: Partial<StepTrace> = {}): StepTrace {
  return {
    traceId: 't-1',
    runId: 'r-1',
    workflowType: 'equity-predict',
    stepType: 'trending_up',
    stepIndex: 0,
    profileSlug: 'AAPL',
    sector: 'tech',
    model: 'default',
    inputFingerprint: 'fp-abc',
    passed: true,
    revised: false,
    tokensIn: 0,
    tokensOut: 0,
    cost: 0,
    latencyMs: 1,
    timestamp: new Date('2026-01-01T00:00:00Z').toISOString(),
    ...overrides,
  };
}

describe('EquityStore (in-memory)', () => {
  let store: EquityStore;

  beforeEach(() => {
    store = new EquityStore();
  });

  afterEach(() => {
    store.close();
  });

  describe('traces', () => {
    it('round-trips a trace with all optional fields populated', async () => {
      const trace = makeTrace({
        traceId: 't-full',
        acrCapabilities: ['cap-a', 'cap-b'],
        acrCoverageRatio: 0.8,
        acrLodLevel: 'standard',
        acrMissingCapabilities: ['cap-c'],
        revised: true,
        revisionReason: 'Missing impact stats',
        approvalResult: 'revised',
        outputFingerprint: 'out-1',
        outputSimilarityToPrior: 0.42,
        awmPrediction: {
          outcome: 'pass',
          confidence: 0.7,
          suggestedModel: 'default',
          skipRecommendation: false,
          constraints: [],
          reasoning: 'test',
          historicalBasis: 5,
        },
        awmWasCorrect: true,
      });
      await store.storeTrace(trace);

      const got = await store.queryTraces({ stepType: 'trending_up' });
      expect(got).toHaveLength(1);
      expect(got[0]).toEqual(trace);
    });

    it('filters by step_type, profile_slug, passed, revised', async () => {
      await store.storeTrace(makeTrace({ traceId: 't1', stepType: 'trending_up', profileSlug: 'AAPL', passed: true }));
      await store.storeTrace(makeTrace({ traceId: 't2', stepType: 'trending_up', profileSlug: 'MSFT', passed: false }));
      await store.storeTrace(makeTrace({ traceId: 't3', stepType: 'quiet', profileSlug: 'AAPL', passed: true, revised: true, revisionReason: 'x' }));

      expect(await store.queryTraces({ stepType: 'trending_up' })).toHaveLength(2);
      expect(await store.queryTraces({ profileSlug: 'AAPL' })).toHaveLength(2);
      expect(await store.queryTraces({ passed: false })).toHaveLength(1);
      expect(await store.queryTraces({ revised: true })).toHaveLength(1);
    });

    it('orders by timestamp DESC and respects limit', async () => {
      await store.storeTrace(makeTrace({ traceId: 't-old', timestamp: '2026-01-01T00:00:00Z' }));
      await store.storeTrace(makeTrace({ traceId: 't-new', timestamp: '2026-02-01T00:00:00Z' }));
      await store.storeTrace(makeTrace({ traceId: 't-mid', timestamp: '2026-01-15T00:00:00Z' }));

      const all = await store.queryTraces({});
      expect(all.map((t) => t.traceId)).toEqual(['t-new', 't-mid', 't-old']);

      const top = await store.queryTraces({ limit: 2 });
      expect(top.map((t) => t.traceId)).toEqual(['t-new', 't-mid']);
    });

    it('deduplicates by trace_id (INSERT OR REPLACE)', async () => {
      await store.storeTrace(makeTrace({ traceId: 'dup', passed: true }));
      await store.storeTrace(makeTrace({ traceId: 'dup', passed: false }));
      const got = await store.queryTraces({});
      expect(got).toHaveLength(1);
      expect(got[0].passed).toBe(false);
    });
  });

  describe('beliefs', () => {
    it('returns null when no belief exists', async () => {
      expect(await store.getBelief('trending_up', 'AAPL')).toBeNull();
    });

    it('upserts and retrieves a belief', async () => {
      const belief: StepBelief = {
        stepType: 'trending_up',
        profileSlug: 'AAPL',
        alpha: 5,
        beta: 2,
        observations: 5,
        updatedAt: '2026-01-01T00:00:00Z',
      };
      await store.setBelief(belief);
      expect(await store.getBelief('trending_up', 'AAPL')).toEqual(belief);

      // Update
      await store.setBelief({ ...belief, alpha: 6, observations: 6 });
      const updated = await store.getBelief('trending_up', 'AAPL');
      expect(updated?.alpha).toBe(6);
      expect(updated?.observations).toBe(6);
    });

    it('isolates beliefs across profiles', async () => {
      await store.setBelief({ stepType: 'trending_up', profileSlug: 'AAPL', alpha: 5, beta: 1, observations: 5, updatedAt: 'now' });
      await store.setBelief({ stepType: 'trending_up', profileSlug: 'MSFT', alpha: 1, beta: 5, observations: 5, updatedAt: 'now' });
      const aapl = await store.getBelief('trending_up', 'AAPL');
      const msft = await store.getBelief('trending_up', 'MSFT');
      expect(aapl?.alpha).toBe(5);
      expect(msft?.beta).toBe(5);
    });
  });

  describe('arms', () => {
    it('returns empty array when no arms exist', async () => {
      expect(await store.getArms('trending_up', 'AAPL')).toEqual([]);
    });

    it('upserts and retrieves arms scoped to step+profile', async () => {
      const arm: ModelArm = {
        model: 'default',
        stepType: 'trending_up',
        profileSlug: 'AAPL',
        alpha: 3,
        beta: 1,
        pulls: 3,
        avgCost: 0.001,
        avgLatencyMs: 100,
      };
      await store.setArm(arm);
      const got = await store.getArms('trending_up', 'AAPL');
      expect(got).toEqual([arm]);

      await store.setArm({ ...arm, alpha: 5, pulls: 5 });
      const updated = await store.getArms('trending_up', 'AAPL');
      expect(updated).toHaveLength(1);
      expect(updated[0].pulls).toBe(5);
    });

    it('keeps multiple models for the same step+profile', async () => {
      await store.setArm({ model: 'a', stepType: 'trending_up', profileSlug: 'AAPL', alpha: 1, beta: 1, pulls: 0, avgCost: 0, avgLatencyMs: 0 });
      await store.setArm({ model: 'b', stepType: 'trending_up', profileSlug: 'AAPL', alpha: 1, beta: 1, pulls: 0, avgCost: 0, avgLatencyMs: 0 });
      const arms = await store.getArms('trending_up', 'AAPL');
      expect(arms.map((a) => a.model).sort()).toEqual(['a', 'b']);
    });
  });

  describe('patterns', () => {
    it('returns global patterns when querying with a profile', async () => {
      const global: ConstraintPattern = {
        patternId: 'p-global',
        stepType: 'trending_up',
        profileSlug: '__global__',
        constraint: 'Include impact statistics',
        occurrences: 5,
        frequency: 0.5,
        sourceReasons: ['Missing impact stats'],
      };
      const specific: ConstraintPattern = { ...global, patternId: 'p-aapl', profileSlug: 'AAPL' };
      await store.setPattern(global);
      await store.setPattern(specific);

      const got = await store.getPatterns('trending_up', 'AAPL');
      expect(got.map((p) => p.patternId).sort()).toEqual(['p-aapl', 'p-global']);
    });

    it('returns all patterns for a step when profile is undefined', async () => {
      await store.setPattern({ patternId: 'p1', stepType: 'trending_up', profileSlug: 'AAPL', constraint: 'c1', occurrences: 2, frequency: 0.2, sourceReasons: ['r1'] });
      await store.setPattern({ patternId: 'p2', stepType: 'trending_up', profileSlug: 'MSFT', constraint: 'c2', occurrences: 2, frequency: 0.2, sourceReasons: ['r2'] });
      const got = await store.getPatterns('trending_up');
      expect(got).toHaveLength(2);
    });

    it('round-trips effectivenessScore and sourceReasons', async () => {
      const pat: ConstraintPattern = {
        patternId: 'p-x',
        stepType: 'quiet',
        profileSlug: 'AAPL',
        constraint: 'X',
        occurrences: 3,
        frequency: 0.6,
        effectivenessScore: 0.8,
        sourceReasons: ['a', 'b', 'c'],
      };
      await store.setPattern(pat);
      const got = await store.getPatterns('quiet', 'AAPL');
      expect(got[0]).toEqual(pat);
    });
  });

  describe('reset', () => {
    it('clears all tables', async () => {
      await store.storeTrace(makeTrace());
      await store.setBelief({ stepType: 's', profileSlug: 'p', alpha: 1, beta: 1, observations: 1, updatedAt: 'now' });
      await store.setArm({ model: 'm', stepType: 's', profileSlug: 'p', alpha: 1, beta: 1, pulls: 0, avgCost: 0, avgLatencyMs: 0 });
      await store.setPattern({ patternId: 'pid', stepType: 's', profileSlug: 'p', constraint: 'c', occurrences: 1, frequency: 0.1, sourceReasons: [] });

      store.reset();

      expect(await store.queryTraces({})).toEqual([]);
      expect(await store.getBelief('s', 'p')).toBeNull();
      expect(await store.getArms('s', 'p')).toEqual([]);
      expect(await store.getPatterns('s', 'p')).toEqual([]);
    });
  });
});

describe('EquityStore (file-backed)', () => {
  let dir: string;
  let dbPath: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'equity-store-'));
    dbPath = join(dir, 'test.db');
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('persists data across instances', async () => {
    const a = new EquityStore({ dbPath });
    await a.setBelief({ stepType: 'trending_up', profileSlug: 'AAPL', alpha: 7, beta: 2, observations: 7, updatedAt: 'now' });
    a.close();

    const b = new EquityStore({ dbPath });
    const got = await b.getBelief('trending_up', 'AAPL');
    b.close();
    expect(got?.alpha).toBe(7);
  });
});
