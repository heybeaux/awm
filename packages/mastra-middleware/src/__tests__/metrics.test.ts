import { describe, it, expect, beforeEach } from 'vitest';
import { MetricsEngine } from '../metrics.js';
import { InMemoryStore, Oracle } from '../../../core/src/index.js';
import type { StepTrace } from '../../../core/src/types.js';

function makeTrace(overrides: Partial<StepTrace> = {}): StepTrace {
  return {
    traceId: `trace-${Math.random().toString(36).slice(2)}`,
    runId: `run-${Math.floor(Math.random() * 100)}`,
    workflowType: 'creative-campaign',
    stepType: 'creative-director',
    stepIndex: 1,
    profileSlug: 'acme-nonprofit',
    model: 'claude-sonnet-4',
    inputFingerprint: 'abc123',
    passed: true,
    revised: false,
    tokensIn: 1500,
    tokensOut: 2000,
    cost: 0.12,
    latencyMs: 3500,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

describe('MetricsEngine', () => {
  let store: InMemoryStore;
  let metrics: MetricsEngine;

  beforeEach(() => {
    store = new InMemoryStore();
    metrics = new MetricsEngine(store);
  });

  it('returns empty metrics with no data', async () => {
    const result = await metrics.computeMetrics();
    expect(result.summary.totalTraces).toBe(0);
    expect(result.steps).toHaveLength(0);
    expect(result.profiles).toHaveLength(0);
  });

  it('computes correct summary metrics', async () => {
    // 8 passed, 2 failed
    for (let i = 0; i < 8; i++) await store.storeTrace(makeTrace({ runId: `run-${i}` }));
    for (let i = 0; i < 2; i++) await store.storeTrace(makeTrace({ passed: false, runId: `run-${8 + i}` }));

    const result = await metrics.computeMetrics();
    expect(result.summary.totalTraces).toBe(10);
    expect(result.summary.overallPassRate).toBe(0.8);
    expect(result.summary.totalCost).toBeCloseTo(1.2, 1);
  });

  it('computes per-step metrics', async () => {
    for (let i = 0; i < 5; i++) await store.storeTrace(makeTrace({ stepType: 'strategist' }));
    for (let i = 0; i < 3; i++) await store.storeTrace(makeTrace({ stepType: 'red-team', passed: false }));

    const result = await metrics.computeMetrics();
    expect(result.steps).toHaveLength(2);

    const strategist = result.steps.find((s) => s.stepType === 'strategist');
    expect(strategist?.passRate).toBe(1.0);

    const redTeam = result.steps.find((s) => s.stepType === 'red-team');
    expect(redTeam?.passRate).toBe(0.0);
  });

  it('computes per-profile metrics', async () => {
    for (let i = 0; i < 4; i++) await store.storeTrace(makeTrace({ profileSlug: 'acme' }));
    for (let i = 0; i < 6; i++) await store.storeTrace(makeTrace({ profileSlug: 'globex' }));

    const result = await metrics.computeMetrics();
    expect(result.profiles).toHaveLength(2);

    const acme = result.profiles.find((p) => p.profileSlug === 'acme');
    expect(acme?.totalRuns).toBe(4);

    const globex = result.profiles.find((p) => p.profileSlug === 'globex');
    expect(globex?.totalRuns).toBe(6);
  });

  it('computes per-model metrics', async () => {
    for (let i = 0; i < 5; i++) await store.storeTrace(makeTrace({ model: 'claude-sonnet-4', cost: 0.15 }));
    for (let i = 0; i < 5; i++) await store.storeTrace(makeTrace({ model: 'gemini-flash', cost: 0.04 }));

    const result = await metrics.computeMetrics();
    expect(result.models).toHaveLength(2);

    const sonnet = result.models.find((m) => m.model === 'claude-sonnet-4');
    expect(sonnet?.avgCost).toBeCloseTo(0.15, 2);

    const gemini = result.models.find((m) => m.model === 'gemini-flash');
    expect(gemini?.avgCost).toBeCloseTo(0.04, 2);
  });

  it('tracks revision reasons per step', async () => {
    for (let i = 0; i < 5; i++) {
      await store.storeTrace(makeTrace({
        revised: true,
        revisionReason: 'Missing impact statistics',
      }));
    }
    for (let i = 0; i < 2; i++) {
      await store.storeTrace(makeTrace({
        revised: true,
        revisionReason: 'Tone too aggressive',
      }));
    }

    const result = await metrics.computeMetrics();
    const step = result.steps[0];
    expect(step.topRevisionReasons[0].reason).toBe('Missing impact statistics');
    expect(step.topRevisionReasons[0].count).toBe(5);
  });

  it('computes daily trend', async () => {
    const today = new Date().toISOString().slice(0, 10);
    const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);

    for (let i = 0; i < 3; i++) {
      await store.storeTrace(makeTrace({ timestamp: `${today}T12:00:00Z` }));
    }
    for (let i = 0; i < 5; i++) {
      await store.storeTrace(makeTrace({ timestamp: `${yesterday}T12:00:00Z` }));
    }

    const result = await metrics.computeMetrics();
    expect(result.trend).toHaveLength(2);
    expect(result.trend[0].date).toBe(yesterday);
    expect(result.trend[0].traces).toBe(5);
    expect(result.trend[1].date).toBe(today);
    expect(result.trend[1].traces).toBe(3);
  });

  it('tracks prediction accuracy', async () => {
    for (let i = 0; i < 7; i++) {
      await store.storeTrace(makeTrace({ awmWasCorrect: true }));
    }
    for (let i = 0; i < 3; i++) {
      await store.storeTrace(makeTrace({ awmWasCorrect: false }));
    }

    const result = await metrics.computeMetrics();
    expect(result.summary.predictionAccuracy).toBeCloseTo(0.7, 1);
  });
});
