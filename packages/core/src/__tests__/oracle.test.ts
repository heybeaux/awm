import { describe, it, expect, beforeEach } from 'vitest';
import { Oracle } from '../oracle.js';
import { InMemoryStore } from '../store.js';
import type { StepTrace } from '../types.js';

function makeTrace(overrides: Partial<StepTrace> = {}): StepTrace {
  return {
    traceId: `trace-${Math.random().toString(36).slice(2)}`,
    runId: 'run-1',
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

describe('Oracle', () => {
  let oracle: Oracle;

  beforeEach(() => {
    oracle = new Oracle({ store: new InMemoryStore() });
  });

  it('returns low-confidence prediction with no data', async () => {
    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4', 'gemini-2.5-flash'],
    });

    // Beta(2,2) prior gives moderate confidence even with no observations
    expect(prediction.confidence).toBeLessThan(0.9);
    expect(prediction.historicalBasis).toBe(0);
    expect(prediction.reasoning).toContain('No historical data');
  });

  it('improves prediction confidence with more data', async () => {
    // Record 20 successful runs
    for (let i = 0; i < 20; i++) {
      await oracle.record(makeTrace());
    }

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4'],
    });

    expect(prediction.confidence).toBeGreaterThan(0.5);
    expect(prediction.outcome).toBe('pass');
    expect(prediction.historicalBasis).toBe(20);
  });

  it('predicts failure when most runs fail', async () => {
    // Record mostly failures
    for (let i = 0; i < 3; i++) await oracle.record(makeTrace({ passed: true }));
    for (let i = 0; i < 15; i++) await oracle.record(makeTrace({ passed: false }));

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4'],
    });

    expect(prediction.outcome).toBe('fail');
  });

  it('extracts constraints from revision patterns', async () => {
    // Record several revisions with the same reason
    for (let i = 0; i < 5; i++) {
      await oracle.record(
        makeTrace({
          passed: false,
          revised: true,
          revisionReason: 'Missing impact statistics',
        }),
      );
    }

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4'],
    });

    expect(prediction.constraints.length).toBeGreaterThan(0);
    expect(prediction.constraints[0].toLowerCase()).toContain('impact statistics');
  });

  it('does not recommend skipping with low confidence', async () => {
    // Only 3 data points — not enough for skip confidence
    for (let i = 0; i < 3; i++) {
      await oracle.record(makeTrace());
    }

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4'],
    });

    expect(prediction.skipRecommendation).toBe(false);
  });

  it('includes ACR context in reasoning when provided', async () => {
    for (let i = 0; i < 5; i++) await oracle.record(makeTrace());

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: ['claude-sonnet-4'],
      acrContext: {
        loadedCapabilities: ['brand-voice', 'audience'],
        coverageRatio: 0.67,
        lodLevel: 'standard',
        missingCapabilities: ['compliance'],
      },
    });

    expect(prediction.reasoning).toContain('67%');
    expect(prediction.reasoning).toContain('compliance');
  });
});
