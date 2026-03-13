import { describe, it, expect, beforeEach } from 'vitest';
import { ConstraintExtractor } from '../constraints.js';
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
    passed: false,
    revised: true,
    tokensIn: 1500,
    tokensOut: 2000,
    cost: 0.12,
    latencyMs: 3500,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

describe('ConstraintExtractor', () => {
  let extractor: ConstraintExtractor;
  let store: InMemoryStore;

  beforeEach(() => {
    store = new InMemoryStore();
    extractor = new ConstraintExtractor(store);
  });

  it('returns no constraints with no data', async () => {
    const constraints = await extractor.getConstraints('creative-director', 'acme');
    expect(constraints).toHaveLength(0);
  });

  it('extracts patterns from repeated revision reasons', async () => {
    // Store 5 traces with the same revision reason
    for (let i = 0; i < 5; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-${i}`,
        revisionReason: 'Missing impact statistics',
      }));
    }
    // Store 3 successful traces (no revision)
    for (let i = 0; i < 3; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-ok-${i}`,
        passed: true,
        revised: false,
        revisionReason: undefined,
      }));
    }

    const patterns = await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    expect(patterns.length).toBeGreaterThan(0);
    expect(patterns[0].constraint.toLowerCase()).toContain('impact statistics');
    expect(patterns[0].occurrences).toBe(5);
  });

  it('returns constraints sorted by frequency', async () => {
    // 6 traces with reason A
    for (let i = 0; i < 6; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-a-${i}`,
        revisionReason: 'Missing impact statistics',
      }));
    }
    // 3 traces with reason B
    for (let i = 0; i < 3; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-b-${i}`,
        revisionReason: 'Tone too aggressive',
      }));
    }

    await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    const constraints = await extractor.getConstraints('creative-director', 'acme-nonprofit');

    expect(constraints.length).toBe(2);
    // Higher frequency first
    expect(constraints[0].toLowerCase()).toContain('impact statistics');
  });

  it('ignores reasons below minimum occurrence threshold', async () => {
    // Only 1 trace with a reason (below threshold of 3)
    await store.storeTrace(makeTrace({
      revisionReason: 'Rare issue that happened once',
    }));
    // Add enough other traces to make frequency low
    for (let i = 0; i < 10; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-ok-${i}`,
        passed: true,
        revised: false,
        revisionReason: undefined,
      }));
    }

    await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    const constraints = await extractor.getConstraints('creative-director', 'acme-nonprofit');
    expect(constraints).toHaveLength(0);
  });

  it('converts revision reasons into actionable constraints', async () => {
    const reasons = [
      { input: 'Missing impact statistics', expected: 'impact statistics' },
      { input: 'Tone too aggressive for nonprofit audience', expected: 'too aggressive' },
      { input: 'Not enough supporting data', expected: 'supporting data' },
      { input: 'Lacking brand consistency', expected: 'brand consistency' },
    ];

    for (const { input } of reasons) {
      // Need 3+ occurrences to pass threshold
      for (let i = 0; i < 4; i++) {
        await store.storeTrace(makeTrace({
          traceId: `trace-${input}-${i}`,
          revisionReason: input,
        }));
      }
    }

    await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    const constraints = await extractor.getConstraints('creative-director', 'acme-nonprofit');

    expect(constraints.length).toBeGreaterThanOrEqual(reasons.length);
    for (const { expected } of reasons) {
      const found = constraints.some((c) => c.toLowerCase().includes(expected));
      expect(found, `Expected constraint containing "${expected}"`).toBe(true);
    }
  });

  it('scopes patterns by step type', async () => {
    for (let i = 0; i < 4; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-cd-${i}`,
        stepType: 'creative-director',
        revisionReason: 'Missing impact statistics',
      }));
    }
    for (let i = 0; i < 4; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-rt-${i}`,
        stepType: 'red-team',
        revisionReason: 'Compliance violation',
      }));
    }

    await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    await extractor.analyzeRevisions('red-team', 'acme-nonprofit');

    const cdConstraints = await extractor.getConstraints('creative-director', 'acme-nonprofit');
    const rtConstraints = await extractor.getConstraints('red-team', 'acme-nonprofit');

    expect(cdConstraints.some((c) => c.toLowerCase().includes('impact'))).toBe(true);
    expect(cdConstraints.some((c) => c.toLowerCase().includes('compliance'))).toBe(false);

    expect(rtConstraints.some((c) => c.toLowerCase().includes('compliance'))).toBe(true);
    expect(rtConstraints.some((c) => c.toLowerCase().includes('impact'))).toBe(false);
  });

  it('generates deterministic pattern IDs for dedup', async () => {
    for (let i = 0; i < 4; i++) {
      await store.storeTrace(makeTrace({
        traceId: `trace-${i}`,
        revisionReason: 'Missing impact statistics',
      }));
    }

    const patterns1 = await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');
    const patterns2 = await extractor.analyzeRevisions('creative-director', 'acme-nonprofit');

    expect(patterns1[0].patternId).toBe(patterns2[0].patternId);
  });
});
