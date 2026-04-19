import { describe, it, expect, afterEach } from 'vitest';
import { Oracle } from '@heybeaux/awm-core';
import type { StepTrace } from '@heybeaux/awm-core';
import { EquityStore } from '../equity-store.js';
import { tickerToProfile } from '../session.js';

function trace(overrides: Partial<StepTrace>): StepTrace {
  return {
    traceId: `t-${Math.random().toString(36).slice(2)}`,
    runId: 'r-1',
    workflowType: 'equity-predict',
    stepType: 'trending_up',
    stepIndex: 0,
    profileSlug: 'AAPL',
    model: 'default',
    inputFingerprint: 'fp',
    passed: true,
    revised: false,
    tokensIn: 0,
    tokensOut: 0,
    cost: 0,
    latencyMs: 1,
    timestamp: new Date().toISOString(),
    ...overrides,
  };
}

describe('Oracle + EquityStore integration', () => {
  let store: EquityStore | null = null;

  afterEach(() => {
    store?.close();
    store = null;
  });

  it('cold start has zero historical basis', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });

    const pred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug: tickerToProfile('AAPL'),
      availableModels: ['default'],
    });

    expect(pred.historicalBasis).toBe(0);
    expect(pred.reasoning).toContain('No historical data');
  });

  it('beliefs update after recording outcomes — predict→record→predict cycle', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });
    const profileSlug = tickerToProfile('AAPL');
    const stepType = 'trending_up';

    // Feed 20 wins and 5 losses
    for (let i = 0; i < 25; i++) {
      const passed = i < 20;
      await oracle.record(trace({
        traceId: `t-${i}`,
        stepType,
        profileSlug,
        passed,
      }));
    }

    const pred = await oracle.predict({
      stepType,
      profileSlug,
      availableModels: ['default'],
    });

    expect(pred.historicalBasis).toBe(25);
    // Posterior mean ≈ (1.5 + 20) / (1.5 + 1 + 25) = 21.5 / 27.5 ≈ 0.78
    // Outcome should be 'pass' (>0.7)
    expect(pred.outcome).toBe('pass');
    expect(pred.confidence).toBeGreaterThan(0.7);
  });

  it('isolates beliefs per ticker (profile)', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });

    // AAPL passes consistently
    for (let i = 0; i < 10; i++) {
      await oracle.record(trace({
        traceId: `aapl-${i}`,
        profileSlug: 'AAPL',
        passed: true,
      }));
    }
    // TSLA fails consistently
    for (let i = 0; i < 10; i++) {
      await oracle.record(trace({
        traceId: `tsla-${i}`,
        profileSlug: 'TSLA',
        passed: false,
      }));
    }

    const aaplPred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug: 'AAPL',
      availableModels: ['default'],
    });
    const tslaPred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug: 'TSLA',
      availableModels: ['default'],
    });

    expect(aaplPred.outcome).toBe('pass');
    expect(tslaPred.outcome).toBe('fail');
    // Both should reflect their own historical basis
    expect(aaplPred.historicalBasis).toBe(10);
    expect(tslaPred.historicalBasis).toBe(10);
  });

  it('isolates beliefs per regime (stepType)', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });
    const profileSlug = 'AAPL';

    // trending_up regime: passes
    for (let i = 0; i < 10; i++) {
      await oracle.record(trace({
        traceId: `up-${i}`,
        stepType: 'trending_up',
        profileSlug,
        passed: true,
      }));
    }
    // trending_down regime: fails
    for (let i = 0; i < 10; i++) {
      await oracle.record(trace({
        traceId: `down-${i}`,
        stepType: 'trending_down',
        profileSlug,
        passed: false,
      }));
    }

    const upPred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug,
      availableModels: ['default'],
    });
    const downPred = await oracle.predict({
      stepType: 'trending_down',
      profileSlug,
      availableModels: ['default'],
    });

    expect(upPred.outcome).toBe('pass');
    expect(downPred.outcome).toBe('fail');
  });

  it('reset() between sweeps clears beliefs back to cold start', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });

    for (let i = 0; i < 10; i++) {
      await oracle.record(trace({ traceId: `t-${i}`, passed: true }));
    }

    let pred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug: 'AAPL',
      availableModels: ['default'],
    });
    expect(pred.historicalBasis).toBe(10);

    store.reset();

    pred = await oracle.predict({
      stepType: 'trending_up',
      profileSlug: 'AAPL',
      availableModels: ['default'],
    });
    expect(pred.historicalBasis).toBe(0);
  });

  it('records bandit arm pulls alongside belief updates', async () => {
    store = new EquityStore();
    const oracle = new Oracle({ store });

    for (let i = 0; i < 5; i++) {
      await oracle.record(trace({
        traceId: `t-${i}`,
        model: 'default',
        cost: 0.0001 * (i + 1),
        passed: true,
      }));
    }

    const arms = await store.getArms('trending_up', 'AAPL');
    expect(arms).toHaveLength(1);
    expect(arms[0].pulls).toBe(5);
    expect(arms[0].alpha).toBeGreaterThan(1); // got incremented from prior
    expect(arms[0].avgCost).toBeGreaterThan(0);
  });
});
