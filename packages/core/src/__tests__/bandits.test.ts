import { describe, it, expect, beforeEach } from 'vitest';
import { ModelRouter } from '../bandits.js';
import { InMemoryStore } from '../store.js';

describe('ModelRouter', () => {
  let router: ModelRouter;
  let store: InMemoryStore;

  beforeEach(() => {
    store = new InMemoryStore();
    router = new ModelRouter(store);
  });

  it('selects a model from available options with no prior data', async () => {
    const result = await router.selectModel('creative-director', 'acme', [
      'claude-sonnet-4',
      'gemini-2.5-flash',
      'claude-opus-4',
    ]);

    expect(result.model).toBeDefined();
    expect(['claude-sonnet-4', 'gemini-2.5-flash', 'claude-opus-4']).toContain(result.model);
    expect(result.reasoning).toContain('Exploring');
  });

  it('records outcomes and updates arms', async () => {
    // Record several successes for one model
    for (let i = 0; i < 10; i++) {
      await router.recordOutcome('creative-director', 'acme', 'claude-sonnet-4', true, 0.15, 3000);
    }

    // Record failures for another
    for (let i = 0; i < 10; i++) {
      await router.recordOutcome('creative-director', 'acme', 'gemini-2.5-flash', false, 0.03, 1500);
    }

    // Over many samples, should prefer the successful model
    const selections: Record<string, number> = { 'claude-sonnet-4': 0, 'gemini-2.5-flash': 0 };
    for (let i = 0; i < 100; i++) {
      const result = await router.selectModel('creative-director', 'acme', [
        'claude-sonnet-4',
        'gemini-2.5-flash',
      ]);
      selections[result.model]++;
    }

    // Sonnet should be selected significantly more often
    expect(selections['claude-sonnet-4']).toBeGreaterThan(selections['gemini-2.5-flash']);
  });

  it('tracks average cost and latency', async () => {
    await router.recordOutcome('step-a', 'profile-1', 'model-a', true, 0.10, 2000);
    await router.recordOutcome('step-a', 'profile-1', 'model-a', true, 0.20, 4000);

    const arms = await store.getArms('step-a', 'profile-1');
    const arm = arms.find((a) => a.model === 'model-a');

    expect(arm).toBeDefined();
    expect(arm!.avgCost).toBeCloseTo(0.15, 2);
    expect(arm!.avgLatencyMs).toBeCloseTo(3000, 0);
    expect(arm!.pulls).toBe(2);
  });

  it('applies cost weighting to prefer cheaper models when quality is similar', async () => {
    // Both models succeed equally
    for (let i = 0; i < 20; i++) {
      await router.recordOutcome('step-b', 'profile-1', 'expensive-model', true, 0.50, 5000);
      await router.recordOutcome('step-b', 'profile-1', 'cheap-model', true, 0.05, 2000);
    }

    // With high cost weight, cheap model should win more
    const selections: Record<string, number> = { 'expensive-model': 0, 'cheap-model': 0 };
    const highCostRouter = new ModelRouter(store);

    for (let i = 0; i < 100; i++) {
      const result = await highCostRouter.selectModel(
        'step-b',
        'profile-1',
        ['expensive-model', 'cheap-model'],
        0.5, // high cost weight
      );
      selections[result.model]++;
    }

    expect(selections['cheap-model']).toBeGreaterThan(selections['expensive-model']);
  });
});
