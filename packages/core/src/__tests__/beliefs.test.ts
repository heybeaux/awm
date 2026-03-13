import { describe, it, expect, beforeEach } from 'vitest';
import { BeliefEngine } from '../beliefs.js';
import { InMemoryStore } from '../store.js';

describe('BeliefEngine', () => {
  let engine: BeliefEngine;
  let store: InMemoryStore;

  beforeEach(() => {
    store = new InMemoryStore();
    engine = new BeliefEngine(store);
  });

  it('returns uninformative prior for unknown step/profile', async () => {
    const belief = await engine.getBelief('creative-director', 'acme');
    expect(belief.alpha).toBe(2);
    expect(belief.beta).toBe(2);
    expect(belief.observations).toBe(0);
    expect(engine.successProbability(belief)).toBe(0.5);
  });

  it('updates belief on success', async () => {
    const belief = await engine.update('creative-director', 'acme', true);
    expect(belief.alpha).toBe(3);
    expect(belief.beta).toBe(2);
    expect(belief.observations).toBe(1);
    expect(engine.successProbability(belief)).toBe(0.6);
  });

  it('updates belief on failure', async () => {
    const belief = await engine.update('creative-director', 'acme', false);
    expect(belief.alpha).toBe(2);
    expect(belief.beta).toBe(3);
    expect(belief.observations).toBe(1);
    expect(engine.successProbability(belief)).toBe(0.4);
  });

  it('increases confidence with more observations', async () => {
    const initial = await engine.getBelief('creative-director', 'acme');
    const initialConf = engine.confidence(initial);

    // Add 20 successes
    for (let i = 0; i < 20; i++) {
      await engine.update('creative-director', 'acme', true);
    }

    const updated = await engine.getBelief('creative-director', 'acme');
    const updatedConf = engine.confidence(updated);

    expect(updatedConf).toBeGreaterThan(initialConf);
  });

  it('predicts pass for high success rate', async () => {
    for (let i = 0; i < 15; i++) await engine.update('step-a', 'profile-1', true);
    for (let i = 0; i < 2; i++) await engine.update('step-a', 'profile-1', false);

    const belief = await engine.getBelief('step-a', 'profile-1');
    expect(engine.predictOutcome(belief)).toBe('pass');
  });

  it('predicts fail for low success rate', async () => {
    for (let i = 0; i < 2; i++) await engine.update('step-b', 'profile-1', true);
    for (let i = 0; i < 15; i++) await engine.update('step-b', 'profile-1', false);

    const belief = await engine.getBelief('step-b', 'profile-1');
    expect(engine.predictOutcome(belief)).toBe('fail');
  });

  it('predicts revise for mixed results', async () => {
    for (let i = 0; i < 5; i++) await engine.update('step-c', 'profile-1', true);
    for (let i = 0; i < 5; i++) await engine.update('step-c', 'profile-1', false);

    const belief = await engine.getBelief('step-c', 'profile-1');
    expect(engine.predictOutcome(belief)).toBe('revise');
  });

  it('also updates global belief alongside profile-specific', async () => {
    await engine.update('creative-director', 'acme', true);

    const profileBelief = await engine.getBelief('creative-director', 'acme');
    const globalBelief = await engine.getBelief('creative-director');

    expect(profileBelief.alpha).toBe(3);
    expect(globalBelief.alpha).toBe(3);
  });

  it('falls back to global when profile has no data', async () => {
    // Add data for global only
    for (let i = 0; i < 10; i++) {
      await engine.update('step-x', undefined, true);
    }

    // Query for a specific profile that has no data
    const belief = await engine.getBelief('step-x', 'unknown-profile');
    // Should fall back to global
    expect(belief.observations).toBeGreaterThan(0);
  });
});
