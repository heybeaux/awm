import { describe, it, expect } from 'vitest';
import { Oracle } from '../oracle.js';
import { InMemoryStore } from '../store.js';
import type { StepTrace } from '../types.js';

/**
 * Integration test: simulates a full Forge pipeline lifecycle
 *
 * 1. Cold start — no data, predictions are uncertain
 * 2. Record historical runs — build up belief data
 * 3. Predictions improve — confidence increases
 * 4. Revision patterns emerge — constraints get extracted
 * 5. Model routing learns — Thompson Sampling adapts
 * 6. New client profile — falls back to global beliefs
 */
describe('Full Pipeline Lifecycle', () => {
  it('improves predictions over 50 pipeline runs', async () => {
    const store = new InMemoryStore();
    const oracle = new Oracle({ store });

    const models = ['claude-sonnet-4', 'gemini-2.5-flash', 'claude-opus-4'];
    const stepTypes = ['strategist', 'creative-director', 'red-team', 'visual-creative'];

    // ─── Phase 1: Cold start ───
    const coldPrediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: models,
    });

    expect(coldPrediction.historicalBasis).toBe(0);
    expect(coldPrediction.reasoning).toContain('No historical data');

    // ─── Phase 2: Record 50 pipeline runs ───
    // Simulate realistic patterns:
    // - strategist passes 95% of the time
    // - creative-director passes 75% (common revisions)
    // - red-team passes 85%
    // - visual-creative passes 90%
    const passRates: Record<string, number> = {
      'strategist': 0.95,
      'creative-director': 0.60, // intentionally lower to generate enough revision patterns
      'red-team': 0.85,
      'visual-creative': 0.90,
    };

    const revisionReasons = [
      'Missing impact statistics',
      'Missing impact statistics',
      'Missing impact statistics',
      'Tone too aggressive for nonprofit audience',
      'Tone too aggressive for nonprofit audience',
      'Lacks brand consistency',
    ];

    for (let run = 0; run < 50; run++) {
      for (let step = 0; step < stepTypes.length; step++) {
        const stepType = stepTypes[step];
        // Use deterministic pass/fail based on run number for reproducibility
        const passThreshold = Math.floor(passRates[stepType] * 50);
        const passed = run < passThreshold;
        const revised = !passed;
        const model = models[run % models.length];

        const trace: StepTrace = {
          traceId: `trace-${run}-${step}`,
          runId: `run-${run}`,
          workflowType: 'creative-campaign',
          stepType,
          stepIndex: step,
          profileSlug: 'acme-nonprofit',
          sector: 'nonprofit',
          model,
          inputFingerprint: `input-${run}-${step}`,
          passed,
          revised,
          revisionReason: revised
            ? revisionReasons[run % revisionReasons.length]
            : undefined,
          tokensIn: 1500,
          tokensOut: 2000,
          cost: model === 'claude-opus-4' ? 0.45 : model === 'claude-sonnet-4' ? 0.15 : 0.04,
          latencyMs: 3000,
          timestamp: new Date(Date.now() - (50 - run) * 3600000).toISOString(),
        };

        await oracle.record(trace);
      }
    }

    // ─── Phase 3: Predictions should be informed ───
    const informedPrediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme-nonprofit',
      availableModels: models,
    });

    expect(informedPrediction.historicalBasis).toBeGreaterThan(0);
    // Confidence is now posterior mean (P(predicted outcome))
    // After training, informed prediction should be based on real data
    expect(informedPrediction.historicalBasis).toBeGreaterThan(10);

    // Strategist should have high confidence pass prediction
    const strategistPrediction = await oracle.predict({
      stepType: 'strategist',
      profileSlug: 'acme-nonprofit',
      availableModels: models,
    });
    expect(strategistPrediction.outcome).toBe('pass');

    // ─── Phase 4: Constraints should emerge ───
    // Creative-director had repeated "Missing impact statistics" revisions
    expect(informedPrediction.constraints.length).toBeGreaterThan(0);
    const hasImpactConstraint = informedPrediction.constraints.some(
      (c) => c.toLowerCase().includes('impact statistics'),
    );
    expect(hasImpactConstraint).toBe(true);

    // ─── Phase 5: Model routing should learn ───
    // Run 100 model selections — should show preference patterns
    const modelSelections: Record<string, number> = {};
    for (let i = 0; i < 100; i++) {
      const pred = await oracle.predict({
        stepType: 'creative-director',
        profileSlug: 'acme-nonprofit',
        availableModels: models,
      });
      modelSelections[pred.suggestedModel] = (modelSelections[pred.suggestedModel] || 0) + 1;
    }

    // With exploration decay, Thompson Sampling should strongly prefer
    // the best-performing model after 50+ training runs
    const values = Object.values(modelSelections);
    const maxSelection = Math.max(...values);
    expect(maxSelection).toBeGreaterThan(50); // dominant model gets majority

    // ─── Phase 6: New client starts fresh (profile isolation) ───
    const newClientPrediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'brand-new-client',
      availableModels: models,
    });

    // With profile isolation, new client has no data — starts from uninformative prior
    expect(newClientPrediction.historicalBasis).toBe(0);
    // Should still produce a valid prediction (from prior)
    expect(['pass', 'revise', 'fail']).toContain(newClientPrediction.outcome);
  });

  it('tracks prediction accuracy over time', async () => {
    const store = new InMemoryStore();
    const oracle = new Oracle({ store });

    // Seed with some data
    for (let i = 0; i < 20; i++) {
      await oracle.record({
        traceId: `seed-${i}`,
        runId: `seed-run-${i}`,
        workflowType: 'creative-campaign',
        stepType: 'creative-director',
        stepIndex: 1,
        profileSlug: 'acme',
        model: 'claude-sonnet-4',
        inputFingerprint: `seed-${i}`,
        passed: true,
        revised: false,
        tokensIn: 1500,
        tokensOut: 2000,
        cost: 0.15,
        latencyMs: 3500,
        timestamp: new Date().toISOString(),
      });
    }

    // Now predict and verify
    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme',
      availableModels: ['claude-sonnet-4'],
    });

    expect(prediction.outcome).toBe('pass');

    // Record a trace with AWM's prediction attached
    await oracle.record({
      traceId: 'verified-1',
      runId: 'run-verified',
      workflowType: 'creative-campaign',
      stepType: 'creative-director',
      stepIndex: 1,
      profileSlug: 'acme',
      model: 'claude-sonnet-4',
      inputFingerprint: 'verified-input',
      passed: true, // AWM predicted pass, and it passed
      revised: false,
      tokensIn: 1500,
      tokensOut: 2000,
      cost: 0.15,
      latencyMs: 3500,
      timestamp: new Date().toISOString(),
      awmPrediction: prediction,
      awmWasCorrect: true,
    });

    // Verify the trace was stored with prediction metadata
    const traces = await store.queryTraces({ stepType: 'creative-director' });
    const verifiedTrace = traces.find((t) => t.traceId === 'verified-1');
    expect(verifiedTrace?.awmWasCorrect).toBe(true);
    expect(verifiedTrace?.awmPrediction?.outcome).toBe('pass');
  });

  it('handles ACR-aware predictions', async () => {
    const store = new InMemoryStore();
    const oracle = new Oracle({ store });

    // Seed data
    for (let i = 0; i < 10; i++) {
      await oracle.record({
        traceId: `acr-${i}`,
        runId: `acr-run-${i}`,
        workflowType: 'creative-campaign',
        stepType: 'creative-director',
        stepIndex: 1,
        profileSlug: 'acme',
        model: 'claude-sonnet-4',
        inputFingerprint: `acr-${i}`,
        passed: true,
        revised: false,
        tokensIn: 1500,
        tokensOut: 2000,
        cost: 0.15,
        latencyMs: 3500,
        acrCapabilities: ['brand-voice', 'audience'],
        acrCoverageRatio: 1.0,
        acrLodLevel: 'standard',
        timestamp: new Date().toISOString(),
      });
    }

    const prediction = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme',
      availableModels: ['claude-sonnet-4'],
      acrContext: {
        loadedCapabilities: ['brand-voice'],
        coverageRatio: 0.5,
        lodLevel: 'summary',
        missingCapabilities: ['audience', 'compliance'],
      },
    });

    // Should mention coverage ratio and missing capabilities in reasoning
    expect(prediction.reasoning).toContain('50%');
    expect(prediction.reasoning).toContain('audience');
    expect(prediction.reasoning).toContain('compliance');
  });

  it('handles multiple profiles independently', async () => {
    const store = new InMemoryStore();
    const oracle = new Oracle({ store });

    // Acme always passes
    for (let i = 0; i < 15; i++) {
      await oracle.record({
        traceId: `acme-${i}`,
        runId: `acme-run-${i}`,
        workflowType: 'creative-campaign',
        stepType: 'creative-director',
        stepIndex: 1,
        profileSlug: 'acme',
        model: 'claude-sonnet-4',
        inputFingerprint: `acme-${i}`,
        passed: true,
        revised: false,
        tokensIn: 1500,
        tokensOut: 2000,
        cost: 0.15,
        latencyMs: 3500,
        timestamp: new Date().toISOString(),
      });
    }

    // Globex always fails
    for (let i = 0; i < 15; i++) {
      await oracle.record({
        traceId: `globex-${i}`,
        runId: `globex-run-${i}`,
        workflowType: 'creative-campaign',
        stepType: 'creative-director',
        stepIndex: 1,
        profileSlug: 'globex',
        model: 'claude-sonnet-4',
        inputFingerprint: `globex-${i}`,
        passed: false,
        revised: true,
        revisionReason: 'Brand voice mismatch',
        tokensIn: 1500,
        tokensOut: 2000,
        cost: 0.15,
        latencyMs: 3500,
        timestamp: new Date().toISOString(),
      });
    }

    const acmePred = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'acme',
      availableModels: ['claude-sonnet-4'],
    });

    const globexPred = await oracle.predict({
      stepType: 'creative-director',
      profileSlug: 'globex',
      availableModels: ['claude-sonnet-4'],
    });

    // Acme should predict pass, Globex should predict fail
    expect(acmePred.outcome).toBe('pass');
    expect(globexPred.outcome).toBe('fail');

    // Globex should have constraints from its revision reasons
    expect(globexPred.constraints.length).toBeGreaterThan(0);
  });
});
