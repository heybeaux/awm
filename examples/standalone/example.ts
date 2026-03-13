/**
 * AWM Standalone Example
 *
 * Demonstrates the full AWM cycle without any external dependencies:
 * 1. Create an Oracle with in-memory storage
 * 2. Simulate pipeline runs and record outcomes
 * 3. Watch predictions improve as data accumulates
 * 4. See constraint pre-injection from revision patterns
 * 5. See model routing optimize for cost
 */

import { Oracle, InMemoryStore, type StepTrace } from '@heybeaux/awm-core';

async function main() {
  const oracle = new Oracle({ store: new InMemoryStore() });

  console.log('═══════════════════════════════════════════');
  console.log(' AWM — Agent Workflow Model Demo');
  console.log('═══════════════════════════════════════════\n');

  // ─── Phase 1: Simulate historical runs ───
  console.log('📊 Phase 1: Recording 30 historical pipeline runs...\n');

  const models = ['claude-sonnet-4', 'gemini-2.5-flash', 'claude-opus-4'];

  for (let i = 0; i < 30; i++) {
    // Creative director step — passes 80% of the time
    const passed = Math.random() < 0.8;
    const revised = !passed && Math.random() < 0.7;

    const trace: StepTrace = {
      traceId: `trace-${i}`,
      runId: `run-${i}`,
      workflowType: 'creative-campaign',
      stepType: 'creative-director',
      stepIndex: 1,
      profileSlug: 'acme-nonprofit',
      sector: 'nonprofit',
      model: models[i % 3],
      inputFingerprint: `input-${i}`,
      passed,
      revised,
      revisionReason: revised
        ? i % 2 === 0
          ? 'Missing impact statistics'
          : 'Tone too aggressive for nonprofit audience'
        : undefined,
      tokensIn: 1500 + Math.random() * 500,
      tokensOut: 2000 + Math.random() * 1000,
      cost: models[i % 3] === 'claude-opus-4' ? 0.45 : models[i % 3] === 'claude-sonnet-4' ? 0.15 : 0.04,
      latencyMs: 3000 + Math.random() * 2000,
      timestamp: new Date(Date.now() - (30 - i) * 86400000).toISOString(),
    };

    await oracle.record(trace);
  }

  console.log('   ✅ 30 runs recorded\n');

  // ─── Phase 2: Get a prediction ───
  console.log('🔮 Phase 2: Predicting next run outcome...\n');

  const prediction = await oracle.predict({
    stepType: 'creative-director',
    profileSlug: 'acme-nonprofit',
    availableModels: models,
  });

  console.log(`   Predicted outcome:  ${prediction.outcome}`);
  console.log(`   Confidence:         ${(prediction.confidence * 100).toFixed(1)}%`);
  console.log(`   Suggested model:    ${prediction.suggestedModel}`);
  console.log(`   Skip recommended:   ${prediction.skipRecommendation}`);
  console.log(`   Historical basis:   ${prediction.historicalBasis} runs`);
  console.log(`   Reasoning:          ${prediction.reasoning}`);

  if (prediction.constraints.length > 0) {
    console.log('\n   📋 Constraints to pre-inject:');
    for (const c of prediction.constraints) {
      console.log(`      → ${c}`);
    }
  }

  // ─── Phase 3: Compare with unknown step ───
  console.log('\n\n🆕 Phase 3: Predicting for a step with NO history...\n');

  const unknownPrediction = await oracle.predict({
    stepType: 'visual-creative',
    profileSlug: 'brand-new-client',
    availableModels: models,
  });

  console.log(`   Predicted outcome:  ${unknownPrediction.outcome}`);
  console.log(`   Confidence:         ${(unknownPrediction.confidence * 100).toFixed(1)}%`);
  console.log(`   Historical basis:   ${unknownPrediction.historicalBasis} runs`);
  console.log(`   Reasoning:          ${unknownPrediction.reasoning}`);

  // ─── Phase 4: Show ACR-aware prediction ───
  console.log('\n\n🧩 Phase 4: ACR-aware prediction (missing capabilities)...\n');

  const acrPrediction = await oracle.predict({
    stepType: 'creative-director',
    profileSlug: 'acme-nonprofit',
    availableModels: models,
    acrContext: {
      loadedCapabilities: ['brand-voice', 'audience-research'],
      coverageRatio: 0.67,
      lodLevel: 'standard',
      missingCapabilities: ['compliance', 'campaign-history'],
    },
  });

  console.log(`   Predicted outcome:  ${acrPrediction.outcome}`);
  console.log(`   Reasoning:          ${acrPrediction.reasoning}`);

  console.log('\n═══════════════════════════════════════════');
  console.log(' Demo complete. AWM learns from every run.');
  console.log('═══════════════════════════════════════════\n');
}

main().catch(console.error);
