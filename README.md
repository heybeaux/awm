# AWM — Agent Workflow Model

> Predictive execution protocol for agent pipelines. Learn from history. Adapt before wasting tokens.

AWM is a prediction and outcome modeling layer for multi-agent pipelines. It observes what happened in past runs, predicts what's likely to happen next, and recommends runtime adaptations — model routing, constraint injection, step skipping — before each step executes.

**Not a world model.** No neural simulation. No billion-dollar compute. AWM uses Bayesian statistics, multi-armed bandits, and historical pattern matching to make pipelines smarter over time.

---

## What It Does

```
Without AWM:
  Step 1 → Step 2 → Step 3 (rejected) → Step 3 again → Step 4 → Done
  Cost: $0.85 | Time: 6 min | Revisions: 1

With AWM:
  AWM: "Step 3 gets rejected 73% of the time for this client. Pre-injecting constraints."
  AWM: "Step 4 output unchanged for this profile 95% of the time. Using cheaper model."
  Step 1 → Step 2 → Step 3 (passes first try) → Step 4 (cheap model) → Done
  Cost: $0.41 | Time: 4 min | Revisions: 0
```

### Core capabilities

- **Outcome prediction** — Before each pipeline step, predict pass/revise/fail probability based on similar past runs
- **Model routing** — Thompson Sampling (multi-armed bandit) picks the cheapest model likely to succeed
- **Constraint pre-injection** — Extract patterns from past approval gate rejections, inject them as constraints into upcoming steps
- **Step skipping** — When prediction confidence is high and output is expected unchanged, skip or reuse
- **Closed-loop learning** — Every run's outcomes feed back into predictions. AWM gets smarter with use.

---

## Architecture

AWM is three packages. Use what you need.

```
@heybeaux/awm-core          Zero-dependency prediction engine
@heybeaux/awm-engram        Storage adapter for Engram memory API
@heybeaux/awm-mastra        Middleware for Mastra/Forge pipelines
```

```
┌─────────────────────────────────────┐
│         Pipeline Runner              │
│  (Forge, LangGraph, custom, etc.)   │
│                                      │
│  beforeStep() → AWM.predict()       │
│  afterStep()  → AWM.record()        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          AWM Core                    │
│                                      │
│  Oracle        → predict() / recommend()
│  Beliefs       → Bayesian Beta distributions
│  Bandits       → Thompson Sampling router
│  Constraints   → Pattern extraction
│  Traces        → Structured outcome schema
└──────────────┬──────────────────────┘
               │ (optional)
┌──────────────▼──────────────────────┐
│     Storage Backend                  │
│  Engram API  |  SQLite  |  Custom   │
└─────────────────────────────────────┘
```

### Works standalone

AWM Core has zero dependencies. Bring your own storage:

```typescript
import { Oracle, InMemoryStore } from '@heybeaux/awm-core';

const oracle = new Oracle({ store: new InMemoryStore() });

// Record outcomes
oracle.record({
  stepType: 'creative-director',
  profileSlug: 'acme-nonprofit',
  model: 'claude-sonnet-4',
  passed: true,
  revised: false,
  cost: 0.12,
  latency: 8500,
});

// Get predictions
const prediction = oracle.predict({
  stepType: 'creative-director',
  profileSlug: 'acme-nonprofit',
  availableModels: ['claude-sonnet-4', 'gemini-2.5-flash', 'claude-opus-4'],
});

// prediction = {
//   outcome: 'pass',
//   confidence: 0.82,
//   suggestedModel: 'gemini-2.5-flash',  // cheaper, high historical success
//   skipRecommendation: false,
//   constraints: ['Include impact statistics', 'Avoid guilt-based language'],
//   reasoning: 'Based on 47 similar runs. Sonnet passes 91% for this profile.',
//   historicalBasis: 47,
// }
```

### With Engram

```typescript
import { Oracle } from '@heybeaux/awm-core';
import { EngramStore } from '@heybeaux/awm-engram';

const oracle = new Oracle({
  store: new EngramStore({
    apiUrl: 'https://api.openengram.ai',
    apiKey: process.env.ENGRAM_API_KEY,
    namespace: 'awm',
  }),
});
```

### With Forge/Mastra

```typescript
import { awmMiddleware } from '@heybeaux/awm-mastra';

// Add to your Mastra workflow
const workflow = createWorkflow({ ... })
  .use(awmMiddleware({ oracle, mode: 'active' }))  // or 'shadow' for logging only
  .then(strategist)
  .then(creative)
  .then(redTeam)
  .commit();
```

---

## How It Works

### Bayesian Beliefs

AWM maintains a [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) for each (step_type, profile) pair. Every outcome updates the distribution:

```
Prior:   Beta(α=2, β=2)     → 50% expected success (no data)
After 8 successes, 2 failures:
Updated: Beta(α=10, β=4)    → 71% expected success
After 50 successes, 5 failures:
Updated: Beta(α=52, β=7)    → 88% expected success, high confidence
```

Confidence increases with more data. Predictions with low confidence trigger conservative behavior (use best model, don't skip).

### Thompson Sampling (Model Routing)

For each model available at a step, AWM maintains a reward distribution. At decision time, it samples from each distribution and picks the model with the highest sample. This naturally balances:

- **Exploitation:** Use the model that historically works best
- **Exploration:** Occasionally try cheaper models to discover if they're good enough

### Constraint Pre-Injection

AWM analyzes past approval gate rejections:

```
Pattern detected: 73% of 'creative-director' rejections for nonprofit clients
mention "missing impact statistics" in revision feedback.

Action: Pre-inject constraint "Include at least 2 verified impact statistics"
into the creative-director step prompt.
```

This turns reactive revision cycles into proactive constraint satisfaction.

---

## Trace Schema

AWM defines a standard format for recording pipeline execution outcomes:

```typescript
interface StepTrace {
  // Identity
  traceId: string;
  runId: string;
  workflowType: string;
  stepType: string;
  stepIndex: number;

  // Context
  profileSlug?: string;
  sector?: string;
  model: string;
  inputFingerprint: string;

  // ACR context (optional)
  acrCapabilities?: string[];
  acrCoverageRatio?: number;
  acrLodLevel?: string;
  acrMissingCapabilities?: string[];

  // Outcome
  passed: boolean;
  revised: boolean;
  revisionReason?: string;
  approvalResult?: 'approved' | 'rejected' | 'revised' | 'auto-approved';
  outputFingerprint?: string;
  outputSimilarityToPrior?: number;

  // Cost
  tokensIn: number;
  tokensOut: number;
  cost: number;
  latencyMs: number;

  // AWM predictions (if AWM was active)
  awmPrediction?: StepPrediction;
  awmWasCorrect?: boolean;

  // Metadata
  timestamp: string;
}
```

This schema is designed to be **portable** — any pipeline runner can produce traces in this format, and any AWM-compatible prediction engine can consume them.

---

## Part of the Stack

AWM integrates with but does not depend on:

| Project | Role | Integration |
|---------|------|-------------|
| [ACR](https://github.com/heybeaux/acr) | Capability management | ACR coverage ratio as prediction feature |
| [Engram](https://github.com/heybeaux/engram) | Agent memory | Storage backend + similar run retrieval |
| [Forge](https://github.com/beaux-riel/forge) | Pipeline platform | Primary consumer via Mastra middleware |

```
ACR: what agents CAN do (capability management)
AWM: what agents WILL do (outcome prediction)
Engram: what agents DID (persistent memory)
```

---

## Roadmap

### Phase 1: Core Engine (current)
- [x] Trace schema definition
- [ ] Bayesian belief tracking
- [ ] Thompson Sampling model router
- [ ] Constraint pattern extraction
- [ ] Oracle predict/recommend API
- [ ] In-memory store
- [ ] Test suite

### Phase 2: Integration
- [ ] Engram storage adapter
- [ ] Mastra/Forge middleware
- [ ] Shadow mode (log predictions, don't act)
- [ ] Active mode (modify pipeline execution)

### Phase 3: Intelligence
- [ ] Step-skipping recommendations
- [ ] Parallel branching for uncertain steps
- [ ] Dashboard metrics API
- [ ] Prediction accuracy self-evaluation

### Phase 4: Spec Extraction
- [ ] Formalize trace schema as open standard
- [ ] Publish prediction contract spec
- [ ] Outcome taxonomy specification

---

## Getting Started

```bash
npm install @heybeaux/awm-core
```

See [examples/standalone](./examples/standalone) for a complete example without any external dependencies.

---

## Contributing

AWM is early-stage. The best way to help:

1. **Try it** — integrate with your pipeline runner, report what works
2. **Trace data** — share anonymized pipeline traces to improve prediction models
3. **Adapters** — build storage adapters for other memory systems
4. **Middleware** — build integration middleware for other orchestration frameworks

---

## License

MIT — built by [heybeaux](https://github.com/heybeaux) with ☁️ Cirrus
