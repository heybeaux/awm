/**
 * AWM Benchmark Scenarios
 *
 * Each scenario tests specific AWM capabilities under
 * realistic conditions. Scenarios are deterministic —
 * same seed always produces same results.
 */

import type { BenchScenario, StepConfig, ProfileConfig, ModelConfig, PipelineConfig } from '../types.js';

// ─── Shared Pipeline Components ──────────────────────────

const STANDARD_MODELS: ModelConfig[] = [
  { name: 'opus',   costPerStep: 0.45, latencyMs: 5000, qualityMultiplier: 1.15 },
  { name: 'sonnet', costPerStep: 0.15, latencyMs: 3000, qualityMultiplier: 1.0 },
  { name: 'flash',  costPerStep: 0.04, latencyMs: 1500, qualityMultiplier: 0.85 },
  { name: 'haiku',  costPerStep: 0.01, latencyMs: 800,  qualityMultiplier: 0.70 },
];

const STANDARD_STEPS: StepConfig[] = [
  {
    stepType: 'strategist',
    basePassRate: 0.90,
    baseCost: 1.0,
    baseLatency: 4000,
    profileSensitivity: 0.3,
    modelSensitivity: 0.2,
    revisionReasons: [
      { reason: 'Strategy lacks specificity', probability: 0.4 },
      { reason: 'Wrong audience segment targeted', probability: 0.3 },
      { reason: 'Missing competitive analysis', probability: 0.3 },
    ],
  },
  {
    stepType: 'creative-director',
    basePassRate: 0.72,
    baseCost: 1.0,
    baseLatency: 3500,
    profileSensitivity: 0.5,
    modelSensitivity: 0.3,
    revisionReasons: [
      { reason: 'Missing impact statistics', probability: 0.30 },
      { reason: 'Tone inappropriate for audience', probability: 0.25 },
      { reason: 'Brand voice mismatch', probability: 0.20 },
      { reason: 'Copy too long for format', probability: 0.15 },
      { reason: 'Call to action unclear', probability: 0.10 },
    ],
  },
  {
    stepType: 'red-team',
    basePassRate: 0.85,
    baseCost: 1.0,
    baseLatency: 3000,
    profileSensitivity: 0.4,
    modelSensitivity: 0.15,
    revisionReasons: [
      { reason: 'Compliance violation detected', probability: 0.35 },
      { reason: 'Cultural sensitivity issue', probability: 0.30 },
      { reason: 'Factual accuracy concern', probability: 0.20 },
      { reason: 'Legal review required', probability: 0.15 },
    ],
  },
  {
    stepType: 'visual-creative',
    basePassRate: 0.82,
    baseCost: 0.8,
    baseLatency: 2500,
    profileSensitivity: 0.3,
    modelSensitivity: 0.25,
    revisionReasons: [
      { reason: 'Brand colors not applied correctly', probability: 0.30 },
      { reason: 'Image composition poor', probability: 0.25 },
      { reason: 'Accessibility contrast failure', probability: 0.25 },
      { reason: 'Wrong aspect ratio for platform', probability: 0.20 },
    ],
  },
];

const STANDARD_PROFILES: ProfileConfig[] = [
  {
    slug: 'nonprofit-a',
    sector: 'nonprofit',
    stepModifiers: {
      'strategist': 0.05,
      'creative-director': -0.15,  // harder — strict brand voice
      'red-team': -0.10,           // more compliance issues
      'visual-creative': 0.0,
    },
    extraRevisionReasons: [
      { stepType: 'creative-director', reason: 'Guilt-based language detected', probability: 0.25 },
      { stepType: 'red-team', reason: 'Theological sensitivity issue', probability: 0.20 },
    ],
  },
  {
    slug: 'tech-consulting',
    sector: 'technology',
    stepModifiers: {
      'strategist': 0.0,
      'creative-director': 0.10,   // easier — more flexible brand
      'red-team': 0.05,
      'visual-creative': 0.05,
    },
  },
  {
    slug: 'healthcare',
    sector: 'healthcare',
    stepModifiers: {
      'strategist': -0.05,
      'creative-director': -0.10,
      'red-team': -0.20,           // very strict compliance
      'visual-creative': -0.05,
    },
    extraRevisionReasons: [
      { stepType: 'red-team', reason: 'HIPAA compliance concern', probability: 0.30 },
      { stepType: 'creative-director', reason: 'Medical claim not substantiated', probability: 0.20 },
    ],
  },
];

const STANDARD_PIPELINE: PipelineConfig = {
  workflowType: 'creative-campaign',
  steps: STANDARD_STEPS,
  profiles: STANDARD_PROFILES,
  models: STANDARD_MODELS,
};

// ─── Scenarios ────────────────────────────────────────────

export const SCENARIOS: BenchScenario[] = [
  // ── S1: Basic Prediction Accuracy ──
  {
    id: 'basic-prediction',
    name: 'Basic Prediction Accuracy',
    description: 'Can AWM learn pass/fail patterns from consistent pipeline behavior? Single profile, stable environment.',
    difficulty: 'easy',
    category: 'prediction_accuracy',
    pipeline: {
      ...STANDARD_PIPELINE,
      profiles: [STANDARD_PROFILES[0]], // single profile
    },
    warmupRuns: 30,
    evalRuns: 50,
    seed: 42,
    standards: ['S1_PREDICTION_ACCURACY', 'S4_CONVERGENCE_SPEED'],
  },

  // ── S2: Cost Optimization via Model Routing ──
  {
    id: 'model-routing',
    name: 'Model Routing Efficiency',
    description: 'Does Thompson Sampling find cheaper models that still pass? Measures cost reduction vs always using the most expensive model.',
    difficulty: 'medium',
    category: 'model_routing',
    pipeline: STANDARD_PIPELINE,
    warmupRuns: 40,
    evalRuns: 60,
    seed: 137,
    standards: ['S2_COST_REDUCTION', 'S5_MODEL_ROUTING_EFFICIENCY'],
  },

  // ── S3: Constraint Learning ──
  {
    id: 'constraint-learning',
    name: 'Constraint Learning & Injection',
    description: 'Does AWM extract revision patterns and pre-inject constraints that reduce revisions? Measures revision rate before and after constraints are active.',
    difficulty: 'medium',
    category: 'constraint_learning',
    pipeline: {
      ...STANDARD_PIPELINE,
      profiles: [STANDARD_PROFILES[0]], // nonprofit with strict requirements
    },
    warmupRuns: 40,
    evalRuns: 60,
    seed: 256,
    standards: ['S3_REVISION_REDUCTION', 'S6_CONSTRAINT_EFFECTIVENESS'],
  },

  // ── S4: Cold Start Recovery ──
  {
    id: 'cold-start',
    name: 'Cold Start Recovery',
    description: 'How quickly does AWM learn a new profile from zero data? Measures runs until predictions beat random baseline.',
    difficulty: 'medium',
    category: 'cold_start',
    pipeline: {
      ...STANDARD_PIPELINE,
      profiles: [STANDARD_PROFILES[2]], // healthcare — challenging
    },
    warmupRuns: 0, // no warmup — this IS the test
    evalRuns: 50,
    seed: 512,
    standards: ['S7_COLD_START_RECOVERY', 'S4_CONVERGENCE_SPEED'],
  },

  // ── S5: Multi-Profile Isolation ──
  {
    id: 'profile-isolation',
    name: 'Multi-Profile Isolation',
    description: 'Does AWM maintain independent beliefs per profile? Predictions for Profile A should not be contaminated by Profile B data.',
    difficulty: 'hard',
    category: 'multi_profile',
    pipeline: STANDARD_PIPELINE, // all 3 profiles
    warmupRuns: 30,
    evalRuns: 90, // 30 per profile
    seed: 1024,
    standards: ['S8_PROFILE_ISOLATION', 'S1_PREDICTION_ACCURACY'],
  },

  // ── S6: Adversarial Distribution Shift ──
  {
    id: 'distribution-shift',
    name: 'Adversarial Distribution Shift',
    description: 'Pipeline behavior changes abruptly mid-evaluation. A step that used to pass 90% now passes 40%. Can AWM adapt?',
    difficulty: 'adversarial',
    category: 'adversarial',
    pipeline: {
      ...STANDARD_PIPELINE,
      profiles: [STANDARD_PROFILES[0]],
    },
    warmupRuns: 40,
    evalRuns: 80, // shift happens at run 40
    seed: 2048,
    standards: ['S9_ADVERSARIAL_ROBUSTNESS'],
  },

  // ── S7: Prediction Calibration ──
  {
    id: 'calibration',
    name: 'Prediction Calibration',
    description: 'Are AWM confidence scores well-calibrated? When AWM says 80% confidence, does the prediction succeed ~80% of the time?',
    difficulty: 'hard',
    category: 'prediction_accuracy',
    pipeline: STANDARD_PIPELINE,
    warmupRuns: 50,
    evalRuns: 100,
    seed: 4096,
    standards: ['S10_CALIBRATION', 'S1_PREDICTION_ACCURACY'],
  },

  // ── S8: End-to-End Full Pipeline ──
  {
    id: 'end-to-end',
    name: 'End-to-End Pipeline Optimization',
    description: 'Full evaluation: prediction accuracy, cost optimization, constraint learning, model routing, multi-profile — all at once.',
    difficulty: 'hard',
    category: 'end_to_end',
    pipeline: STANDARD_PIPELINE,
    warmupRuns: 50,
    evalRuns: 150,
    seed: 8192,
    standards: [
      'S1_PREDICTION_ACCURACY',
      'S2_COST_REDUCTION',
      'S3_REVISION_REDUCTION',
      'S5_MODEL_ROUTING_EFFICIENCY',
      'S10_CALIBRATION',
    ],
  },
];

export function getScenario(id: string): BenchScenario | undefined {
  return SCENARIOS.find((s) => s.id === id);
}

export function getScenariosByCategory(category: string): BenchScenario[] {
  return SCENARIOS.filter((s) => s.category === category);
}
