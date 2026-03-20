/**
 * AWM Autoresearch Meta-Optimizer
 *
 * Tunes AWM's hyperparameters against its own benchmark suite.
 * Uses the autoresearch pattern: mutate one thing, run benchmark,
 * keep improvements, revert failures.
 *
 * Tunable parameters:
 * - Belief priors (alpha, beta)
 * - Bandit priors (alpha, beta)
 * - Exploration decay (start, rate)
 * - Oracle thresholds (costWeight, skipConfidence, routingConfidence)
 *
 * Run: npx tsx packages/bench/src/autoresearch.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// We need to modify source files, run the benchmark, then restore.
// This is the autoresearch pattern applied to AWM itself.

interface AWMConfig {
  // Belief engine
  beliefAlpha: number;
  beliefBeta: number;
  // Bandit router
  banditAlpha: number;
  banditBeta: number;
  explorationDecayStart: number;
  explorationDecayRate: number;
  // Oracle
  costWeight: number;
  skipConfidenceThreshold: number;
  routingConfidenceThreshold: number;
}

const BASELINE_CONFIG: AWMConfig = {
  beliefAlpha: 1.5,
  beliefBeta: 1.5,
  banditAlpha: 1,
  banditBeta: 1,
  explorationDecayStart: 15,
  explorationDecayRate: 0.05,
  costWeight: 0.3,
  skipConfidenceThreshold: 0.85,
  routingConfidenceThreshold: 0.7,
};

// File paths
const BELIEFS_PATH = path.join(__dirname, '..', '..', 'core', 'src', 'beliefs.ts');
const BANDITS_PATH = path.join(__dirname, '..', '..', 'core', 'src', 'bandits.ts');
const ORACLE_PATH = path.join(__dirname, '..', '..', 'core', 'src', 'oracle.ts');

// ── Source File Manipulation ─────────────────────────────────────────

function readFile(p: string): string {
  return fs.readFileSync(p, 'utf-8');
}

function writeFile(p: string, content: string): void {
  fs.writeFileSync(p, content, 'utf-8');
}

// Store originals for restoration
const originalBeliefs = readFile(BELIEFS_PATH);
const originalBandits = readFile(BANDITS_PATH);
const originalOracle = readFile(ORACLE_PATH);

function applyConfig(config: AWMConfig): void {
  // Beliefs
  let beliefs = originalBeliefs;
  beliefs = beliefs.replace(
    /const DEFAULT_ALPHA = [\d.]+;/,
    `const DEFAULT_ALPHA = ${config.beliefAlpha};`
  );
  beliefs = beliefs.replace(
    /const DEFAULT_BETA = [\d.]+;/,
    `const DEFAULT_BETA = ${config.beliefBeta};`
  );
  writeFile(BELIEFS_PATH, beliefs);

  // Bandits
  let bandits = originalBandits;
  bandits = bandits.replace(
    /const DEFAULT_ALPHA = [\d.]+;/,
    `const DEFAULT_ALPHA = ${config.banditAlpha};`
  );
  bandits = bandits.replace(
    /const DEFAULT_BETA = [\d.]+;/,
    `const DEFAULT_BETA = ${config.banditBeta};`
  );
  bandits = bandits.replace(
    /const EXPLORATION_DECAY_START = [\d.]+;/,
    `const EXPLORATION_DECAY_START = ${config.explorationDecayStart};`
  );
  bandits = bandits.replace(
    /const EXPLORATION_DECAY_RATE = [\d.]+;/,
    `const EXPLORATION_DECAY_RATE = ${config.explorationDecayRate};`
  );
  writeFile(BANDITS_PATH, bandits);

  // Oracle defaults
  let oracle = originalOracle;
  oracle = oracle.replace(
    /costWeight: [\d.]+,/,
    `costWeight: ${config.costWeight},`
  );
  oracle = oracle.replace(
    /skipConfidenceThreshold: [\d.]+,/,
    `skipConfidenceThreshold: ${config.skipConfidenceThreshold},`
  );
  oracle = oracle.replace(
    /routingConfidenceThreshold: [\d.]+,/,
    `routingConfidenceThreshold: ${config.routingConfidenceThreshold},`
  );
  writeFile(ORACLE_PATH, oracle);
}

function restoreOriginals(): void {
  writeFile(BELIEFS_PATH, originalBeliefs);
  writeFile(BANDITS_PATH, originalBandits);
  writeFile(ORACLE_PATH, originalOracle);
}

// ── Benchmark Execution ──────────────────────────────────────────────

async function runBenchmark(): Promise<{
  grade: string;
  passed: number;
  total: number;
  scenarioDetails: { name: string; passed: boolean; standards: { id: string; passed: boolean; value: number; detail: string }[] }[];
}> {
  // Fork a child process to run the benchmark with fresh module loading
  // This ensures modified source files are picked up
  const { execSync } = await import('child_process');
  const benchScript = path.join(__dirname, 'run-bench.ts');
  const resultFile = path.join(__dirname, '..', '..', '..', 'autoresearch-awm', 'bench-result.json');

  // Create a small script that runs the benchmark and writes results
  fs.writeFileSync(benchScript, `
import { BenchmarkRunner } from './runner.js';
import { SCENARIOS } from './scenarios/index.js';
import * as fs from 'fs';

const runner = new BenchmarkRunner({ verbose: false });
runner.runAll(SCENARIOS).then(report => {
  const result = {
    grade: report.overallGrade,
    passed: report.passed,
    total: report.totalScenarios,
    scenarioDetails: report.scenarios.map(s => ({
      name: s.scenario.name,
      passed: s.passed,
      standards: s.standards.map(st => ({
        id: st.standard.id,
        passed: st.passed,
        value: st.value,
        detail: st.detail,
      })),
    })),
  };
  fs.writeFileSync('${resultFile.replace(/\\/g, '\\\\')}', JSON.stringify(result));
  process.exit(0);
}).catch(e => {
  console.error(e);
  process.exit(1);
});
`);

  try {
    execSync(`npx tsx ${benchScript}`, {
      cwd: path.join(__dirname),
      timeout: 60000,
      stdio: 'pipe',
    });
    const result = JSON.parse(fs.readFileSync(resultFile, 'utf-8'));
    return result;
  } finally {
    try { fs.unlinkSync(benchScript); } catch {}
  }
}

// ── Mutations ────────────────────────────────────────────────────────

interface Mutation {
  name: string;
  description: string;
  apply: (config: AWMConfig) => AWMConfig;
}

function generateMutations(config: AWMConfig): Mutation[] {
  return [
    // Belief priors
    { name: 'belief_alpha_up', description: `Belief prior alpha: ${config.beliefAlpha} → ${config.beliefAlpha + 0.5} (stronger prior)`,
      apply: c => ({ ...c, beliefAlpha: c.beliefAlpha + 0.5 }) },
    { name: 'belief_alpha_down', description: `Belief prior alpha: ${config.beliefAlpha} → ${Math.max(0.5, config.beliefAlpha - 0.5)} (weaker prior)`,
      apply: c => ({ ...c, beliefAlpha: Math.max(0.5, c.beliefAlpha - 0.5) }) },
    { name: 'belief_beta_up', description: `Belief prior beta: ${config.beliefBeta} → ${config.beliefBeta + 0.5}`,
      apply: c => ({ ...c, beliefBeta: c.beliefBeta + 0.5 }) },
    { name: 'belief_beta_down', description: `Belief prior beta: ${config.beliefBeta} → ${Math.max(0.5, config.beliefBeta - 0.5)}`,
      apply: c => ({ ...c, beliefBeta: Math.max(0.5, c.beliefBeta - 0.5) }) },

    // Bandit priors
    { name: 'bandit_alpha_up', description: `Bandit prior alpha: ${config.banditAlpha} → ${config.banditAlpha + 0.5}`,
      apply: c => ({ ...c, banditAlpha: c.banditAlpha + 0.5 }) },
    { name: 'bandit_beta_up', description: `Bandit prior beta: ${config.banditBeta} → ${config.banditBeta + 0.5}`,
      apply: c => ({ ...c, banditBeta: c.banditBeta + 0.5 }) },

    // Exploration
    { name: 'explore_start_up', description: `Exploration decay start: ${config.explorationDecayStart} → ${config.explorationDecayStart + 5} (explore longer)`,
      apply: c => ({ ...c, explorationDecayStart: c.explorationDecayStart + 5 }) },
    { name: 'explore_start_down', description: `Exploration decay start: ${config.explorationDecayStart} → ${Math.max(5, config.explorationDecayStart - 5)} (exploit sooner)`,
      apply: c => ({ ...c, explorationDecayStart: Math.max(5, c.explorationDecayStart - 5) }) },
    { name: 'explore_rate_up', description: `Exploration decay rate: ${config.explorationDecayRate} → ${config.explorationDecayRate + 0.02} (faster decay)`,
      apply: c => ({ ...c, explorationDecayRate: c.explorationDecayRate + 0.02 }) },
    { name: 'explore_rate_down', description: `Exploration decay rate: ${config.explorationDecayRate} → ${Math.max(0.01, config.explorationDecayRate - 0.02)} (slower decay)`,
      apply: c => ({ ...c, explorationDecayRate: Math.max(0.01, c.explorationDecayRate - 0.02) }) },

    // Oracle thresholds
    { name: 'cost_weight_up', description: `Cost weight: ${config.costWeight} → ${Math.min(0.8, config.costWeight + 0.1)} (more cost-sensitive)`,
      apply: c => ({ ...c, costWeight: Math.min(0.8, c.costWeight + 0.1) }) },
    { name: 'cost_weight_down', description: `Cost weight: ${config.costWeight} → ${Math.max(0.0, config.costWeight - 0.1)} (more quality-focused)`,
      apply: c => ({ ...c, costWeight: Math.max(0.0, c.costWeight - 0.1) }) },
    { name: 'skip_threshold_up', description: `Skip confidence: ${config.skipConfidenceThreshold} → ${Math.min(0.95, config.skipConfidenceThreshold + 0.05)} (more conservative skipping)`,
      apply: c => ({ ...c, skipConfidenceThreshold: Math.min(0.95, c.skipConfidenceThreshold + 0.05) }) },
    { name: 'skip_threshold_down', description: `Skip confidence: ${config.skipConfidenceThreshold} → ${Math.max(0.70, config.skipConfidenceThreshold - 0.05)} (more aggressive skipping)`,
      apply: c => ({ ...c, skipConfidenceThreshold: Math.max(0.70, c.skipConfidenceThreshold - 0.05) }) },
    { name: 'routing_threshold_up', description: `Routing confidence: ${config.routingConfidenceThreshold} → ${Math.min(0.9, config.routingConfidenceThreshold + 0.05)}`,
      apply: c => ({ ...c, routingConfidenceThreshold: Math.min(0.9, c.routingConfidenceThreshold + 0.05) }) },
    { name: 'routing_threshold_down', description: `Routing confidence: ${config.routingConfidenceThreshold} → ${Math.max(0.5, config.routingConfidenceThreshold - 0.05)}`,
      apply: c => ({ ...c, routingConfidenceThreshold: Math.max(0.5, c.routingConfidenceThreshold - 0.05) }) },
  ];
}

// ── Grade to numeric score ───────────────────────────────────────────

function gradeToScore(grade: string, passed: number, total: number): number {
  // Primary: scenarios passed. Secondary: grade as tiebreaker.
  const gradeBonus: Record<string, number> = { A: 0.1, B: 0.05, C: 0, D: -0.05, F: -0.1 };
  return passed + (gradeBonus[grade] || 0);
}

// ── Main Loop ────────────────────────────────────────────────────────

async function main() {
  const outDir = path.join(__dirname, '..', '..', '..', 'autoresearch-awm');
  fs.mkdirSync(outDir, { recursive: true });

  const resultsPath = path.join(outDir, 'results.tsv');
  const changelogPath = path.join(outDir, 'changelog.md');

  fs.writeFileSync(resultsPath, 'experiment\tgrade\tpassed\ttotal\tscore\tstatus\tdescription\n');
  fs.writeFileSync(changelogPath, '# Autoresearch Changelog — AWM Hyperparameter Optimization\n\n');

  console.log('=== AWM Autoresearch Meta-Optimizer ===\n');

  // Baseline
  console.log('📊 Experiment 0: BASELINE');
  applyConfig(BASELINE_CONFIG);
  const baseline = await runBenchmark();
  const baseScore = gradeToScore(baseline.grade, baseline.passed, baseline.total);

  console.log(`   Grade: ${baseline.grade} (${baseline.passed}/${baseline.total})`);
  for (const s of baseline.scenarioDetails) {
    console.log(`   ${s.passed ? '✅' : '❌'} ${s.name}`);
    for (const st of s.standards.filter(st => !st.passed)) {
      console.log(`      ↳ ${st.id}: ${st.detail}`);
    }
  }

  fs.appendFileSync(resultsPath, `0\t${baseline.grade}\t${baseline.passed}\t${baseline.total}\t${baseScore.toFixed(2)}\tbaseline\toriginal config\n`);
  fs.appendFileSync(changelogPath, `## Experiment 0 — baseline\n**Grade:** ${baseline.grade} (${baseline.passed}/${baseline.total})\n**Score:** ${baseScore.toFixed(2)}\n\n`);

  let currentConfig = { ...BASELINE_CONFIG };
  let bestScore = baseScore;
  let bestGrade = baseline.grade;
  let bestPassed = baseline.passed;
  const MAX_EXPERIMENTS = 16;
  let consecutivePerfect = 0;

  const allMutations = generateMutations(currentConfig);
  
  for (let exp = 1; exp <= MAX_EXPERIMENTS; exp++) {
    // Regenerate mutations based on current config
    const mutations = generateMutations(currentConfig);
    const mutation = mutations[(exp - 1) % mutations.length];
    const candidateConfig = mutation.apply(currentConfig);

    console.log(`\n🔬 Experiment ${exp}: ${mutation.description}`);

    applyConfig(candidateConfig);
    
    let result;
    try {
      result = await runBenchmark();
    } catch (e) {
      console.log(`   💥 CRASH: ${e}`);
      fs.appendFileSync(resultsPath, `${exp}\t-\t0\t0\t0\tcrash\t${mutation.description}\n`);
      fs.appendFileSync(changelogPath, `## Experiment ${exp} — crash\n**Change:** ${mutation.description}\n**Error:** ${e}\n\n`);
      restoreOriginals();
      applyConfig(currentConfig);
      continue;
    }

    const score = gradeToScore(result.grade, result.passed, result.total);
    let status: string;

    if (score > bestScore) {
      status = 'keep';
      bestScore = score;
      bestGrade = result.grade;
      bestPassed = result.passed;
      currentConfig = { ...candidateConfig };
      consecutivePerfect = result.passed === result.total ? consecutivePerfect + 1 : 0;
      console.log(`   ✅ KEEP — Grade: ${result.grade} (${result.passed}/${result.total}) — IMPROVED`);
    } else if (score === bestScore && result.passed > bestPassed) {
      status = 'keep';
      bestPassed = result.passed;
      currentConfig = { ...candidateConfig };
      console.log(`   ✅ KEEP — Grade: ${result.grade} (${result.passed}/${result.total}) — same score, more scenarios`);
    } else {
      status = 'discard';
      consecutivePerfect = 0;
      // Restore previous config
      applyConfig(currentConfig);
      console.log(`   ❌ DISCARD — Grade: ${result.grade} (${result.passed}/${result.total})`);
    }

    // Show failing scenarios
    for (const s of result.scenarioDetails.filter(s => !s.passed)) {
      console.log(`      ❌ ${s.name}`);
      for (const st of s.standards.filter(st => !st.passed)) {
        console.log(`         ↳ ${st.id}: ${st.detail}`);
      }
    }

    fs.appendFileSync(resultsPath, `${exp}\t${result.grade}\t${result.passed}\t${result.total}\t${score.toFixed(2)}\t${status}\t${mutation.description}\n`);
    fs.appendFileSync(changelogPath, `## Experiment ${exp} — ${status}\n**Grade:** ${result.grade} (${result.passed}/${result.total})\n**Score:** ${score.toFixed(2)}\n**Change:** ${mutation.description}\n**Config:** beliefA=${candidateConfig.beliefAlpha}, beliefB=${candidateConfig.beliefBeta}, banditA=${candidateConfig.banditAlpha}, banditB=${candidateConfig.banditBeta}, exploreStart=${candidateConfig.explorationDecayStart}, exploreRate=${candidateConfig.explorationDecayRate}, costW=${candidateConfig.costWeight}, skipConf=${candidateConfig.skipConfidenceThreshold}, routeConf=${candidateConfig.routingConfidenceThreshold}\n\n`);

    if (consecutivePerfect >= 3) {
      console.log('\n🏆 Perfect score 3x in a row — stopping.');
      break;
    }
  }

  // Restore originals before applying best config
  restoreOriginals();

  console.log('\n\n=== AUTORESEARCH COMPLETE ===\n');
  console.log(`Baseline: Grade ${baseline.grade} (${baseline.passed}/${baseline.total})`);
  console.log(`Best:     Grade ${bestGrade} (${bestPassed}/${baseline.total})`);
  console.log(`\nOptimal config:`);
  console.log(`  beliefAlpha = ${currentConfig.beliefAlpha}`);
  console.log(`  beliefBeta = ${currentConfig.beliefBeta}`);
  console.log(`  banditAlpha = ${currentConfig.banditAlpha}`);
  console.log(`  banditBeta = ${currentConfig.banditBeta}`);
  console.log(`  explorationDecayStart = ${currentConfig.explorationDecayStart}`);
  console.log(`  explorationDecayRate = ${currentConfig.explorationDecayRate}`);
  console.log(`  costWeight = ${currentConfig.costWeight}`);
  console.log(`  skipConfidenceThreshold = ${currentConfig.skipConfidenceThreshold}`);
  console.log(`  routingConfidenceThreshold = ${currentConfig.routingConfidenceThreshold}`);

  if (bestScore > baseScore) {
    console.log(`\n🔧 To apply these improvements, update the constants in:`);
    console.log(`   - packages/core/src/beliefs.ts (DEFAULT_ALPHA, DEFAULT_BETA)`);
    console.log(`   - packages/core/src/bandits.ts (DEFAULT_ALPHA, DEFAULT_BETA, EXPLORATION_DECAY_*)`);
    console.log(`   - packages/core/src/oracle.ts (costWeight, skipConfidenceThreshold, routingConfidenceThreshold)`);
  } else {
    console.log(`\nBaseline config is already optimal for current benchmark suite.`);
  }

  console.log(`\nResults: ${resultsPath}`);
  console.log(`Changelog: ${changelogPath}`);
}

main().catch((e) => {
  console.error('Fatal:', e);
  restoreOriginals();
  process.exit(1);
});

// Always restore on exit
process.on('exit', restoreOriginals);
process.on('SIGINT', () => { restoreOriginals(); process.exit(1); });
process.on('uncaughtException', (e) => { console.error(e); restoreOriginals(); process.exit(1); });
