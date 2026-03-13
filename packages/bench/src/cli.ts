#!/usr/bin/env node
/**
 * AWM Benchmark CLI
 *
 * Usage:
 *   npx awm-bench                    # run all scenarios
 *   npx awm-bench --scenario basic   # run specific scenario
 *   npx awm-bench --json             # output JSON report
 *   npx awm-bench --verbose          # show progress
 */

import { BenchmarkRunner } from './runner.js';
import { SCENARIOS, getScenario } from './scenarios/index.js';
import { formatReport, formatJSON } from './report.js';

async function main() {
  const args = process.argv.slice(2);
  const jsonOutput = args.includes('--json');
  const verbose = args.includes('--verbose') || !jsonOutput;

  // Filter scenarios
  const scenarioArg = args.find((_a: string, i: number) => args[i - 1] === '--scenario');
  let scenarios = SCENARIOS;

  if (scenarioArg) {
    const scenario = getScenario(scenarioArg)!;
    if (!scenario) {
      console.error(`Unknown scenario: ${scenarioArg}`);
      console.error(`Available: ${SCENARIOS.map((s) => s.id).join(', ')}`);
      process.exit(1);
    }
    scenarios = [scenario];
  }

  if (verbose) {
    console.log('');
    console.log('  AWM Benchmark Suite v0.1.0');
    console.log(`  Running ${scenarios.length} scenario(s)...`);
  }

  const runner = new BenchmarkRunner({ verbose });
  const report = await runner.runAll(scenarios);

  if (jsonOutput) {
    console.log(formatJSON(report));
  } else {
    console.log(formatReport(report));
  }

  process.exit(report.overallGrade === 'F' ? 1 : 0);
}

main().catch((err) => {
  console.error('Benchmark failed:', err);
  process.exit(1);
});
