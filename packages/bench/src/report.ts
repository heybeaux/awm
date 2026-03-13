/**
 * Benchmark Report Formatter
 *
 * Produces human-readable and machine-readable reports
 * from benchmark results.
 */

import type { BenchmarkReport, ScenarioResult, StandardResult } from './types.js';
import { STANDARDS } from './types.js';

export function formatReport(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push('');
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('  AWM Benchmark Report');
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('');
  lines.push(`  Version:    ${report.awmVersion}`);
  lines.push(`  Timestamp:  ${report.timestamp}`);
  lines.push(`  Duration:   ${(report.durationMs / 1000).toFixed(1)}s`);
  lines.push(`  Grade:      ${gradeEmoji(report.overallGrade)} ${report.overallGrade}`);
  lines.push('');
  lines.push(`  Scenarios:  ${report.passed}/${report.totalScenarios} passed`);
  lines.push('');

  // ─── Standards Summary ───
  lines.push('───────────────────────────────────────────────────────────────');
  lines.push('  Standards Summary');
  lines.push('───────────────────────────────────────────────────────────────');
  lines.push('');

  for (const [sid, summary] of Object.entries(report.standardsSummary)) {
    const standard = STANDARDS[sid as keyof typeof STANDARDS];
    if (!standard) continue;
    const total = summary.passed + summary.failed;
    const icon = summary.failed === 0 ? '✅' : summary.passed > 0 ? '⚠️' : '❌';
    const avgDisplay = standard.unit === '%'
      ? `${(summary.avgValue * 100).toFixed(1)}%`
      : standard.unit === 'runs'
        ? `${summary.avgValue.toFixed(0)} runs`
        : `${summary.avgValue.toFixed(4)}`;

    lines.push(`  ${icon} ${standard.name}`);
    lines.push(`     ${summary.passed}/${total} passed | avg: ${avgDisplay} | threshold: ${formatThreshold(standard)}`);
    lines.push('');
  }

  // ─── Scenario Details ───
  lines.push('───────────────────────────────────────────────────────────────');
  lines.push('  Scenario Results');
  lines.push('───────────────────────────────────────────────────────────────');

  for (const scenario of report.scenarios) {
    lines.push('');
    const icon = scenario.passed ? '✅' : '❌';
    lines.push(`  ${icon} ${scenario.scenario.name} [${scenario.scenario.difficulty}]`);
    lines.push(`     ${scenario.scenario.description}`);
    lines.push(`     Duration: ${scenario.durationMs}ms | Warmup: ${scenario.warmupRuns.length} | Eval: ${scenario.evalRuns.length} runs`);
    lines.push('');

    for (const std of scenario.standards) {
      const stdIcon = std.passed ? '  ✓' : '  ✗';
      lines.push(`     ${stdIcon} ${std.standard.name}: ${std.detail}`);
    }
  }

  // ─── Footer ───
  lines.push('');
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push(`  Overall: ${report.overallGrade} | ${report.passed}/${report.totalScenarios} scenarios | ${(report.durationMs / 1000).toFixed(1)}s`);
  lines.push('═══════════════════════════════════════════════════════════════');
  lines.push('');

  return lines.join('\n');
}

export function formatJSON(report: BenchmarkReport): string {
  // Strip raw run data for compact JSON output
  const compact = {
    ...report,
    scenarios: report.scenarios.map((s) => ({
      id: s.scenario.id,
      name: s.scenario.name,
      difficulty: s.scenario.difficulty,
      passed: s.passed,
      durationMs: s.durationMs,
      standards: s.standards.map((std) => ({
        id: std.standard.id,
        name: std.standard.name,
        value: std.value,
        threshold: std.standard.threshold,
        passed: std.passed,
        detail: std.detail,
      })),
    })),
  };
  return JSON.stringify(compact, null, 2);
}

function gradeEmoji(grade: string): string {
  switch (grade) {
    case 'A': return '🏆';
    case 'B': return '🥈';
    case 'C': return '🥉';
    case 'D': return '⚠️';
    case 'F': return '❌';
    default: return '❓';
  }
}

function formatThreshold(standard: { threshold: number; unit: string; higherIsBetter: boolean }): string {
  const dir = standard.higherIsBetter ? '≥' : '≤';
  if (standard.unit === '%') return `${dir} ${(standard.threshold * 100).toFixed(0)}%`;
  if (standard.unit === 'runs') return `${dir} ${standard.threshold} runs`;
  return `${dir} ${standard.threshold}`;
}
