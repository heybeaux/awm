/**
 * Constraint Pattern Extractor
 *
 * Analyzes revision reasons from approval gates to extract
 * recurring patterns. These patterns become constraints that
 * are pre-injected into step prompts to prevent known failures.
 */

import type { ConstraintPattern, StepTrace, AWMStore } from './types.js';

/** Minimum occurrences before a pattern becomes actionable */
const MIN_OCCURRENCES = 2;

/** Minimum frequency (0-1) relative to total runs for this step/profile */
const MIN_FREQUENCY = 0.08;

export class ConstraintExtractor {
  constructor(private store: AWMStore) {}

  /**
   * Get applicable constraints for a step about to execute.
   * Returns constraints sorted by frequency (most common first).
   */
  async getConstraints(stepType: string, profileSlug?: string): Promise<string[]> {
    // Get patterns for specific profile AND global
    const patterns = await this.store.getPatterns(stepType, profileSlug);

    return patterns
      .filter((p) => p.occurrences >= MIN_OCCURRENCES && p.frequency >= MIN_FREQUENCY)
      .sort((a, b) => b.frequency - a.frequency)
      .map((p) => p.constraint);
  }

  /**
   * Force a constraint analysis pass for all step/profile combinations
   * in the store. Useful after warmup to ensure patterns are extracted.
   */
  async analyzeAll(stepTypes: string[], profileSlugs: string[]): Promise<ConstraintPattern[]> {
    const allPatterns: ConstraintPattern[] = [];
    for (const stepType of stepTypes) {
      for (const slug of profileSlugs) {
        const patterns = await this.analyzeRevisions(stepType, slug);
        allPatterns.push(...patterns);
      }
    }
    return allPatterns;
  }

  /**
   * Analyze recent traces to extract or update constraint patterns.
   * Call this after recording a revision outcome.
   */
  async analyzeRevisions(stepType: string, profileSlug?: string): Promise<ConstraintPattern[]> {
    const slug = profileSlug || '__global__';

    // Get all traces for this step/profile
    const traces = await this.store.queryTraces({
      stepType,
      profileSlug: slug !== '__global__' ? slug : undefined,
      limit: 200,
    });

    if (traces.length === 0) return [];

    // Count revision reasons
    const reasonCounts = new Map<string, { count: number; traces: StepTrace[] }>();
    const revisedTraces = traces.filter((t) => t.revised && t.revisionReason);

    for (const trace of revisedTraces) {
      const reason = normalizeReason(trace.revisionReason!);
      const existing = reasonCounts.get(reason) || { count: 0, traces: [] };
      existing.count += 1;
      existing.traces.push(trace);
      reasonCounts.set(reason, existing);
    }

    // Convert to constraint patterns
    const patterns: ConstraintPattern[] = [];
    for (const [reason, { count, traces: reasonTraces }] of reasonCounts) {
      const frequency = count / traces.length;
      if (count >= MIN_OCCURRENCES && frequency >= MIN_FREQUENCY) {
        const constraint = reasonToConstraint(reason);
        const pattern: ConstraintPattern = {
          patternId: generatePatternId(stepType, slug, reason),
          stepType,
          profileSlug: slug,
          constraint,
          occurrences: count,
          frequency,
          sourceReasons: reasonTraces.map((t) => t.revisionReason!),
        };
        await this.store.setPattern(pattern);
        patterns.push(pattern);
      }
    }

    return patterns;
  }
}

/**
 * Normalize a revision reason for pattern matching.
 * Lowercases, trims, removes common prefixes.
 */
function normalizeReason(reason: string): string {
  return reason
    .toLowerCase()
    .trim()
    .replace(/^(please |need to |should |must |missing |lacks? |no )/i, '')
    .replace(/[.!?]+$/, '')
    .trim();
}

/**
 * Convert a revision reason into an actionable constraint.
 * Transforms "missing impact statistics" → "Include impact statistics in the output"
 */
function reasonToConstraint(reason: string): string {
  // Common patterns → constraint templates
  const patterns: [RegExp, string][] = [
    [/missing (.+)/, 'Include $1 in the output'],
    [/lack(?:s|ing)? (.+)/, 'Ensure $1 is present'],
    [/too (.+)/, 'Avoid being too $1'],
    [/not enough (.+)/, 'Provide sufficient $1'],
    [/(.+) not (?:included|present|mentioned)/, 'Include $1'],
    [/wrong (.+)/, 'Verify $1 is correct'],
    [/inappropriate (.+)/, 'Ensure $1 is appropriate for the audience'],
  ];

  for (const [regex, template] of patterns) {
    const match = reason.match(regex);
    if (match) {
      return reason.replace(regex, template);
    }
  }

  // Fallback: prefix with "Ensure" or "Include"
  if (reason.startsWith('include') || reason.startsWith('add')) {
    return reason.charAt(0).toUpperCase() + reason.slice(1);
  }

  return `Ensure: ${reason}`;
}

/**
 * Generate a deterministic pattern ID for dedup.
 */
function generatePatternId(stepType: string, profileSlug: string, reason: string): string {
  const input = `${stepType}:${profileSlug}:${reason}`;
  // Simple hash — not crypto-grade, just for dedup
  let hash = 0;
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return `cp_${Math.abs(hash).toString(36)}`;
}
