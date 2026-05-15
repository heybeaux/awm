/**
 * SQLite-backed AWMStore for the equity benchmark.
 *
 * better-sqlite3 is synchronous; we wrap returns in Promise.resolve()
 * to satisfy the AWMStore async interface. This keeps the benchmark
 * deterministic (no async ordering) while remaining drop-in compatible
 * with the Oracle.
 */

import Database from 'better-sqlite3';
import type {
  AWMStore,
  StepTrace,
  StepBelief,
  ModelArm,
  ConstraintPattern,
  TraceQuery,
  ApprovalResult,
  StepPrediction,
} from '@heybeaux/awm-core';
import { ConfigValidationError } from '@heybeaux/awm-core';

export interface EquityStoreOptions {
  /** Path to the SQLite file. Use ':memory:' for tests. Default: ':memory:'. */
  dbPath?: string;
  /** If true, run PRAGMA journal_mode=WAL for the file-backed store. Default: true. */
  wal?: boolean;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  workflow_type TEXT NOT NULL,
  step_type TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  profile_slug TEXT,
  sector TEXT,
  model TEXT NOT NULL,
  input_fingerprint TEXT NOT NULL,
  acr_capabilities TEXT,
  acr_coverage_ratio REAL,
  acr_lod_level TEXT,
  acr_missing_capabilities TEXT,
  passed INTEGER NOT NULL,
  revised INTEGER NOT NULL,
  revision_reason TEXT,
  approval_result TEXT,
  output_fingerprint TEXT,
  output_similarity_to_prior REAL,
  tokens_in INTEGER NOT NULL,
  tokens_out INTEGER NOT NULL,
  cost REAL NOT NULL,
  latency_ms INTEGER NOT NULL,
  awm_prediction TEXT,
  awm_was_correct INTEGER,
  timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_step_profile ON traces(step_type, profile_slug);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);

CREATE TABLE IF NOT EXISTS beliefs (
  step_type TEXT NOT NULL,
  profile_slug TEXT NOT NULL,
  alpha REAL NOT NULL,
  beta REAL NOT NULL,
  observations INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (step_type, profile_slug)
);

CREATE TABLE IF NOT EXISTS arms (
  model TEXT NOT NULL,
  step_type TEXT NOT NULL,
  profile_slug TEXT NOT NULL,
  alpha REAL NOT NULL,
  beta REAL NOT NULL,
  pulls INTEGER NOT NULL,
  avg_cost REAL NOT NULL,
  avg_latency_ms REAL NOT NULL,
  PRIMARY KEY (model, step_type, profile_slug)
);

CREATE TABLE IF NOT EXISTS patterns (
  pattern_id TEXT PRIMARY KEY,
  step_type TEXT NOT NULL,
  profile_slug TEXT NOT NULL,
  constraint_text TEXT NOT NULL,
  occurrences INTEGER NOT NULL,
  frequency REAL NOT NULL,
  effectiveness_score REAL,
  source_reasons TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patterns_step_profile ON patterns(step_type, profile_slug);
`;

interface TraceRow {
  trace_id: string;
  run_id: string;
  workflow_type: string;
  step_type: string;
  step_index: number;
  profile_slug: string | null;
  sector: string | null;
  model: string;
  input_fingerprint: string;
  acr_capabilities: string | null;
  acr_coverage_ratio: number | null;
  acr_lod_level: string | null;
  acr_missing_capabilities: string | null;
  passed: number;
  revised: number;
  revision_reason: string | null;
  approval_result: string | null;
  output_fingerprint: string | null;
  output_similarity_to_prior: number | null;
  tokens_in: number;
  tokens_out: number;
  cost: number;
  latency_ms: number;
  awm_prediction: string | null;
  awm_was_correct: number | null;
  timestamp: string;
}

interface BeliefRow {
  step_type: string;
  profile_slug: string;
  alpha: number;
  beta: number;
  observations: number;
  updated_at: string;
}

interface ArmRow {
  model: string;
  step_type: string;
  profile_slug: string;
  alpha: number;
  beta: number;
  pulls: number;
  avg_cost: number;
  avg_latency_ms: number;
}

interface PatternRow {
  pattern_id: string;
  step_type: string;
  profile_slug: string;
  constraint_text: string;
  occurrences: number;
  frequency: number;
  effectiveness_score: number | null;
  source_reasons: string;
}

/**
 * Validate EquityStoreOptions at construction time.
 * Throws ConfigValidationError if any field is invalid.
 */
function validateEquityStoreOptions(options: EquityStoreOptions): void {
  if (options.dbPath !== undefined) {
    if (typeof options.dbPath !== 'string') {
      throw new ConfigValidationError(
        `"dbPath" must be a string, got ${typeof options.dbPath}.`,
      );
    }
    if (options.dbPath.trim() === '') {
      throw new ConfigValidationError(
        '"dbPath" must not be an empty string. Use ":memory:" for an in-memory database.',
      );
    }
  }

  if (options.wal !== undefined && typeof options.wal !== 'boolean') {
    throw new ConfigValidationError(
      `"wal" must be a boolean, got ${typeof options.wal}.`,
    );
  }
}

export class EquityStore implements AWMStore {
  private db: Database.Database;

  private insertTrace: Database.Statement;
  private getBeliefStmt: Database.Statement;
  private upsertBeliefStmt: Database.Statement;
  private getArmsStmt: Database.Statement;
  private upsertArmStmt: Database.Statement;
  private getPatternsScoped: Database.Statement;
  private getPatternsStepOnly: Database.Statement;
  private upsertPatternStmt: Database.Statement;

  constructor(options: EquityStoreOptions = {}) {
    validateEquityStoreOptions(options);

    const dbPath = options.dbPath ?? ':memory:';
    this.db = new Database(dbPath);
    if (dbPath !== ':memory:' && options.wal !== false) {
      this.db.pragma('journal_mode = WAL');
    }
    this.db.pragma('foreign_keys = ON');
    this.db.exec(SCHEMA);

    this.insertTrace = this.db.prepare(`
      INSERT OR REPLACE INTO traces (
        trace_id, run_id, workflow_type, step_type, step_index,
        profile_slug, sector, model, input_fingerprint,
        acr_capabilities, acr_coverage_ratio, acr_lod_level, acr_missing_capabilities,
        passed, revised, revision_reason, approval_result,
        output_fingerprint, output_similarity_to_prior,
        tokens_in, tokens_out, cost, latency_ms,
        awm_prediction, awm_was_correct, timestamp
      ) VALUES (
        @trace_id, @run_id, @workflow_type, @step_type, @step_index,
        @profile_slug, @sector, @model, @input_fingerprint,
        @acr_capabilities, @acr_coverage_ratio, @acr_lod_level, @acr_missing_capabilities,
        @passed, @revised, @revision_reason, @approval_result,
        @output_fingerprint, @output_similarity_to_prior,
        @tokens_in, @tokens_out, @cost, @latency_ms,
        @awm_prediction, @awm_was_correct, @timestamp
      )
    `);

    this.getBeliefStmt = this.db.prepare(
      `SELECT * FROM beliefs WHERE step_type = ? AND profile_slug = ?`,
    );
    this.upsertBeliefStmt = this.db.prepare(`
      INSERT INTO beliefs (step_type, profile_slug, alpha, beta, observations, updated_at)
      VALUES (@step_type, @profile_slug, @alpha, @beta, @observations, @updated_at)
      ON CONFLICT(step_type, profile_slug) DO UPDATE SET
        alpha = excluded.alpha,
        beta = excluded.beta,
        observations = excluded.observations,
        updated_at = excluded.updated_at
    `);

    this.getArmsStmt = this.db.prepare(
      `SELECT * FROM arms WHERE step_type = ? AND profile_slug = ?`,
    );
    this.upsertArmStmt = this.db.prepare(`
      INSERT INTO arms (model, step_type, profile_slug, alpha, beta, pulls, avg_cost, avg_latency_ms)
      VALUES (@model, @step_type, @profile_slug, @alpha, @beta, @pulls, @avg_cost, @avg_latency_ms)
      ON CONFLICT(model, step_type, profile_slug) DO UPDATE SET
        alpha = excluded.alpha,
        beta = excluded.beta,
        pulls = excluded.pulls,
        avg_cost = excluded.avg_cost,
        avg_latency_ms = excluded.avg_latency_ms
    `);

    this.getPatternsScoped = this.db.prepare(
      `SELECT * FROM patterns WHERE step_type = ? AND (profile_slug = ? OR profile_slug = '__global__')`,
    );
    this.getPatternsStepOnly = this.db.prepare(
      `SELECT * FROM patterns WHERE step_type = ?`,
    );
    this.upsertPatternStmt = this.db.prepare(`
      INSERT INTO patterns (pattern_id, step_type, profile_slug, constraint_text, occurrences, frequency, effectiveness_score, source_reasons)
      VALUES (@pattern_id, @step_type, @profile_slug, @constraint_text, @occurrences, @frequency, @effectiveness_score, @source_reasons)
      ON CONFLICT(pattern_id) DO UPDATE SET
        step_type = excluded.step_type,
        profile_slug = excluded.profile_slug,
        constraint_text = excluded.constraint_text,
        occurrences = excluded.occurrences,
        frequency = excluded.frequency,
        effectiveness_score = excluded.effectiveness_score,
        source_reasons = excluded.source_reasons
    `);
  }

  // ─── Lifecycle ───

  /** Drop and recreate all tables. Use between threshold-sweep runs. */
  reset(): void {
    this.db.exec(`
      DROP TABLE IF EXISTS traces;
      DROP TABLE IF EXISTS beliefs;
      DROP TABLE IF EXISTS arms;
      DROP TABLE IF EXISTS patterns;
    `);
    this.db.exec(SCHEMA);
  }

  close(): void {
    this.db.close();
  }

  // ─── Traces ───

  async storeTrace(trace: StepTrace): Promise<void> {
    this.insertTrace.run({
      trace_id: trace.traceId,
      run_id: trace.runId,
      workflow_type: trace.workflowType,
      step_type: trace.stepType,
      step_index: trace.stepIndex,
      profile_slug: trace.profileSlug ?? null,
      sector: trace.sector ?? null,
      model: trace.model,
      input_fingerprint: trace.inputFingerprint,
      acr_capabilities: trace.acrCapabilities ? JSON.stringify(trace.acrCapabilities) : null,
      acr_coverage_ratio: trace.acrCoverageRatio ?? null,
      acr_lod_level: trace.acrLodLevel ?? null,
      acr_missing_capabilities: trace.acrMissingCapabilities
        ? JSON.stringify(trace.acrMissingCapabilities)
        : null,
      passed: trace.passed ? 1 : 0,
      revised: trace.revised ? 1 : 0,
      revision_reason: trace.revisionReason ?? null,
      approval_result: trace.approvalResult ?? null,
      output_fingerprint: trace.outputFingerprint ?? null,
      output_similarity_to_prior: trace.outputSimilarityToPrior ?? null,
      tokens_in: trace.tokensIn,
      tokens_out: trace.tokensOut,
      cost: trace.cost,
      latency_ms: trace.latencyMs,
      awm_prediction: trace.awmPrediction ? JSON.stringify(trace.awmPrediction) : null,
      awm_was_correct: trace.awmWasCorrect === undefined ? null : (trace.awmWasCorrect ? 1 : 0),
      timestamp: trace.timestamp,
    });
    return Promise.resolve();
  }

  async queryTraces(query: TraceQuery): Promise<StepTrace[]> {
    const clauses: string[] = [];
    const params: unknown[] = [];

    if (query.stepType !== undefined) {
      clauses.push('step_type = ?');
      params.push(query.stepType);
    }
    if (query.profileSlug !== undefined) {
      clauses.push('profile_slug = ?');
      params.push(query.profileSlug);
    }
    if (query.sector !== undefined) {
      clauses.push('sector = ?');
      params.push(query.sector);
    }
    if (query.model !== undefined) {
      clauses.push('model = ?');
      params.push(query.model);
    }
    if (query.passed !== undefined) {
      clauses.push('passed = ?');
      params.push(query.passed ? 1 : 0);
    }
    if (query.revised !== undefined) {
      clauses.push('revised = ?');
      params.push(query.revised ? 1 : 0);
    }
    if (query.similarTo !== undefined) {
      clauses.push('input_fingerprint = ?');
      params.push(query.similarTo);
    }

    const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : '';
    const limit = query.limit && query.limit > 0 ? `LIMIT ${query.limit | 0}` : '';
    const sql = `SELECT * FROM traces ${where} ORDER BY timestamp DESC ${limit}`;
    const rows = this.db.prepare(sql).all(...params) as TraceRow[];
    return Promise.resolve(rows.map(rowToTrace));
  }

  // ─── Beliefs ───

  async getBelief(stepType: string, profileSlug: string): Promise<StepBelief | null> {
    const row = this.getBeliefStmt.get(stepType, profileSlug) as BeliefRow | undefined;
    if (!row) return Promise.resolve(null);
    return Promise.resolve({
      stepType: row.step_type,
      profileSlug: row.profile_slug,
      alpha: row.alpha,
      beta: row.beta,
      observations: row.observations,
      updatedAt: row.updated_at,
    });
  }

  async setBelief(belief: StepBelief): Promise<void> {
    this.upsertBeliefStmt.run({
      step_type: belief.stepType,
      profile_slug: belief.profileSlug,
      alpha: belief.alpha,
      beta: belief.beta,
      observations: belief.observations,
      updated_at: belief.updatedAt,
    });
    return Promise.resolve();
  }

  // ─── Arms ───

  async getArms(stepType: string, profileSlug: string): Promise<ModelArm[]> {
    const rows = this.getArmsStmt.all(stepType, profileSlug) as ArmRow[];
    return Promise.resolve(rows.map(rowToArm));
  }

  async setArm(arm: ModelArm): Promise<void> {
    this.upsertArmStmt.run({
      model: arm.model,
      step_type: arm.stepType,
      profile_slug: arm.profileSlug,
      alpha: arm.alpha,
      beta: arm.beta,
      pulls: arm.pulls,
      avg_cost: arm.avgCost,
      avg_latency_ms: arm.avgLatencyMs,
    });
    return Promise.resolve();
  }

  // ─── Patterns ───

  async getPatterns(stepType: string, profileSlug?: string): Promise<ConstraintPattern[]> {
    const rows = profileSlug !== undefined
      ? (this.getPatternsScoped.all(stepType, profileSlug) as PatternRow[])
      : (this.getPatternsStepOnly.all(stepType) as PatternRow[]);
    return Promise.resolve(rows.map(rowToPattern));
  }

  async setPattern(pattern: ConstraintPattern): Promise<void> {
    this.upsertPatternStmt.run({
      pattern_id: pattern.patternId,
      step_type: pattern.stepType,
      profile_slug: pattern.profileSlug,
      constraint_text: pattern.constraint,
      occurrences: pattern.occurrences,
      frequency: pattern.frequency,
      effectiveness_score: pattern.effectivenessScore ?? null,
      source_reasons: JSON.stringify(pattern.sourceReasons),
    });
    return Promise.resolve();
  }
}

// ─── Row → object converters ───

function rowToTrace(row: TraceRow): StepTrace {
  const trace: StepTrace = {
    traceId: row.trace_id,
    runId: row.run_id,
    workflowType: row.workflow_type,
    stepType: row.step_type,
    stepIndex: row.step_index,
    profileSlug: row.profile_slug ?? undefined,
    sector: row.sector ?? undefined,
    model: row.model,
    inputFingerprint: row.input_fingerprint,
    acrCapabilities: row.acr_capabilities ? (JSON.parse(row.acr_capabilities) as string[]) : undefined,
    acrCoverageRatio: row.acr_coverage_ratio ?? undefined,
    acrLodLevel: (row.acr_lod_level as StepTrace['acrLodLevel']) ?? undefined,
    acrMissingCapabilities: row.acr_missing_capabilities
      ? (JSON.parse(row.acr_missing_capabilities) as string[])
      : undefined,
    passed: row.passed === 1,
    revised: row.revised === 1,
    revisionReason: row.revision_reason ?? undefined,
    approvalResult: (row.approval_result as ApprovalResult) ?? undefined,
    outputFingerprint: row.output_fingerprint ?? undefined,
    outputSimilarityToPrior: row.output_similarity_to_prior ?? undefined,
    tokensIn: row.tokens_in,
    tokensOut: row.tokens_out,
    cost: row.cost,
    latencyMs: row.latency_ms,
    awmPrediction: row.awm_prediction
      ? (JSON.parse(row.awm_prediction) as StepPrediction)
      : undefined,
    awmWasCorrect: row.awm_was_correct === null ? undefined : row.awm_was_correct === 1,
    timestamp: row.timestamp,
  };
  return trace;
}

function rowToArm(row: ArmRow): ModelArm {
  return {
    model: row.model,
    stepType: row.step_type,
    profileSlug: row.profile_slug,
    alpha: row.alpha,
    beta: row.beta,
    pulls: row.pulls,
    avgCost: row.avg_cost,
    avgLatencyMs: row.avg_latency_ms,
  };
}

function rowToPattern(row: PatternRow): ConstraintPattern {
  return {
    patternId: row.pattern_id,
    stepType: row.step_type,
    profileSlug: row.profile_slug,
    constraint: row.constraint_text,
    occurrences: row.occurrences,
    frequency: row.frequency,
    effectivenessScore: row.effectiveness_score ?? undefined,
    sourceReasons: JSON.parse(row.source_reasons) as string[],
  };
}
