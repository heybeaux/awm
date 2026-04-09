/**
 * @heybeaux/awm-engram
 *
 * Engram storage adapter for AWM. Persists traces, beliefs,
 * bandit arms, and constraint patterns via the Engram API.
 *
 * Uses Engram's memory types:
 * - 'episodic' for step traces (what happened)
 * - 'semantic' for beliefs and arms (what we've learned)
 * - 'procedural' for constraint patterns (what to do about it)
 *
 * Leverages pgvector similarity search for finding similar past runs.
 */

import type {
  AWMStore,
  StepTrace,
  StepBelief,
  ModelArm,
  ConstraintPattern,
  TraceQuery,
} from '@heybeaux/awm-core';

export interface EngramStoreConfig {
  /** Engram API base URL */
  apiUrl: string;
  /** Engram API key (X-AM-API-Key header) */
  apiKey: string;
  /** Namespace for AWM data (isolates from other Engram users) */
  namespace?: string;
  /** User ID for Engram requests */
  userId?: string;
}

export class EngramStore implements AWMStore {
  private apiUrl: string;
  private apiKey: string;
  private namespace: string;
  private userId: string;

  constructor(config: EngramStoreConfig) {
    this.apiUrl = config.apiUrl.replace(/\/$/, '');
    this.apiKey = config.apiKey;
    this.namespace = config.namespace || 'awm';
    this.userId = config.userId || 'awm-oracle';
  }

  // ─── HTTP Client ───

  private async request<T>(path: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.apiUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'X-AM-API-Key': this.apiKey,
        'X-AM-User-ID': this.userId,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const body = await response.text().catch(() => '');
      throw new Error(`Engram API error ${response.status}: ${body}`);
    }

    return response.json() as Promise<T>;
  }

  // ─── Traces ───

  async storeTrace(trace: StepTrace): Promise<void> {
    await this.request('/v1/memories', {
      method: 'POST',
      body: JSON.stringify({
        content: this.serializeTrace(trace),
        type: 'episodic',
        layer: 'SESSION',
        source: 'AGENT_OBSERVATION',
        agentId: this.namespace,
        metadata: {
          awm_type: 'step_trace',
          traceId: trace.traceId,
          runId: trace.runId,
          workflowType: trace.workflowType,
          stepType: trace.stepType,
          profileSlug: trace.profileSlug || '__global__',
          model: trace.model,
          passed: trace.passed,
          revised: trace.revised,
          cost: trace.cost,
          latencyMs: trace.latencyMs,
          approvalResult: trace.approvalResult,
        },
        tags: [
          'awm:trace',
          `step:${trace.stepType}`,
          `profile:${trace.profileSlug || 'global'}`,
          `model:${trace.model}`,
          trace.passed ? 'outcome:pass' : 'outcome:fail',
          trace.revised ? 'revised:yes' : 'revised:no',
        ],
      }),
    });
  }

  async queryTraces(query: TraceQuery): Promise<StepTrace[]> {
    // Use Engram's recall endpoint with semantic search
    const searchQuery = this.buildSearchQuery(query);

    const result = await this.request<{ memories: Array<{ content: string; metadata: Record<string, unknown>; relevance?: number }> }>(
      '/v1/recall',
      {
        method: 'POST',
        body: JSON.stringify({
          query: searchQuery,
          limit: query.limit || 50,
          filters: this.buildFilters(query),
        }),
      },
    );

    return result.memories
      .map((m) => this.deserializeTrace(m.content))
      .filter((t): t is StepTrace => t !== null);
  }

  // ─── Beliefs ───

  async getBelief(stepType: string, profileSlug: string): Promise<StepBelief | null> {
    const key = `awm:belief:${stepType}:${profileSlug}`;

    try {
      const result = await this.request<{ memories: Array<{ content: string }> }>(
        '/v1/recall',
        {
          method: 'POST',
          body: JSON.stringify({
            query: key,
            limit: 1,
            filters: {
              tags: ['awm:belief', `step:${stepType}`, `profile:${profileSlug}`],
            },
          }),
        },
      );

      if (result.memories.length === 0) return null;
      return JSON.parse(result.memories[0].content) as StepBelief;
    } catch {
      return null;
    }
  }

  async setBelief(belief: StepBelief): Promise<void> {
    const key = `awm:belief:${belief.stepType}:${belief.profileSlug}`;

    await this.request('/v1/memories', {
      method: 'POST',
      body: JSON.stringify({
        content: JSON.stringify(belief),
        type: 'semantic',
        layer: 'INSIGHT',
        source: 'AGENT_OBSERVATION',
        agentId: this.namespace,
        metadata: {
          awm_type: 'step_belief',
          key,
          stepType: belief.stepType,
          profileSlug: belief.profileSlug,
        },
        tags: [
          'awm:belief',
          `step:${belief.stepType}`,
          `profile:${belief.profileSlug}`,
        ],
      }),
    });
  }

  // ─── Bandit Arms ───

  async getArms(stepType: string, profileSlug: string): Promise<ModelArm[]> {
    try {
      const result = await this.request<{ memories: Array<{ content: string }> }>(
        '/v1/recall',
        {
          method: 'POST',
          body: JSON.stringify({
            query: `awm:arms:${stepType}:${profileSlug}`,
            limit: 20,
            filters: {
              tags: ['awm:arm', `step:${stepType}`, `profile:${profileSlug}`],
            },
          }),
        },
      );

      return result.memories.map((m) => JSON.parse(m.content) as ModelArm);
    } catch {
      return [];
    }
  }

  async setArm(arm: ModelArm): Promise<void> {
    await this.request('/v1/memories', {
      method: 'POST',
      body: JSON.stringify({
        content: JSON.stringify(arm),
        type: 'semantic',
        layer: 'INSIGHT',
        source: 'AGENT_OBSERVATION',
        agentId: this.namespace,
        metadata: {
          awm_type: 'model_arm',
          model: arm.model,
          stepType: arm.stepType,
          profileSlug: arm.profileSlug,
        },
        tags: [
          'awm:arm',
          `step:${arm.stepType}`,
          `profile:${arm.profileSlug}`,
          `model:${arm.model}`,
        ],
      }),
    });
  }

  // ─── Constraint Patterns ───

  async getPatterns(stepType: string, profileSlug?: string): Promise<ConstraintPattern[]> {
    const tags = ['awm:pattern', `step:${stepType}`];
    if (profileSlug) tags.push(`profile:${profileSlug}`);

    try {
      const result = await this.request<{ memories: Array<{ content: string }> }>(
        '/v1/recall',
        {
          method: 'POST',
          body: JSON.stringify({
            query: `awm:patterns:${stepType}:${profileSlug || 'global'}`,
            limit: 50,
            filters: { tags },
          }),
        },
      );

      return result.memories.map((m) => JSON.parse(m.content) as ConstraintPattern);
    } catch {
      return [];
    }
  }

  async setPattern(pattern: ConstraintPattern): Promise<void> {
    await this.request('/v1/memories', {
      method: 'POST',
      body: JSON.stringify({
        content: JSON.stringify(pattern),
        type: 'procedural',
        layer: 'INSIGHT',
        source: 'AGENT_OBSERVATION',
        agentId: this.namespace,
        metadata: {
          awm_type: 'constraint_pattern',
          patternId: pattern.patternId,
          stepType: pattern.stepType,
          profileSlug: pattern.profileSlug,
        },
        tags: [
          'awm:pattern',
          `step:${pattern.stepType}`,
          `profile:${pattern.profileSlug}`,
        ],
      }),
    });
  }

  // ─── Serialization Helpers ───

  private serializeTrace(trace: StepTrace): string {
    return JSON.stringify(trace);
  }

  private deserializeTrace(content: string): StepTrace | null {
    try {
      return JSON.parse(content) as StepTrace;
    } catch {
      return null;
    }
  }

  private buildSearchQuery(query: TraceQuery): string {
    const parts: string[] = ['AWM step trace'];
    if (query.stepType) parts.push(`step:${query.stepType}`);
    if (query.profileSlug) parts.push(`profile:${query.profileSlug}`);
    if (query.sector) parts.push(`sector:${query.sector}`);
    if (query.similarTo) parts.push(query.similarTo);
    return parts.join(' ');
  }

  private buildFilters(query: TraceQuery): Record<string, unknown> {
    const tags: string[] = ['awm:trace'];
    if (query.stepType) tags.push(`step:${query.stepType}`);
    if (query.profileSlug) tags.push(`profile:${query.profileSlug}`);
    if (query.model) tags.push(`model:${query.model}`);
    if (query.passed !== undefined) tags.push(query.passed ? 'outcome:pass' : 'outcome:fail');

    return { tags };
  }
}
