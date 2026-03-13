/**
 * AWM Step Wrapper for Mastra/Forge
 *
 * Wraps existing Mastra createStep() calls to add AWM
 * prediction and outcome recording hooks.
 *
 * Usage in Forge:
 *   import { wrapStep } from '@heybeaux/awm-mastra';
 *
 *   const strategistStep = wrapStep(createStep({
 *     id: 'strategist',
 *     execute: async ({ inputData }) => { ... }
 *   }), {
 *     workflowType: 'creative-campaign',
 *     defaultModel: 'claude-opus-4',
 *   });
 *
 * The wrapper intercepts execute() to:
 * 1. Call AWM.beforeStep() — get predictions + constraints
 * 2. Inject constraints into the step context
 * 3. Execute the original step
 * 4. Call AWM.afterStep() — record the outcome
 */

import type { AWMMiddleware, AWMStepContext } from './index.js';

/**
 * Configuration for wrapping a step with AWM.
 */
export interface WrapStepConfig {
  /** AWM middleware instance */
  awm: AWMMiddleware;
  /** Workflow type identifier */
  workflowType: string;
  /** Default model used by this step (for tracking) */
  defaultModel: string;
  /** Available models for routing (if mode=active) */
  availableModels?: string[];
  /** Extract profile slug from step input */
  getProfileSlug?: (input: unknown) => string | undefined;
  /** Extract sector from step input */
  getSector?: (input: unknown) => string | undefined;
}

/**
 * Step definition shape (minimal Mastra createStep interface)
 */
export interface StepDefinition<TInput = unknown, TOutput = unknown> {
  id: string;
  description?: string;
  inputSchema?: unknown;
  outputSchema?: unknown;
  execute: (context: StepExecuteContext<TInput>) => Promise<TOutput>;
  [key: string]: unknown;
}

export interface StepExecuteContext<T = unknown> {
  inputData: T;
  [key: string]: unknown;
}

/**
 * The AWM context that gets attached to step execution.
 * Steps can read this to access predictions and constraints.
 */
export interface AWMStepData {
  awm: AWMStepContext;
  constraintText: string;
}

/**
 * Wrap a Mastra step definition with AWM prediction hooks.
 * Returns a new step definition with the same interface.
 */
export function wrapStep<TInput extends Record<string, unknown>, TOutput>(
  stepDef: StepDefinition<TInput, TOutput>,
  config: WrapStepConfig,
): StepDefinition<TInput, TOutput> {
  const originalExecute = stepDef.execute;

  return {
    ...stepDef,
    execute: async (context: StepExecuteContext<TInput>): Promise<TOutput> => {
      const startTime = Date.now();
      const runId = extractRunId(context) || `run-${Date.now()}`;
      const stepIndex = extractStepIndex(context) || 0;

      const profileSlug = config.getProfileSlug
        ? config.getProfileSlug(context.inputData)
        : extractProfileSlug(context.inputData);

      const sector = config.getSector
        ? config.getSector(context.inputData)
        : undefined;

      // ─── Before Step: Get AWM prediction ───
      let awmContext: AWMStepContext | null = null;
      try {
        awmContext = await config.awm.beforeStep({
          runId,
          workflowType: config.workflowType,
          stepType: stepDef.id,
          stepIndex,
          profileSlug,
          sector,
          currentModel: config.defaultModel,
          input: context.inputData,
          availableModels: config.availableModels,
        });
      } catch (err) {
        // AWM failure should never block pipeline execution
        console.warn(`[AWM] beforeStep failed for ${stepDef.id}:`, err);
      }

      // ─── Skip if AWM recommends ───
      if (awmContext?.skip) {
        console.log(`[AWM] Skipping step ${stepDef.id} — prediction confidence high, output unchanged`);
        // For now, still execute. Full skip requires cached output lookup.
        // TODO: implement output caching for skip support
      }

      // ─── Execute the original step ───
      // Attach AWM context so the step can read constraints if needed
      const enrichedContext = {
        ...context,
        awm: awmContext,
        awmConstraints: awmContext
          ? config.awm.constructor.prototype.constructor === AWMMiddlewareRef
            ? formatConstraintsStatic(awmContext.constraints)
            : ''
          : '',
      };

      let result: TOutput;
      let passed = true;
      let error: Error | undefined;

      try {
        result = await originalExecute(enrichedContext);
      } catch (err) {
        passed = false;
        error = err as Error;
        throw err;
      } finally {
        const latencyMs = Date.now() - startTime;

        // ─── After Step: Record outcome ───
        try {
          await config.awm.afterStep({
            runId,
            workflowType: config.workflowType,
            stepType: stepDef.id,
            stepIndex,
            profileSlug,
            sector,
            model: awmContext?.model || config.defaultModel,
            input: context.inputData,
            passed,
            tokensIn: 0,  // TODO: extract from Mastra observability
            tokensOut: 0,
            cost: 0,       // TODO: extract from Mastra observability
            latencyMs,
            awmPrediction: awmContext?.prediction,
          });
        } catch (recordErr) {
          console.warn(`[AWM] afterStep failed for ${stepDef.id}:`, recordErr);
        }
      }

      return result!;
    },
  };
}

// ─── Helpers ───

/** Placeholder ref to avoid circular import for formatConstraints */
const AWMMiddlewareRef = class {};

function formatConstraintsStatic(constraints: string[]): string {
  if (constraints.length === 0) return '';
  return [
    '',
    '## ⚠️ AWM Constraints (from historical analysis)',
    'The following constraints are based on patterns from previous pipeline runs.',
    'Following these will reduce the likelihood of revision:',
    '',
    ...constraints.map((c, i) => `${i + 1}. ${c}`),
    '',
  ].join('\n');
}

function extractRunId(context: Record<string, unknown>): string | undefined {
  // Mastra passes run context — try common paths
  const ctx = context as Record<string, unknown>;
  if (typeof ctx.runId === 'string') return ctx.runId;
  if (ctx.context && typeof (ctx.context as Record<string, unknown>).runId === 'string') {
    return (ctx.context as Record<string, unknown>).runId as string;
  }
  return undefined;
}

function extractStepIndex(context: Record<string, unknown>): number | undefined {
  const ctx = context as Record<string, unknown>;
  if (typeof ctx.stepIndex === 'number') return ctx.stepIndex;
  return undefined;
}

function extractProfileSlug(input: unknown): string | undefined {
  if (input && typeof input === 'object') {
    const obj = input as Record<string, unknown>;
    if (typeof obj.profileSlug === 'string') return obj.profileSlug;
    if (typeof obj.profile_slug === 'string') return obj.profile_slug;
    if (typeof obj.profile === 'string') return obj.profile;
  }
  return undefined;
}
