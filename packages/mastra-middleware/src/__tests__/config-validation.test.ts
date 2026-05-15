/**
 * GIN-36 — AWMMiddlewareConfig and WrapStepConfig validation tests.
 *
 * Covers rejection of invalid dynamic config inputs passed to
 * AWMMiddleware and wrapStep at construction/call time.
 */

import { describe, it, expect, vi } from 'vitest';
import { AWMMiddleware } from '../index.js';
import { wrapStep } from '../step-wrapper.js';
import { Oracle, InMemoryStore } from '../../../core/src/index.js';
import { ConfigValidationError } from '../../../core/src/validation.js';

// ─── Fixtures ─────────────────────────────────────────────────

function makeOracle(): Oracle {
  return new Oracle({ store: new InMemoryStore() });
}

function expectConfigError(fn: () => void, messagePart?: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  expect(caught).toBeInstanceOf(ConfigValidationError);
  if (messagePart !== undefined) {
    expect((caught as ConfigValidationError).message).toContain(messagePart);
  }
}

function makeValidMiddleware(): AWMMiddleware {
  return new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow' });
}

// ─── AWMMiddlewareConfig ──────────────────────────────────────

describe('AWMMiddleware — config validation', () => {
  describe('oracle (required)', () => {
    it('throws when oracle is missing', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new AWMMiddleware({ mode: 'shadow' }), '"oracle" is required');
    });

    it('throws when oracle is null', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new AWMMiddleware({ oracle: null, mode: 'shadow' }), '"oracle" is required');
    });

    it('accepts a valid Oracle instance', () => {
      expect(() => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow' })).not.toThrow();
    });
  });

  describe('mode (required, must be shadow|advisory|active)', () => {
    it('throws when mode is missing', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new AWMMiddleware({ oracle: makeOracle() }), '"mode" is required');
    });

    it('throws when mode is an invalid string', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'passive' }),
        '"mode" must be one of',
      );
    });

    it('throws when mode is a number', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new AWMMiddleware({ oracle: makeOracle(), mode: 1 }), '"mode" must be one of');
    });

    it('accepts all valid modes', () => {
      const oracle = makeOracle();
      expect(() => new AWMMiddleware({ oracle, mode: 'shadow' })).not.toThrow();
      expect(() => new AWMMiddleware({ oracle, mode: 'advisory' })).not.toThrow();
      expect(() => new AWMMiddleware({ oracle, mode: 'active' })).not.toThrow();
    });
  });

  describe('defaultModels (optional, non-empty string array)', () => {
    it('accepts a valid non-empty array of model names', () => {
      expect(() => new AWMMiddleware({
        oracle: makeOracle(),
        mode: 'shadow',
        defaultModels: ['claude-sonnet-4', 'gpt-4'],
      })).not.toThrow();
    });

    it('accepts undefined (uses built-in defaults)', () => {
      expect(() => new AWMMiddleware({
        oracle: makeOracle(),
        mode: 'shadow',
        defaultModels: undefined,
      })).not.toThrow();
    });

    it('throws when defaultModels is an empty array', () => {
      expectConfigError(
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', defaultModels: [] }),
        '"defaultModels" must be a non-empty array',
      );
    });

    it('throws when defaultModels contains an empty string', () => {
      expectConfigError(
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', defaultModels: ['claude-sonnet-4', ''] }),
        '"defaultModels[1]" must be a non-empty string',
      );
    });

    it('throws when defaultModels is not an array', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', defaultModels: 'claude-sonnet-4' }),
        '"defaultModels" must be an array',
      );
    });
  });

  describe('callback functions (optional, must be functions)', () => {
    it('accepts valid function for onPrediction', () => {
      expect(() => new AWMMiddleware({
        oracle: makeOracle(),
        mode: 'shadow',
        onPrediction: vi.fn(),
      })).not.toThrow();
    });

    it('throws when onPrediction is not a function', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', onPrediction: 'log' }),
        '"onPrediction" must be a function',
      );
    });

    it('accepts valid function for onOutcome', () => {
      expect(() => new AWMMiddleware({
        oracle: makeOracle(),
        mode: 'shadow',
        onOutcome: vi.fn(),
      })).not.toThrow();
    });

    it('throws when onOutcome is not a function', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', onOutcome: {} }),
        '"onOutcome" must be a function',
      );
    });

    it('accepts valid function for fingerprintFn', () => {
      expect(() => new AWMMiddleware({
        oracle: makeOracle(),
        mode: 'shadow',
        fingerprintFn: (input: unknown) => String(input),
      })).not.toThrow();
    });

    it('throws when fingerprintFn is not a function', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'shadow', fingerprintFn: 'sha256' }),
        '"fingerprintFn" must be a function',
      );
    });
  });

  describe('enableSkipping (optional, must be boolean)', () => {
    it('accepts true and false', () => {
      const oracle = makeOracle();
      expect(() => new AWMMiddleware({ oracle, mode: 'active', enableSkipping: true })).not.toThrow();
      expect(() => new AWMMiddleware({ oracle, mode: 'active', enableSkipping: false })).not.toThrow();
    });

    it('throws when enableSkipping is not a boolean', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => new AWMMiddleware({ oracle: makeOracle(), mode: 'active', enableSkipping: 1 }),
        '"enableSkipping" must be a boolean',
      );
    });
  });
});

// ─── WrapStepConfig ───────────────────────────────────────────

describe('wrapStep — config validation', () => {
  function makeStep(id = 'test-step') {
    return {
      id,
      execute: async () => ({ result: 'ok' }),
    };
  }

  describe('awm (required)', () => {
    it('throws when awm is missing', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => wrapStep(makeStep(), { workflowType: 'my-workflow', defaultModel: 'claude-sonnet-4' }),
        '"awm" is required',
      );
    });

    it('throws when awm is null', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => wrapStep(makeStep(), { awm: null, workflowType: 'my-workflow', defaultModel: 'claude-sonnet-4' }),
        '"awm" is required',
      );
    });
  });

  describe('workflowType (required, non-empty string)', () => {
    it('throws when workflowType is missing', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => wrapStep(makeStep(), { awm: makeValidMiddleware(), defaultModel: 'claude-sonnet-4' }),
        '"workflowType" must be a string',
      );
    });

    it('throws when workflowType is empty', () => {
      expectConfigError(
        () => wrapStep(makeStep(), { awm: makeValidMiddleware(), workflowType: '', defaultModel: 'claude-sonnet-4' }),
        '"workflowType" must not be an empty string',
      );
    });

    it('throws when workflowType is not a string', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => wrapStep(makeStep(), { awm: makeValidMiddleware(), workflowType: 42, defaultModel: 'claude-sonnet-4' }),
        '"workflowType" must be a string',
      );
    });

    it('accepts a valid workflow type', () => {
      expect(() => wrapStep(makeStep(), {
        awm: makeValidMiddleware(),
        workflowType: 'creative-campaign',
        defaultModel: 'claude-sonnet-4',
      })).not.toThrow();
    });
  });

  describe('defaultModel (required, non-empty string)', () => {
    it('throws when defaultModel is missing', () => {
      expectConfigError(
        // @ts-expect-error — testing runtime behavior
        () => wrapStep(makeStep(), { awm: makeValidMiddleware(), workflowType: 'wf' }),
        '"defaultModel" must be a string',
      );
    });

    it('throws when defaultModel is empty', () => {
      expectConfigError(
        () => wrapStep(makeStep(), { awm: makeValidMiddleware(), workflowType: 'wf', defaultModel: '' }),
        '"defaultModel" must not be an empty string',
      );
    });

    it('accepts a valid model name', () => {
      expect(() => wrapStep(makeStep(), {
        awm: makeValidMiddleware(),
        workflowType: 'wf',
        defaultModel: 'claude-opus-4',
      })).not.toThrow();
    });
  });

  describe('availableModels (optional, non-empty string array)', () => {
    it('accepts a valid list', () => {
      expect(() => wrapStep(makeStep(), {
        awm: makeValidMiddleware(),
        workflowType: 'wf',
        defaultModel: 'claude-sonnet-4',
        availableModels: ['claude-sonnet-4', 'gpt-4'],
      })).not.toThrow();
    });

    it('throws when availableModels is empty', () => {
      expectConfigError(
        () => wrapStep(makeStep(), {
          awm: makeValidMiddleware(),
          workflowType: 'wf',
          defaultModel: 'claude-sonnet-4',
          availableModels: [],
        }),
        '"availableModels" must be a non-empty array',
      );
    });

    it('throws when availableModels contains a blank model name', () => {
      expectConfigError(
        () => wrapStep(makeStep(), {
          awm: makeValidMiddleware(),
          workflowType: 'wf',
          defaultModel: 'claude-sonnet-4',
          availableModels: ['claude-sonnet-4', '  '],
        }),
        '"availableModels[1]" must be a non-empty string',
      );
    });
  });

  describe('getProfileSlug / getSector (optional, must be functions)', () => {
    it('accepts valid extractor functions', () => {
      expect(() => wrapStep(makeStep(), {
        awm: makeValidMiddleware(),
        workflowType: 'wf',
        defaultModel: 'claude-sonnet-4',
        getProfileSlug: () => 'acme',
        getSector: () => 'tech',
      })).not.toThrow();
    });

    it('throws when getProfileSlug is not a function', () => {
      expectConfigError(
        () => wrapStep(makeStep(), {
          awm: makeValidMiddleware(),
          workflowType: 'wf',
          defaultModel: 'claude-sonnet-4',
          getProfileSlug: 'acme' as any,
        }),
        '"getProfileSlug" must be a function',
      );
    });

    it('throws when getSector is not a function', () => {
      expectConfigError(
        () => wrapStep(makeStep(), {
          awm: makeValidMiddleware(),
          workflowType: 'wf',
          defaultModel: 'claude-sonnet-4',
          getSector: 'tech' as any,
        }),
        '"getSector" must be a function',
      );
    });
  });

  describe('stepDef.id (required)', () => {
    it('throws when step id is empty', () => {
      let caught: unknown;
      try {
        wrapStep({ ...makeStep(), id: '' }, {
          awm: makeValidMiddleware(),
          workflowType: 'wf',
          defaultModel: 'claude-sonnet-4',
        });
      } catch (e) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(Error);
      expect((caught as Error).message).toContain('stepDef.id');
    });
  });
});
