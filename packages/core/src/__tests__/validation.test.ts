import { describe, it, expect } from 'vitest';
import {
  ConfigValidationError,
  requireField,
  requireString,
  requireNumber,
  requireBoolean,
  requireFunction,
  requireRange,
  requireMin,
  requirePositive,
  requireNonEmptyArray,
  requireStringArray,
  requireOneOf,
} from '../validation.js';

// ─── Helper ───────────────────────────────────────────────────

function expectValidationError(fn: () => void, messagePart?: string): void {
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

// ─── ConfigValidationError ────────────────────────────────────

describe('ConfigValidationError', () => {
  it('is an instance of Error', () => {
    const err = new ConfigValidationError('oops');
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe('ConfigValidationError');
    expect(err.message).toBe('oops');
  });
});

// ─── requireField ─────────────────────────────────────────────

describe('requireField', () => {
  it('passes when value is present', () => {
    expect(() => requireField('hello', 'myField')).not.toThrow();
    expect(() => requireField(0, 'myField')).not.toThrow();
    expect(() => requireField(false, 'myField')).not.toThrow();
    expect(() => requireField({}, 'myField')).not.toThrow();
  });

  it('throws when value is undefined', () => {
    expectValidationError(() => requireField(undefined, 'store'), '"store" is required');
  });

  it('throws when value is null', () => {
    expectValidationError(() => requireField(null, 'oracle'), '"oracle" is required');
    expect(() => {
      try { requireField(null, 'oracle'); } catch (e) {
        throw e;
      }
    }).toThrow('null');
  });

  it('names the field in the error message', () => {
    expectValidationError(() => requireField(undefined, 'mySpecialField'), 'mySpecialField');
  });
});

// ─── requireString ────────────────────────────────────────────

describe('requireString', () => {
  it('passes for a non-empty string', () => {
    expect(() => requireString('hello', 'name')).not.toThrow();
    expect(() => requireString('  spaces  ', 'name')).not.toThrow();
  });

  it('throws for non-string types', () => {
    expectValidationError(() => requireString(42, 'apiKey'), '"apiKey" must be a string');
    expectValidationError(() => requireString(null, 'apiUrl'), '"apiUrl" must be a string');
    expectValidationError(() => requireString(undefined, 'apiUrl'), '"apiUrl" must be a string');
    expectValidationError(() => requireString([], 'apiUrl'), '"apiUrl" must be a string');
  });

  it('throws for empty string', () => {
    expectValidationError(() => requireString('', 'apiKey'), '"apiKey" must not be an empty string');
  });

  it('throws for whitespace-only string', () => {
    expectValidationError(() => requireString('   ', 'apiKey'), '"apiKey" must not be an empty string');
  });
});

// ─── requireNumber ────────────────────────────────────────────

describe('requireNumber', () => {
  it('passes for finite numbers', () => {
    expect(() => requireNumber(0, 'timeout')).not.toThrow();
    expect(() => requireNumber(3.14, 'timeout')).not.toThrow();
    expect(() => requireNumber(-1, 'costWeight')).not.toThrow();
  });

  it('throws for non-numbers', () => {
    expectValidationError(() => requireNumber('10', 'timeout'), '"timeout" must be a finite number');
    expectValidationError(() => requireNumber(null, 'timeout'), '"timeout" must be a finite number');
    expectValidationError(() => requireNumber(undefined, 'timeout'), '"timeout" must be a finite number');
  });

  it('throws for NaN', () => {
    expectValidationError(() => requireNumber(NaN, 'batchSize'), '"batchSize" must be a finite number');
  });

  it('throws for Infinity', () => {
    expectValidationError(() => requireNumber(Infinity, 'timeout'), '"timeout" must be a finite number');
    expectValidationError(() => requireNumber(-Infinity, 'timeout'), '"timeout" must be a finite number');
  });
});

// ─── requireBoolean ───────────────────────────────────────────

describe('requireBoolean', () => {
  it('passes for true and false', () => {
    expect(() => requireBoolean(true, 'enableSkipping')).not.toThrow();
    expect(() => requireBoolean(false, 'enableSkipping')).not.toThrow();
  });

  it('throws for non-boolean', () => {
    expectValidationError(() => requireBoolean(1, 'enableSkipping'), '"enableSkipping" must be a boolean');
    expectValidationError(() => requireBoolean('true', 'enableSkipping'), '"enableSkipping" must be a boolean');
    expectValidationError(() => requireBoolean(null, 'enableSkipping'), '"enableSkipping" must be a boolean');
  });
});

// ─── requireFunction ──────────────────────────────────────────

describe('requireFunction', () => {
  it('passes for functions', () => {
    expect(() => requireFunction(() => {}, 'callback')).not.toThrow();
    expect(() => requireFunction(async () => {}, 'callback')).not.toThrow();
    expect(() => requireFunction(Math.random, 'rng')).not.toThrow();
  });

  it('throws for non-functions', () => {
    expectValidationError(() => requireFunction('fn', 'callback'), '"callback" must be a function');
    expectValidationError(() => requireFunction({}, 'callback'), '"callback" must be a function');
    expectValidationError(() => requireFunction(null, 'callback'), '"callback" must be a function');
  });
});

// ─── requireRange ─────────────────────────────────────────────

describe('requireRange', () => {
  it('passes at boundary values', () => {
    expect(() => requireRange(0, 'threshold', 0, 1)).not.toThrow();
    expect(() => requireRange(1, 'threshold', 0, 1)).not.toThrow();
    expect(() => requireRange(0.5, 'threshold', 0, 1)).not.toThrow();
    expect(() => requireRange(-1, 'costWeight', -1, 1)).not.toThrow();
  });

  it('throws when below minimum', () => {
    expectValidationError(
      () => requireRange(-0.01, 'skipConfidenceThreshold', 0, 1),
      '"skipConfidenceThreshold" must be between 0 and 1',
    );
  });

  it('throws when above maximum', () => {
    expectValidationError(
      () => requireRange(1.01, 'skipConfidenceThreshold', 0, 1),
      '"skipConfidenceThreshold" must be between 0 and 1',
    );
  });

  it('includes the actual value in the error', () => {
    expectValidationError(
      () => requireRange(2.5, 'costWeight', -1, 1),
      '2.5',
    );
  });
});

// ─── requireMin ───────────────────────────────────────────────

describe('requireMin', () => {
  it('passes at or above minimum', () => {
    expect(() => requireMin(0, 'timeout', 0)).not.toThrow();
    expect(() => requireMin(1, 'timeout', 0)).not.toThrow();
    expect(() => requireMin(100, 'batchSize', 1)).not.toThrow();
  });

  it('throws below minimum', () => {
    expectValidationError(() => requireMin(-1, 'timeout', 0), '"timeout" must be >= 0');
    expectValidationError(() => requireMin(0, 'batchSize', 1), '"batchSize" must be >= 1');
  });
});

// ─── requirePositive ──────────────────────────────────────────

describe('requirePositive', () => {
  it('passes for positive numbers', () => {
    expect(() => requirePositive(1, 'batchSize')).not.toThrow();
    expect(() => requirePositive(0.001, 'batchSize')).not.toThrow();
    expect(() => requirePositive(1000, 'timeout')).not.toThrow();
  });

  it('throws for zero', () => {
    expectValidationError(() => requirePositive(0, 'batchSize'), '"batchSize" must be a positive number');
  });

  it('throws for negative numbers', () => {
    expectValidationError(() => requirePositive(-5, 'timeout'), '"timeout" must be a positive number');
  });
});

// ─── requireNonEmptyArray ─────────────────────────────────────

describe('requireNonEmptyArray', () => {
  it('passes for non-empty arrays', () => {
    expect(() => requireNonEmptyArray(['a'], 'models')).not.toThrow();
    expect(() => requireNonEmptyArray([1, 2, 3], 'ids')).not.toThrow();
  });

  it('throws for empty array', () => {
    expectValidationError(() => requireNonEmptyArray([], 'models'), '"models" must be a non-empty array');
  });

  it('throws for non-array', () => {
    expectValidationError(() => requireNonEmptyArray('models', 'models'), '"models" must be an array');
    expectValidationError(() => requireNonEmptyArray(null, 'models'), '"models" must be an array');
  });
});

// ─── requireStringArray ───────────────────────────────────────

describe('requireStringArray', () => {
  it('passes for arrays of non-empty strings', () => {
    expect(() => requireStringArray(['claude-sonnet-4', 'gpt-4'], 'models')).not.toThrow();
  });

  it('throws if an element is not a string', () => {
    expectValidationError(
      // @ts-expect-error — intentionally passing wrong type to test runtime behavior
      () => requireStringArray([42, 'valid'], 'models'),
      '"models[0]" must be a non-empty string',
    );
  });

  it('throws if an element is an empty string', () => {
    expectValidationError(
      () => requireStringArray(['valid', ''], 'models'),
      '"models[1]" must be a non-empty string',
    );
  });
});

// ─── requireOneOf ─────────────────────────────────────────────

describe('requireOneOf', () => {
  const modes = ['shadow', 'advisory', 'active'] as const;

  it('passes for allowed values', () => {
    expect(() => requireOneOf('shadow', 'mode', modes)).not.toThrow();
    expect(() => requireOneOf('advisory', 'mode', modes)).not.toThrow();
    expect(() => requireOneOf('active', 'mode', modes)).not.toThrow();
  });

  it('throws for a value not in the allowed set', () => {
    expectValidationError(
      () => requireOneOf('passive', 'mode', modes),
      '"mode" must be one of',
    );
    expectValidationError(
      () => requireOneOf('passive', 'mode', modes),
      '"shadow"',
    );
  });

  it('includes the invalid value in the error', () => {
    expectValidationError(
      () => requireOneOf('unknown', 'mode', modes),
      '"unknown"',
    );
  });

  it('throws for undefined', () => {
    expectValidationError(() => requireOneOf(undefined, 'mode', modes), '"mode" must be one of');
  });
});
