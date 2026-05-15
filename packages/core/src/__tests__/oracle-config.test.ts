/**
 * GIN-36 — OracleConfig validation tests.
 *
 * Covers rejection of invalid dynamic config inputs passed to Oracle
 * at construction time.
 */

import { describe, it, expect } from 'vitest';
import { Oracle } from '../oracle.js';
import { InMemoryStore } from '../store.js';
import { ConfigValidationError } from '../validation.js';

const store = new InMemoryStore();

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

describe('Oracle — config validation', () => {
  // ─── store (required) ───

  describe('store', () => {
    it('throws when store is missing', () => {
      // @ts-expect-error — testing runtime behavior with missing required field
      expectConfigError(() => new Oracle({}), '"store" is required');
    });

    it('throws when store is null', () => {
      // @ts-expect-error — testing runtime behavior with null
      expectConfigError(() => new Oracle({ store: null }), '"store" is required');
    });

    it('accepts a valid store', () => {
      expect(() => new Oracle({ store })).not.toThrow();
    });
  });

  // ─── costWeight (optional, -1 to 1) ───

  describe('costWeight', () => {
    it('accepts valid values in range', () => {
      expect(() => new Oracle({ store, costWeight: 0 })).not.toThrow();
      expect(() => new Oracle({ store, costWeight: 0.3 })).not.toThrow();
      expect(() => new Oracle({ store, costWeight: 1 })).not.toThrow();
      expect(() => new Oracle({ store, costWeight: -1 })).not.toThrow();
    });

    it('accepts undefined (uses default)', () => {
      expect(() => new Oracle({ store, costWeight: undefined })).not.toThrow();
    });

    it('throws when costWeight is above 1', () => {
      expectConfigError(() => new Oracle({ store, costWeight: 1.1 }), '"costWeight" must be between -1 and 1');
    });

    it('throws when costWeight is below -1', () => {
      expectConfigError(() => new Oracle({ store, costWeight: -1.5 }), '"costWeight" must be between -1 and 1');
    });

    it('throws when costWeight is NaN', () => {
      expectConfigError(() => new Oracle({ store, costWeight: NaN }), '"costWeight" must be a finite number');
    });

    it('throws when costWeight is Infinity', () => {
      expectConfigError(() => new Oracle({ store, costWeight: Infinity }), '"costWeight" must be a finite number');
    });

    it('throws when costWeight is a string', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new Oracle({ store, costWeight: '0.3' }), '"costWeight" must be a finite number');
    });
  });

  // ─── skipConfidenceThreshold (optional, 0-1) ───

  describe('skipConfidenceThreshold', () => {
    it('accepts valid values', () => {
      expect(() => new Oracle({ store, skipConfidenceThreshold: 0 })).not.toThrow();
      expect(() => new Oracle({ store, skipConfidenceThreshold: 0.85 })).not.toThrow();
      expect(() => new Oracle({ store, skipConfidenceThreshold: 1 })).not.toThrow();
    });

    it('throws when below 0', () => {
      expectConfigError(
        () => new Oracle({ store, skipConfidenceThreshold: -0.1 }),
        '"skipConfidenceThreshold" must be between 0 and 1',
      );
    });

    it('throws when above 1', () => {
      expectConfigError(
        () => new Oracle({ store, skipConfidenceThreshold: 1.5 }),
        '"skipConfidenceThreshold" must be between 0 and 1',
      );
    });

    it('throws when not a number', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => new Oracle({ store, skipConfidenceThreshold: 'high' }), '"skipConfidenceThreshold" must be a finite number');
    });
  });

  // ─── routingConfidenceThreshold (optional, 0-1) ───

  describe('routingConfidenceThreshold', () => {
    it('accepts valid values', () => {
      expect(() => new Oracle({ store, routingConfidenceThreshold: 0 })).not.toThrow();
      expect(() => new Oracle({ store, routingConfidenceThreshold: 0.7 })).not.toThrow();
      expect(() => new Oracle({ store, routingConfidenceThreshold: 1 })).not.toThrow();
    });

    it('throws when below 0', () => {
      expectConfigError(
        () => new Oracle({ store, routingConfidenceThreshold: -0.01 }),
        '"routingConfidenceThreshold" must be between 0 and 1',
      );
    });

    it('throws when above 1', () => {
      expectConfigError(
        () => new Oracle({ store, routingConfidenceThreshold: 2 }),
        '"routingConfidenceThreshold" must be between 0 and 1',
      );
    });

    it('throws when NaN', () => {
      expectConfigError(
        () => new Oracle({ store, routingConfidenceThreshold: NaN }),
        '"routingConfidenceThreshold" must be a finite number',
      );
    });
  });

  // ─── Multiple invalid fields ───

  it('names the specific invalid field in the error', () => {
    let msg = '';
    try {
      new Oracle({ store, costWeight: 99 });
    } catch (e) {
      msg = (e as ConfigValidationError).message;
    }
    expect(msg).toContain('costWeight');
    expect(msg).toContain('99');
  });
});
