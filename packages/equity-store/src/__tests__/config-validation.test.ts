/**
 * GIN-36 — EquityStore config validation tests.
 *
 * Covers rejection of invalid dynamic config inputs passed to
 * EquityStore at construction time.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { EquityStore } from '../equity-store.js';
import { ConfigValidationError } from '@heybeaux/awm-core';

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

describe('EquityStore — config validation', () => {
  const stores: EquityStore[] = [];

  afterEach(() => {
    for (const s of stores) {
      try { s.close(); } catch { /* already closed */ }
    }
    stores.length = 0;
  });

  function make(opts?: ConstructorParameters<typeof EquityStore>[0]): EquityStore {
    const s = new EquityStore(opts);
    stores.push(s);
    return s;
  }

  describe('default (no options)', () => {
    it('constructs successfully with no options', () => {
      expect(() => make()).not.toThrow();
    });

    it('constructs with empty options object', () => {
      expect(() => make({})).not.toThrow();
    });
  });

  describe('dbPath (optional, non-empty string)', () => {
    it('accepts ":memory:" explicitly', () => {
      expect(() => make({ dbPath: ':memory:' })).not.toThrow();
    });

    it('throws when dbPath is an empty string', () => {
      expectConfigError(
        () => make({ dbPath: '' }),
        '"dbPath" must not be an empty string',
      );
    });

    it('throws when dbPath is a whitespace-only string', () => {
      expectConfigError(
        () => make({ dbPath: '   ' }),
        '"dbPath" must not be an empty string',
      );
    });

    it('throws when dbPath is not a string', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => make({ dbPath: 42 }), '"dbPath" must be a string');
    });

    it('throws when dbPath is null', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => make({ dbPath: null }), '"dbPath" must be a string');
    });
  });

  describe('wal (optional, boolean)', () => {
    it('accepts true', () => {
      // Only valid with ":memory:" to avoid filesystem writes in tests
      expect(() => make({ dbPath: ':memory:', wal: true })).not.toThrow();
    });

    it('accepts false', () => {
      expect(() => make({ dbPath: ':memory:', wal: false })).not.toThrow();
    });

    it('accepts undefined', () => {
      expect(() => make({ dbPath: ':memory:', wal: undefined })).not.toThrow();
    });

    it('throws when wal is a string', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => make({ wal: 'true' }), '"wal" must be a boolean');
    });

    it('throws when wal is a number', () => {
      // @ts-expect-error — testing runtime behavior
      expectConfigError(() => make({ wal: 1 }), '"wal" must be a boolean');
    });
  });

  describe('error message quality', () => {
    it('names the invalid field in the error', () => {
      let msg = '';
      try {
        // @ts-expect-error — testing runtime behavior
        make({ dbPath: 123 });
      } catch (e) {
        msg = (e as ConfigValidationError).message;
      }
      expect(msg).toContain('dbPath');
    });
  });
});
