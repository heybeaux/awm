/**
 * AWM Config Validation
 *
 * Shared validation helpers for dynamic configuration inputs
 * (model names, thresholds, file paths, timeouts, batch sizes, etc.)
 * passed at runtime to data-processing modules.
 *
 * All validators throw a descriptive ConfigValidationError that names the
 * invalid field and expected value/range so callers can fix the problem.
 */

export class ConfigValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigValidationError';
  }
}

// ─── Primitive checks ─────────────────────────────────────────

/**
 * Assert that a required field is present and not null/undefined.
 */
export function requireField<T>(
  value: T | null | undefined,
  fieldName: string,
): asserts value is T {
  if (value === null || value === undefined) {
    throw new ConfigValidationError(
      `"${fieldName}" is required but was ${value === null ? 'null' : 'undefined'}.`,
    );
  }
}

/**
 * Assert that a value is a non-empty string.
 */
export function requireString(value: unknown, fieldName: string): asserts value is string {
  if (typeof value !== 'string') {
    throw new ConfigValidationError(
      `"${fieldName}" must be a string, got ${typeof value}.`,
    );
  }
  if (value.trim() === '') {
    throw new ConfigValidationError(
      `"${fieldName}" must not be an empty string.`,
    );
  }
}

/**
 * Assert that a value is a number (not NaN, not Infinity).
 */
export function requireNumber(value: unknown, fieldName: string): asserts value is number {
  if (typeof value !== 'number' || !isFinite(value)) {
    throw new ConfigValidationError(
      `"${fieldName}" must be a finite number, got ${JSON.stringify(value)}.`,
    );
  }
}

/**
 * Assert that a value is a boolean.
 */
export function requireBoolean(value: unknown, fieldName: string): asserts value is boolean {
  if (typeof value !== 'boolean') {
    throw new ConfigValidationError(
      `"${fieldName}" must be a boolean, got ${typeof value}.`,
    );
  }
}

/**
 * Assert that a value is a function.
 */
// eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
export function requireFunction(value: unknown, fieldName: string): asserts value is Function {
  if (typeof value !== 'function') {
    throw new ConfigValidationError(
      `"${fieldName}" must be a function, got ${typeof value}.`,
    );
  }
}

// ─── Range checks ─────────────────────────────────────────────

/**
 * Assert that a numeric value falls within [min, max] inclusive.
 */
export function requireRange(
  value: number,
  fieldName: string,
  min: number,
  max: number,
): void {
  if (value < min || value > max) {
    throw new ConfigValidationError(
      `"${fieldName}" must be between ${min} and ${max} (inclusive), got ${value}.`,
    );
  }
}

/**
 * Assert that a numeric value is >= min.
 */
export function requireMin(value: number, fieldName: string, min: number): void {
  if (value < min) {
    throw new ConfigValidationError(
      `"${fieldName}" must be >= ${min}, got ${value}.`,
    );
  }
}

/**
 * Assert that a numeric value is > 0 (i.e. a positive count/size).
 */
export function requirePositive(value: number, fieldName: string): void {
  if (value <= 0) {
    throw new ConfigValidationError(
      `"${fieldName}" must be a positive number (> 0), got ${value}.`,
    );
  }
}

// ─── Array checks ─────────────────────────────────────────────

/**
 * Assert that a value is a non-empty array.
 */
export function requireNonEmptyArray<T>(
  value: unknown,
  fieldName: string,
): asserts value is T[] {
  if (!Array.isArray(value)) {
    throw new ConfigValidationError(
      `"${fieldName}" must be an array, got ${typeof value}.`,
    );
  }
  if (value.length === 0) {
    throw new ConfigValidationError(
      `"${fieldName}" must be a non-empty array.`,
    );
  }
}

/**
 * Assert that every element of an array is a non-empty string.
 * Useful for model name lists.
 */
export function requireStringArray(value: string[], fieldName: string): void {
  value.forEach((item, i) => {
    if (typeof item !== 'string' || item.trim() === '') {
      throw new ConfigValidationError(
        `"${fieldName}[${i}]" must be a non-empty string, got ${JSON.stringify(item)}.`,
      );
    }
  });
}

// ─── Enum checks ──────────────────────────────────────────────

/**
 * Assert that a value is one of a set of allowed literals.
 */
export function requireOneOf<T extends string>(
  value: unknown,
  fieldName: string,
  allowed: readonly T[],
): asserts value is T {
  if (!allowed.includes(value as T)) {
    throw new ConfigValidationError(
      `"${fieldName}" must be one of [${allowed.map((v) => JSON.stringify(v)).join(', ')}], got ${JSON.stringify(value)}.`,
    );
  }
}
