/**
 * Seeded Random Number Generator
 *
 * Deterministic PRNG for reproducible benchmarks.
 * Uses Mulberry32 — fast, simple, good distribution.
 */

export class SeededRNG {
  private state: number;
  private initialSeed: number;

  constructor(seed: number) {
    this.initialSeed = seed;
    this.state = seed;
  }

  /**
   * Returns a random float in [0, 1).
   */
  next(): number {
    this.state |= 0;
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * Returns a random integer in [min, max).
   */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min)) + min;
  }

  /**
   * Pick a random element from an array.
   */
  pick<T>(arr: T[]): T {
    return arr[this.nextInt(0, arr.length)];
  }

  /**
   * Get the initial seed (for reproducibility reporting).
   */
  getSeed(): number {
    return this.initialSeed;
  }

  /**
   * Reset to initial seed.
   */
  reset(): void {
    this.state = this.initialSeed;
  }
}
