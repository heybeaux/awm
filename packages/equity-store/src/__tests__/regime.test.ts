import { describe, it, expect } from 'vitest';
import { classifyRegime } from '../regime.js';

describe('classifyRegime', () => {
  const medianVol = 0.02;

  it('classifies trending_up when 20d return > 5%', () => {
    expect(classifyRegime(0.06, 0.01, medianVol)).toBe('trending_up');
    expect(classifyRegime(0.20, 0.05, medianVol)).toBe('trending_up');
  });

  it('classifies trending_down when 20d return < -5%', () => {
    expect(classifyRegime(-0.06, 0.01, medianVol)).toBe('trending_down');
    expect(classifyRegime(-0.20, 0.05, medianVol)).toBe('trending_down');
  });

  it('classifies mean_reverting when not trending and vol > median', () => {
    expect(classifyRegime(0.01, 0.03, medianVol)).toBe('mean_reverting');
    expect(classifyRegime(-0.04, 0.05, medianVol)).toBe('mean_reverting');
  });

  it('classifies quiet when not trending and vol <= median', () => {
    expect(classifyRegime(0.01, 0.01, medianVol)).toBe('quiet');
    expect(classifyRegime(-0.03, 0.02, medianVol)).toBe('quiet');
    expect(classifyRegime(0.0, 0.0, medianVol)).toBe('quiet');
  });

  it('treats trend as dominant over volatility', () => {
    // Strong trend with very high vol → still trending, not mean_reverting
    expect(classifyRegime(0.10, 0.10, medianVol)).toBe('trending_up');
    expect(classifyRegime(-0.10, 0.10, medianVol)).toBe('trending_down');
  });

  it('handles boundary at exactly +/- 5% (strict inequality)', () => {
    // exactly 0.05 is not > 0.05 → falls through to vol check
    expect(classifyRegime(0.05, 0.01, medianVol)).toBe('quiet');
    expect(classifyRegime(-0.05, 0.01, medianVol)).toBe('quiet');
    expect(classifyRegime(0.05, 0.03, medianVol)).toBe('mean_reverting');
  });

  it('handles boundary at exactly median vol (strict inequality)', () => {
    // vol equal to median is not > median → quiet
    expect(classifyRegime(0.01, medianVol, medianVol)).toBe('quiet');
  });
});
