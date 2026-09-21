import { describe, it, expect } from 'vitest';
import { diverging, sequential, gradientMap, maxAbs, robustMax, isMasked, MASK_COLOR } from './colormap';

describe('diverging', () => {
  it('is near-white at 0', () => {
    const [r, g, b] = diverging(0, 1);
    expect(r).toBeCloseTo(247); expect(g).toBeCloseTo(247); expect(b).toBeCloseTo(247);
  });
  it('is red-dominant at +1', () => {
    const [r, g, b] = diverging(1, 1);
    expect(r).toBeCloseTo(178); expect(g).toBeCloseTo(24); expect(b).toBeCloseTo(43);
    expect(r).toBeGreaterThan(b);
  });
  it('is blue-dominant at -1', () => {
    const [r, g, b] = diverging(-1, 1);
    expect(r).toBeCloseTo(33); expect(g).toBeCloseTo(102); expect(b).toBeCloseTo(172);
    expect(b).toBeGreaterThan(r);
  });
  it('clamps beyond maxAbs', () => {
    expect(diverging(5, 1)).toEqual(diverging(1, 1));
    expect(diverging(-5, 1)).toEqual(diverging(-1, 1));
  });
  it('maxAbs 0 renders white regardless of v', () => {
    expect(diverging(7, 0)).toEqual([247, 247, 247]);
  });
});

describe('sequential', () => {
  it('is white at 0 and dark teal at 1', () => {
    expect(sequential(0)).toEqual([255, 255, 255]);
    expect(sequential(1)).toEqual([1, 102, 94]);
  });
  it('clamps to [0, 1]', () => {
    expect(sequential(-3)).toEqual(sequential(0));
    expect(sequential(3)).toEqual(sequential(1));
  });
});

describe('gradientMap', () => {
  it('is purple-dominant at -1 and green-dominant at +1', () => {
    const neg = gradientMap(-1, 1);
    const pos = gradientMap(1, 1);
    expect(neg).toEqual([118, 42, 131]);
    expect(pos).toEqual([27, 120, 55]);
  });
  it('maxAbs 0 renders white', () => {
    expect(gradientMap(4, 0)).toEqual([247, 247, 247]);
  });
});

describe('maxAbs', () => {
  it('handles empty data', () => {
    expect(maxAbs(new Float64Array([]))).toBe(0);
  });
  it('returns the largest absolute value', () => {
    expect(maxAbs(new Float64Array([-1, 3, -7, 2]))).toBe(7);
  });
});

describe('isMasked', () => {
  it('is true at or below -1e8', () => {
    expect(isMasked(-1e8)).toBe(true);
    expect(isMasked(-2e8)).toBe(true);
  });
  it('is false above -1e8', () => {
    expect(isMasked(-1e7)).toBe(false);
    expect(isMasked(0)).toBe(false);
  });
});

describe('MASK_COLOR', () => {
  it('is mid grey', () => {
    expect(MASK_COLOR).toEqual([160, 160, 160]);
  });
});

describe('robustMax', () => {
  it('ignores a single outlier in a large tensor', () => {
    const d = new Float64Array(1000).fill(0.5); d[7] = 100;
    expect(robustMax(d)).toBe(0.5);
  });
  it('is the plain max for small tensors', () => {
    expect(robustMax(new Float64Array([1, -3, 2]))).toBe(3);
  });
});
