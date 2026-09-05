import { describe, it, expect } from 'vitest';
import { splitTokens, windows, shuffled, minTokens } from './windows';
import { Rng } from '../rng';

describe('splitTokens', () => {
  it('splits contiguously at the ratio', () => {
    const { train, test } = splitTokens([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.6);
    expect(train).toEqual([0, 1, 2, 3, 4, 5]);
    expect(test).toEqual([6, 7, 8, 9]);
  });
});

describe('windows', () => {
  it('produces every stride-1 window with targets shifted by one', () => {
    const w = windows([10, 11, 12, 13, 14], 3);
    expect(w).toEqual([
      { input: [10, 11, 12], target: [11, 12, 13] },
      { input: [11, 12, 13], target: [12, 13, 14] },
    ]);
  });
  it('returns nothing when the stream is too short', () => { expect(windows([1, 2, 3], 3)).toEqual([]); });
});

describe('shuffled', () => {
  it('is a permutation and deterministic for a seed', () => {
    const items = [1, 2, 3, 4, 5, 6, 7, 8];
    const a = shuffled(items, new Rng(9)), b = shuffled(items, new Rng(9));
    expect(a).toEqual(b);
    expect([...a].sort((x, y) => x - y)).toEqual(items);
    expect(items).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

it('minTokens is 2T+2', () => { expect(minTokens(24)).toBe(50); });
