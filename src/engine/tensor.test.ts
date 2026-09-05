import { describe, it, expect } from 'vitest';
import { tensor, size, zerosLike } from './tensor';

describe('tensor', () => {
  it('size multiplies the shape', () => { expect(size([3, 4])).toBe(12); expect(size([])).toBe(1); });
  it('tensor() allocates zeros of the right length', () => {
    const t = tensor('x', [2, 3]);
    expect(t.data.length).toBe(6); expect(Array.from(t.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });
  it('tensor() rejects mismatched data', () => { expect(() => tensor('x', [2], new Float64Array(3))).toThrow(); });
  it('zerosLike copies shape, not data', () => {
    const t = tensor('x', [2], new Float64Array([1, 2]));
    const z = zerosLike(t, 'z');
    expect(z.shape).toEqual([2]); expect(Array.from(z.data)).toEqual([0, 0]); expect(z.name).toBe('z');
  });
});
