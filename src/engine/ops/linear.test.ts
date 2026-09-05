import { describe, it, expect } from 'vitest';
import { linear } from './linear';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'lin', label: 'lin', formula: '', explain: '' };

describe('linear', () => {
  it('computes x W + b', () => {
    const ctx = randomCtx({ x: [2, 2], W: [2, 3], b: [3] }, 1);
    get(ctx, 'x').data.set([1, 2, 3, 4]);
    get(ctx, 'W').data.set([1, 0, 1, 0, 1, 1]);
    get(ctx, 'b').data.set([10, 20, 30]);
    linear(meta, 'x', 'W', 'b', 'y').forward(ctx);
    expect(Array.from(get(ctx, 'y').data)).toEqual([11, 22, 33, 13, 24, 37]);
    expect(get(ctx, 'y').shape).toEqual([2, 3]);
  });
  it('passes the gradient check for x, W and b', () => {
    const ctx = randomCtx({ x: [3, 4], W: [4, 5], b: [5] }, 2);
    expect(gradCheck(linear(meta, 'x', 'W', 'b', 'y'), ctx, ['x', 'W', 'b'])).toEqual([]);
  });
});
