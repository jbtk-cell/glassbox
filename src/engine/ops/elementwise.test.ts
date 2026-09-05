import { describe, it, expect } from 'vitest';
import { add, relu } from './elementwise';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'op', label: 'op', formula: '', explain: '' };

describe('add', () => {
  it('adds elementwise', () => {
    const ctx = randomCtx({ a: [2, 2], b: [2, 2] }, 1);
    const op = add(meta, 'a', 'b', 'y');
    op.forward(ctx);
    const a = get(ctx, 'a').data, b = get(ctx, 'b').data, y = get(ctx, 'y').data;
    for (let i = 0; i < 4; i++) expect(y[i]).toBeCloseTo(a[i] + b[i], 12);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ a: [3, 4], b: [3, 4] }, 2);
    expect(gradCheck(add(meta, 'a', 'b', 'y'), ctx, ['a', 'b'])).toEqual([]);
  });
});

describe('relu', () => {
  it('clamps negatives to zero', () => {
    const ctx = randomCtx({ x: [1, 3] }, 3);
    get(ctx, 'x').data.set([-1, 0.5, 2]);
    relu(meta, 'x', 'y').forward(ctx);
    expect(Array.from(get(ctx, 'y').data)).toEqual([0, 0.5, 2]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ x: [3, 5] }, 4);
    expect(gradCheck(relu(meta, 'x', 'y'), ctx, ['x'])).toEqual([]);
  });
});
