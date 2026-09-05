import { describe, it, expect } from 'vitest';
import { scores, causalMask, softmaxRows, attnApply, MASK_VALUE } from './attention';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const m = { id: 'a', label: 'a', formula: '', explain: '' };

describe('scores', () => {
  it('is q k^T scaled by 1/sqrt(d)', () => {
    const ctx = randomCtx({ q: [2, 4], k: [2, 4] }, 1);
    get(ctx, 'q').data.set([1, 0, 0, 0, 0, 1, 0, 0]);
    get(ctx, 'k').data.set([2, 0, 0, 0, 0, 4, 0, 0]);
    scores(m, 'q', 'k', 's').forward(ctx);
    expect(Array.from(get(ctx, 's').data)).toEqual([1, 0, 0, 2]);  // /sqrt(4) = /2
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ q: [3, 4], k: [3, 4] }, 2);
    expect(gradCheck(scores(m, 'q', 'k', 's'), ctx, ['q', 'k'])).toEqual([]);
  });
});

describe('causalMask', () => {
  it('masks strictly-future positions', () => {
    const ctx = randomCtx({ s: [3, 3] }, 3);
    get(ctx, 's').data.fill(1);
    causalMask(m, 's', 'o').forward(ctx);
    expect(Array.from(get(ctx, 'o').data)).toEqual([1, MASK_VALUE, MASK_VALUE, 1, 1, MASK_VALUE, 1, 1, 1]);
  });
  it('passes the gradient check (zero grad through masked cells)', () => {
    const ctx = randomCtx({ s: [4, 4] }, 4);
    // MASK_VALUE (-1e9) dwarfs the unmasked signal in the objective's sum, so the
    // default eps=1e-6 central difference is swept by float64 rounding noise at that
    // magnitude; a larger eps keeps the (still-exact, since this op is piecewise
    // linear) numeric derivative above that noise floor.
    expect(gradCheck(causalMask(m, 's', 'o'), ctx, ['s'], { eps: 1e-2 })).toEqual([]);
  });
});

describe('softmaxRows', () => {
  it('rows sum to one and masked entries become exactly zero', () => {
    const ctx = randomCtx({ s: [2, 3] }, 5);
    get(ctx, 's').data.set([0, MASK_VALUE, MASK_VALUE, 1, 2, MASK_VALUE]);
    softmaxRows(m, 's', 'a').forward(ctx);
    const a = Array.from(get(ctx, 'a').data);
    expect(a[0]).toBe(1); expect(a[1]).toBe(0); expect(a[2]).toBe(0);
    expect(a[3] + a[4]).toBeCloseTo(1, 12); expect(a[5]).toBe(0);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ s: [3, 5] }, 6);
    expect(gradCheck(softmaxRows(m, 's', 'a'), ctx, ['s'])).toEqual([]);
  });
});

describe('attnApply', () => {
  it('mixes value rows by attention weights', () => {
    const ctx = randomCtx({ a: [2, 2], v: [2, 2] }, 7);
    get(ctx, 'a').data.set([1, 0, 0.5, 0.5]);
    get(ctx, 'v').data.set([2, 4, 6, 8]);
    attnApply(m, 'a', 'v', 'o').forward(ctx);
    expect(Array.from(get(ctx, 'o').data)).toEqual([2, 4, 4, 6]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ a: [3, 3], v: [3, 4] }, 8);
    expect(gradCheck(attnApply(m, 'a', 'v', 'o'), ctx, ['a', 'v'])).toEqual([]);
  });
});
