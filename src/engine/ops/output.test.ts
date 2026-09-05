import { describe, it, expect } from 'vitest';
import { softmaxOut, crossEntropy } from './output';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const m = { id: 'o', label: 'o', formula: '', explain: '' };

describe('softmaxOut', () => {
  it('has kind softmax_out and rows summing to one', () => {
    const ctx = randomCtx({ logits: [2, 4] }, 1);
    const op = softmaxOut(m, 'logits', 'probs');
    expect(op.kind).toBe('softmax_out');
    op.forward(ctx);
    const p = get(ctx, 'probs').data;
    expect(p[0] + p[1] + p[2] + p[3]).toBeCloseTo(1, 12);
  });
});

describe('crossEntropy', () => {
  it('is the mean negative log probability of the targets', () => {
    const ctx = randomCtx({ probs: [2, 3] }, 2);
    get(ctx, 'probs').data.set([0.5, 0.25, 0.25, 0.1, 0.8, 0.1]);
    ctx.targets = [0, 1]; ctx.T = 2;
    crossEntropy(m, 'probs', 'loss').forward(ctx);
    const L = get(ctx, 'loss');
    expect(L.shape).toEqual([1]);
    expect(L.data[0]).toBeCloseTo(-(Math.log(0.5) + Math.log(0.8)) / 2, 12);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ probs: [3, 4] }, 3);
    const p = get(ctx, 'probs').data; for (let i = 0; i < p.length; i++) p[i] = Math.abs(p[i]) + 0.1;
    ctx.targets = [1, 3, 0]; ctx.T = 3;
    expect(gradCheck(crossEntropy(m, 'probs', 'loss'), ctx, ['probs'])).toEqual([]);
  });
});
