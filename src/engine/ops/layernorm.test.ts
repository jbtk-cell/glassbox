import { describe, it, expect } from 'vitest';
import { layerNorm } from './layernorm';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'ln', label: 'ln', formula: '', explain: '' };

describe('layerNorm', () => {
  it('normalises each row to mean 0, variance 1 when gamma=1, beta=0', () => {
    const ctx = randomCtx({ x: [3, 6], g: [6], b: [6] }, 1, 3);
    get(ctx, 'g').data.fill(1); get(ctx, 'b').data.fill(0);
    layerNorm(meta, 'x', 'g', 'b', 'y').forward(ctx);
    const y = get(ctx, 'y').data;
    for (let r = 0; r < 3; r++) {
      let m = 0, v = 0;
      for (let c = 0; c < 6; c++) m += y[r * 6 + c] / 6;
      for (let c = 0; c < 6; c++) v += (y[r * 6 + c] - m) ** 2 / 6;
      expect(m).toBeCloseTo(0, 10); expect(v).toBeCloseTo(1, 4);
    }
  });
  it('passes the gradient check for x, gamma and beta', () => {
    const ctx = randomCtx({ x: [3, 5], g: [5], b: [5] }, 2);
    expect(gradCheck(layerNorm(meta, 'x', 'g', 'b', 'y'), ctx, ['x', 'g', 'b'])).toEqual([]);
  });
});
