import { describe, it, expect } from 'vitest';
import { networkLayout, rowY, X_STEP } from './networkLayout';

const shapes = (T: number, V = 289, d = 20, dFF = 30): Record<string, number[]> => ({
  tok: [T, d], pos: [T, d], x0: [T, d], h1: [T, d], q: [T, d], k: [T, d], v: [T, d], scores: [T, T], masked: [T, T], attn: [T, T],
  ctxv: [T, d], attn_out: [T, d], x1: [T, d], h2: [T, d], ff_pre: [T, dFF], ff_act: [T, dFF], ff_out: [T, d], x2: [T, d], hf: [T, d],
  logits: [T, V], probs: [T, V], W_q: [d, d], W_k: [d, d], W_v: [d, d], W_o: [d, d], W_1: [d, dFF], W_2: [dFF, d], U: [d, V],
});

describe('networkLayout', () => {
  it('puts every stage in a column, stacking tok/pos and q/k/v', () => {
    const L = networkLayout(shapes(24));
    expect(L.groups.map(g => g.key)).toEqual(['tok', 'pos', 'x0', 'h1', 'q', 'k', 'v', 'attn', 'ctxv', 'attn_out', 'x1', 'h2', 'ff_pre', 'ff_act', 'ff_out', 'x2', 'hf', 'logits', 'probs']);
    expect(L.byKey.get('tok')!.x).toBe(L.byKey.get('pos')!.x);
    expect(L.byKey.get('q')!.x).toBe(L.byKey.get('v')!.x);
    expect(L.byKey.get('x0')!.x).toBe(X_STEP);
    expect(L.byKey.get('probs')!.x).toBe(15 * X_STEP);
  });
  it('connects every neighbouring column with a band', () => {
    const L = networkLayout(shapes(24));
    const cols = [...new Set(L.groups.map(g => g.x))].sort((a, b) => a - b);
    for (let i = 1; i < cols.length; i++) {
      const to = L.groups.filter(g => g.x === cols[i]).map(g => g.key);
      expect(L.bands.some(b => to.includes(b.to)), `column ${i}`).toBe(true);
    }
    expect(L.bands.filter(b => b.kind === 'matrix').map(b => b.param)).toEqual(['W_q', 'W_k', 'W_v', 'W_o', 'W_1', 'W_2', 'U']);
  });
  it('stacks groups without overlap and sizes the attention column by T', () => {
    const L = networkLayout(shapes(5));
    const attn = L.byKey.get('attn')!; expect(attn.n).toBe(5); expect(attn.words).toBe(true);
    const q = L.byKey.get('q')!, k = L.byKey.get('k')!;
    expect(rowY(k, 0) - k.r).toBeGreaterThan(rowY(q, q.n - 1) + q.r);
    const logits = L.byKey.get('logits')!; expect(logits.compact).toBe(true); expect(logits.n * logits.dy).toBeLessThanOrEqual(620);
  });
});
