import { describe, it, expect } from 'vitest';
import { GPT, T_ } from './gpt';
import { initParams, paramCount, PARAM_NAMES } from './params';
import { get } from '../ops/types';

const cfg = { vocabSize: 7, contextSize: 5, dModel: 4, dFF: 6, seed: 3 };

describe('params', () => {
  it('creates every named parameter with the documented shapes', () => {
    const p = initParams(cfg);
    expect([...p.keys()]).toEqual([...PARAM_NAMES]);
    expect(p.get('E')!.shape).toEqual([7, 4]);
    expect(p.get('P')!.shape).toEqual([5, 4]);
    expect(p.get('W_1')!.shape).toEqual([4, 6]);
    expect(p.get('U')!.shape).toEqual([4, 7]);
    expect(Array.from(p.get('ln1_g')!.data)).toEqual([1, 1, 1, 1]);
    expect(Array.from(p.get('b_q')!.data)).toEqual([0, 0, 0, 0]);
  });
  it('counts parameters', () => {
    // E 28 + P 20 + 4*(16+4) + LN 3*8 + W_1 24 + b_1 6 + W_2 24 + b_2 4 + U 28 + b_u 7
    expect(paramCount(initParams(cfg))).toBe(28 + 20 + 80 + 24 + 24 + 6 + 24 + 4 + 28 + 7);
  });
});

describe('GPT.forward', () => {
  it('produces probabilities over the vocabulary for each position', () => {
    const m = new GPT(cfg);
    const ctx = m.forward([1, 2, 3]);
    const probs = get(ctx, T_.probs);
    expect(probs.shape).toEqual([3, 7]);
    for (let t = 0; t < 3; t++) { let s = 0; for (let v = 0; v < 7; v++) s += probs.data[t * 7 + v]; expect(s).toBeCloseTo(1, 12); }
    expect(ctx.tensors.has(T_.loss)).toBe(false);
  });
  it('starts near uniform: loss ~ ln(V) at init', () => {
    const m = new GPT(cfg);
    const ctx = m.forward([1, 2, 3, 4], [2, 3, 4, 5]);
    expect(Math.abs(get(ctx, T_.loss).data[0] - Math.log(7))).toBeLessThan(0.1);
  });
  it('is deterministic for a seed', () => {
    const a = new GPT(cfg).forward([1, 2]), b = new GPT(cfg).forward([1, 2]);
    expect(Array.from(get(a, T_.probs).data)).toEqual(Array.from(get(b, T_.probs).data));
  });
  it('rejects prompts longer than the context', () => {
    expect(() => new GPT(cfg).forward([1, 2, 3, 4, 5, 6])).toThrow();
  });
});

describe('GPT.backward (end-to-end gradient check)', () => {
  it('matches central finite differences of the loss for every parameter', () => {
    const m = new GPT(cfg);
    const tokens = [1, 2, 3, 4, 5], targets = [2, 3, 4, 5, 6];
    const ctx = m.forward(tokens, targets);
    m.backward(ctx);
    const eps = 1e-6, tol = 1e-5;
    const failures: string[] = [];
    for (const name of PARAM_NAMES) {
      const p = m.params.get(name)!; const g = ctx.grads.get(name)!;
      for (let i = 0; i < p.data.length; i++) {
        const v = p.data[i];
        p.data[i] = v + eps; const lp = get(m.forward(tokens, targets), T_.loss).data[0];
        p.data[i] = v - eps; const lm = get(m.forward(tokens, targets), T_.loss).data[0];
        p.data[i] = v;
        const num = (lp - lm) / (2 * eps);
        if (Math.abs(g.data[i] - num) > tol * Math.max(1, Math.abs(num))) failures.push(`${name}[${i}] analytic=${g.data[i]} numeric=${num}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('GPT params round trip', () => {
  it('exports and imports plain arrays', () => {
    const a = new GPT(cfg); a.setParam('W_q', 3, 0.5);
    const b = new GPT(cfg); b.importParams(a.exportParams());
    expect(b.params.get('W_q')!.data[3]).toBe(0.5);
    expect(Array.from(get(b.forward([1, 2]), T_.probs).data)).toEqual(Array.from(get(a.forward([1, 2]), T_.probs).data));
  });
});
