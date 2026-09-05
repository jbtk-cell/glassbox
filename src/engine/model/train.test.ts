import { describe, it, expect } from 'vitest';
import { GPT } from './gpt';
import { Adam } from './adam';
import { trainStep, evaluate } from './train';
import { windows, shuffled } from '../corpus/windows';
import { Rng } from '../rng';

describe('Adam', () => {
  it('moves each parameter against its gradient on the first step, by about lr', () => {
    const cfg = { vocabSize: 5, contextSize: 3, dModel: 4, dFF: 4, seed: 1 };
    const m = new GPT(cfg);
    const before = m.params.get('W_q')!.data[0];
    const ctx = m.forward([1, 2, 3], [2, 3, 4]); m.backward(ctx);
    const g = ctx.grads.get('W_q')!.data[0];
    const adam = new Adam(m.params, { lr: 0.01 });
    const deltas = adam.step(ctx.grads);
    const after = m.params.get('W_q')!.data[0];
    expect(Math.sign(after - before)).toBe(-Math.sign(g));
    expect(Math.abs(after - before)).toBeCloseTo(0.01, 3);   // Adam's first step is ~lr in magnitude
    expect(deltas.get('W_q')![0]).toBeCloseTo(after - before, 12);
  });
});

describe('training', () => {
  it('overfits a tiny corpus to near-zero loss', () => {
    // 30 tokens: a repeating 9-token cycle, so every next token is fully determined and loss can reach ~0.
    const tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3];
    const cfg = { vocabSize: 10, contextSize: 6, dModel: 8, dFF: 16, seed: 2 };
    const m = new GPT(cfg);
    const adam = new Adam(m.params, { lr: 0.01 });
    const ws = windows(tokens, 6);
    const rng = new Rng(3);
    let steps = 0, loss = Infinity;
    while (steps < 3000) {
      for (const w of shuffled(ws, rng)) { loss = trainStep(m, adam, w).loss; steps++; }
      loss = evaluate(m, ws).loss;
      if (loss < 0.05) break;
    }
    expect(loss).toBeLessThan(0.05);
    expect(evaluate(m, ws).accuracy).toBeGreaterThan(0.95);
    console.log(`overfit reached loss ${loss.toFixed(4)} after ${steps} window steps`);
  });
  it('evaluate reports loss and accuracy in range', () => {
    const cfg = { vocabSize: 6, contextSize: 3, dModel: 4, dFF: 4, seed: 4 };
    const r = evaluate(new GPT(cfg), windows([1, 2, 3, 4, 5, 1, 2], 3));
    expect(r.loss).toBeGreaterThan(0); expect(r.accuracy).toBeGreaterThanOrEqual(0); expect(r.accuracy).toBeLessThanOrEqual(1);
  });
});
