import { describe, it, expect } from 'vitest';
import { sampleFromLogits, generate, SAMPLE_DEFAULTS } from './generate';
import { GPT } from './gpt';
import { Rng } from '../rng';

const logits = new Float64Array([0.1, 2.0, 1.0, -1.0, 1.5]);   // argmax = 1, order 1,4,2,0,3

describe('sampleFromLogits', () => {
  it('greedy returns the argmax', () => {
    expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'greedy' }, new Rng(1)).token).toBe(1);
  });
  it('a near-zero temperature is effectively greedy', () => {
    const rng = new Rng(2);
    for (let i = 0; i < 50; i++) expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, temperature: 1e-6, strategy: 'top-p', p: 1 }, rng).token).toBe(1);
  });
  it('top-k never samples outside the top k', () => {
    const rng = new Rng(3);
    for (let i = 0; i < 200; i++) expect([1, 4]).toContain(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-k', k: 2 }, rng).token);
  });
  it('top-p with a tiny p is greedy; with p=1 every token is possible', () => {
    expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-p', p: 0.01 }, new Rng(4)).token).toBe(1);
    const seen = new Set<number>(); const rng = new Rng(5);
    for (let i = 0; i < 2000; i++) seen.add(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-p', p: 1, temperature: 3 }, rng).token);
    expect(seen.size).toBe(5);
  });
  it('returned probs sum to one', () => {
    const { probs } = sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-k', k: 3 }, new Rng(6));
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(probs[3]).toBe(0);
  });
});

describe('generate', () => {
  const cfg = { vocabSize: 9, contextSize: 4, dModel: 4, dFF: 4, seed: 7 };
  it('returns the prompt followed by n tokens inside the vocabulary', () => {
    const out = generate(new GPT(cfg), [1, 2], 5, SAMPLE_DEFAULTS, new Rng(8));
    expect(out.length).toBe(7); expect(out.slice(0, 2)).toEqual([1, 2]);
    for (const t of out) { expect(t).toBeGreaterThanOrEqual(0); expect(t).toBeLessThan(9); }
  });
  it('handles prompts longer than the context by using the last contextSize tokens', () => {
    const out = generate(new GPT(cfg), [1, 2, 3, 4, 5, 6], 2, { ...SAMPLE_DEFAULTS, strategy: 'greedy' }, new Rng(9));
    expect(out.length).toBe(8);
  });
});
