import { describe, it, expect } from 'vitest';
import { EpochRunner } from './epoch';
import { GPT } from '../../engine/model/gpt';
import { Adam } from '../../engine/model/adam';
import { windows } from '../../engine/corpus/windows';
import { Rng } from '../../engine/rng';

describe('EpochRunner', () => {
  const cfg = { vocabSize: 10, contextSize: 4, dModel: 8, dFF: 8, seed: 1 };
  const tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7];
  it('reports metrics only when an epoch completes, and counts iterations', () => {
    const m = new GPT(cfg);
    const r = new EpochRunner(m, new Adam(m.params, { lr: 0.01 }), windows(tokens.slice(0, 12), 4), windows(tokens.slice(8), 4), new Rng(2));
    expect(r.step(3)).toBeNull();                       // 8 windows: 3 done
    expect(r.step(3)).toBeNull();                       // 6 done
    const out = r.step(100);                            // finishes the epoch
    expect(out && 'metrics' in out && out.metrics.iteration).toBe(1);
    if (out && 'metrics' in out) { expect(out.metrics.trainLoss).toBeGreaterThan(0); expect(out.metrics.testLoss).toBeGreaterThan(0); }
    expect(r.iteration).toBe(1);
  });
  it('gives NaN test metrics when there are no test windows', () => {
    const m = new GPT(cfg);
    const r = new EpochRunner(m, new Adam(m.params), windows(tokens, 4), [], new Rng(3));
    const out = r.step(1000);
    expect(out && 'metrics' in out && Number.isNaN(out.metrics.testLoss)).toBe(true);
  });
  it('reports divergence on a non-finite loss', () => {
    const m = new GPT(cfg);
    m.params.get('E')!.data.fill(NaN);
    const r = new EpochRunner(m, new Adam(m.params), windows(tokens, 4), [], new Rng(4));
    expect(r.step(1000)).toEqual({ diverged: true });
  });
});
