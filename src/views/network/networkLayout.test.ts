import { describe, it, expect } from 'vitest';
import { networkLayout, COLUMN_ORDER } from './networkLayout';
import { GPT } from '../../engine/model/gpt';
import { recordForward } from '../../engine/trace/trace';
import { PARAM_NAMES } from '../../engine/model/params';

const cfg = { vocabSize: 300, contextSize: 24, dModel: 20, dFF: 30, seed: 1 };

function shapesOf(ctx: { tensors: Map<string, { shape: number[] }> }): Record<string, number[]> {
  return Object.fromEntries([...ctx.tensors].map(([k, t]) => [k, t.shape]));
}

describe('networkLayout', () => {
  const model = new GPT(cfg);
  const trace = recordForward(model, [1, 2, 3, 4, 5]);
  const shapes = shapesOf(trace.ctx);
  const viewport = { w: 1400, h: 700 };
  const { columns, wires } = networkLayout(shapes, viewport);

  it('has one column per activation tensor, in the order the plan lists', () => {
    expect(columns.map(c => c.key)).toEqual([...COLUMN_ORDER]);
  });

  it('never overlaps two columns horizontally', () => {
    for (let i = 0; i < columns.length; i++) {
      for (let j = i + 1; j < columns.length; j++) {
        const a = columns[i], b = columns[j];
        expect(Math.abs(a.x - b.x), `${a.key} vs ${b.key}`).toBeGreaterThan(a.r + b.r);
      }
    }
  });

  it('wires reference existing columns and real parameter names', () => {
    const keys = new Set(columns.map(c => c.key));
    expect(wires.length).toBeGreaterThan(0);
    const paramNames: readonly string[] = PARAM_NAMES;
    for (const w of wires) {
      expect(keys.has(w.from)).toBe(true);
      expect(keys.has(w.to)).toBe(true);
      expect(paramNames.includes(w.param)).toBe(true);
    }
  });

  it('still avoids overlap in a narrow viewport', () => {
    const { columns: narrow } = networkLayout(shapes, { w: 300, h: 400 });
    for (let i = 0; i < narrow.length; i++) {
      for (let j = i + 1; j < narrow.length; j++) {
        const a = narrow[i], b = narrow[j];
        expect(Math.abs(a.x - b.x)).toBeGreaterThan(a.r + b.r);
      }
    }
  });
});
