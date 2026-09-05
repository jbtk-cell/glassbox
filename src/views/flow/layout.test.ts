import { describe, it, expect } from 'vitest';
import { flowLayout, tileBox } from './layout';
import { GPT } from '../../engine/model/gpt';
import { recordForward, recordTrainingStep } from '../../engine/trace/trace';
import { Adam } from '../../engine/model/adam';

const cfg = { vocabSize: 300, contextSize: 24, dModel: 20, dFF: 30, seed: 1 };
const shapesOf = (ctx: { tensors: Map<string, { shape: number[] }> }) =>
  Object.fromEntries([...ctx.tensors].map(([k, t]) => [k, t.shape]));

const overlaps = (a: { x: number; y: number; w: number; h: number }, b: typeof a) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

describe('tileBox', () => {
  it('gives 6 units per cell for ordinary tensors and fixed boxes for vocabulary-sized ones', () => {
    expect(tileBox('x0', [24, 20])).toEqual({ w: 120, h: 144 });
    expect(tileBox('E', [300, 20])).toEqual({ w: 80, h: 240 });
    expect(tileBox('U', [20, 300])).toEqual({ w: 240, h: 120 });
    expect(tileBox('probs', [24, 300])).toEqual({ w: 240, h: 144 });
    expect(tileBox('b_q', [20])).toEqual({ w: 120, h: 8 });
  });
});

describe('flowLayout', () => {
  const m = new GPT(cfg);
  const fwd = recordForward(m, [1, 2, 3, 4, 5]);
  const L = flowLayout(m.ops, shapesOf(fwd.ctx));

  it('has a node for every op, activation and parameter, each exactly once', () => {
    const ids = L.nodes.map(n => n.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const op of m.ops.slice(0, -1)) {
      expect(L.byId.has('op:' + op.id)).toBe(true);
      expect(L.byId.has('t:' + op.output)).toBe(true);
      for (const p of op.params) expect(L.byId.get('t:' + p)?.kind).toBe('param');
    }
    expect(L.byId.has('op:loss')).toBe(false);
  });
  it('never overlaps two tiles', () => {
    const tiles = L.nodes.filter(n => n.kind !== 'op');
    for (let i = 0; i < tiles.length; i++) for (let j = i + 1; j < tiles.length; j++) {
      expect(overlaps(tiles[i].rect, tiles[j].rect), `${tiles[i].id} overlaps ${tiles[j].id}`).toBe(false);
    }
  });
  it('has edges whose endpoints exist, and rails only for the two residual inputs', () => {
    for (const e of L.edges) { expect(L.byId.has(e.from), e.from).toBe(true); expect(L.byId.has(e.to), e.to).toBe(true); }
    expect(L.edges.filter(e => e.rail).map(e => e.from + '>' + e.to).sort()).toEqual(['t:x0>op:residual1', 't:x1>op:residual2']);
  });
  it('stacks the residual stream bottom-up on the spine', () => {
    const spine = ['x0', 'x1', 'x2', 'hf', 'logits', 'probs'].map(n => L.byId.get('t:' + n)!);
    for (const n of spine) expect(n.rect.x).toBeCloseTo(-n.rect.w / 2, 9);
    for (let i = 1; i < spine.length; i++) expect(spine[i].rect.y).toBeLessThan(spine[i - 1].rect.y);
    const [q, k, v] = ['q', 'k', 'v'].map(n => L.byId.get('t:' + n)!);
    expect(q.rect.y).toBe(k.rect.y); expect(k.rect.y).toBe(v.rect.y);
  });
  it('nests attention and feed-forward inside the transformer block', () => {
    const block = L.groups.find(g => g.id === 'block')!;
    for (const id of ['attention', 'ff']) {
      const g = L.groups.find(x => x.id === id)!;
      expect(g.parent).toBe('block');
      expect(g.rect.x).toBeGreaterThanOrEqual(block.rect.x); expect(g.rect.y).toBeGreaterThanOrEqual(block.rect.y);
      expect(g.rect.x + g.rect.w).toBeLessThanOrEqual(block.rect.x + block.rect.w);
    }
    expect(L.bounds.w).toBeGreaterThan(0); expect(L.bounds.h).toBeGreaterThan(0);
  });
  it('adds the loss node for a training trace', () => {
    const tr = recordTrainingStep(m, new Adam(m.params), { input: [1, 2, 3], target: [2, 3, 4] });
    const Lt = flowLayout(m.ops, shapesOf(tr.ctx));
    expect(Lt.byId.get('t:loss')?.rect).toMatchObject({ w: 28, h: 28 });
    expect(Lt.byId.has('op:loss')).toBe(true);
  });
});
