import { describe, it, expect } from 'vitest';
import { mapLayout, predOrder } from './mapLayout';

const shapes = (T: number, V = 289, d = 20, dFF = 30, Tmax = 24): Record<string, number[]> => ({
  x0: [T, d], h1: [T, d], q: [T, d], k: [T, d], v: [T, d], scores: [T, T], masked: [T, T], attn: [T, T], ctxv: [T, d], attn_out: [T, d],
  x1: [T, d], h2: [T, d], ff_pre: [T, dFF], ff_act: [T, dFF], ff_out: [T, d], x2: [T, d], hf: [T, d], logits: [T, V], probs: [T, V],
  E: [V, d], P: [Tmax, d], W_q: [d, d], W_k: [d, d], W_v: [d, d], W_o: [d, d], W_1: [d, dFF], W_2: [dFF, d], U: [d, V],
});

const overlaps = (a: { x: number; y: number; w: number; h: number }, b: typeof a) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

describe('mapLayout', () => {
  it('places every tile and no two tiles overlap', () => {
    const L = mapLayout(shapes(24));
    const ids = L.slots.map(s => s.id);
    for (const id of ['in', 'W_q', 'W_k', 'W_v', 'q', 'k', 'v', 'attn', 'ctx', 'W_o', 'ff_in', 'W_1', 'ff_hid', 'W_2', 'ff_out', 'out', 'U', 'softmax', 'E', 'P']) expect(ids).toContain(id);
    expect(ids).not.toContain('loss');
    for (let i = 0; i < L.slots.length; i++) for (let j = i + 1; j < L.slots.length; j++) expect(overlaps(L.slots[i].rect, L.slots[j].rect), `${L.slots[i].id} vs ${L.slots[j].id}`).toBe(false);
  });
  it('keeps groups apart and inside the bounds', () => {
    const L = mapLayout(shapes(24));
    for (let i = 0; i < L.groups.length; i++) for (let j = i + 1; j < L.groups.length; j++) expect(overlaps(L.groups[i].rect, L.groups[j].rect), `${L.groups[i].id} vs ${L.groups[j].id}`).toBe(false);
    for (const g of L.groups) expect(overlaps(g.rect, L.bounds)).toBe(true);
    for (const s of L.slots) { const g = L.groups.find(g => g.id === s.group)!; expect(overlaps(s.rect, g.rect), s.id).toBe(true); }
  });
  it('shows the loss tile only for training traces and adapts to short contexts', () => {
    const L = mapLayout({ ...shapes(3), loss: [] });
    expect(L.slotById.has('loss')).toBe(true);
    expect(L.inputs.rows).toBe(3);
    expect(L.slotById.get('q')!.rect.h).toBe(12);
    const W2 = L.slotById.get('W_2')!.rect, W1 = L.slotById.get('W_1')!.rect;
    expect(W1.y).toBeGreaterThanOrEqual(W2.y + W2.h + 20);
  });
  it('orders the prediction grid alphabetically without the unknown token', () => {
    expect(predOrder(4, ['<unk>', 'b', 'a', 'c'])).toEqual([2, 1, 3]);
    const L = mapLayout(shapes(24, 5), ['<unk>', 'd', 'a', 'c', 'b']);
    expect(L.pred.order).toEqual([2, 4, 3, 1]);
    expect(L.pred.cols * L.pred.rows).toBeGreaterThanOrEqual(4);
  });
});
