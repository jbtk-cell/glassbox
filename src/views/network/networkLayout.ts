/** Geometry for the Neurons view: one position's numbers as columns of circles, with a band
 *  between every pair of neighbouring columns saying how one becomes the next. World units,
 *  y grows downward, columns centred on y = 0. Drawing lives in NetworkView.tsx. */
import { T_ } from '../../engine/model/gpt';

export interface Rect { x: number; y: number; w: number; h: number }
export interface NeuronGroup {
  key: string; label: string; sub: string; n: number;
  x: number; y0: number; dy: number; r: number;
  compact: boolean;        // many rows: drawn as a strip instead of circles
  words: boolean;          // label each row with the word at that position (attention weights)
}
export type BandKind = 'matrix' | 'elementwise' | 'attend' | 'mix';
/** How the `to` group is computed from the `from` groups. */
export interface Band { kind: BandKind; from: string[]; to: string; param?: string; label: string }
/** A dashed arc showing a value carried forward and added back in. */
export interface SkipArc { from: string; to: string; label: string }
export interface NetLayout { groups: NeuronGroup[]; byKey: Map<string, NeuronGroup>; bands: Band[]; skips: SkipArc[]; bounds: Rect; top: number }

export const X_STEP = 140;
export const DY = 20, R = 7;
export const STACK_GAP = 60;
export const COMPACT_THRESHOLD = 48;
export const COMPACT_HALF_W = 9;
const COMPACT_MAX_H = 620;

const CAPTIONS: Record<string, string> = {
  [T_.tok]: 'word row', [T_.pos]: 'position row', [T_.x0]: 'sum', [T_.h1]: 'tidied', [T_.q]: 'question', [T_.k]: 'label', [T_.v]: 'content',
  [T_.attn]: 'attention', [T_.ctxv]: 'mix', [T_.attn_out]: 'written back', [T_.x1]: 'after attention', [T_.h2]: 'tidied', [T_.ff_pre]: 'hidden',
  [T_.ff_act]: 'after ReLU', [T_.ff_out]: 'FF output', [T_.x2]: 'block output', [T_.hf]: 'tidied', [T_.logits]: 'scores', [T_.probs]: 'probabilities',
};

/** Column slots, left to right; a slot with several keys stacks them vertically. */
const SLOTS: string[][] = [
  [T_.tok, T_.pos], [T_.x0], [T_.h1], [T_.q, T_.k, T_.v], [T_.attn], [T_.ctxv], [T_.attn_out], [T_.x1],
  [T_.h2], [T_.ff_pre], [T_.ff_act], [T_.ff_out], [T_.x2], [T_.hf], [T_.logits], [T_.probs],
];

const BANDS: Band[] = [
  { kind: 'elementwise', from: [T_.tok, T_.pos], to: T_.x0, label: 'add' },
  { kind: 'elementwise', from: [T_.x0], to: T_.h1, label: 'norm' },
  { kind: 'matrix', from: [T_.h1], to: T_.q, param: 'W_q', label: 'W_q' },
  { kind: 'matrix', from: [T_.h1], to: T_.k, param: 'W_k', label: 'W_k' },
  { kind: 'matrix', from: [T_.h1], to: T_.v, param: 'W_v', label: 'W_v' },
  { kind: 'attend', from: [T_.q, T_.k], to: T_.attn, label: 'q . k, softmax' },
  { kind: 'mix', from: [T_.attn, T_.v], to: T_.ctxv, label: 'mix v by attention' },
  { kind: 'matrix', from: [T_.ctxv], to: T_.attn_out, param: 'W_o', label: 'W_o' },
  { kind: 'elementwise', from: [T_.attn_out], to: T_.x1, label: 'add x0' },
  { kind: 'elementwise', from: [T_.x1], to: T_.h2, label: 'norm' },
  { kind: 'matrix', from: [T_.h2], to: T_.ff_pre, param: 'W_1', label: 'W_1' },
  { kind: 'elementwise', from: [T_.ff_pre], to: T_.ff_act, label: 'ReLU' },
  { kind: 'matrix', from: [T_.ff_act], to: T_.ff_out, param: 'W_2', label: 'W_2' },
  { kind: 'elementwise', from: [T_.ff_out], to: T_.x2, label: 'add x1' },
  { kind: 'elementwise', from: [T_.x2], to: T_.hf, label: 'norm' },
  { kind: 'matrix', from: [T_.hf], to: T_.logits, param: 'U', label: 'U' },
  { kind: 'elementwise', from: [T_.logits], to: T_.probs, label: 'softmax' },
];

const SKIPS: SkipArc[] = [
  { from: T_.x0, to: T_.x1, label: 'x0 carried forward' },
  { from: T_.x1, to: T_.x2, label: 'x1 carried forward' },
];

/** Rows shown for a tensor at one position: its last dimension (T for the attention row). */
function rowsOf(shape: number[]): number { return shape.length === 0 ? 1 : shape[shape.length - 1]; }

export function networkLayout(shapes: Record<string, number[]>): NetLayout {
  const groups: NeuronGroup[] = []; const byKey = new Map<string, NeuronGroup>();
  const slots = SLOTS.map(keys => keys.filter(k => shapes[k] !== undefined)).filter(keys => keys.length > 0);
  slots.forEach((keys, col) => {
    const x = col * X_STEP;
    const specs = keys.map(key => {
      const n = Math.max(1, rowsOf(shapes[key])); const compact = n > COMPACT_THRESHOLD;
      const dy = compact ? Math.min(2.4, COMPACT_MAX_H / n) : keys.length > 2 ? 15 : keys.length > 1 ? 18 : DY;
      return { key, n, compact, dy, h: n * dy };
    });
    const total = specs.reduce((a, s) => a + s.h, 0) + STACK_GAP * (specs.length - 1);
    let y = -total / 2;
    for (const s of specs) {
      const g: NeuronGroup = { key: s.key, label: s.key, sub: CAPTIONS[s.key] ?? '', n: s.n, x, y0: y + (s.compact ? 0 : s.dy / 2), dy: s.dy, r: s.compact ? 0 : Math.min(R, s.dy / 2 - 1), compact: s.compact, words: s.key === T_.attn };
      groups.push(g); byKey.set(g.key, g);
      y += s.h + STACK_GAP;
    }
  });
  const present = (k: string) => byKey.has(k);
  const bands = BANDS.filter(b => present(b.to) && b.from.every(present) && (!b.param || shapes[b.param] !== undefined));
  const skips = SKIPS.filter(s => present(s.from) && present(s.to));
  const top = Math.min(...groups.map(g => g.y0 - g.r - 30));
  const bottom = Math.max(...groups.map(g => g.y0 + g.n * g.dy + 10));
  const bounds: Rect = { x: -X_STEP / 2, y: top - 150, w: slots.length * X_STEP, h: bottom - top + 170 };
  return { groups, byKey, bands, skips, bounds, top };
}

/** y of row i in a group (circle centre, or the top of a strip row). */
export function rowY(g: NeuronGroup, i: number): number { return g.y0 + i * g.dy; }
