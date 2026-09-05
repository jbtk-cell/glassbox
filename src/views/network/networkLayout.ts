/** Pure layout maths for the Network view: one column per activation tensor at the current
 *  position, and the wires (weight matrices) between consecutive linear stages. Drawing lives
 *  in NetworkView.tsx; this file only computes geometry. */
import { T_ } from '../../engine/model/gpt';

export interface NetColumn { key: string; label: string; n: number; x: number; y0: number; dy: number; r: number }
export interface NetWire { from: string; to: string; param: string }

/** Left-to-right order of the activation-tensor columns. */
export const COLUMN_ORDER = [
  T_.tok, T_.pos, T_.x0, T_.h1, T_.q, T_.k, T_.v, T_.ctxv, T_.attn_out, T_.x1,
  T_.h2, T_.ff_pre, T_.ff_act, T_.ff_out, T_.x2, T_.hf, T_.logits, T_.probs,
] as const;

/** Weight matrix carried by the wires between two consecutive linear stages. */
const WIRE_SPEC: NetWire[] = [
  { from: T_.h1, to: T_.q, param: 'W_q' },
  { from: T_.h1, to: T_.k, param: 'W_k' },
  { from: T_.h1, to: T_.v, param: 'W_v' },
  { from: T_.ctxv, to: T_.attn_out, param: 'W_o' },
  { from: T_.h2, to: T_.ff_pre, param: 'W_1' },
  { from: T_.ff_act, to: T_.ff_out, param: 'W_2' },
  { from: T_.hf, to: T_.logits, param: 'U' },
];

/** Columns with more than this many neurons are drawn as a compact strip of 1 px rows. */
export const COMPACT_THRESHOLD = 48;

const HEADER = 36;      // reserved height for the position picker
const ATTN_STRIP = 64;  // reserved height for the attention arcs below the columns
const MARGIN = 24;

/** Neurons shown for one row of a tensor: its feature width (the last dimension of its shape). */
function neuronCount(shape: number[]): number {
  return shape.length === 0 ? 0 : shape[shape.length - 1];
}

/** Columns and wires for the current trace's shapes, positioned to fit inside `viewport`. */
export function networkLayout(shapes: Record<string, number[]>, viewport: { w: number; h: number }): { columns: NetColumn[]; wires: NetWire[] } {
  const names = COLUMN_ORDER.filter(name => shapes[name] !== undefined);
  const availH = Math.max(40, viewport.h - HEADER - ATTN_STRIP - MARGIN);
  const xStep = names.length > 0 ? (viewport.w - 2 * MARGIN) / names.length : 0;

  const columns: NetColumn[] = names.map((name, i) => {
    const n = Math.max(1, neuronCount(shapes[name]));
    const compact = n > COMPACT_THRESHOLD;
    const rawDy = availH / n;
    const dy = compact ? rawDy : Math.min(22, Math.max(4, rawDy));
    const rawR = compact ? dy / 2 : Math.max(1.5, Math.min(9, dy / 2 - 1));
    const r = Math.max(0.5, Math.min(rawR, xStep / 2 - 2));
    return { key: name, label: name, n, x: MARGIN + xStep * (i + 0.5), y0: HEADER + MARGIN, dy, r };
  });

  const present = new Set<string>(names);
  const wires = WIRE_SPEC.filter(w => present.has(w.from) && present.has(w.to));

  return { columns, wires };
}
