/** Hand-placed layout for the map view, modelled on Simbrain's Tiny LM diagram: Inputs at the
 *  bottom left, Embedding above it, the Transformer block as a tall box on the right (input at the
 *  bottom, output at the top), Unembedding above the block, and the prediction at the top left.
 *  Pure geometry; drawing lives in MapView.tsx. World units, y grows downward. */
import { T_ } from '../../engine/model/gpt';

export interface Rect { x: number; y: number; w: number; h: number }
export type CaptionSide = 'below' | 'left' | 'right' | 'above';
/** A tile that shows one of several tensors: the latest one the cursor has computed. */
export interface MapSlot { id: string; label: string; kind: 'tensor' | 'param'; tensors: string[]; rect: Rect; group: string; caption: CaptionSide }
export interface MapGroup { id: string; label: string; rect: Rect }
/** A big arrow between groups: a quadratic curve. */
export interface MapArrow { from: [number, number]; ctrl: [number, number]; to: [number, number] }
/** A thin polyline inside the block, optionally with a junction dot at its first point. */
export interface MapLine { pts: [number, number][]; dot?: boolean }
export interface CircleGrid { rect: Rect; cols: number; rows: number; cellW: number; cellH: number; order: number[] }
export interface InputGrid { rect: Rect; rows: number; cols: number }
export interface MapLayout {
  slots: MapSlot[]; groups: MapGroup[]; arrows: MapArrow[]; lines: MapLine[];
  pred: CircleGrid; topList: Rect; inputs: InputGrid; bounds: Rect; slotById: Map<string, MapSlot>;
}

export const U = 4;              // world units per cell in activation and weight tiles
const CAP = 24;                  // space under a tile for a caption below it
const GAP = 16;                  // vertical gap between tiers in the block
const PAD = 26;                  // block padding
const COL_GAP = 56;              // gap between q, k, v columns
const SIDE_GAP = 44;             // gap between a tile and the weight tile beside it
const ATTN_MIN = 72;             // the attention tile never gets smaller than this
export const PRED_CELL = { w: 36, h: 27 };
export const TOP_LIST = { w: 250, rowH: 32, rows: 8 };

const union = (rs: Rect[]): Rect => {
  const x0 = Math.min(...rs.map(r => r.x)), y0 = Math.min(...rs.map(r => r.y));
  const x1 = Math.max(...rs.map(r => r.x + r.w)), y1 = Math.max(...rs.map(r => r.y + r.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
};
const grow = (r: Rect, p: number, top = p, bottom = p, left = p, right = p): Rect => ({ x: r.x - left, y: r.y - top, w: r.w + left + right, h: r.h + top + bottom });
const cx = (r: Rect) => r.x + r.w / 2, cy = (r: Rect) => r.y + r.h / 2;

/** Alphabetical order of the vocabulary, id 0 (unknown) excluded. */
export function predOrder(V: number, words?: string[]): number[] {
  const ids = Array.from({ length: Math.max(0, V - 1) }, (_, i) => i + 1);
  if (!words) return ids;
  return ids.sort((a, b) => (words[a] ?? '').localeCompare(words[b] ?? ''));
}

interface Item { id: string; label: string; kind: MapSlot['kind']; tensors: string[]; w: number; h: number; caption?: CaptionSide }

/** The rectangle a caption occupies, so groups can make room for it. */
export function captionBox(slot: MapSlot): Rect {
  const r = slot.rect; const w = Math.max(60, slot.label.length * 6 + 12), h = CAP;
  switch (slot.caption) {
    case 'below': return { x: cx(r) - w / 2, y: r.y + r.h + 2, w, h };
    case 'above': return { x: cx(r) - w / 2, y: r.y - h - 2, w, h };
    case 'left': return { x: r.x - w - 6, y: cy(r) - h / 2, w, h };
    case 'right': return { x: r.x + r.w + 6, y: cy(r) - h / 2, w, h };
  }
}

export function mapLayout(shapes: Record<string, number[]>, words?: string[]): MapLayout {
  const need = (n: string) => { const s = shapes[n]; if (!s) throw new Error(`mapLayout: no shape for '${n}'`); return s; };
  const [T, d] = need(T_.x0); const dFF = need(T_.ff_pre)[1]; const V = need(T_.probs)[1]; const Tmax = need('P')[0];
  const slots: MapSlot[] = []; const slotById = new Map<string, MapSlot>();
  const place = (it: Item, group: string, x: number, y: number): MapSlot | null => {
    const tensors = it.tensors.filter(t => shapes[t] !== undefined);
    if (tensors.length === 0) return null;
    const s: MapSlot = { id: it.id, label: it.label, kind: it.kind, tensors, rect: { x, y, w: it.w, h: it.h }, group, caption: it.caption ?? 'below' };
    slots.push(s); slotById.set(s.id, s); return s;
  };
  const act = (id: string, label: string, tensors: string[], cols = d, caption?: CaptionSide): Item => ({ id, label, kind: 'tensor', tensors, w: cols * U, h: T * U, caption });
  const weight = (id: string, label: string, rows: number, cols: number): Item => ({ id, label, kind: 'param', tensors: [id], w: cols * U, h: rows * U, caption: 'right' });

  // ---- Transformer block (right column), built top-down in tiers. A tier is a row of tiles
  //      centred on the block's axis, with weight tiles to the right; every tile is centred
  //      vertically in the tier so tall weight tiles never collide with the tier above. ----
  const BCX = 1230, BY = 300;
  let y = BY + PAD + 8;
  const tier = (main: Item[], side: Item[] = []): MapSlot[] => {
    const rowH = Math.max(...[...main, ...side].map(i => i.h));
    const mainW = main.reduce((a, i) => a + i.w, 0) + COL_GAP * (main.length - 1);
    let x = BCX - mainW / 2; const out: MapSlot[] = [];
    for (const it of main) { const s = place(it, 'block', x, y + (rowH - it.h) / 2); if (s) out.push(s); x += it.w + COL_GAP; }
    x += SIDE_GAP - COL_GAP;
    for (const it of side) { const s = place(it, 'block', x, y + (rowH - it.h) / 2); if (s) out.push(s); x += it.w + SIDE_GAP; }
    const belowCaption = main.some(i => (i.caption ?? 'below') === 'below');
    y += rowH + (belowCaption ? CAP : 8) + GAP;
    return out;
  };
  const [out] = tier([act('out', 'Output', [T_.x2, T_.hf], d, 'left')]);
  const [ffOut, W2] = tier([act('ff_out', 'FF output', [T_.ff_out])], [weight('W_2', 'Hidden -> output', dFF, d)]);
  const [ffHid, W1] = tier([act('ff_hid', 'FF hidden', [T_.ff_pre, T_.ff_act], dFF)], [weight('W_1', 'Input -> hidden', d, dFF)]);
  const [ffIn] = tier([act('ff_in', 'FF input', [T_.x1, T_.h2])]);
  y += 12;
  const [ctx, Wo] = tier([act('ctx', 'Attention output', [T_.ctxv, T_.attn_out])], [weight('W_o', 'Write back', d, d)]);
  const attnSize = Math.max(T * U, ATTN_MIN);
  const [attn] = tier([{ id: 'attn', label: 'Attention', kind: 'tensor', tensors: [T_.scores, T_.masked, T_.attn], w: attnSize, h: attnSize }]);
  const [q, k, v] = tier([act('q', 'q', [T_.q]), act('k', 'k', [T_.k]), act('v', 'v', [T_.v])]);
  const [Wq, Wk, Wv] = tier([{ ...weight('W_q', 'Q', d, d), caption: 'below' }, { ...weight('W_k', 'K', d, d), caption: 'below' }, { ...weight('W_v', 'V', d, d), caption: 'below' }]);
  const [inp] = tier([act('in', 'Input', [T_.x0, T_.h1], d, 'left')]);
  const blockRect = grow(union(slots.map(s => [s.rect, captionBox(s)]).flat()), PAD, PAD + 8, PAD);
  const BX = blockRect.x;

  // ---- Unembedding above the block; probabilities and the prediction to the left. ----
  const uT = { w: 240, h: 100 };
  const Uslot = place({ id: 'U', label: 'Unembedding', kind: 'param', tensors: ['U'], w: uT.w, h: uT.h, caption: 'right' }, 'unembedding', BCX - uT.w / 2, 40)!;
  const sm = { w: 240, h: Math.max(T * U, 20) };
  const smx = BCX - uT.w / 2 - 150 - sm.w;
  const soft = place({ id: 'softmax', label: 'One probability per word', kind: 'tensor', tensors: [T_.logits, T_.probs], w: sm.w, h: sm.h }, 'probs', smx, 40 + (uT.h - sm.h) / 2)!;
  place({ id: 'loss', label: 'Loss', kind: 'tensor', tensors: [T_.loss], w: 28, h: 28, caption: 'right' }, 'probs', smx, soft.rect.y + sm.h + CAP + 14);
  const cols = Math.max(4, Math.min(16, Math.ceil(Math.sqrt((V - 1) * 1.4))));
  const rows = Math.max(1, Math.ceil((V - 1) / cols));
  const listW = TOP_LIST.w, predW = cols * PRED_CELL.w;
  const topList: Rect = { x: smx - 110 - listW, y: 40, w: listW, h: TOP_LIST.rows * TOP_LIST.rowH + 30 };
  const pred: CircleGrid = { rect: { x: topList.x - 24 - predW, y: 40, w: predW, h: rows * PRED_CELL.h }, cols, rows, cellW: PRED_CELL.w, cellH: PRED_CELL.h, order: predOrder(V, words) };

  // ---- Embedding and Inputs (left column, under the prediction grid). ----
  const ey = Math.max(pred.rect.y + pred.rect.h + 150, 640);
  const ex = smx + 10;
  const E = place({ id: 'E', label: 'Word table', kind: 'param', tensors: ['E'], w: d * U, h: 200, caption: 'left' }, 'embedding', ex, ey)!;
  const P = place({ id: 'P', label: 'Position table', kind: 'param', tensors: ['P'], w: d * U, h: Tmax * U, caption: 'above' }, 'embedding', ex + d * U + 60, ey + (200 - Tmax * U) / 2 + 12)!;
  const inputs: InputGrid = { rect: { x: E.rect.x - 40, y: ey + 200 + 150, w: 240, h: Math.max(T * 7, 24) }, rows: T, cols: V };

  // ---- Groups. ----
  const withCaptions = (ids: string[]) => union(ids.flatMap(id => { const s = slotById.get(id); return s ? [s.rect, captionBox(s)] : []; }));
  const groups: MapGroup[] = [
    { id: 'pred', label: 'Predicted next token', rect: grow(union([pred.rect, topList]), 14) },
    { id: 'probs', label: 'Probabilities', rect: grow(withCaptions(['softmax', 'loss']), 14) },
    { id: 'unembedding', label: 'Unembedding', rect: grow(withCaptions(['U']), 14) },
    { id: 'block', label: 'Transformer block', rect: blockRect },
    { id: 'embedding', label: 'Embedding', rect: grow(withCaptions(['E', 'P']), 16, 20, 16) },
    { id: 'inputs', label: 'Inputs', rect: grow(inputs.rect, 14, 14, 14 + CAP) },
  ];
  const G = Object.fromEntries(groups.map(g => [g.id, g.rect]));

  // ---- Big arrows, tile to tile. ----
  const arrows: MapArrow[] = [
    { from: [cx(inputs.rect), inputs.rect.y - 4], ctrl: [cx(inputs.rect), (inputs.rect.y + E.rect.y + E.rect.h) / 2], to: [cx(E.rect), E.rect.y + E.rect.h + 6] },
    { from: [G.embedding.x + G.embedding.w, cy(P.rect)], ctrl: [BCX - 120, blockRect.y + blockRect.h + 40], to: [cx(inp.rect), inp.rect.y + inp.rect.h + 8] },
    { from: [cx(out.rect), out.rect.y - 8], ctrl: [cx(out.rect), (out.rect.y + Uslot.rect.y + Uslot.rect.h) / 2], to: [cx(Uslot.rect), Uslot.rect.y + Uslot.rect.h + 8] },
    { from: [Uslot.rect.x - 8, cy(Uslot.rect)], ctrl: [(Uslot.rect.x + soft.rect.x + soft.rect.w) / 2, cy(Uslot.rect)], to: [soft.rect.x + soft.rect.w + 8, cy(soft.rect)] },
    { from: [soft.rect.x - 8, cy(soft.rect)], ctrl: [(soft.rect.x + topList.x + topList.w) / 2, cy(soft.rect)], to: [topList.x + topList.w + 8, cy(soft.rect)] },
  ];

  // ---- Thin lines inside the block. ----
  const lines: MapLine[] = [];
  const top = (r: Rect): [number, number] => [cx(r), r.y], bot = (r: Rect): [number, number] => [cx(r), r.y + r.h];
  const right = (r: Rect): [number, number] => [r.x + r.w, cy(r)], left = (r: Rect): [number, number] => [r.x, cy(r)];
  const between = (above: Rect, below: Rect, cap = CAP) => (above.y + above.h + cap + below.y) / 2;   // y of the gap between two tiers
  const busY = between(Wq.rect, inp.rect);
  const railX = BX + 12;
  lines.push({ pts: [top(inp.rect), [BCX, busY]] });
  lines.push({ pts: [[cx(Wq.rect), busY], [cx(Wv.rect), busY]] });
  for (const W of [Wq, Wk, Wv]) lines.push({ pts: [[cx(W.rect), busY], bot(W.rect)] });
  for (const [W, a] of [[Wq, q], [Wk, k], [Wv, v]] as const) lines.push({ pts: [top(W.rect), bot(a.rect)] });
  const attnInY = between(attn.rect, q.rect);
  lines.push({ pts: [top(q.rect), [cx(q.rect), attnInY], [cx(attn.rect) - 8, attnInY], [cx(attn.rect) - 8, attn.rect.y + attn.rect.h]] });
  lines.push({ pts: [top(k.rect), [cx(k.rect), attnInY], [cx(attn.rect) + 8, attnInY], [cx(attn.rect) + 8, attn.rect.y + attn.rect.h]] });
  const vRail = Math.max(attn.rect.x + attn.rect.w, ctx.rect.x + ctx.rect.w) + 22;
  lines.push({ pts: [top(v.rect), [cx(v.rect), attnInY + 6], [vRail, attnInY + 6], [vRail, cy(ctx.rect)], right(ctx.rect)] });
  lines.push({ pts: [top(attn.rect), bot(ctx.rect)] });
  lines.push({ pts: [left(Wo.rect), [vRail, cy(Wo.rect)]] });
  lines.push({ pts: [top(ctx.rect), bot(ffIn.rect)] });
  const res1Y = between(ffIn.rect, ctx.rect);
  lines.push({ pts: [[BCX, busY], [railX, busY], [railX, res1Y], [BCX, res1Y]], dot: true });
  lines.push({ pts: [top(ffIn.rect), bot(ffHid.rect)] });
  lines.push({ pts: [right(ffHid.rect), left(W1.rect)] });
  lines.push({ pts: [top(ffHid.rect), bot(ffOut.rect)] });
  lines.push({ pts: [right(ffOut.rect), left(W2.rect)] });
  lines.push({ pts: [top(ffOut.rect), bot(out.rect)] });
  const res2Y = between(ffHid.rect, ffIn.rect), res2Top = between(out.rect, ffOut.rect, 8);
  lines.push({ pts: [[BCX, res2Y], [railX, res2Y], [railX, res2Top], [BCX, res2Top]], dot: true });

  const bounds = grow(union(groups.map(g => g.rect)), 50, 60, 50);
  return { slots, groups, arrows, lines, pred, topList, inputs, bounds, slotById };
}
