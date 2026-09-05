import type { Op } from '../../engine/ops/types';

export interface TileRect { x: number; y: number; w: number; h: number }
export type NodeKind = 'tensor' | 'param' | 'op';
export interface LayoutNode {
  id: string; kind: NodeKind; key?: string; opId?: string; label: string;
  rect: TileRect; shape?: number[]; group: string;
}
export interface LayoutEdge { from: string; to: string; rail?: boolean }
export interface LayoutGroup { id: string; label: string; rect: TileRect; parent?: string }
export interface FlowLayout {
  nodes: LayoutNode[]; edges: LayoutEdge[]; groups: LayoutGroup[];
  bounds: TileRect; byId: Map<string, LayoutNode>; tokenStrip: TileRect;
}

const PILL_W = 120, PILL_H = 28, GAP = 24, RIGHT_X = 300, PARAM_COL = 170, RAIL_X = -260, PAD = 60;

/** Fixed on-screen box for a tensor; cells may be non-square so big vocabularies stay readable. */
export function tileBox(name: string, shape: number[]): { w: number; h: number } {
  const rows = shape.length === 1 ? 1 : shape[0];
  const cols = shape.length === 1 ? shape[0] : shape[1];
  if (name === 'loss') return { w: 28, h: 28 };
  if (shape.length === 1) return { w: cols > 64 ? 240 : 120, h: 8 };          // bias strips
  if (rows > 64 && cols <= 64) return { w: 80, h: 240 };                        // E
  if (cols > 64) return { w: 240, h: rows > 64 ? 240 : Math.max(48, rows * 6) }; // U, logits, probs
  return { w: cols * 6, h: rows * 6 };                                          // everything else, 6 units per cell
}

/** Where each activation goes: the spine (x=0) or a side lane. */
const LANE: Record<string, number> = { tok: -100, pos: 100, q: -170, k: 0, v: 170 };

interface Level { opId: string; out: string; x: number; group: string }

function levelsFor(hasLoss: boolean): Level[] {
  const L: Level[] = [
    { opId: 'embed', out: 'tok', x: LANE.tok, group: 'embedding' },
    { opId: 'pos_embed', out: 'pos', x: LANE.pos, group: 'embedding' },
    { opId: 'add_pos', out: 'x0', x: 0, group: 'embedding' },
    { opId: 'ln1', out: 'h1', x: 0, group: 'attention' },
    { opId: 'q_proj', out: 'q', x: LANE.q, group: 'attention' },
    { opId: 'k_proj', out: 'k', x: LANE.k, group: 'attention' },
    { opId: 'v_proj', out: 'v', x: LANE.v, group: 'attention' },
    { opId: 'scores', out: 'scores', x: 0, group: 'attention' },
    { opId: 'causal_mask', out: 'masked', x: 0, group: 'attention' },
    { opId: 'attn_softmax', out: 'attn', x: 0, group: 'attention' },
    { opId: 'attn_apply', out: 'ctxv', x: 0, group: 'attention' },
    { opId: 'o_proj', out: 'attn_out', x: 0, group: 'attention' },
    { opId: 'residual1', out: 'x1', x: 0, group: 'attention' },
    { opId: 'ln2', out: 'h2', x: 0, group: 'ff' },
    { opId: 'ff_up', out: 'ff_pre', x: 0, group: 'ff' },
    { opId: 'relu', out: 'ff_act', x: 0, group: 'ff' },
    { opId: 'ff_down', out: 'ff_out', x: 0, group: 'ff' },
    { opId: 'residual2', out: 'x2', x: 0, group: 'ff' },
    { opId: 'ln_final', out: 'hf', x: 0, group: 'unembedding' },
    { opId: 'unembed', out: 'logits', x: 0, group: 'unembedding' },
    { opId: 'softmax_out', out: 'probs', x: 0, group: 'output' },
  ];
  if (hasLoss) L.push({ opId: 'loss', out: 'loss', x: 0, group: 'output' });
  return L;
}

/** Ops that share a row: their pills and tiles sit at the same y. */
const ROW_MATES: string[][] = [['embed', 'pos_embed'], ['q_proj', 'k_proj', 'v_proj']];

export function flowLayout(ops: Op[], shapes: Record<string, number[]>): FlowLayout {
  const opById = new Map(ops.map(o => [o.id, o]));
  const nodes: LayoutNode[] = [];
  const edges: LayoutEdge[] = [];
  const byId = new Map<string, LayoutNode>();
  const add = (n: LayoutNode) => { nodes.push(n); byId.set(n.id, n); return n; };

  const levels = levelsFor(shapes['loss'] !== undefined).filter(l => opById.has(l.opId));
  const tokenStrip: TileRect = { x: -220, y: -60, w: 440, h: 40 };
  let y = tokenStrip.y - GAP;        // running top edge of the spine, moving upward (decreasing)
  const groupMembers = new Map<string, LayoutNode[]>();
  const member = (g: string, n: LayoutNode) => { (groupMembers.get(g) ?? groupMembers.set(g, []).get(g)!).push(n); };

  let i = 0;
  while (i < levels.length) {
    const mates = ROW_MATES.find(m => m.includes(levels[i].opId));
    const row = mates ? levels.slice(i, i + mates.length) : [levels[i]];
    const rowTop = y;
    let rowHeight = 0;
    for (const lv of row) {
      const op = opById.get(lv.opId)!;
      const shape = shapes[lv.out];
      if (!shape) throw new Error(`flowLayout: no shape for tensor '${lv.out}'`);
      const box = tileBox(lv.out, shape);
      const pill = add({ id: 'op:' + op.id, kind: 'op', opId: op.id, label: op.label, group: lv.group,
        rect: { x: lv.x - PILL_W / 2, y: rowTop - PILL_H, w: PILL_W, h: PILL_H } });
      const tile = add({ id: 't:' + lv.out, kind: 'tensor', key: 't:' + lv.out, label: lv.out, shape, group: lv.group,
        rect: { x: lv.x - box.w / 2, y: rowTop - PILL_H - GAP - box.h, w: box.w, h: box.h } });
      member(lv.group, pill); member(lv.group, tile);
      rowHeight = Math.max(rowHeight, PILL_H + GAP + box.h);
      for (const inp of op.inputs) {
        const rail = (op.id === 'residual1' && inp === 'x0') || (op.id === 'residual2' && inp === 'x1');
        edges.push({ from: 't:' + inp, to: pill.id, rail });
      }
      // Parameters sit beside their own row, one sub-column per op, biases stacked above their matrix.
      const sub = row.indexOf(lv);
      let ptop = rowTop - PILL_H;
      for (const p of op.params) {
        const pshape = shapes[p];
        if (!pshape) throw new Error(`flowLayout: no shape for parameter '${p}'`);
        const pbox = tileBox(p, pshape);
        ptop -= pbox.h;
        const pn = add({ id: 't:' + p, kind: 'param', key: 't:' + p, label: p, shape: pshape, group: lv.group,
          rect: { x: RIGHT_X + sub * PARAM_COL, y: ptop, w: pbox.w, h: pbox.h } });
        member(lv.group, pn);
        ptop -= 14;
        edges.push({ from: pn.id, to: pill.id });
      }
      rowHeight = Math.max(rowHeight, rowTop - ptop);
      edges.push({ from: pill.id, to: tile.id });
    }
    y = rowTop - rowHeight - GAP;
    i += row.length;
  }

  const union = (rs: TileRect[]): TileRect => {
    const x0 = Math.min(...rs.map(r => r.x)), y0 = Math.min(...rs.map(r => r.y));
    const x1 = Math.max(...rs.map(r => r.x + r.w)), y1 = Math.max(...rs.map(r => r.y + r.h));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  };
  const pad = (r: TileRect, p: number): TileRect => ({ x: r.x - p, y: r.y - p, w: r.w + 2 * p, h: r.h + 2 * p });
  const groupRect = (ids: string[], p: number) => pad(union(ids.flatMap(g => (groupMembers.get(g) ?? []).map(n => n.rect))), p);

  const groups: LayoutGroup[] = [
    { id: 'embedding', label: 'Embedding', rect: groupRect(['embedding'], 20) },
    { id: 'attention', label: 'Attention', rect: groupRect(['attention'], 20), parent: 'block' },
    { id: 'ff', label: 'Feed-forward', rect: groupRect(['ff'], 20), parent: 'block' },
    { id: 'unembedding', label: 'Unembedding', rect: groupRect(['unembedding'], 20) },
    { id: 'output', label: 'Output', rect: groupRect(['output'], 20) },
  ];
  const blockInner = union(groups.filter(g => g.parent === 'block').map(g => g.rect));
  groups.splice(1, 0, { id: 'block', label: 'Transformer block', rect: { x: blockInner.x - 24, y: blockInner.y - 44, w: blockInner.w + 48, h: blockInner.h + 68 } });
  // Widen groups to include the rail lane so rails are drawn inside the block.
  for (const g of groups) if (g.id === 'block' || g.parent === 'block') { const right = g.rect.x + g.rect.w; g.rect.x = Math.min(g.rect.x, RAIL_X - 20); g.rect.w = right - g.rect.x; }

  const bounds = pad(union([...nodes.map(n => n.rect), ...groups.map(g => g.rect), tokenStrip]), PAD);
  return { nodes, edges, groups, bounds, byId, tokenStrip };
}

export const RAIL_LANE_X = RAIL_X;
