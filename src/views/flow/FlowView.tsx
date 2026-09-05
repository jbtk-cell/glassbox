import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../app/store';
import { flowLayout, RAIL_LANE_X, type FlowLayout, type LayoutNode } from './layout';
import { TileCache, cellAt, type TileMode } from '../tiles';
import { toScreen, toWorld, zoomAt, pan, fit, type Camera } from '../camera';
import { isMasked } from '../colormap';
import { Cursor, type Status } from '../../engine/trace/cursor';
import { gKey, type Trace } from '../../engine/trace/trace';
import { T_ } from '../../engine/model/gpt';

const ACCENT = '#d97706', ORANGE = '#f59e0b', INK = '#1f2937', DIM = '#9ca3af';
const COLLAPSE_ZOOM = 0.1, LABEL_ZOOM = 0.5, NUMBER_PX = 22;

function fmt(v: number): string { return isMasked(v) ? '-inf' : Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toPrecision(3); }

interface Tip { x: number; y: number; text: string }

export function FlowView() {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trace = useStore(s => s.trace);
  const cursorIndex = useStore(s => s.cursorIndex);
  const selection = useStore(s => s.selection);
  const hover = useStore(s => s.hover);
  const model = useStore(s => s.model);
  const words = useStore(s => s.corpus?.vocab.words);
  const position = useStore(s => s.position);
  const setHover = useStore(s => s.setHover);
  const select = useStore(s => s.select);
  const [cam, setCam] = useState<Camera>({ x: -400, y: -1200, zoom: 0.5 });
  const [tip, setTip] = useState<Tip | null>(null);
  const cache = useMemo(() => new TileCache(), []);
  const versionRef = useRef(0);
  const lastTrace = useRef<Trace | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const fittedKind = useRef<string | null>(null);

  // Layout depends only on the op list and the tensor shapes of this trace.
  const layout: FlowLayout | null = useMemo(() => {
    if (!trace || !model) return null;
    const shapes: Record<string, number[]> = {};
    for (const [k, t] of trace.ctx.tensors) shapes[k] = t.shape;
    return flowLayout(model.ops, shapes);
  }, [trace, model]);

  if (trace !== lastTrace.current) { lastTrace.current = trace; versionRef.current++; }

  const size = useCallback(() => {
    const el = wrapRef.current; return { w: el?.clientWidth ?? 800, h: el?.clientHeight ?? 600 };
  }, []);

  // Fit when a trace first appears or its kind changes.
  useEffect(() => {
    if (!layout || !trace) return;
    if (fittedKind.current !== trace.kind) { fittedKind.current = trace.kind; setCam(fit(layout.bounds, size(), 30)); }
  }, [layout, trace, size]);

  // Follow the active op: when a step moves it out of view, pan so it is centred (zoom unchanged).
  const followedTrace = useRef<Trace | null>(null);
  useEffect(() => {
    if (!layout || !trace) return;
    if (followedTrace.current !== trace) { followedTrace.current = trace; return; }   // a new trace is fitted, not followed
    const step = cursorIndex >= 0 ? trace.steps[cursorIndex] : null; if (!step) return;
    const node = layout.byId.get('op:' + step.opId); if (!node) return;
    const { w, h } = size();
    const [sx, sy] = toScreen(cam, node.rect.x + node.rect.w / 2, node.rect.y + node.rect.h / 2);
    const margin = 60;
    if (sx < margin || sx > w - margin || sy < margin || sy > h - margin) {
      setCam(c => ({ ...c, x: node.rect.x + node.rect.w / 2 - w / (2 * c.zoom), y: node.rect.y + node.rect.h / 2 - h / (2 * c.zoom) }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorIndex, layout, trace]);

  // Draw.
  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current; if (!canvas || !wrap) return;
    const { w, h } = size(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const g = canvas.getContext('2d')!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#fafaf9'; g.fillRect(0, 0, w, h);
    if (!trace || !layout || !model) {
      g.fillStyle = DIM; g.font = '15px system-ui'; g.textAlign = 'center';
      g.fillText('Predict a word or record a training step to see the computation.', w / 2, h / 2); return;
    }
    const cursor = new Cursor(trace); cursor.index = cursorIndex;
    const step = cursor.current(); const phase = step?.phase ?? 'forward';
    const opById = new Map(model.ops.map(o => [o.id, o]));
    const S = (wx: number, wy: number) => toScreen(cam, wx, wy);
    const Z = cam.zoom;
    const R = (r: { x: number; y: number; w: number; h: number }) => { const [x, y] = S(r.x, r.y); return { x, y, w: r.w * Z, h: r.h * Z }; };
    const rr = (x: number, y: number, ww: number, hh: number, rad: number) => { g.beginPath(); g.roundRect(x, y, ww, hh, rad); };

    // Groups.
    const drawGroup = (id: string, withLabel: boolean) => {
      const grp = layout.groups.find(x => x.id === id)!; const r = R(grp.rect);
      rr(r.x, r.y, r.w, r.h, 10 * Z + 4);
      g.fillStyle = id === 'block' ? 'rgba(217,119,6,0.05)' : 'rgba(0,0,0,0.025)'; g.fill();
      g.strokeStyle = id === 'block' ? 'rgba(217,119,6,0.5)' : 'rgba(0,0,0,0.18)'; g.lineWidth = id === 'block' ? 1.5 : 1; g.stroke();
      if (!withLabel) return;
      g.font = `${Z < COLLAPSE_ZOOM ? 16 : 13}px system-ui`; g.textAlign = 'left'; g.textBaseline = 'top';
      const tw = g.measureText(grp.label).width;
      g.fillStyle = 'rgba(250,250,249,0.9)'; g.fillRect(r.x + 4, r.y + 3, tw + 10, 18);
      g.fillStyle = id === 'block' ? '#b45309' : INK; g.fillText(grp.label, r.x + 9, r.y + 5);
    };
    for (const id of ['embedding', 'block', 'attention', 'ff', 'unembedding', 'output']) if (layout.groups.some(x => x.id === id)) drawGroup(id, Z >= 0.3 || (id !== 'attention' && id !== 'ff'));

    // Token strip.
    const strip = R(layout.tokenStrip); const T = trace.ctx.T; const bw = strip.w / Math.max(T, 1);
    for (let t = 0; t < T; t++) {
      const x = strip.x + t * bw; const on = t === position;
      rr(x + 1, strip.y, bw - 2, strip.h, 3); g.fillStyle = on ? 'rgba(217,119,6,0.18)' : 'white'; g.fill(); g.strokeStyle = on ? ACCENT : '#d1d5db'; g.lineWidth = 1; g.stroke();
      if (bw > 18) { g.fillStyle = INK; g.font = `${Math.min(12, bw / 3)}px system-ui`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(words?.[trace.ctx.tokens[t]] ?? String(trace.ctx.tokens[t]), x + bw / 2, strip.y + strip.h / 2, bw - 4); }
    }

    const statusOf = (key: string): Status => cursor.status(key);
    const outputOf = (opId: string) => opId === 'adam_update' ? null : opById.get(opId)?.output ?? null;

    if (Z < COLLAPSE_ZOOM) {
      // Collapsed: spine tiles as solid bars only.
      for (const n of layout.nodes) if (n.kind === 'tensor' && Math.abs(n.rect.x + n.rect.w / 2) < 1) {
        const r = R(n.rect); g.fillStyle = statusOf(n.key!) === 'pending' ? '#e5e7eb' : '#9ca3af'; g.fillRect(r.x, r.y, r.w, r.h);
      }
      return;
    }

    // Edges.
    const activeOp = step ? 'op:' + step.opId : null;
    for (const e of layout.edges) {
      const a = layout.byId.get(e.from)!, b = layout.byId.get(e.to)!;
      const into = e.to === activeOp || e.from === activeOp;
      g.strokeStyle = into ? ACCENT : 'rgba(0,0,0,0.22)'; g.lineWidth = into ? 2 : 1;
      g.beginPath();
      if (e.rail) {
        const [rx] = S(RAIL_LANE_X, 0); const [, ay] = S(0, a.rect.y); const [, by] = S(0, b.rect.y + b.rect.h / 2); const [bx] = S(b.rect.x, 0);
        const [ax] = S(a.rect.x, 0);
        g.moveTo(ax, ay + a.rect.h * Z / 2); g.lineTo(rx, ay + a.rect.h * Z / 2); g.lineTo(rx, by); g.lineTo(bx, by);
      } else if (a.kind === 'param') {
        const [ax, ay] = S(a.rect.x, a.rect.y + a.rect.h / 2); const [bx, by] = S(b.rect.x + b.rect.w, b.rect.y + b.rect.h / 2);
        const dx = Math.max(20 * Z, Math.abs(ax - bx) / 2);
        g.moveTo(ax, ay); g.bezierCurveTo(ax - dx, ay, bx + dx, by, bx, by);
      } else {
        const [ax, ay] = S(a.rect.x + a.rect.w / 2, a.rect.y); const [bx, by] = S(b.rect.x + b.rect.w / 2, b.rect.y + b.rect.h);
        const dy = Math.max(20 * Z, Math.abs(ay - by) / 2);
        g.moveTo(ax, ay); g.bezierCurveTo(ax, ay - dy, bx, by + dy, bx, by);
      }
      g.stroke();
    }

    // Op pills.
    for (const n of layout.nodes) if (n.kind === 'op') {
      const r = R(n.rect); const out = outputOf(n.opId!); const st: Status = out ? statusOf('t:' + out) : 'done';
      const active = activeOp === n.id;
      if (active) { g.shadowColor = ACCENT; g.shadowBlur = 16; }
      rr(r.x, r.y, r.w, r.h, r.h / 2);
      g.fillStyle = active ? ACCENT : st === 'pending' ? '#f3f4f6' : 'white'; g.fill();
      g.shadowBlur = 0; g.strokeStyle = active ? ACCENT : st === 'pending' ? '#d1d5db' : '#6b7280'; g.lineWidth = 1; g.stroke();
      if (Z >= 0.35) { g.fillStyle = active ? 'white' : st === 'pending' ? DIM : INK; g.font = `${Math.min(12, 10 * Z + 4)}px system-ui`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(n.label, r.x + r.w / 2, r.y + r.h / 2, r.w - 8); }
    }

    // Tiles.
    const gradPhase = phase === 'backward' || phase === 'update';
    for (const n of layout.nodes) if (n.kind !== 'op') {
      const name = n.label; const key = n.key!; const r = R(n.rect);
      const value = trace.ctx.tensors.get(name)!;
      let data = value.data, mode: TileMode = 'value', badge = '';
      let st = statusOf(key);
      if (gradPhase) {
        if (phase === 'update' && n.kind === 'param' && trace.deltas?.has(name)) { data = trace.deltas.get(name)!; mode = 'grad'; badge = 'delta'; st = statusOf(key); }
        else if (trace.ctx.grads.has(name) && statusOf(gKey(name)) !== 'pending') { data = trace.ctx.grads.get(name)!.data; mode = 'grad'; badge = 'grad'; st = statusOf(gKey(name)); }
        else if (n.kind === 'param') st = 'done';
      }
      const img = cache.get(name + (mode === 'grad' ? '#' + badge : ''), value.shape, data, mode, versionRef.current);
      g.globalAlpha = st === 'pending' ? 0.3 : st === 'partial' ? 0.75 : 1;
      g.imageSmoothingEnabled = false;
      if (st === 'active') { g.shadowColor = ACCENT; g.shadowBlur = 18; g.fillStyle = 'white'; g.fillRect(r.x, r.y, r.w, r.h); g.shadowBlur = 0; }
      g.drawImage(img, r.x, r.y, r.w, r.h);
      if (st === 'partial') { g.strokeStyle = 'rgba(0,0,0,0.35)'; g.lineWidth = 1; g.beginPath(); for (let d = -r.h; d < r.w; d += 8) { g.moveTo(r.x + d, r.y + r.h); g.lineTo(r.x + d + r.h, r.y); } g.save(); g.beginPath(); g.rect(r.x, r.y, r.w, r.h); g.clip(); g.stroke(); g.restore(); }
      g.strokeStyle = n.kind === 'param' ? ORANGE : INK; g.lineWidth = n.kind === 'param' ? 2 : 1; g.strokeRect(r.x, r.y, r.w, r.h);
      if (st === 'active') { g.strokeStyle = ACCENT; g.lineWidth = 2.5; g.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4); }
      g.globalAlpha = 1;
      if (selection?.key === key) { g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.strokeRect(r.x - 3, r.y - 3, r.w + 6, r.h + 6); }
      else if (hover?.key === key) { g.strokeStyle = '#60a5fa'; g.lineWidth = 1.5; g.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4); }
      if (Z >= LABEL_ZOOM) {
        g.fillStyle = n.kind === 'param' ? '#b45309' : INK; g.font = `${Math.min(12, 8 * Z + 4)}px ui-monospace, monospace`; g.textAlign = 'left'; g.textBaseline = 'bottom';
        g.fillText(`${name}  ${value.shape.join('x')}${badge ? '  ' + badge : ''}`, r.x, r.y - 2);
      }
      // Cell numbers and cell outlines when zoomed in far enough.
      const rows = value.shape.length === 1 ? 1 : value.shape[0], cols = value.shape.length === 1 ? value.shape[0] : value.shape[1];
      const cw = r.w / cols, ch = r.h / rows;
      if (Math.min(cw, ch) >= NUMBER_PX) {
        const c0 = Math.max(0, Math.floor((0 - r.x) / cw)), c1 = Math.min(cols, Math.ceil((w - r.x) / cw));
        const r0 = Math.max(0, Math.floor((0 - r.y) / ch)), r1 = Math.min(rows, Math.ceil((h - r.y) / ch));
        g.font = `${Math.min(13, cw / 4)}px ui-monospace, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle';
        for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) {
          const v = data[i * cols + j]; g.fillStyle = 'rgba(0,0,0,0.85)';
          g.fillText(fmt(v), r.x + (j + 0.5) * cw, r.y + (i + 0.5) * ch, cw - 4);
        }
        g.strokeStyle = 'rgba(0,0,0,0.08)'; g.lineWidth = 1; g.beginPath();
        for (let i = r0; i <= r1; i++) { g.moveTo(r.x, r.y + i * ch); g.lineTo(r.x + r.w, r.y + i * ch); }
        for (let j = c0; j <= c1; j++) { g.moveTo(r.x + j * cw, r.y); g.lineTo(r.x + j * cw, r.y + r.h); }
        g.stroke();
      }
      // Selected / hovered cell outline.
      for (const [ref, colour] of [[selection, '#2563eb'], [hover, '#60a5fa']] as const) {
        if (ref?.key === key) { const i = Math.floor(ref.index / cols), j = ref.index % cols; g.strokeStyle = colour; g.lineWidth = 2; g.strokeRect(r.x + j * cw, r.y + i * ch, Math.max(cw, 2), Math.max(ch, 2)); }
      }
    }

    // Prediction readout above the probabilities tile.
    const probsNode = layout.byId.get('t:' + T_.probs);
    if (probsNode && statusOf('t:' + T_.probs) !== 'pending') {
      const P = trace.ctx.tensors.get(T_.probs)!; const V = P.shape[1]; const row = (T - 1) * V;
      let best = 0; for (let v = 1; v < V; v++) if (P.data[row + v] > P.data[row + best]) best = v;
      const r = R(probsNode.rect); g.fillStyle = INK; g.font = `${Math.min(16, 12 * Z + 6)}px system-ui`; g.textAlign = 'center'; g.textBaseline = 'bottom';
      const target = trace.ctx.targets ? `   target: ${words?.[trace.ctx.targets[T - 1]] ?? trace.ctx.targets[T - 1]}` : '';
      g.fillText(`next: ${words?.[best] ?? best}  (${(P.data[row + best] * 100).toFixed(1)}%)${target}`, r.x + r.w / 2, r.y - 18);
    }
  }, [trace, layout, model, cursorIndex, selection, hover, cam, words, position, cache, size]);

  // Redraw on resize.
  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(() => setCam(c => ({ ...c })));
    ro.observe(el); return () => ro.disconnect();
  }, []);

  const hit = useCallback((sx: number, sy: number): { node: LayoutNode; index: number } | null => {
    if (!layout) return null;
    const [wx, wy] = toWorld(cam, sx, sy);
    for (const n of layout.nodes) if (n.kind !== 'op') {
      const i = cellAt(n.rect, n.shape!, wx, wy); if (i !== null) return { node: n, index: i };
    }
    return null;
  }, [layout, cam]);

  const local = (e: React.MouseEvent) => { const b = wrapRef.current!.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top] as const; };

  const onMove = (e: React.MouseEvent) => {
    const [sx, sy] = local(e);
    if (drag.current) {
      const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true;
      setCam(c => pan(c, dx, dy)); drag.current.x = e.clientX; drag.current.y = e.clientY; return;
    }
    const hh = hit(sx, sy);
    if (!hh || !trace) { setHover(null); setTip(null); return; }
    const name = hh.node.label; const t = trace.ctx.tensors.get(name)!;
    const cols = t.shape.length === 1 ? t.shape[0] : t.shape[1]; const i = Math.floor(hh.index / cols), j = hh.index % cols;
    const grad = trace.ctx.grads.get(name)?.data[hh.index];
    setHover({ key: hh.node.key!, index: hh.index });
    setTip({ x: sx + 14, y: sy + 14, text: `${name}[${i}, ${j}] = ${fmt(t.data[hh.index])}${grad !== undefined ? `   grad ${fmt(grad)}` : ''}` });
  };
  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; };
  const onUp = (e: React.MouseEvent) => {
    const moved = drag.current?.moved; drag.current = null; if (moved) return;
    const [sx, sy] = local(e); const hh = hit(sx, sy);
    select(hh ? { key: hh.node.key!, index: hh.index } : null);
  };
  const onWheel = (e: React.WheelEvent) => { const [sx, sy] = local(e); setCam(c => zoomAt(c, sx, sy, Math.pow(1.1, -e.deltaY / 100))); };
  const onDouble = () => { if (layout) setCam(fit(layout.bounds, size(), 30)); };

  return (
    <div ref={wrapRef} className="flow-view" style={{ position: 'relative', width: '100%', height: '100%', overflow: 'hidden', cursor: drag.current ? 'grabbing' : 'default' }}
      onMouseMove={onMove} onMouseDown={onDown} onMouseUp={onUp} onMouseLeave={() => { drag.current = null; setHover(null); setTip(null); }} onWheel={onWheel} onDoubleClick={onDouble}>
      <canvas ref={canvasRef} />
      {tip && <div className="tip" style={{ position: 'absolute', left: tip.x, top: tip.y, pointerEvents: 'none', background: 'rgba(31,41,55,0.92)', color: 'white', font: '12px ui-monospace, monospace', padding: '4px 7px', borderRadius: 4, whiteSpace: 'nowrap' }}>{tip.text}</div>}
    </div>
  );
}
