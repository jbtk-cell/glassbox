/** The map: Simbrain-style diagram of the whole model with a few big tiles in labelled groups.
 *  Pan by dragging, zoom with the wheel or the buttons, double-click to fit. Tiles show the latest
 *  tensor the cursor has computed; gradients replace values during the backward pass. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../app/store';
import { mapLayout, type MapLayout, type MapSlot, type Rect } from './mapLayout';
import { TileCache, cellAt, type TileMode } from '../tiles';
import { toScreen, toWorld, zoomAt, pan, fit, type Camera } from '../camera';
import { isMasked } from '../colormap';
import { Cursor, type Status } from '../../engine/trace/cursor';
import { gKey, tKey, type Trace } from '../../engine/trace/trace';
import { T_ } from '../../engine/model/gpt';

const ACCENT = '#d97706', ORANGE = '#e08a1e', INK = '#222', DIM = '#8a8a8a';
const TAG_BG = '#f3e79a', TAG_BORDER = '#c8b04a', ARROW = 'rgba(240, 200, 90, 0.75)', ARROW_EDGE = 'rgba(200, 160, 50, 0.9)';
const CAPTION_ZOOM = 0.28, SHAPE_ZOOM = 0.7, NUMBER_PX = 22, WORD_ZOOM = 0.85;

function fmt(v: number): string { return isMasked(v) ? '-inf' : Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : v.toPrecision(3); }

interface Tip { x: number; y: number; text: string }
interface Shown { name: string; data: Float64Array; shape: number[]; mode: TileMode; badge: string; status: Status; active: boolean }

/** Which tensor a slot shows right now, and how. */
function resolve(slot: MapSlot, trace: Trace, cursor: Cursor, phase: string, activeOp: { params: string[]; output: string } | null): Shown {
  const st = (k: string) => cursor.status(k);
  const grad = phase === 'backward' || phase === 'update';
  const shown = (name: string, data: Float64Array, mode: TileMode, badge: string, status: Status, active = status === 'active'): Shown =>
    ({ name, data, shape: trace.ctx.tensors.get(name)!.shape, mode, badge, status, active });
  if (slot.kind === 'param') {
    const name = slot.tensors[0]; const t = trace.ctx.tensors.get(name)!;
    if (phase === 'update' && trace.deltas?.has(name)) return shown(name, trace.deltas.get(name)!, 'grad', 'update', st(tKey(name)));
    if (grad && trace.ctx.grads.has(name) && st(gKey(name)) !== 'pending') return shown(name, trace.ctx.grads.get(name)!.data, 'grad', 'grad', st(gKey(name)));
    return shown(name, t.data, 'value', '', 'done', !!activeOp && activeOp.params.includes(name));
  }
  if (grad) {
    for (const name of slot.tensors) if (trace.ctx.grads.has(name) && st(gKey(name)) !== 'pending') return shown(name, trace.ctx.grads.get(name)!.data, 'grad', 'grad', st(gKey(name)));
  }
  let pick = slot.tensors[0], status: Status = 'pending';
  for (const name of slot.tensors) { const s = st(tKey(name)); if (s !== 'pending') { pick = name; status = s; } }
  if (grad && status === 'pending') status = 'done';
  return shown(pick, trace.ctx.tensors.get(pick)!.data, 'value', '', status);
}

export function MapView() {
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
  const setPosition = useStore(s => s.setPosition);
  const [cam, setCam] = useState<Camera>({ x: 0, y: 0, zoom: 0.5 });
  const [tip, setTip] = useState<Tip | null>(null);
  const cache = useMemo(() => new TileCache(), []);
  const versionRef = useRef(0);
  const lastTrace = useRef<Trace | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const fittedKind = useRef<string | null>(null);

  const layout: MapLayout | null = useMemo(() => {
    if (!trace) return null;
    const shapes: Record<string, number[]> = {};
    for (const [k, t] of trace.ctx.tensors) shapes[k] = t.shape;
    return mapLayout(shapes, words);
  }, [trace, words]);

  if (trace !== lastTrace.current) { lastTrace.current = trace; versionRef.current++; }

  const size = useCallback(() => { const el = wrapRef.current; return { w: el?.clientWidth ?? 800, h: el?.clientHeight ?? 600 }; }, []);
  const fitAll = useCallback(() => { if (layout) setCam(fit(layout.bounds, size(), 24)); }, [layout, size]);

  useEffect(() => {
    if (!layout || !trace) return;
    if (fittedKind.current !== trace.kind) { fittedKind.current = trace.kind; setCam(fit(layout.bounds, size(), 24)); }
  }, [layout, trace, size]);

  // Follow the active tile when a step moves it out of view.
  const followedTrace = useRef<Trace | null>(null);
  useEffect(() => {
    if (!layout || !trace || !model) return;
    if (followedTrace.current !== trace) { followedTrace.current = trace; return; }
    const step = cursorIndex >= 0 ? trace.steps[cursorIndex] : null; if (!step) return;
    const op = model.ops.find(o => o.id === step.opId); if (!op) return;
    const slot = layout.slots.find(s => s.tensors.includes(op.output)); if (!slot) return;
    const { w, h } = size(); const r = slot.rect;
    const [sx, sy] = toScreen(cam, r.x + r.w / 2, r.y + r.h / 2);
    if (sx < 60 || sx > w - 60 || sy < 60 || sy > h - 60) setCam(c => ({ ...c, x: r.x + r.w / 2 - w / (2 * c.zoom), y: r.y + r.h / 2 - h / (2 * c.zoom) }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cursorIndex, layout, trace]);

  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current; if (!canvas || !wrap) return;
    const { w, h } = size(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr); canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const g = canvas.getContext('2d')!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#fbfbfa'; g.fillRect(0, 0, w, h);
    if (!trace || !layout || !model) {
      g.fillStyle = DIM; g.font = '15px system-ui'; g.textAlign = 'center'; g.fillText('Press Step to predict a word and see the model.', w / 2, h / 2); return;
    }
    const cursor = new Cursor(trace); cursor.index = cursorIndex;
    const step = cursor.current(); const phase = step?.phase ?? 'forward';
    const activeOp = step && step.opId !== 'adam_update' ? model.ops.find(o => o.id === step.opId) ?? null : null;
    const Z = cam.zoom;
    const S = (wx: number, wy: number) => toScreen(cam, wx, wy);
    const R = (r: Rect) => { const [x, y] = S(r.x, r.y); return { x, y, w: r.w * Z, h: r.h * Z }; };
    const T = trace.ctx.T;

    // Groups: light boxes with a yellow tag at the top left.
    for (const grp of layout.groups) {
      const r = R(grp.rect);
      g.beginPath(); g.roundRect(r.x, r.y, r.w, r.h, 6);
      g.fillStyle = grp.id === 'block' ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.6)'; g.fill();
      g.strokeStyle = grp.id === 'block' ? '#666' : '#b5b5b5'; g.lineWidth = 1; g.stroke();
    }

    // Big arrows between groups.
    for (const a of layout.arrows) {
      const [x0, y0] = S(...a.from), [cxp, cyp] = S(...a.ctrl), [x1, y1] = S(...a.to);
      const width = Math.max(6, 22 * Z);
      g.lineCap = 'round'; g.lineJoin = 'round';
      g.beginPath(); g.moveTo(x0, y0); g.quadraticCurveTo(cxp, cyp, x1, y1);
      g.strokeStyle = ARROW_EDGE; g.lineWidth = width + 2; g.stroke();
      g.strokeStyle = ARROW; g.lineWidth = width; g.stroke();
      const ang = Math.atan2(y1 - cyp, x1 - cxp); const L = width * 1.6;
      g.beginPath(); g.moveTo(x1 + Math.cos(ang) * L * 0.6, y1 + Math.sin(ang) * L * 0.6);
      g.lineTo(x1 + Math.cos(ang + 2.4) * L, y1 + Math.sin(ang + 2.4) * L); g.lineTo(x1 + Math.cos(ang - 2.4) * L, y1 + Math.sin(ang - 2.4) * L); g.closePath();
      g.fillStyle = ARROW; g.fill(); g.strokeStyle = ARROW_EDGE; g.lineWidth = 1.5; g.stroke();
    }

    // Thin lines inside the block.
    g.strokeStyle = '#555'; g.lineWidth = 1; g.lineCap = 'butt';
    for (const ln of layout.lines) {
      g.beginPath(); ln.pts.forEach((p, i) => { const [x, y] = S(...p); if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }); g.stroke();
      if (ln.dot) { const [x, y] = S(...ln.pts[0]); g.beginPath(); g.arc(x, y, Math.max(2, 3 * Z), 0, Math.PI * 2); g.fillStyle = '#fff'; g.fill(); g.stroke(); }
    }

    // Tiles.
    const shownById = new Map<string, Shown>();
    for (const slot of layout.slots) {
      const sh = resolve(slot, trace, cursor, phase, activeOp); shownById.set(slot.id, sh);
      const r = R(slot.rect); const key = tKey(sh.name);
      const img = cache.get(sh.name + (sh.mode === 'grad' ? '#' + sh.badge : ''), sh.shape, sh.data, sh.mode, versionRef.current);
      g.globalAlpha = sh.status === 'pending' ? 0.25 : sh.status === 'partial' ? 0.7 : 1;
      g.imageSmoothingEnabled = false;
      if (sh.active) { g.shadowColor = ACCENT; g.shadowBlur = 18; g.fillStyle = 'white'; g.fillRect(r.x, r.y, r.w, r.h); g.shadowBlur = 0; }
      g.drawImage(img, r.x, r.y, r.w, r.h);
      g.strokeStyle = slot.kind === 'param' ? ORANGE : INK; g.lineWidth = slot.kind === 'param' ? 2 : 1; g.strokeRect(r.x, r.y, r.w, r.h);
      if (sh.active) { g.strokeStyle = ACCENT; g.lineWidth = 2.5; g.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4); }
      g.globalAlpha = 1;
      if (selection?.key === key) { g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.strokeRect(r.x - 3, r.y - 3, r.w + 6, r.h + 6); }
      else if (hover?.key === key) { g.strokeStyle = '#60a5fa'; g.lineWidth = 1.5; g.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4); }
      if (Z >= CAPTION_ZOOM) {
        const rows = sh.shape.length <= 1 ? 1 : sh.shape[0], cols = sh.shape.length <= 1 ? (sh.shape[0] ?? 1) : sh.shape[1];
        g.textAlign = 'center'; g.textBaseline = 'top'; g.fillStyle = sh.status === 'pending' ? DIM : slot.kind === 'param' ? '#a3600f' : INK;
        g.font = `${Z < 0.5 ? 9 : 10}px system-ui`; g.fillText(slot.label, r.x + r.w / 2, r.y + r.h + 3, Math.max(r.w + 44, 80));
        if (Z >= SHAPE_ZOOM) { g.font = '9px ui-monospace, monospace'; g.fillStyle = DIM;
        g.fillText(`${sh.name} ${rows}x${cols}${sh.badge ? ' ' + sh.badge : ''}`, r.x + r.w / 2, r.y + r.h + 14, Math.max(r.w + 40, 80)); }
      }
      const rows = sh.shape.length <= 1 ? 1 : sh.shape[0], cols = sh.shape.length <= 1 ? (sh.shape[0] ?? 1) : sh.shape[1];
      const cw = r.w / cols, ch = r.h / rows;
      if (Math.min(cw, ch) >= NUMBER_PX) {
        const c0 = Math.max(0, Math.floor(-r.x / cw)), c1 = Math.min(cols, Math.ceil((w - r.x) / cw));
        const r0 = Math.max(0, Math.floor(-r.y / ch)), r1 = Math.min(rows, Math.ceil((h - r.y) / ch));
        g.font = `${Math.min(13, cw / 4)}px ui-monospace, monospace`; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillStyle = 'rgba(0,0,0,0.85)';
        for (let i = r0; i < r1; i++) for (let j = c0; j < c1; j++) g.fillText(fmt(sh.data[i * cols + j]), r.x + (j + 0.5) * cw, r.y + (i + 0.5) * ch, cw - 4);
        g.strokeStyle = 'rgba(0,0,0,0.08)'; g.lineWidth = 1; g.beginPath();
        for (let i = r0; i <= r1; i++) { g.moveTo(r.x, r.y + i * ch); g.lineTo(r.x + r.w, r.y + i * ch); }
        for (let j = c0; j <= c1; j++) { g.moveTo(r.x + j * cw, r.y); g.lineTo(r.x + j * cw, r.y + r.h); }
        g.stroke();
      }
      for (const [ref, colour] of [[selection, '#2563eb'], [hover, '#60a5fa']] as const) {
        if (ref?.key === key) { const i = Math.floor(ref.index / cols), j = ref.index % cols; g.strokeStyle = colour; g.lineWidth = 2; g.strokeRect(r.x + j * cw, r.y + i * ch, Math.max(cw, 2), Math.max(ch, 2)); }
      }
    }

    // Inputs: one row per word in the context, a dot in the column of its id.
    {
      const r = R(layout.inputs.rect); const V = layout.inputs.cols; const rh = r.h / T;
      const active = activeOp?.id === 'embed';
      if (active) { g.shadowColor = ACCENT; g.shadowBlur = 18; g.fillStyle = 'white'; g.fillRect(r.x, r.y, r.w, r.h); g.shadowBlur = 0; }
      g.fillStyle = 'white'; g.fillRect(r.x, r.y, r.w, r.h);
      for (let t = 0; t < T; t++) {
        const y = r.y + t * rh;
        if (t === position) { g.fillStyle = 'rgba(217,119,6,0.15)'; g.fillRect(r.x, y, r.w, rh); }
        const id = trace.ctx.tokens[t];
        g.beginPath(); g.arc(r.x + ((id + 0.5) / V) * r.w, y + rh / 2, Math.max(1.5, 2.6 * Z), 0, Math.PI * 2); g.fillStyle = '#333'; g.fill();
        if (Z >= WORD_ZOOM) { g.fillStyle = INK; g.font = `${Math.min(11, rh * 0.7)}px system-ui`; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(words?.[id] ?? String(id), r.x - 6, y + rh / 2, 90); }
      }
      g.strokeStyle = active ? ACCENT : INK; g.lineWidth = active ? 2.5 : 1; g.strokeRect(r.x, r.y, r.w, r.h);
      if (Z >= CAPTION_ZOOM) { g.textAlign = 'center'; g.textBaseline = 'top'; g.fillStyle = INK; g.font = '10px system-ui'; g.fillText(`Word ids ${T}x${V}`, r.x + r.w / 2, r.y + r.h + 3); }
    }

    // Predicted next token: one circle per word, filled by the probability for the last position.
    {
      const P = trace.ctx.tensors.get(T_.probs)!; const V = P.shape[1]; const row = (T - 1) * V;
      const ready = cursor.status(tKey(T_.probs)) !== 'pending';
      const active = cursor.status(tKey(T_.probs)) === 'active';
      let best = 1; for (let v = 2; v < V; v++) if (P.data[row + v] > P.data[row + best]) best = v;
      const { pred } = layout; const r = R(pred.rect); const cw = pred.cellW * Z, ch = pred.cellH * Z;
      g.fillStyle = 'white'; g.fillRect(r.x, r.y, r.w, r.h);
      const rad = Math.max(1.5, Math.min(cw, ch) * 0.28);
      pred.order.forEach((id, n) => {
        const i = Math.floor(n / pred.cols), j = n % pred.cols;
        const x = r.x + (j + 0.5) * cw, y = r.y + (i + 0.5) * ch + (Z >= WORD_ZOOM ? ch * 0.12 : 0);
        const p = ready ? P.data[row + id] : 0;
        g.beginPath(); g.arc(x, y, rad, 0, Math.PI * 2);
        g.fillStyle = `rgba(214, 40, 40, ${Math.min(1, p)})`; g.fill();
        g.strokeStyle = ready && id === best ? '#b91c1c' : '#666'; g.lineWidth = ready && id === best ? 2 : 1; g.stroke();
        if (Z >= WORD_ZOOM) { g.fillStyle = INK; g.font = `${Math.min(10, cw * 0.28)}px system-ui`; g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(words?.[id] ?? String(id), x, y - rad - 2, cw - 4); }
        if (selection?.key === tKey(T_.probs) && selection.index === row + id) { g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.beginPath(); g.arc(x, y, rad + 3, 0, Math.PI * 2); g.stroke(); }
      });
      g.strokeStyle = active ? ACCENT : INK; g.lineWidth = active ? 2.5 : 1; g.strokeRect(r.x, r.y, r.w, r.h);
      if (ready) {
        g.fillStyle = INK; g.font = `${Math.max(11, Math.min(15, 13 * Z + 4))}px system-ui`; g.textAlign = 'left'; g.textBaseline = 'top';
        const target = trace.ctx.targets ? `    target: ${words?.[trace.ctx.targets[T - 1]] ?? trace.ctx.targets[T - 1]}` : '';
        g.fillText(`next: ${words?.[best] ?? best}  (${(P.data[row + best] * 100).toFixed(1)}%)${target}`, r.x, r.y + r.h + 6);
      }
    }

    // Group tags, drawn last so nothing covers them. Screen-space text at world anchors.
    g.font = '600 12px system-ui'; g.textBaseline = 'middle';
    for (const grp of layout.groups) {
      const [x, y] = S(grp.rect.x, grp.rect.y); const tw = g.measureText(grp.label).width + 16;
      g.beginPath(); g.roundRect(x, y - 22, tw, 20, 3); g.fillStyle = TAG_BG; g.fill(); g.strokeStyle = TAG_BORDER; g.lineWidth = 1; g.stroke();
      g.fillStyle = INK; g.textAlign = 'left'; g.fillText(grp.label, x + 8, y - 12);
    }
    const shownLoss = shownById.get('loss');
    if (shownLoss && shownLoss.status !== 'pending') { const r = R(layout.slotById.get('loss')!.rect); g.fillStyle = INK; g.font = '11px ui-monospace, monospace'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(`loss ${fmt(trace.ctx.tensors.get(T_.loss)!.data[0])}`, r.x + r.w + 6, r.y + r.h / 2); }
  }, [trace, layout, model, cursorIndex, selection, hover, cam, words, position, cache, size]);

  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(() => setCam(c => ({ ...c }))); ro.observe(el); return () => ro.disconnect();
  }, []);

  type Hit = { kind: 'tile'; slot: MapSlot; name: string; index: number } | { kind: 'pred'; id: number } | { kind: 'input'; t: number };
  const hit = useCallback((sx: number, sy: number): Hit | null => {
    if (!layout || !trace || !model) return null;
    const [wx, wy] = toWorld(cam, sx, sy);
    const cursor = new Cursor(trace); cursor.index = cursorIndex;
    const step = cursor.current(); const phase = step?.phase ?? 'forward';
    const activeOp = step && step.opId !== 'adam_update' ? model.ops.find(o => o.id === step.opId) ?? null : null;
    for (const slot of layout.slots) {
      const sh = resolve(slot, trace, cursor, phase, activeOp);
      const i = cellAt(slot.rect, sh.shape, wx, wy); if (i !== null) return { kind: 'tile', slot, name: sh.name, index: i };
    }
    const p = layout.pred;
    if (wx >= p.rect.x && wx < p.rect.x + p.rect.w && wy >= p.rect.y && wy < p.rect.y + p.rect.h) {
      const j = Math.floor((wx - p.rect.x) / p.cellW), i = Math.floor((wy - p.rect.y) / p.cellH); const n = i * p.cols + j;
      if (n < p.order.length) return { kind: 'pred', id: p.order[n] };
    }
    const r = layout.inputs.rect;
    if (wx >= r.x && wx < r.x + r.w && wy >= r.y && wy < r.y + r.h) return { kind: 'input', t: Math.min(trace.ctx.T - 1, Math.floor((wy - r.y) / (r.h / trace.ctx.T))) };
    return null;
  }, [layout, trace, model, cam, cursorIndex]);

  const local = (e: React.MouseEvent) => { const b = wrapRef.current!.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top] as const; };
  const onMove = (e: React.MouseEvent) => {
    const [sx, sy] = local(e);
    if (drag.current) { const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y; if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true; setCam(c => pan(c, dx, dy)); drag.current.x = e.clientX; drag.current.y = e.clientY; return; }
    const hh = hit(sx, sy);
    if (!hh || !trace) { setHover(null); setTip(null); return; }
    if (hh.kind === 'tile') {
      const t = trace.ctx.tensors.get(hh.name)!; const cols = t.shape.length <= 1 ? (t.shape[0] ?? 1) : t.shape[1]; const i = Math.floor(hh.index / cols), j = hh.index % cols;
      const grad = trace.ctx.grads.get(hh.name)?.data[hh.index];
      setHover({ key: tKey(hh.name), index: hh.index });
      setTip({ x: sx + 14, y: sy + 14, text: `${hh.name}[${i}, ${j}] = ${fmt(t.data[hh.index])}${grad !== undefined ? `   grad ${fmt(grad)}` : ''}` });
    } else if (hh.kind === 'pred') {
      const P = trace.ctx.tensors.get(T_.probs)!; const V = P.shape[1]; const p = P.data[(trace.ctx.T - 1) * V + hh.id];
      setHover({ key: tKey(T_.probs), index: (trace.ctx.T - 1) * V + hh.id });
      setTip({ x: sx + 14, y: sy + 14, text: `${words?.[hh.id] ?? hh.id}  ${(p * 100).toFixed(2)}%` });
    } else { setHover(null); setTip({ x: sx + 14, y: sy + 14, text: `position ${hh.t}: ${words?.[trace.ctx.tokens[hh.t]] ?? trace.ctx.tokens[hh.t]}` }); }
  };
  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; };
  const onUp = (e: React.MouseEvent) => {
    const moved = drag.current?.moved; drag.current = null; if (moved) return;
    const [sx, sy] = local(e); const hh = hit(sx, sy);
    if (!hh || !trace) { select(null); return; }
    if (hh.kind === 'tile') select({ key: tKey(hh.name), index: hh.index });
    else if (hh.kind === 'pred') select({ key: tKey(T_.probs), index: (trace.ctx.T - 1) * trace.ctx.tensors.get(T_.probs)!.shape[1] + hh.id });
    else setPosition(hh.t);
  };
  const onWheel = (e: React.WheelEvent) => { const [sx, sy] = local(e); setCam(c => zoomAt(c, sx, sy, Math.pow(1.1, -e.deltaY / 100))); };
  const zoomBy = (f: number) => { const { w, h } = size(); setCam(c => zoomAt(c, w / 2, h / 2, f)); };

  return (
    <div ref={wrapRef} className="map-view" style={{ cursor: drag.current ? 'grabbing' : 'default' }}
      onMouseMove={onMove} onMouseDown={onDown} onMouseUp={onUp} onMouseLeave={() => { drag.current = null; setHover(null); setTip(null); }} onWheel={onWheel} onDoubleClick={fitAll}>
      <canvas ref={canvasRef} />
      <div className="map-zoom">
        <button onClick={() => zoomBy(1.3)} title="Zoom in">+</button>
        <button onClick={() => zoomBy(1 / 1.3)} title="Zoom out">-</button>
        <button onClick={fitAll} title="Fit everything in view">Fit</button>
      </div>
      {tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  );
}
