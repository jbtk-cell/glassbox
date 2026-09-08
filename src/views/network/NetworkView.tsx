/** Neurons: one position's numbers as columns of circles, every neighbouring pair of columns
 *  joined by the step that turns one into the next. Matrix steps draw their whole fan of weights,
 *  faint until you hover a neuron; elementwise steps draw one-to-one links with the operation
 *  named above them. Pan by dragging, zoom with the wheel or the buttons, double-click to fit. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../app/store';
import { tKey, type Trace } from '../../engine/trace/trace';
import { Cursor } from '../../engine/trace/cursor';
import { T_ } from '../../engine/model/gpt';
import { diverging, sequential, maxAbs, isMasked, type RGB } from '../colormap';
import { toScreen, toWorld, zoomAt, pan, fit, type Camera } from '../camera';
import { networkLayout, rowY, COMPACT_HALF_W, type NetLayout, type NeuronGroup, type Band } from './networkLayout';

const INK = '#222', DIM = '#8a8a8a';
const RED: RGB = [214, 40, 40];
const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
const fmt = (v: number): string => (isMasked(v) ? '-inf' : String(Number(v.toPrecision(4))));

interface Wire { band: Band; from: string; i: number; to: string; j: number; w: number; maxW: number }
interface Tip { x: number; y: number; text: string }
interface Focus { key: string; i: number }

/** Every wire of every band, at the current position. */
function buildWires(trace: Trace, L: NetLayout, position: number): Wire[] {
  const out: Wire[] = [];
  const T = trace.ctx.T;
  for (const band of L.bands) {
    const to = L.byKey.get(band.to)!;
    if (band.kind === 'matrix') {
      const from = L.byKey.get(band.from[0])!; const W = trace.ctx.tensors.get(band.param!)!; const maxW = maxAbs(W.data);
      for (let i = 0; i < from.n; i++) for (let j = 0; j < to.n; j++) out.push({ band, from: from.key, i, to: to.key, j, w: W.data[i * to.n + j], maxW });
    } else if (band.kind === 'elementwise') {
      if (to.compact) continue;                      // strip to strip: drawn as a band, not wires
      for (const f of band.from) { const from = L.byKey.get(f)!; for (let i = 0; i < Math.min(from.n, to.n); i++) out.push({ band, from: from.key, i, to: to.key, j: i, w: 0, maxW: 1 }); }
    } else if (band.kind === 'attend') {
      for (const f of band.from) { const from = L.byKey.get(f)!; for (let i = 0; i < from.n; i++) for (let j = 0; j < to.n; j++) out.push({ band, from: from.key, i, to: to.key, j, w: 0, maxW: 1 }); }
    } else if (band.kind === 'mix') {
      const attn = L.byKey.get(T_.attn)!; const A = trace.ctx.tensors.get(T_.attn)!;
      for (let j = 0; j < attn.n; j++) { if (j > position) continue; const a = A.data[position * T + j]; for (let i = 0; i < to.n; i++) out.push({ band, from: attn.key, i: j, to: to.key, j: i, w: a, maxW: 1 }); }
    }
  }
  return out;
}

function wireStyle(w: Wire, focus: Focus | null): { color: string; width: number; hot: boolean } | null {
  const hot = focus !== null && ((w.from === focus.key && w.i === focus.i) || (w.to === focus.key && w.j === focus.i));
  const dimmed = focus !== null && !hot;
  const rel = w.maxW === 0 ? 0 : Math.abs(w.w) / w.maxW;
  switch (w.band.kind) {
    case 'matrix': return { color: rgba(diverging(w.w, w.maxW), hot ? 0.95 : dimmed ? 0.015 : 0.03 + 0.22 * rel), width: (0.6 + 2.2 * rel) * (hot ? 1.4 : 1), hot };
    case 'elementwise': return { color: `rgba(70, 70, 70, ${hot ? 0.9 : dimmed ? 0.06 : 0.3})`, width: hot ? 2 : 1, hot };
    case 'attend': return { color: `rgba(70, 70, 70, ${hot ? 0.8 : dimmed ? 0.012 : 0.04})`, width: hot ? 1.5 : 0.8, hot };
    case 'mix': return { color: rgba(sequential(0.35 + 0.65 * w.w), hot ? 0.95 : dimmed ? 0.03 : 0.08 + 0.75 * w.w), width: (0.8 + 4 * w.w) * (hot ? 1.3 : 1), hot };
  }
}

function neuronColor(key: string, v: number, m: number): RGB {
  if (key === T_.probs) return [255 - (255 - RED[0]) * Math.min(1, v), 255 - (255 - RED[1]) * Math.min(1, v), 255 - (255 - RED[2]) * Math.min(1, v)];
  if (key === T_.attn) return sequential(v);
  return diverging(v, m);
}

export function NetworkView() {
  const trace = useStore(s => s.trace);
  const cursorIndex = useStore(s => s.cursorIndex);
  const selection = useStore(s => s.selection);
  const hover = useStore(s => s.hover);
  const position = useStore(s => s.position);
  const words = useStore(s => s.corpus?.vocab.words);
  const select = useStore(s => s.select);
  const setHover = useStore(s => s.setHover);
  const setPosition = useStore(s => s.setPosition);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cam, setCam] = useState<Camera>({ x: -100, y: -400, zoom: 0.6 });
  const [tip, setTip] = useState<Tip | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const fittedFor = useRef<string>('');

  const layout = useMemo(() => {
    if (!trace) return null;
    const shapes: Record<string, number[]> = {};
    for (const [k, t] of trace.ctx.tensors) shapes[k] = t.shape;
    return networkLayout(shapes);
  }, [trace]);
  const wires = useMemo(() => (trace && layout ? buildWires(trace, layout, position) : []), [trace, layout, position]);

  const size = useCallback(() => { const el = wrapRef.current; return { w: el?.clientWidth ?? 800, h: el?.clientHeight ?? 600 }; }, []);
  const fitAll = useCallback(() => { if (layout) setCam(fit(layout.bounds, size(), 20)); }, [layout, size]);
  useEffect(() => {
    if (!layout || !trace) return;
    const key = `${trace.ctx.T}`;
    if (fittedFor.current !== key) { fittedFor.current = key; setCam(fit(layout.bounds, size(), 20)); }
  }, [layout, trace, size]);
  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(() => setCam(c => ({ ...c }))); ro.observe(el); return () => ro.disconnect();
  }, []);

  /** The neuron whose wires are shown: hovered, else selected, if it belongs to this position. */
  const focus = useMemo((): Focus | null => {
    if (!layout) return null;
    for (const ref of [hover, selection]) {
      if (!ref) continue;
      const key = ref.key.startsWith('t:') ? ref.key.slice(2) : ref.key; const g = layout.byKey.get(key); if (!g) continue;
      const i = ref.index - position * g.n; if (i >= 0 && i < g.n) return { key, i };
    }
    return null;
  }, [hover, selection, layout, position]);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const { w, h } = size(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const g = canvas.getContext('2d')!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#fbfbfa'; g.fillRect(0, 0, w, h);
    if (!trace || !layout) { g.fillStyle = DIM; g.font = '15px system-ui'; g.textAlign = 'center'; g.fillText('Press Step to predict a word and see the neurons.', w / 2, h / 2); return; }
    const Z = cam.zoom; const S = (wx: number, wy: number) => toScreen(cam, wx, wy);
    const cursor = new Cursor(trace); cursor.index = cursorIndex;
    const status = new Map(layout.groups.map(gr => [gr.key, cursor.status(tKey(gr.key))]));
    const alphaOf = (key: string) => (status.get(key) === 'pending' ? 0.25 : 1);
    const centre = (gr: NeuronGroup, i: number): [number, number] => S(gr.x, gr.compact ? rowY(gr, i) + gr.dy / 2 : rowY(gr, i));

    // Skip arcs: values carried forward and added back in.
    g.setLineDash([5, 4]); g.strokeStyle = 'rgba(80,80,80,0.5)'; g.lineWidth = 1;
    for (const sk of layout.skips) {
      const a = layout.byKey.get(sk.from)!, b = layout.byKey.get(sk.to)!;
      const [x0, y0] = S(a.x, layout.top - 10), [x1, y1] = S(b.x, layout.top - 10); const [mx, my] = S((a.x + b.x) / 2, layout.top - 200);
      g.beginPath(); g.moveTo(x0, y0); g.quadraticCurveTo(mx, my, x1, y1); g.stroke();
      g.fillStyle = '#555'; g.font = '10px system-ui'; g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(sk.label, (x0 + x1) / 2, (y0 + my) / 2 - 2);
    }
    g.setLineDash([]);

    // Wires. Faint ones first, the focused neuron's fan on top.
    const hot: Wire[] = [];
    for (const wr of wires) {
      const st = wireStyle(wr, focus); if (!st) continue;
      if (st.hot) { hot.push(wr); continue; }
      const a = layout.byKey.get(wr.from)!, b = layout.byKey.get(wr.to)!;
      const [x0, y0] = centre(a, wr.i), [x1, y1] = centre(b, wr.j);
      g.strokeStyle = st.color; g.lineWidth = st.width; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    }
    for (const wr of hot) {
      const st = wireStyle(wr, focus)!; const a = layout.byKey.get(wr.from)!, b = layout.byKey.get(wr.to)!;
      const [x0, y0] = centre(a, wr.i), [x1, y1] = centre(b, wr.j);
      g.strokeStyle = st.color; g.lineWidth = st.width; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    }
    // Strip-to-strip band (logits -> probs).
    for (const band of layout.bands) if (band.kind === 'elementwise' && layout.byKey.get(band.to)!.compact) {
      const a = layout.byKey.get(band.from[0])!, b = layout.byKey.get(band.to)!;
      const [ax0, ay0] = S(a.x + COMPACT_HALF_W, a.y0), [, ay1] = S(0, a.y0 + a.n * a.dy); const [bx0, by0] = S(b.x - COMPACT_HALF_W, b.y0), [, by1] = S(0, b.y0 + b.n * b.dy);
      g.fillStyle = 'rgba(70,70,70,0.08)'; g.beginPath(); g.moveTo(ax0, ay0); g.lineTo(bx0, by0); g.lineTo(bx0, by1); g.lineTo(ax0, ay1); g.closePath(); g.fill();
    }

    // Neurons.
    for (const gr of layout.groups) {
      const t = trace.ctx.tensors.get(gr.key)!; const base = position * gr.n; const m = maxAbs(t.data); const alpha = alphaOf(gr.key);
      const [gx, gy0] = S(gr.x, gr.y0); const heightPx = gr.n * gr.dy * Z;
      if (status.get(gr.key) === 'active') {
        g.save(); g.shadowColor = 'rgba(255,176,0,0.9)'; g.shadowBlur = 18; g.fillStyle = 'rgba(255,176,0,0.12)';
        const gw = (gr.compact ? COMPACT_HALF_W * 2 : gr.r * 2) * Z + 14; g.fillRect(gx - gw / 2, gy0 - (gr.compact ? 0 : gr.r * Z) - 7, gw, heightPx + 14); g.restore();
      }
      if (gr.compact) {
        const hw = COMPACT_HALF_W * Z;
        for (let i = 0; i < gr.n; i++) { const v = t.data[base + i]; g.fillStyle = rgba(neuronColor(gr.key, v, m), alpha); g.fillRect(gx - hw, gy0 + i * gr.dy * Z, hw * 2, Math.max(1, gr.dy * Z)); }
        g.strokeStyle = `rgba(0,0,0,${0.5 * alpha})`; g.lineWidth = 1; g.strokeRect(gx - hw, gy0, hw * 2, heightPx);
        // Labels for the largest rows, spaced so they never overlap.
        if (words) {
          const order = [...Array(gr.n).keys()].sort((a, b) => t.data[base + b] - t.data[base + a]); const used: number[] = [];
          const left = gr.key === T_.logits; const side = left ? -1 : 1;
          g.font = '10px system-ui'; g.textAlign = left ? 'right' : 'left'; g.textBaseline = 'middle'; g.fillStyle = `rgba(30,30,30,${alpha})`;
          for (const i of order) { if (used.length >= 8) break; const y = gy0 + (i + 0.5) * gr.dy * Z; if (used.some(u => Math.abs(u - y) < 11)) continue; used.push(y); g.fillText(words[i] ?? String(i), gx + side * (hw + 5), y); g.beginPath(); g.moveTo(gx + side * hw, y); g.lineTo(gx + side * (hw + 3), y); g.strokeStyle = 'rgba(0,0,0,0.5)'; g.stroke(); }
        }
        for (const [ref, colour] of [[selection, '#2563eb'], [hover, '#60a5fa']] as const) {
          if (ref && ref.key === tKey(gr.key) && ref.index >= base && ref.index < base + gr.n) { const i = ref.index - base; g.strokeStyle = colour; g.lineWidth = 2; g.strokeRect(gx - hw - 2, gy0 + i * gr.dy * Z - 1, hw * 2 + 4, Math.max(2, gr.dy * Z) + 2); }
        }
      } else {
        const r = Math.max(1.5, gr.r * Z);
        for (let i = 0; i < gr.n; i++) {
          const v = t.data[base + i]; const [x, y] = centre(gr, i);
          const masked = gr.words && i > position;
          g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2);
          g.fillStyle = masked ? 'rgba(255,255,255,0.9)' : rgba(neuronColor(gr.key, v, m), alpha); g.fill();
          if (masked) g.setLineDash([2, 2]);
          g.strokeStyle = `rgba(60,60,60,${(masked ? 0.5 : 0.7) * alpha})`; g.lineWidth = 1; g.stroke(); g.setLineDash([]);
          const isFocus = focus && focus.key === gr.key && focus.i === i;
          if (isFocus) { g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.beginPath(); g.arc(x, y, r + 2.5, 0, Math.PI * 2); g.stroke(); }
          if (gr.words && words && Z * gr.dy >= 9) { g.fillStyle = masked ? DIM : INK; g.font = `${Math.min(11, Z * gr.dy * 0.6)}px system-ui`; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(words[trace.ctx.tokens[i]] ?? String(i), x - r - 5, y, 70); }
        }
      }
      // Caption above the group.
      const [cx, cy] = S(gr.x, gr.y0 - (gr.compact ? 0 : gr.r) - 6);
      g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillStyle = `rgba(30,30,30,${alpha})`; g.font = '600 11px system-ui'; g.fillText(gr.label, cx, cy - 12);
      g.fillStyle = `rgba(110,110,110,${alpha})`; g.font = '9px system-ui'; g.fillText(gr.sub, cx, cy - 1);
    }

    // Band labels above the columns.
    g.font = '10px system-ui'; g.textBaseline = 'middle';
    let row = 0;
    for (const band of layout.bands) {
      if (!band.label) continue;
      const a = layout.byKey.get(band.from[0])!, b = layout.byKey.get(band.to)!;
      const [x, y] = S((a.x + b.x) / 2, layout.top - 36 - (row++ % 2) * 24); const tw = g.measureText(band.label).width + 10;
      g.fillStyle = band.kind === 'matrix' ? '#fdf1dc' : band.kind === 'mix' || band.kind === 'attend' ? '#e3f1ef' : '#eeeeee';
      g.strokeStyle = band.kind === 'matrix' ? '#e0a94a' : band.kind === 'mix' || band.kind === 'attend' ? '#6aa89f' : '#bbbbbb';
      g.beginPath(); g.roundRect(x - tw / 2, y - 8, tw, 16, 3); g.fill(); g.stroke();
      g.fillStyle = INK; g.textAlign = 'center'; g.fillText(band.label, x, y);
    }
  }, [trace, layout, wires, focus, cursorIndex, selection, hover, position, cam, words, size]);

  const hitNeuron = useCallback((wx: number, wy: number): { gr: NeuronGroup; i: number } | null => {
    if (!layout) return null;
    for (const gr of layout.groups) {
      if (gr.compact) { if (Math.abs(wx - gr.x) > COMPACT_HALF_W + 2) continue; const i = Math.floor((wy - gr.y0) / gr.dy); if (i >= 0 && i < gr.n) return { gr, i }; continue; }
      if (Math.abs(wx - gr.x) > gr.r + 3) continue;
      const i = Math.round((wy - gr.y0) / gr.dy); if (i < 0 || i >= gr.n) continue;
      if (Math.hypot(wx - gr.x, wy - rowY(gr, i)) <= gr.r + 3) return { gr, i };
    }
    return null;
  }, [layout]);
  const hitWire = useCallback((wx: number, wy: number): Wire | null => {
    if (!layout || !focus) return null;
    const tol = 3 / cam.zoom;
    for (const wr of wires) {
      if (!wireStyle(wr, focus)?.hot) continue;
      const a = layout.byKey.get(wr.from)!, b = layout.byKey.get(wr.to)!;
      const x1 = a.x, y1 = a.compact ? rowY(a, wr.i) + a.dy / 2 : rowY(a, wr.i), x2 = b.x, y2 = b.compact ? rowY(b, wr.j) + b.dy / 2 : rowY(b, wr.j);
      const dx = x2 - x1, dy = y2 - y1; const len2 = dx * dx + dy * dy; const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((wx - x1) * dx + (wy - y1) * dy) / len2));
      if (Math.hypot(wx - (x1 + t * dx), wy - (y1 + t * dy)) <= tol) return wr;
    }
    return null;
  }, [layout, focus, wires, cam]);

  const local = (e: React.MouseEvent) => { const b = wrapRef.current!.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top] as const; };
  const onMove = (e: React.MouseEvent) => {
    const [sx, sy] = local(e);
    if (drag.current) { const dx = e.clientX - drag.current.x, dy = e.clientY - drag.current.y; if (Math.abs(dx) + Math.abs(dy) > 2) drag.current.moved = true; setCam(c => pan(c, dx, dy)); drag.current.x = e.clientX; drag.current.y = e.clientY; return; }
    if (!trace) return;
    const [wx, wy] = toWorld(cam, sx, sy);
    const n = hitNeuron(wx, wy);
    if (n) {
      const v = trace.ctx.tensors.get(n.gr.key)!.data[position * n.gr.n + n.i];
      const what = n.gr.key === T_.attn ? `attention on "${words?.[trace.ctx.tokens[n.i]] ?? n.i}" = ${fmt(v)}` : n.gr.compact && words ? `${n.gr.key} for "${words[n.i] ?? n.i}" = ${fmt(v)}` : `${n.gr.key}[${n.i}] = ${fmt(v)}`;
      setHover({ key: tKey(n.gr.key), index: position * n.gr.n + n.i }); setTip({ x: sx + 14, y: sy + 14, text: what }); return;
    }
    const wr = hitWire(wx, wy);
    if (wr) {
      const text = wr.band.kind === 'matrix' ? `${wr.band.param}[${wr.i}, ${wr.j}] = ${fmt(wr.w)}` : wr.band.kind === 'mix' ? `weight ${fmt(wr.w)} from "${words?.[trace.ctx.tokens[wr.i]] ?? wr.i}"` : wr.band.label;
      setTip({ x: sx + 14, y: sy + 14, text }); return;
    }
    if (hover) setHover(null); setTip(null);
  };
  const onDown = (e: React.MouseEvent) => { drag.current = { x: e.clientX, y: e.clientY, moved: false }; };
  const onUp = (e: React.MouseEvent) => {
    const moved = drag.current?.moved; drag.current = null; if (moved || !trace) return;
    const [sx, sy] = local(e); const [wx, wy] = toWorld(cam, sx, sy); const n = hitNeuron(wx, wy);
    select(n ? { key: tKey(n.gr.key), index: position * n.gr.n + n.i } : null);
  };
  const onWheel = (e: React.WheelEvent) => { const [sx, sy] = local(e); setCam(c => zoomAt(c, sx, sy, Math.pow(1.1, -e.deltaY / 100))); };
  const zoomBy = (f: number) => { const { w, h } = size(); setCam(c => zoomAt(c, w / 2, h / 2, f)); };

  const T = trace?.ctx.T ?? 0;
  return (
    <div className="network-view">
      {trace && (
        <div className="seq-strip">
          <span className="seq-label">Position</span>
          {Array.from({ length: T }, (_, t) => (
            <button key={t} type="button" className={'tok' + (t === position ? ' on' : '')} title={`Show the numbers at position ${t}`} onClick={() => setPosition(t)}>{words?.[trace.ctx.tokens[t]] ?? trace.ctx.tokens[t]}</button>
          ))}
        </div>
      )}
      <div ref={wrapRef} className="map-view" data-cam={`${cam.x.toFixed(1)},${cam.y.toFixed(1)},${cam.zoom.toFixed(4)}`} style={{ cursor: drag.current ? 'grabbing' : 'default' }}
        onMouseMove={onMove} onMouseDown={onDown} onMouseUp={onUp} onMouseLeave={() => { drag.current = null; setHover(null); setTip(null); }} onWheel={onWheel} onDoubleClick={fitAll}>
        <canvas ref={canvasRef} />
        <div className="map-zoom">
          <button onClick={() => zoomBy(1.3)} title="Zoom in">+</button>
          <button onClick={() => zoomBy(1 / 1.3)} title="Zoom out">-</button>
          <button onClick={fitAll} title="Fit everything in view">Fit</button>
        </div>
        {tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
      </div>
    </div>
  );
}
