/** Neurons: one position's numbers as columns of circles, every neighbouring pair of columns
 *  joined by the step that turns one into the next. Steps that use a weight grid are drawn as a
 *  soft band with the grid's name; hover a circle to see its own wires and their weights.
 *  Pan by dragging, zoom with the wheel or the buttons, double-click to fit. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../app/store';
import { tKey, type Trace } from '../../engine/trace/trace';
import { Cursor } from '../../engine/trace/cursor';
import { T_ } from '../../engine/model/gpt';
import { diverging, sequential, maxAbs, isMasked, type RGB } from '../colormap';
import { toScreen, toWorld, zoomAt, pan, fit, type Camera } from '../camera';
import { drawLegend, drawBanner } from '../legend';
import { networkLayout, rowY, COMPACT_HALF_W, type NetLayout, type NeuronGroup, type Band } from './networkLayout';
import { displayWord, contextWords } from '../../ui/words';

const INK = '#222', DIM = '#8a8a8a';
const RED: RGB = [214, 40, 40];
const rgba = ([r, g, b]: RGB, a: number) => `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`;
const fmt = (v: number): string => (isMasked(v) ? '-inf' : String(Number(v.toPrecision(4))));

interface Wire { band: Band; from: string; i: number; to: string; j: number; w: number; maxW: number; hot: boolean }
interface Tip { x: number; y: number; text: string }
interface Focus { key: string; i: number }

/** The wires worth drawing: every one-to-one link, plus the fan of the focused neuron. */
function buildWires(trace: Trace, L: NetLayout, position: number, focus: Focus | null): Wire[] {
  const out: Wire[] = [];
  const T = trace.ctx.T;
  const touches = (from: string, i: number, to: string, j: number) => focus !== null && ((focus.key === from && focus.i === i) || (focus.key === to && focus.i === j));
  for (const band of L.bands) {
    const to = L.byKey.get(band.to)!;
    if (band.kind === 'elementwise') {
      if (to.compact) continue;
      for (const f of band.from) { const from = L.byKey.get(f)!; for (let i = 0; i < Math.min(from.n, to.n); i++) out.push({ band, from: from.key, i, to: to.key, j: i, w: 0, maxW: 1, hot: touches(from.key, i, to.key, i) }); }
    } else if (!focus) continue;
    else if (band.kind === 'matrix') {
      const from = L.byKey.get(band.from[0])!; const W = trace.ctx.tensors.get(band.param!)!; const maxW = maxAbs(W.data);
      if (focus.key === from.key) for (let j = 0; j < to.n; j++) out.push({ band, from: from.key, i: focus.i, to: to.key, j, w: W.data[focus.i * to.n + j], maxW, hot: true });
      else if (focus.key === to.key) for (let i = 0; i < from.n; i++) out.push({ band, from: from.key, i, to: to.key, j: focus.i, w: W.data[i * to.n + focus.i], maxW, hot: true });
    } else if (band.kind === 'attend') {
      if (focus.key !== to.key) continue;
      for (const f of band.from) { const from = L.byKey.get(f)!; for (let i = 0; i < from.n; i++) out.push({ band, from: from.key, i, to: to.key, j: focus.i, w: 0, maxW: 1, hot: true }); }
    } else if (band.kind === 'mix') {
      const attn = L.byKey.get(T_.attn)!; const A = trace.ctx.tensors.get(T_.attn)!;
      if (focus.key === to.key) for (let j = 0; j <= position && j < attn.n; j++) out.push({ band, from: attn.key, i: j, to: to.key, j: focus.i, w: A.data[position * T + j], maxW: 1, hot: true });
      else if (focus.key === attn.key && focus.i <= position) for (let i = 0; i < to.n; i++) out.push({ band, from: attn.key, i: focus.i, to: to.key, j: i, w: A.data[position * T + focus.i], maxW: 1, hot: true });
    }
  }
  return out;
}

function wireStyle(w: Wire, dimmed: boolean): { color: string; width: number } {
  const rel = w.maxW === 0 ? 0 : Math.abs(w.w) / w.maxW;
  switch (w.band.kind) {
    case 'matrix': return { color: rgba(diverging(w.w, w.maxW), 0.25 + 0.7 * rel), width: 0.8 + 2.6 * rel };
    case 'elementwise': return { color: `rgba(70, 70, 70, ${w.hot ? 0.9 : dimmed ? 0.08 : 0.3})`, width: w.hot ? 2 : 1 };
    case 'attend': return { color: 'rgba(60, 110, 100, 0.55)', width: 1 };
    case 'mix': return { color: rgba(sequential(0.35 + 0.65 * w.w), 0.3 + 0.7 * w.w), width: 0.8 + 4 * w.w };
  }
}

function neuronColor(key: string, v: number, m: number): RGB {
  if (key === T_.probs) return [255 - (255 - RED[0]) * Math.min(1, v), 255 - (255 - RED[1]) * Math.min(1, v), 255 - (255 - RED[2]) * Math.min(1, v)];
  if (key === T_.attn) return sequential(v);
  return diverging(v, m);
}

const BAND_FILL: Record<Band['kind'], string> = { matrix: 'rgba(224,138,30,0.10)', attend: 'rgba(1,102,94,0.08)', mix: 'rgba(1,102,94,0.10)', elementwise: 'rgba(0,0,0,0.04)' };
const PILL: Record<Band['kind'], [string, string]> = { matrix: ['#fdf1dc', '#e0a94a'], attend: ['#e3f1ef', '#6aa89f'], mix: ['#e3f1ef', '#6aa89f'], elementwise: ['#f1f1f1', '#bbbbbb'] };

export function NetworkView() {
  const trace = useStore(s => s.trace);
  const cursorIndex = useStore(s => s.cursorIndex);
  const selection = useStore(s => s.selection);
  const hover = useStore(s => s.hover);
  const position = useStore(s => s.position);
  const corpus = useStore(s => s.corpus);
  const prompt = useStore(s => s.prompt);
  const select = useStore(s => s.select);
  const setHover = useStore(s => s.setHover);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cam, setCam] = useState<Camera>({ x: -100, y: -400, zoom: 0.6 });
  const [tip, setTip] = useState<Tip | null>(null);
  const drag = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  const fittedFor = useRef<string>('');
  const words = corpus?.vocab.words;
  const ctxWords = useMemo(() => (trace && corpus ? contextWords(trace, prompt, corpus.vocab).words : []), [trace, prompt, corpus]);

  const layout = useMemo(() => {
    if (!trace) return null;
    const shapes: Record<string, number[]> = {};
    for (const [k, t] of trace.ctx.tensors) shapes[k] = t.shape;
    return networkLayout(shapes);
  }, [trace]);

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
  const wires = useMemo(() => (trace && layout ? buildWires(trace, layout, position, focus) : []), [trace, layout, position, focus]);

  const size = useCallback(() => { const el = wrapRef.current; return { w: el?.clientWidth ?? 800, h: el?.clientHeight ?? 600 }; }, []);
  const fitArea = useCallback(() => { const { w, h } = size(); return { w, h: Math.max(200, h - 70) }; }, [size]);
  const fitAll = useCallback(() => { if (layout) setCam(fit(layout.bounds, fitArea(), 20)); }, [layout, fitArea]);
  useEffect(() => {
    if (!layout || !trace) return;
    const key = `${trace.ctx.T}`;
    if (fittedFor.current !== key) { fittedFor.current = key; setCam(fit(layout.bounds, fitArea(), 20)); }
  }, [layout, trace, fitArea]);
  useEffect(() => {
    const el = wrapRef.current; if (!el) return;
    const ro = new ResizeObserver(() => setCam(c => ({ ...c }))); ro.observe(el); return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    const { w, h } = size(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const g = canvas.getContext('2d')!; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#fbfbfa'; g.fillRect(0, 0, w, h);
    if (!trace || !layout) { g.fillStyle = DIM; g.font = '14px system-ui'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('The neurons for one word appear here after you press Step.', w / 2, h / 2); return; }
    const Z = cam.zoom; const S = (wx: number, wy: number) => toScreen(cam, wx, wy);
    const cursor = new Cursor(trace); cursor.index = cursorIndex;
    const step = cursor.current(); const phase = (step?.phase ?? 'forward') as 'forward' | 'backward' | 'update';
    const status = new Map(layout.groups.map(gr => [gr.key, cursor.status(tKey(gr.key))]));
    const alphaOf = (key: string) => (status.get(key) === 'pending' ? 0.25 : 1);
    const centre = (gr: NeuronGroup, i: number): [number, number] => S(gr.x, gr.compact ? rowY(gr, i) + gr.dy / 2 : rowY(gr, i));
    const extent = (gr: NeuronGroup) => ({ top: gr.y0 - (gr.compact ? 0 : gr.r), bottom: gr.y0 + (gr.compact ? gr.n * gr.dy : (gr.n - 1) * gr.dy + gr.r), half: gr.compact ? COMPACT_HALF_W : gr.r });
    const backing = (x: number, y: number, tw: number, th: number) => { g.fillStyle = 'rgba(251,251,250,0.92)'; g.beginPath(); g.roundRect(x, y, tw, th, 3); g.fill(); };

    // Bands: a soft shape from the source column(s) to the target column, for every step that is
    // not a one-to-one link. The hovered neuron's wires are drawn on top later.
    for (const band of layout.bands) {
      const to = layout.byKey.get(band.to)!;
      if (band.kind === 'elementwise' && !to.compact) continue;
      const froms = band.from.map(f => layout.byKey.get(f)!);
      const fTop = Math.min(...froms.map(f => extent(f).top)), fBot = Math.max(...froms.map(f => extent(f).bottom));
      const fx = Math.max(...froms.map(f => f.x + extent(f).half)) + 3, tx = to.x - extent(to).half - 3; const te = extent(to);
      const [x0, y0] = S(fx, fTop), [, y1] = S(fx, fBot), [x1, y2] = S(tx, te.top), [, y3] = S(tx, te.bottom);
      g.fillStyle = BAND_FILL[band.kind]; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y2); g.lineTo(x1, y3); g.lineTo(x0, y1); g.closePath(); g.fill();
    }

    // Skip arcs: values carried forward and added back in.
    g.setLineDash([5, 4]); g.strokeStyle = 'rgba(80,80,80,0.55)'; g.lineWidth = 1;
    // Boxes of every label drawn so far; later captions move to stay clear of them.
    const placed: { x: number; y: number; w: number; h: number }[] = [];
    for (const sk of layout.skips) {
      const a = layout.byKey.get(sk.from)!, b = layout.byKey.get(sk.to)!;
      const [x0, y0] = S(a.x, layout.top - 36), [x1, y1] = S(b.x, layout.top - 36); const [mx, my] = S((a.x + b.x) / 2, layout.top - 150);
      g.beginPath(); g.moveTo(x0, y0); g.quadraticCurveTo(mx, my, x1, y1); g.stroke();
      g.font = '10px system-ui'; const tw = g.measureText(sk.label).width + 8; const ly = (y0 + my) / 2 - 2;
      backing((x0 + x1) / 2 - tw / 2, ly - 12, tw, 13); placed.push({ x: (x0 + x1) / 2 - tw / 2, y: ly - 12, w: tw, h: 13 }); g.fillStyle = '#555'; g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillText(sk.label, (x0 + x1) / 2, ly);
    }
    g.setLineDash([]);

    // Wires: one-to-one links always; the focused neuron's fan on top.
    const dimmed = focus !== null;
    const hot: Wire[] = [];
    for (const wr of wires) {
      if (wr.hot) { hot.push(wr); continue; }
      const st = wireStyle(wr, dimmed); const a = layout.byKey.get(wr.from)!, b = layout.byKey.get(wr.to)!;
      const [x0, y0] = centre(a, wr.i), [x1, y1] = centre(b, wr.j);
      g.strokeStyle = st.color; g.lineWidth = st.width; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    }
    for (const wr of hot) {
      const st = wireStyle(wr, dimmed); const a = layout.byKey.get(wr.from)!, b = layout.byKey.get(wr.to)!;
      const [x0, y0] = centre(a, wr.i), [x1, y1] = centre(b, wr.j);
      g.strokeStyle = st.color; g.lineWidth = st.width; g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
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
        const nextX = Math.min(...layout.groups.filter(o => o.x > gr.x).map(o => o.x), Infinity);
        const room = Math.min(400, (nextX - gr.x) * Z - hw * 2 - 10);   // screen px before the next column starts
        if (words && room >= 44) {
          const order = [...Array(gr.n).keys()].sort((a, b) => t.data[base + b] - t.data[base + a]); const used: number[] = [];
          g.font = '10px system-ui'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillStyle = `rgba(30,30,30,${alpha})`;
          for (const i of order) { if (used.length >= 8) break; const y = gy0 + (i + 0.5) * gr.dy * Z; if (used.some(u => Math.abs(u - y) < 11)) continue; used.push(y); g.fillText(displayWord(words[i]), gx + hw + 5, y, room); g.beginPath(); g.moveTo(gx + hw, y); g.lineTo(gx + hw + 3, y); g.strokeStyle = 'rgba(0,0,0,0.5)'; g.stroke(); }
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
          if (focus && focus.key === gr.key && focus.i === i) { g.strokeStyle = '#2563eb'; g.lineWidth = 2; g.beginPath(); g.arc(x, y, r + 2.5, 0, Math.PI * 2); g.stroke(); }
          if (gr.words && Z * gr.dy >= 9) { g.fillStyle = masked ? DIM : INK; g.font = `${Math.max(8, Math.min(11, Z * gr.dy * 0.75))}px system-ui`; g.textAlign = 'right'; g.textBaseline = 'middle'; g.fillText(ctxWords[i] ?? displayWord(words?.[trace.ctx.tokens[i]]), x - r - 5, y, 70); }
        }
      }
    }

    // Band labels, one per band, in the gap between its columns at the target's height. Their
    // boxes are kept so the captions drawn next can stay clear of them.
    g.font = '10px system-ui'; g.textBaseline = 'middle';
    for (const band of layout.bands) {
      if (!band.label) continue;
      const froms = band.from.map(f => layout.byKey.get(f)!); const to = layout.byKey.get(band.to)!; const te = extent(to);
      const fx = Math.max(...froms.map(f => f.x));
      // Around the short attention column the two labels stack above and below it; elsewhere a
      // label sits in the gap at its target's middle, and softmax goes above the two strips.
      let wx = (fx + to.x) / 2, wy = (te.top + te.bottom) / 2;
      if (band.kind === 'attend') { wx = to.x; wy = te.top - 46; }
      else if (band.kind === 'mix') { const a = layout.byKey.get(T_.attn)!; const ae = extent(a); wx = a.x; wy = ae.bottom + 22; }
      else if (band.kind === 'elementwise') wy = te.bottom + (to.compact ? 22 : 16);
      const [x, y] = S(wx, wy); const tw = g.measureText(band.label).width + 10;
      const [fill, stroke] = PILL[band.kind];
      g.fillStyle = fill; g.strokeStyle = stroke; g.lineWidth = 1;
      g.beginPath(); g.roundRect(x - tw / 2, y - 8, tw, 16, 3); g.fill(); g.stroke();
      g.fillStyle = INK; g.textAlign = 'center'; g.fillText(band.label, x, y);
      placed.push({ x: x - tw / 2, y: y - 8, w: tw, h: 16 });
    }

    // Captions above each column: the plain name, with the symbol used in Explain under it. A
    // caption that would touch a pill or an earlier caption moves up a row, so at any zoom no two
    // pieces of text overlap; at low zoom the columns alternate between two rows.
    const MAIN_FONT = '600 12px system-ui', SYM_FONT = '10px ui-monospace, Menlo, monospace';
    const hits = (r: { x: number; y: number; w: number; h: number }) => placed.some(q => r.x < q.x + q.w && r.x + r.w > q.x && r.y < q.y + q.h && r.y + r.h > q.y);
    for (const gr of layout.groups) {
      const alpha = alphaOf(gr.key);
      const main = gr.sub || gr.label, sym = gr.sub ? gr.label : '';
      g.font = MAIN_FONT; const lw = g.measureText(main).width; g.font = SYM_FONT; const sw = g.measureText(sym).width;
      const bw = Math.max(lw, sw) + 8;
      const [cx, cy] = S(gr.x, gr.y0 - (gr.compact ? 0 : gr.r) - 8);
      let ty = cy; let rect = { x: cx - bw / 2, y: ty - 28, w: bw, h: 28 };
      for (let k = 0; k < 3 && hits(rect); k++) { ty -= 30; rect = { ...rect, y: ty - 28 }; }
      placed.push(rect); backing(rect.x, rect.y, rect.w, rect.h);
      g.textAlign = 'center'; g.textBaseline = 'bottom'; g.fillStyle = `rgba(30,30,30,${alpha})`; g.font = MAIN_FONT; g.fillText(main, cx, ty - 13);
      g.fillStyle = `rgba(110,110,110,${alpha})`; g.font = SYM_FONT; g.fillText(sym, cx, ty - 1);
    }

    // Fixed overlays.
    const posWord = ctxWords[position] ?? displayWord(words?.[trace.ctx.tokens[position]]);
    drawBanner(g, 8, h - 30, `Numbers for position ${position + 1} of ${trace.ctx.T}: "${posWord}". Hover a circle to see its wires.`, '#444');
    if (phase !== 'forward') drawBanner(g, 8, h - 58, 'This view always shows the values; the Network tab shows the gradients');
    drawLegend(g, w - 8, h - 8, 'forward');
  }, [trace, layout, wires, focus, cursorIndex, selection, hover, position, cam, words, ctxWords, size]);

  const hitNeuron = useCallback((wx: number, wy: number): { gr: NeuronGroup; i: number } | null => {
    if (!layout) return null;
    for (const gr of layout.groups) {
      if (gr.compact) { if (Math.abs(wx - gr.x) > COMPACT_HALF_W + 2) continue; const i = Math.floor((wy - gr.y0) / gr.dy); if (i >= 0 && i < gr.n) return { gr, i }; continue; }
      // Snap to a circle from a little way off, so hovering near a column shows its wires.
      if (Math.abs(wx - gr.x) > gr.r + 9) continue;
      const i = Math.round((wy - gr.y0) / gr.dy); if (i < 0 || i >= gr.n) continue;
      if (Math.hypot(wx - gr.x, wy - rowY(gr, i)) <= gr.r + 9) return { gr, i };
    }
    return null;
  }, [layout]);
  const hitWire = useCallback((wx: number, wy: number): Wire | null => {
    if (!layout || !focus) return null;
    const tol = 3 / cam.zoom;
    for (const wr of wires) {
      if (!wr.hot) continue;
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
      const what = n.gr.key === T_.attn ? `attention on "${ctxWords[n.i] ?? '?'}" = ${fmt(v)}` : n.gr.compact && words ? `${n.gr.key} for "${displayWord(words[n.i])}" = ${fmt(v)}` : `${n.gr.sub ? n.gr.sub + ' ' : ''}${n.gr.key}[${n.i}] = ${fmt(v)}`;
      setHover({ key: tKey(n.gr.key), index: position * n.gr.n + n.i }); setTip({ x: sx + 14, y: sy + 14, text: what }); return;
    }
    const wr = hitWire(wx, wy);
    if (wr) {
      const text = wr.band.kind === 'matrix' ? `${wr.band.param}[${wr.i}, ${wr.j}] = ${fmt(wr.w)}` : wr.band.kind === 'mix' ? `weight ${fmt(wr.w)} from "${ctxWords[wr.i] ?? '?'}"` : wr.band.label;
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

  return (
    <div ref={wrapRef} className="map-view" data-cam={`${cam.x.toFixed(1)},${cam.y.toFixed(1)},${cam.zoom.toFixed(4)}`} style={{ cursor: drag.current ? 'grabbing' : hover ? 'pointer' : 'default' }}
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
