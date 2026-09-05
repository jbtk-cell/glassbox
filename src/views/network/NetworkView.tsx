/** Every neuron and connection for one selected sequence position. Pure presentation over the
 *  store; layout maths live in networkLayout.ts. */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '../../app/store';
import { tKey } from '../../engine/trace/trace';
import { Cursor } from '../../engine/trace/cursor';
import { get } from '../../engine/ops/types';
import { T_ } from '../../engine/model/gpt';
import { diverging, sequential, maxAbs, type RGB } from '../colormap';
import { networkLayout, COMPACT_THRESHOLD, type NetColumn, type NetWire } from './networkLayout';

const MAX_WIRES = 2000;
const COMPACT_HALF_W = 12;
const TOP_LABELS = 8;

interface WireGeom { param: string; from: string; to: string; i: number; j: number; x1: number; y1: number; x2: number; y2: number; w: number; maxW: number }

function rgba([r, g, b]: RGB, a: number): string { return `rgba(${r | 0}, ${g | 0}, ${b | 0}, ${a})`; }

const fmt = (v: number): string => (v <= -1e8 ? '-inf' : String(Number(v.toPrecision(4))));

function buildWireGeoms(tensors: Map<string, { data: Float64Array }>, columns: NetColumn[], wires: NetWire[]): WireGeom[] {
  const byKey = new Map(columns.map(c => [c.key, c]));
  const all: WireGeom[] = [];
  for (const wire of wires) {
    const from = byKey.get(wire.from), to = byKey.get(wire.to);
    const W = tensors.get(wire.param);
    if (!from || !to || !W) continue;
    const maxW = maxAbs(W.data as Float64Array);
    for (let i = 0; i < from.n; i++) {
      for (let j = 0; j < to.n; j++) {
        const w = W.data[i * to.n + j];
        all.push({
          param: wire.param, from: wire.from, to: wire.to, i, j,
          x1: from.x, y1: from.y0 + i * from.dy, x2: to.x, y2: to.y0 + j * to.dy, w, maxW,
        });
      }
    }
  }
  if (all.length <= MAX_WIRES) return all;
  return [...all].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, MAX_WIRES);
}

function hitNeuron(columns: NetColumn[], mx: number, my: number): { col: NetColumn; i: number } | null {
  for (const col of columns) {
    const compact = col.n > COMPACT_THRESHOLD;
    const halfW = compact ? COMPACT_HALF_W : col.r + 2;
    if (Math.abs(mx - col.x) > halfW) continue;
    const i = Math.round((my - col.y0) / (col.dy || 1));
    if (i < 0 || i >= col.n) continue;
    if (compact) return { col, i };
    const cy = col.y0 + i * col.dy;
    if (Math.hypot(mx - col.x, my - cy) <= col.r + 2) return { col, i };
  }
  return null;
}

function distToSegment(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / len2));
  const cx = x1 + t * dx, cy = y1 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

function hitWire(wires: WireGeom[], mx: number, my: number): WireGeom | null {
  for (const w of wires) {
    if (distToSegment(mx, my, w.x1, w.y1, w.x2, w.y2) <= 3) return w;
  }
  return null;
}

interface Tip { x: number; y: number; text: string }

export function NetworkView() {
  const trace = useStore(s => s.trace);
  const cursorIndex = useStore(s => s.cursorIndex);
  const selection = useStore(s => s.selection);
  const position = useStore(s => s.position);
  const select = useStore(s => s.select);
  const setHover = useStore(s => s.setHover);
  const setPosition = useStore(s => s.setPosition);

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [tip, setTip] = useState<Tip | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const box = entries[0].contentRect;
      setSize({ w: box.width, h: box.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const shapes = useMemo(() => {
    if (!trace) return null;
    const out: Record<string, number[]> = {};
    for (const [k, t] of trace.ctx.tensors) out[k] = t.shape;
    return out;
  }, [trace]);

  const layout = useMemo(() => {
    if (!shapes || size.w === 0 || size.h === 0) return null;
    return networkLayout(shapes, size);
  }, [shapes, size]);

  const wireGeoms = useMemo(() => {
    if (!trace || !layout) return [];
    return buildWireGeoms(trace.ctx.tensors, layout.columns, layout.wires);
  }, [trace, layout]);

  const T = trace?.ctx.T ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trace || !layout) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    canvas.style.width = size.w + 'px';
    canvas.style.height = size.h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const cursor = new Cursor(trace);
    cursor.index = cursorIndex;

    // Resolve the selected neuron (if it is one of our columns, at this position).
    let selCol: string | null = null, selIdx: number | null = null;
    if (selection) {
      const name = selection.key.startsWith('t:') ? selection.key.slice(2) : selection.key;
      const col = layout.columns.find(c => c.key === name);
      if (col) {
        const idx = selection.index - position * col.n;
        if (idx >= 0 && idx < col.n) { selCol = name; selIdx = idx; }
      }
    }

    // Wires.
    for (const w of wireGeoms) {
      const isFan = selCol !== null && ((w.from === selCol && w.i === selIdx) || (w.to === selCol && w.j === selIdx));
      const alpha = selCol === null ? 0.6 : (isFan ? 0.6 : 0.1);
      const width = 0.5 + 3 * (w.maxW === 0 ? 0 : Math.abs(w.w) / w.maxW);
      ctx.strokeStyle = rgba(diverging(w.w, w.maxW), alpha);
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(w.x1, w.y1);
      ctx.lineTo(w.x2, w.y2);
      ctx.stroke();
    }

    // Columns.
    const words = useStore.getState().corpus?.vocab.words;
    for (const col of layout.columns) {
      const name = col.key;
      const tensor = get(trace.ctx, name);
      const rowN = col.n;
      const colMaxAbs = maxAbs(tensor.data);
      const status = cursor.status(tKey(name));
      const alpha = status === 'pending' ? 0.3 : 1;
      const compact = rowN > COMPACT_THRESHOLD;
      const colHeight = rowN * col.dy;

      if (status === 'active') {
        ctx.save();
        ctx.shadowColor = 'rgba(255, 176, 0, 0.9)';
        ctx.shadowBlur = 18;
        ctx.fillStyle = 'rgba(255, 176, 0, 0.15)';
        const glowW = compact ? COMPACT_HALF_W * 2 + 16 : col.r * 2 + 16;
        ctx.fillRect(col.x - glowW / 2, col.y0 - 10, glowW, colHeight + 20);
        ctx.restore();
      }

      ctx.font = '10px sans-serif';
      ctx.fillStyle = `rgba(60, 60, 60, ${alpha})`;
      ctx.textAlign = 'center';
      ctx.fillText(name, col.x, col.y0 - 12);

      if (compact) {
        const order = [...Array(rowN).keys()].sort((a, b) => tensor.data[position * rowN + b] - tensor.data[position * rowN + a]);
        const top = new Set(order.slice(0, TOP_LABELS));
        for (let i = 0; i < rowN; i++) {
          const v = tensor.data[position * rowN + i];
          const rgb = name === T_.probs ? sequential(v) : diverging(v, colMaxAbs);
          ctx.fillStyle = rgba(rgb, alpha);
          ctx.fillRect(col.x - COMPACT_HALF_W, col.y0 + i * col.dy, COMPACT_HALF_W * 2, Math.max(1, col.dy));
          if (top.has(i) && words) {
            ctx.save();
            ctx.font = '9px sans-serif';
            ctx.fillStyle = `rgba(30, 30, 30, ${alpha})`;
            ctx.textAlign = 'left';
            ctx.fillText(words[i] ?? String(i), col.x + COMPACT_HALF_W + 4, col.y0 + i * col.dy + col.dy / 2 + 3);
            ctx.restore();
          }
        }
      } else {
        for (let i = 0; i < rowN; i++) {
          const v = tensor.data[position * rowN + i];
          const rgb = name === T_.probs ? sequential(v) : diverging(v, colMaxAbs);
          const cy = col.y0 + i * col.dy;
          const highlighted = selCol === name && selIdx === i;
          ctx.beginPath();
          ctx.fillStyle = rgba(rgb, alpha);
          ctx.arc(col.x, cy, col.r, 0, Math.PI * 2);
          ctx.fill();
          if (highlighted) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgba(20, 110, 220, 0.9)';
            ctx.stroke();
          }
        }
      }
    }

    // Attention strip: arcs from the current position to each earlier position.
    if (T > 0 && trace.ctx.tensors.has(T_.attn)) {
      const attn = get(trace.ctx, T_.attn);
      const stripY = size.h - 24;
      const xStep = (size.w - 48) / T;
      const boxX = (t: number) => 24 + xStep * (t + 0.5);
      for (let t = 0; t < T; t++) {
        ctx.beginPath();
        ctx.fillStyle = t === position ? 'rgba(20, 110, 220, 0.9)' : 'rgba(120, 120, 120, 0.6)';
        ctx.arc(boxX(t), stripY, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      for (let j = 0; j < position; j++) {
        const w = attn.data[position * T + j];
        if (w <= 0) continue;
        const x1 = boxX(position), x2 = boxX(j);
        const mid = (x1 + x2) / 2;
        const control = stripY - 16 - w * 40;
        ctx.beginPath();
        ctx.moveTo(x1, stripY);
        ctx.quadraticCurveTo(mid, control, x2, stripY);
        ctx.lineWidth = 1 + 6 * w;
        ctx.strokeStyle = rgba(sequential(w), 0.7);
        ctx.stroke();
      }
    }
  }, [trace, layout, wireGeoms, cursorIndex, selection, position, size, T]);

  if (!trace) {
    return <div className="network-view network-view--empty">Predict a word or record a training step to see the network.</div>;
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const neuron = hitNeuron(layout.columns, mx, my);
    if (neuron) {
      const { col, i } = neuron;
      const tensor = get(trace.ctx, col.key);
      const value = tensor.data[position * col.n + i];
      setTip({ x: e.clientX, y: e.clientY, text: `${col.key}[${position}, ${i}] = ${fmt(value)}` });
      setHover({ key: tKey(col.key), index: position * col.n + i });
      return;
    }
    const wire = hitWire(wireGeoms, mx, my);
    if (wire) {
      setTip({ x: e.clientX, y: e.clientY, text: `${wire.param}[${wire.i}, ${wire.j}] = ${fmt(wire.w)}` });
      setHover(null);
      return;
    }
    setTip(null);
    setHover(null);
  };

  const onMouseLeave = () => { setTip(null); setHover(null); };

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !layout) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const neuron = hitNeuron(layout.columns, mx, my);
    if (neuron) { select({ key: tKey(neuron.col.key), index: position * neuron.col.n + neuron.i }); return; }
    select(null);
  };

  return (
    <div ref={wrapRef} className="network-view">
      <div className="network-view__picker">
        <button type="button" onClick={() => setPosition(position - 1)} disabled={position <= 0}>{'<'}</button>
        <span>{position} / {T}</span>
        <button type="button" onClick={() => setPosition(position + 1)} disabled={position >= T - 1}>{'>'}</button>
      </div>
      <canvas ref={canvasRef} onMouseMove={onMouseMove} onMouseLeave={onMouseLeave} onClick={onClick} />
      {tip && <div className="network-view__tooltip" style={{ left: tip.x + 12, top: tip.y + 12 }}>{tip.text}</div>}
    </div>
  );
}
