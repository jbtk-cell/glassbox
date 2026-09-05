import type { Trace } from '../engine/trace/trace';
import type { Op } from '../engine/ops/types';
import { get } from '../engine/ops/types';

export interface DotTerm { i: number; xi: number; wi: number; prod: number }
export interface DotBreakdown { terms: DotTerm[]; bias: number; total: number; inputName: string; weightName: string; row: number; col: number }

/** For a linear op y = x W + b, explain one output cell y[row, col] as its dot product. */
export function dotProductBreakdown(trace: Trace, op: Op, row: number, col: number): DotBreakdown | null {
  if (op.kind !== 'linear' || op.inputs.length !== 1 || op.params.length !== 2) return null;
  const X = get(trace.ctx, op.inputs[0]), W = get(trace.ctx, op.params[0]), B = get(trace.ctx, op.params[1]);
  const In = X.shape[1], Out = W.shape[1];
  if (row < 0 || row >= X.shape[0] || col < 0 || col >= Out) return null;
  const terms: DotTerm[] = [];
  let total = B.data[col];
  for (let i = 0; i < In; i++) {
    const xi = X.data[row * In + i], wi = W.data[i * Out + col];
    terms.push({ i, xi, wi, prod: xi * wi }); total += xi * wi;
  }
  return { terms, bias: B.data[col], total, inputName: op.inputs[0], weightName: op.params[0], row, col };
}

/** Row/column of a flat index for a tensor shape (1-D shapes are one row). */
export function rowCol(shape: number[], index: number): { row: number; col: number; cols: number } {
  const cols = shape.length === 1 ? shape[0] : shape[1];
  return { row: Math.floor(index / cols), col: index % cols, cols };
}

export function fmtNum(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v <= -1e8) return '-inf';
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1000 || a < 0.001 ? v.toExponential(3) : v.toPrecision(4);
}
