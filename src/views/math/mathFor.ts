import { Trace, TraceStep, tKey } from '../../engine/trace/trace';
import { Op, OpKind, Ctx, get } from '../../engine/ops/types';
import type { CellRef } from '../../app/store';

export interface MathBlock { title: string; latex: string; note?: string }

const HINT = 'Select a cell in the Network tab to see the numbers.';

/** Format a number to 4 significant figures; masked/very negative values render as -infinity. */
function fmt(v: number): string {
  return v <= -1e8 ? '-\\infty' : v.toPrecision(4);
}

/** Row-major (row, col) for a flat index into a tensor of the given shape. 1-D tensors have row 0. */
function rowCol(shape: number[], index: number): [number, number] {
  if (shape.length < 2) return [0, index];
  const cols = shape[shape.length - 1];
  return [Math.floor(index / cols), index % cols];
}

/** Up to 6 terms joined with '+', then an ellipsis if there were more. */
function terms(vals: string[]): string {
  return vals.length > 6 ? [...vals.slice(0, 6), '\\cdots'].join(' + ') : vals.join(' + ');
}

function generalFormula(kind: OpKind): string {
  switch (kind) {
    case 'embed': return '\\text{tok}_{t,i} = E_{\\text{id}_t,\\,i}';
    case 'pos_embed': return '\\text{pos}_{t,i} = P_{t,i}';
    case 'add': return 'y_{t,i} = a_{t,i} + b_{t,i}';
    case 'layernorm': return '\\begin{aligned} \\mu_t &= \\frac{1}{D}\\sum_i x_{t,i} \\\\ \\sigma_t^2 &= \\frac{1}{D}\\sum_i (x_{t,i}-\\mu_t)^2 \\\\ \\hat{x}_{t,i} &= \\frac{x_{t,i}-\\mu_t}{\\sqrt{\\sigma_t^2+\\epsilon}} \\\\ y_{t,i} &= \\gamma_i \\hat{x}_{t,i} + \\beta_i \\end{aligned}';
    case 'linear': return 'y_{t,o} = b_o + \\sum_i x_{t,i}\\,W_{i,o}';
    case 'scores': return 'S_{i,j} = \\frac{1}{\\sqrt{d}}\\sum_c q_{i,c}\\,k_{j,c}';
    case 'mask': return 'S\'_{i,j} = \\begin{cases} -\\infty & j > i \\\\ S_{i,j} & j \\le i \\end{cases}';
    case 'softmax_rows': case 'softmax_out':
      return 'A_{i,j} = \\frac{\\exp(S_{i,j}-\\max_c S_{i,c})}{\\sum_c \\exp(S_{i,c}-\\max_c S_{i,c})}';
    case 'attn_apply': return 'y_{i,c} = \\sum_j A_{i,j}\\,v_{j,c}';
    case 'relu': return 'y_i = \\max(0, x_i)';
    case 'loss': return 'L = -\\frac{1}{T}\\sum_t \\log p_{t,\\text{target}_t}';
    default: { const check: never = kind; return check; }
  }
}

/** Block (b): the general formula with the selected output cell's real numbers plugged in. */
function substituted(ctx: Ctx, op: Op, index: number): string {
  const Y = get(ctx, op.output);
  const [row, col] = rowCol(Y.shape, index);
  switch (op.kind) {
    case 'embed': {
      const id = ctx.tokens[row];
      return `\\text{tok}_{${row},${col}} = E_{${id},${col}} = ${fmt(Y.data[index])}`;
    }
    case 'pos_embed':
      return `\\text{pos}_{${row},${col}} = P_{${row},${col}} = ${fmt(Y.data[index])}`;
    case 'add': {
      const A = get(ctx, op.inputs[0]), B = get(ctx, op.inputs[1]);
      return `y_{${row},${col}} = ${fmt(A.data[index])} + ${fmt(B.data[index])} = ${fmt(Y.data[index])}`;
    }
    case 'layernorm': {
      const X = get(ctx, op.inputs[0]);
      const D = X.shape[1];
      const rowVals: number[] = [];
      for (let i = 0; i < D; i++) rowVals.push(X.data[row * D + i]);
      const mean = rowVals.reduce((a, b) => a + b, 0) / D;
      const variance = rowVals.reduce((a, b) => a + (b - mean) ** 2, 0) / D;
      const xval = X.data[row * D + col];
      const xhat = (xval - mean) / Math.sqrt(variance + 1e-5);
      const G = get(ctx, op.params[0]), Bt = get(ctx, op.params[1]);
      const meanLine = `\\mu_{${row}} = \\frac{${terms(rowVals.map(fmt))}}{${D}} = ${fmt(mean)}`;
      const varLine = `\\sigma^2_{${row}} = \\frac{${terms(rowVals.map(v => `(${fmt(v)}-${fmt(mean)})^2`))}}{${D}} = ${fmt(variance)}`;
      const normLine = `\\hat{x}_{${row},${col}} = \\frac{${fmt(xval)}-${fmt(mean)}}{\\sqrt{${fmt(variance)}+\\epsilon}} = ${fmt(xhat)}`;
      const outLine = `y_{${row},${col}} = ${fmt(G.data[col])}\\cdot ${fmt(xhat)} + ${fmt(Bt.data[col])} = ${fmt(Y.data[index])}`;
      return `\\begin{aligned} ${meanLine} \\\\ ${varLine} \\\\ ${normLine} \\\\ ${outLine} \\end{aligned}`;
    }
    case 'linear': {
      const X = get(ctx, op.inputs[0]), W = get(ctx, op.params[0]), B = get(ctx, op.params[1]);
      const In = X.shape[1], Out = W.shape[1];
      const parts = [`b_{${col}}=${fmt(B.data[col])}`];
      for (let i = 0; i < In; i++) parts.push(`${fmt(X.data[row * In + i])}\\cdot ${fmt(W.data[i * Out + col])}`);
      return `y_{${row},${col}} = ${terms(parts)} = ${fmt(Y.data[index])}`;
    }
    case 'scores': {
      const Q = get(ctx, op.inputs[0]), K = get(ctx, op.inputs[1]);
      const d = Q.shape[1];
      const parts: string[] = [];
      for (let c = 0; c < d; c++) parts.push(`${fmt(Q.data[row * d + c])}\\cdot ${fmt(K.data[col * d + c])}`);
      return `S_{${row},${col}} = \\frac{1}{\\sqrt{${d}}}\\big(${terms(parts)}\\big) = ${fmt(Y.data[index])}`;
    }
    case 'mask':
      return `S'_{${row},${col}} = ${fmt(Y.data[index])} \\quad(${col > row ? 'j > i' : 'j \\le i'})`;
    case 'softmax_rows': case 'softmax_out': {
      const S = get(ctx, op.inputs[0]);
      const C = S.shape[1];
      const rowVals: number[] = [];
      for (let c = 0; c < C; c++) rowVals.push(S.data[row * C + c]);
      const mx = Math.max(...rowVals);
      const exps = rowVals.map(v => Math.exp(v - mx));
      const z = exps.reduce((a, b) => a + b, 0);
      return `A_{${row},${col}} = \\frac{e^{${fmt(rowVals[col])}-${fmt(mx)}}}{${terms(exps.map(fmt))}} = \\frac{${fmt(exps[col])}}{${fmt(z)}} = ${fmt(Y.data[index])}`;
    }
    case 'attn_apply': {
      const A = get(ctx, op.inputs[0]), V = get(ctx, op.inputs[1]);
      const T = A.shape[0], d = V.shape[1];
      const parts: string[] = [];
      for (let j = 0; j < T; j++) parts.push(`${fmt(A.data[row * T + j])}\\cdot ${fmt(V.data[j * d + col])}`);
      return `y_{${row},${col}} = ${terms(parts)} = ${fmt(Y.data[index])}`;
    }
    case 'relu': {
      const X = get(ctx, op.inputs[0]);
      return `y_{${row},${col}} = \\max(0, ${fmt(X.data[index])}) = ${fmt(Y.data[index])}`;
    }
    case 'loss': {
      const P = get(ctx, op.inputs[0]);
      const [T, V] = P.shape;
      const targets = ctx.targets!;
      const parts: number[] = [];
      for (let t = 0; t < T; t++) parts.push(-Math.log(P.data[t * V + targets[t]]));
      return `L = \\frac{${terms(parts.map(fmt))}}{${T}} = ${fmt(Y.data[0])}`;
    }
    default: { const check: never = op.kind; return check; }
  }
}

/** Block (c): the gradient rule for this op's kind, with numbers for the same cell. */
function gradient(ctx: Ctx, op: Op, index: number): string {
  const Y = get(ctx, op.output);
  const [row, col] = rowCol(Y.shape, index);
  const dY = ctx.grads.get(op.output)?.data[index] ?? 0;
  switch (op.kind) {
    case 'embed': {
      const id = ctx.tokens[row];
      const E = get(ctx, op.params[0]);
      const dE = ctx.grads.get(op.params[0])?.data[id * E.shape[1] + col] ?? 0;
      return `\\frac{\\partial L}{\\partial E_{${id},${col}}} = \\sum_{t\\,:\\,\\text{id}_t=${id}} \\frac{\\partial L}{\\partial \\text{tok}_{t,${col}}} = ${fmt(dE)}`;
    }
    case 'pos_embed': {
      const dP = ctx.grads.get(op.params[0])?.data[index] ?? 0;
      return `\\frac{\\partial L}{\\partial P_{${row},${col}}} = \\frac{\\partial L}{\\partial \\text{pos}_{${row},${col}}} = ${fmt(dP)}`;
    }
    case 'add':
      return `\\frac{\\partial L}{\\partial a_{${row},${col}}} = \\frac{\\partial L}{\\partial b_{${row},${col}}} = \\frac{\\partial L}{\\partial y_{${row},${col}}} = ${fmt(dY)}`;
    case 'layernorm': {
      const dX = ctx.grads.get(op.inputs[0])?.data[index] ?? 0;
      return `\\frac{\\partial L}{\\partial x_{${row},${col}}} = \\frac{1}{\\sqrt{\\sigma^2+\\epsilon}}\\Big(\\hat g_{${col}} - \\overline{\\hat g} - \\hat x_{${col}}\\,\\overline{\\hat g \\hat x}\\Big) = ${fmt(dX)}`;
    }
    case 'linear': {
      const W = get(ctx, op.params[0]);
      const In = W.shape[0], Out = W.shape[1];
      const i = Math.min(col, In - 1);
      const dOut = ctx.grads.get(op.output);
      const parts: string[] = [];
      for (let o = 0; o < Out; o++) parts.push(`${fmt(dOut ? dOut.data[row * Out + o] : 0)}\\cdot ${fmt(W.data[i * Out + o])}`);
      const dX = ctx.grads.get(op.inputs[0])?.data[row * In + i] ?? 0;
      return `\\frac{\\partial L}{\\partial x_{${row},${i}}} = ${terms(parts)} = ${fmt(dX)}`;
    }
    case 'scores': {
      const K = get(ctx, op.inputs[1]);
      const d = K.shape[1];
      const S = get(ctx, op.output);
      const T = S.shape[0];
      const dS = ctx.grads.get(op.output);
      const parts: string[] = [];
      for (let j = 0; j < T; j++) parts.push(`${fmt(dS ? dS.data[row * T + j] : 0)}\\cdot ${fmt(K.data[j * d])}`);
      const dQ = ctx.grads.get(op.inputs[0])?.data[row * d] ?? 0;
      return `\\frac{\\partial L}{\\partial q_{${row},0}} = \\frac{1}{\\sqrt{${d}}}\\big(${terms(parts)}\\big) = ${fmt(dQ)}`;
    }
    case 'mask': {
      const passes = col <= row;
      const dS = passes ? dY : 0;
      return `\\frac{\\partial L}{\\partial S_{${row},${col}}} = ${fmt(dS)} \\quad(${passes ? 'j \\le i' : 'j > i,\\ \\text{masked}'})`;
    }
    case 'softmax_rows': case 'softmax_out': {
      const A = get(ctx, op.output);
      const C = A.shape[1];
      const dA = ctx.grads.get(op.output);
      const rowA: number[] = [], rowDA: number[] = [];
      for (let c = 0; c < C; c++) { rowA.push(A.data[row * C + c]); rowDA.push(dA ? dA.data[row * C + c] : 0); }
      const dot = rowA.reduce((s, a, c) => s + a * rowDA[c], 0);
      const dS = ctx.grads.get(op.inputs[0])?.data[index] ?? 0;
      return `\\frac{\\partial L}{\\partial S_{${row},${col}}} = A_{${row},${col}}\\Big(\\frac{\\partial L}{\\partial A_{${row},${col}}} - \\sum_c A_{${row},c}\\frac{\\partial L}{\\partial A_{${row},c}}\\Big) = ${fmt(rowA[col])}\\big(${fmt(rowDA[col])}-${fmt(dot)}\\big) = ${fmt(dS)}`;
    }
    case 'attn_apply': {
      const A = get(ctx, op.inputs[0]);
      const T = A.shape[0];
      const dOut = ctx.grads.get(op.output);
      const parts: string[] = [];
      for (let i = 0; i < T; i++) parts.push(`${fmt(A.data[i * T + row])}\\cdot ${fmt(dOut ? dOut.data[i * Y.shape[1] + col] : 0)}`);
      const dV = ctx.grads.get(op.inputs[1])?.data[row * Y.shape[1] + col] ?? 0;
      return `\\frac{\\partial L}{\\partial v_{${row},${col}}} = ${terms(parts)} = ${fmt(dV)}`;
    }
    case 'relu': {
      const X = get(ctx, op.inputs[0]);
      const dX = ctx.grads.get(op.inputs[0])?.data[index] ?? 0;
      return `\\frac{\\partial L}{\\partial x_{${row},${col}}} = ${X.data[index] > 0 ? '1' : '0'}\\cdot \\frac{\\partial L}{\\partial y_{${row},${col}}} = ${fmt(dX)}`;
    }
    case 'loss': {
      const P = get(ctx, op.inputs[0]);
      const [T, V] = P.shape;
      const targets = ctx.targets!;
      const t0 = 0;
      const p = P.data[t0 * V + targets[t0]];
      const dP = ctx.grads.get(op.inputs[0])?.data[t0 * V + targets[t0]] ?? 0;
      return `\\frac{\\partial L}{\\partial p_{${t0},\\text{target}}} = \\frac{-1}{T\\,p_{${t0},\\text{target}}} = \\frac{-1}{${T}\\cdot ${fmt(p)}} = ${fmt(dP)}`;
    }
    default: { const check: never = op.kind; return check; }
  }
}

/**
 * The maths for the currently-active op. Block (a) is always the general formula. Block (b) adds
 * the selected cell's real numbers when the selection is on this op's output tensor. Block (c) adds
 * the gradient rule for the same cell when the current step is in the backward phase.
 */
export function mathFor(trace: Trace, step: TraceStep | null, op: Op | null, selection: CellRef | null): MathBlock[] {
  if (!op) return [{ title: 'Nothing selected', latex: '', note: HINT }];
  const matches = selection !== null && selection.key === tKey(op.output);
  const blocks: MathBlock[] = [
    matches
      ? { title: op.label, latex: generalFormula(op.kind) }
      : { title: op.label, latex: generalFormula(op.kind), note: HINT },
  ];
  if (matches) {
    blocks.push({ title: 'This cell', latex: substituted(trace.ctx, op, selection!.index) });
    if (step?.phase === 'backward') blocks.push({ title: 'Gradient', latex: gradient(trace.ctx, op, selection!.index) });
  }
  return blocks;
}
