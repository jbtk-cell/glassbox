import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** Large negative stand-in for -infinity; exp() of it underflows to exactly 0 in float64. */
export const MASK_VALUE = -1e9;

/** S[i,j] = sum_d q[i,d] k[j,d] / sqrt(d) */
export function scores(meta: OpMeta, q: string, k: string, out: string): Op {
  return {
    ...meta, kind: 'scores', inputs: [q, k], output: out, params: [],
    forward(ctx: Ctx) {
      const Q = get(ctx, q), K = get(ctx, k); const [T, d] = Q.shape; const inv = 1 / Math.sqrt(d);
      const S = put(ctx, out, [T, T]);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        let s = 0; for (let c = 0; c < d; c++) s += Q.data[i * d + c] * K.data[j * d + c];
        S.data[i * T + j] = s * inv;
      }
    },
    backward(ctx: Ctx) {
      const Q = get(ctx, q), K = get(ctx, k); const [T, d] = Q.shape; const inv = 1 / Math.sqrt(d);
      const dS = gradGet(ctx, out).data, dQ = gradOf(ctx, q).data, dK = gradOf(ctx, k).data;
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const g = dS[i * T + j] * inv;
        for (let c = 0; c < d; c++) { dQ[i * d + c] += g * K.data[j * d + c]; dK[j * d + c] += g * Q.data[i * d + c]; }
      }
    },
  };
}

/** out[i,j] = j > i ? MASK_VALUE : s[i,j] */
export function causalMask(meta: OpMeta, s: string, out: string): Op {
  return {
    ...meta, kind: 'mask', inputs: [s], output: out, params: [],
    forward(ctx: Ctx) {
      const S = get(ctx, s); const [T] = S.shape;
      const O = put(ctx, out, [T, T]);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) O.data[i * T + j] = j > i ? MASK_VALUE : S.data[i * T + j];
    },
    backward(ctx: Ctx) {
      const [T] = get(ctx, s).shape;
      const dO = gradGet(ctx, out).data, dS = gradOf(ctx, s).data;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) dS[i * T + j] += dO[i * T + j];
    },
  };
}

/** Row-wise softmax of an R×C tensor, max-subtracted for stability. */
export function softmaxRows(meta: OpMeta, s: string, out: string): Op {
  return {
    ...meta, kind: 'softmax_rows', inputs: [s], output: out, params: [],
    forward(ctx: Ctx) {
      const S = get(ctx, s); const [R, C] = S.shape;
      const A = put(ctx, out, [R, C]);
      for (let r = 0; r < R; r++) {
        let mx = -Infinity; for (let c = 0; c < C; c++) mx = Math.max(mx, S.data[r * C + c]);
        let z = 0; for (let c = 0; c < C; c++) { const e = Math.exp(S.data[r * C + c] - mx); A.data[r * C + c] = e; z += e; }
        for (let c = 0; c < C; c++) A.data[r * C + c] /= z;
      }
    },
    backward(ctx: Ctx) {
      const A = get(ctx, out); const [R, C] = A.shape;
      const dA = gradGet(ctx, out).data, dS = gradOf(ctx, s).data;
      for (let r = 0; r < R; r++) {
        let dot = 0; for (let c = 0; c < C; c++) dot += dA[r * C + c] * A.data[r * C + c];
        for (let c = 0; c < C; c++) dS[r * C + c] += A.data[r * C + c] * (dA[r * C + c] - dot);
      }
    },
  };
}

/** O = A v.  A: T×T, v: T×d. */
export function attnApply(meta: OpMeta, a: string, v: string, out: string): Op {
  return {
    ...meta, kind: 'attn_apply', inputs: [a, v], output: out, params: [],
    forward(ctx: Ctx) {
      const A = get(ctx, a), V = get(ctx, v); const T = A.shape[0], d = V.shape[1];
      const O = put(ctx, out, [T, d]);
      for (let i = 0; i < T; i++) for (let c = 0; c < d; c++) {
        let s = 0; for (let j = 0; j < T; j++) s += A.data[i * T + j] * V.data[j * d + c];
        O.data[i * d + c] = s;
      }
    },
    backward(ctx: Ctx) {
      const A = get(ctx, a), V = get(ctx, v); const T = A.shape[0], d = V.shape[1];
      const dO = gradGet(ctx, out).data, dA = gradOf(ctx, a).data, dV = gradOf(ctx, v).data;
      for (let i = 0; i < T; i++) for (let c = 0; c < d; c++) {
        const g = dO[i * d + c];
        for (let j = 0; j < T; j++) { dA[i * T + j] += g * V.data[j * d + c]; dV[j * d + c] += A.data[i * T + j] * g; }
      }
    },
  };
}
