import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

export function add(meta: OpMeta, a: string, b: string, out: string): Op {
  return {
    ...meta, kind: 'add', inputs: [a, b], output: out, params: [],
    forward(ctx: Ctx) {
      const A = get(ctx, a), B = get(ctx, b);
      const Y = put(ctx, out, A.shape);
      for (let i = 0; i < Y.data.length; i++) Y.data[i] = A.data[i] + B.data[i];
    },
    backward(ctx: Ctx) {
      const dY = gradGet(ctx, out).data;
      const dA = gradOf(ctx, a).data, dB = gradOf(ctx, b).data;
      for (let i = 0; i < dY.length; i++) { dA[i] += dY[i]; dB[i] += dY[i]; }
    },
  };
}

export function relu(meta: OpMeta, x: string, out: string): Op {
  return {
    ...meta, kind: 'relu', inputs: [x], output: out, params: [],
    forward(ctx: Ctx) {
      const X = get(ctx, x);
      const Y = put(ctx, out, X.shape);
      for (let i = 0; i < Y.data.length; i++) Y.data[i] = X.data[i] > 0 ? X.data[i] : 0;
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x).data, dY = gradGet(ctx, out).data, dX = gradOf(ctx, x).data;
      for (let i = 0; i < dY.length; i++) if (X[i] > 0) dX[i] += dY[i];
    },
  };
}
