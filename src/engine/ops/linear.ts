import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** y = x W + b.  x: T×In, W: In×Out, b: [Out]. */
export function linear(meta: OpMeta, x: string, W: string, b: string, out: string): Op {
  return {
    ...meta, kind: 'linear', inputs: [x], output: out, params: [W, b],
    forward(ctx: Ctx) {
      const X = get(ctx, x), Wt = get(ctx, W), B = get(ctx, b);
      const [T, In] = X.shape, Out = Wt.shape[1];
      if (Wt.shape[0] !== In) throw new Error(`${meta.id}: x has ${In} cols but W has ${Wt.shape[0]} rows`);
      const Y = put(ctx, out, [T, Out]);
      const xv = X.data, w = Wt.data, y = Y.data, bb = B.data;
      for (let t = 0; t < T; t++) {
        const yo = t * Out;
        for (let o = 0; o < Out; o++) y[yo + o] = bb[o];
        for (let i = 0; i < In; i++) {
          const xi = xv[t * In + i]; if (xi === 0) continue;
          const wo = i * Out;
          for (let o = 0; o < Out; o++) y[yo + o] += xi * w[wo + o];
        }
      }
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x), Wt = get(ctx, W);
      const [T, In] = X.shape, Out = Wt.shape[1];
      const dY = gradGet(ctx, out).data;
      const dX = gradOf(ctx, x).data, dW = gradOf(ctx, W).data, dB = gradOf(ctx, b).data;
      const xd = X.data, w = Wt.data;
      for (let t = 0; t < T; t++) {
        const yo = t * Out;
        for (let o = 0; o < Out; o++) dB[o] += dY[yo + o];
        for (let i = 0; i < In; i++) {
          const wo = i * Out, xi = xd[t * In + i];
          let acc = 0;
          for (let o = 0; o < Out; o++) { const g = dY[yo + o]; acc += g * w[wo + o]; dW[wo + o] += xi * g; }
          dX[t * In + i] += acc;
        }
      }
    },
  };
}
