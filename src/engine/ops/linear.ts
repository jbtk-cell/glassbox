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
      for (let t = 0; t < T; t++) for (let o = 0; o < Out; o++) {
        let s = B.data[o];
        for (let i = 0; i < In; i++) s += X.data[t * In + i] * Wt.data[i * Out + o];
        Y.data[t * Out + o] = s;
      }
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x), Wt = get(ctx, W);
      const [T, In] = X.shape, Out = Wt.shape[1];
      const dY = gradGet(ctx, out).data;
      const dX = gradOf(ctx, x).data, dW = gradOf(ctx, W).data, dB = gradOf(ctx, b).data;
      for (let t = 0; t < T; t++) for (let o = 0; o < Out; o++) {
        const g = dY[t * Out + o];
        dB[o] += g;
        for (let i = 0; i < In; i++) {
          dX[t * In + i] += g * Wt.data[i * Out + o];
          dW[i * Out + o] += X.data[t * In + i] * g;
        }
      }
    },
  };
}
