import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** out[t, :] = E[tokens[t], :] */
export function embed(meta: OpMeta, E: string, out: string): Op {
  return {
    ...meta, kind: 'embed', inputs: [], output: out, params: [E],
    forward(ctx: Ctx) {
      const Et = get(ctx, E); const d = Et.shape[1]; const T = ctx.T;
      const X = put(ctx, out, [T, d]);
      for (let t = 0; t < T; t++) {
        const id = ctx.tokens[t];
        if (id < 0 || id >= Et.shape[0]) throw new Error(`${meta.id}: token id ${id} out of range`);
        for (let j = 0; j < d; j++) X.data[t * d + j] = Et.data[id * d + j];
      }
    },
    backward(ctx: Ctx) {
      const Et = get(ctx, E); const d = Et.shape[1]; const T = ctx.T;
      const dX = gradGet(ctx, out).data, dE = gradOf(ctx, E).data;
      for (let t = 0; t < T; t++) { const id = ctx.tokens[t]; for (let j = 0; j < d; j++) dE[id * d + j] += dX[t * d + j]; }
    },
  };
}

/** out[t, :] = P[t, :] for t < T */
export function posEmbed(meta: OpMeta, P: string, out: string): Op {
  return {
    ...meta, kind: 'pos_embed', inputs: [], output: out, params: [P],
    forward(ctx: Ctx) {
      const Pt = get(ctx, P); const d = Pt.shape[1]; const T = ctx.T;
      if (T > Pt.shape[0]) throw new Error(`${meta.id}: sequence length ${T} exceeds context size ${Pt.shape[0]}`);
      const X = put(ctx, out, [T, d]);
      X.data.set(Pt.data.subarray(0, T * d));
    },
    backward(ctx: Ctx) {
      const Pt = get(ctx, P); const d = Pt.shape[1]; const T = ctx.T;
      const dX = gradGet(ctx, out).data, dP = gradOf(ctx, P).data;
      for (let i = 0; i < T * d; i++) dP[i] += dX[i];
    },
  };
}
