import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';
import { softmaxRows } from './attention';

/** Same maths as softmaxRows; a distinct kind so views can treat the vocabulary distribution specially. */
export function softmaxOut(meta: OpMeta, logits: string, out: string): Op {
  return { ...softmaxRows(meta, logits, out), kind: 'softmax_out' };
}

/** loss = -(1/T) * sum_t log probs[t, targets[t]] */
export function crossEntropy(meta: OpMeta, probs: string, out: string): Op {
  return {
    ...meta, kind: 'loss', inputs: [probs], output: out, params: [],
    forward(ctx: Ctx) {
      if (!ctx.targets) throw new Error(`${meta.id}: ctx.targets required`);
      const P = get(ctx, probs); const [T, V] = P.shape;
      const L = put(ctx, out, [1]);
      let s = 0;
      for (let t = 0; t < T; t++) s += -Math.log(P.data[t * V + ctx.targets[t]]);
      L.data[0] = s / T;
    },
    backward(ctx: Ctx) {
      if (!ctx.targets) throw new Error(`${meta.id}: ctx.targets required`);
      const P = get(ctx, probs); const [T, V] = P.shape;
      const dL = gradGet(ctx, out).data[0];
      const dP = gradOf(ctx, probs).data;
      for (let t = 0; t < T; t++) { const idx = t * V + ctx.targets[t]; dP[idx] += -dL / (T * P.data[idx]); }
    },
  };
}
