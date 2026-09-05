import { GPT, T_ } from './gpt';
import { Adam } from './adam';
import { Ctx, get } from '../ops/types';
import { Window } from '../corpus/windows';

export function trainStep(model: GPT, adam: Adam, w: Window, wantDeltas = true): { loss: number; ctx: Ctx; deltas: Map<string, Float64Array> } {
  const ctx = model.forward(w.input, w.target);
  model.backward(ctx);
  const deltas = adam.step(ctx.grads, wantDeltas);
  return { loss: get(ctx, T_.loss).data[0], ctx, deltas };
}

/** Mean loss and next-token accuracy over up to `limit` windows (evenly spaced through the list). */
export function evaluate(model: GPT, ws: Window[], limit = 64): { loss: number; accuracy: number } {
  if (ws.length === 0) return { loss: NaN, accuracy: NaN };
  const stride = Math.max(1, Math.floor(ws.length / limit));
  let lossSum = 0, correct = 0, positions = 0, n = 0;
  for (let i = 0; i < ws.length; i += stride) {
    const w = ws[i];
    const ctx = model.forward(w.input, w.target);
    lossSum += get(ctx, T_.loss).data[0]; n++;
    const P = get(ctx, T_.probs); const [T, V] = P.shape;
    for (let t = 0; t < T; t++) {
      let best = 0; for (let v = 1; v < V; v++) if (P.data[t * V + v] > P.data[t * V + best]) best = v;
      if (best === w.target[t]) correct++;
      positions++;
    }
  }
  return { loss: lossSum / n, accuracy: correct / positions };
}
