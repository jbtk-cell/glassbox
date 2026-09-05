import { GPT, T_ } from './gpt';
import { get } from '../ops/types';
import { Rng } from '../rng';

export type Strategy = 'greedy' | 'top-k' | 'top-p';
export interface SampleOpts { temperature: number; strategy: Strategy; k: number; p: number }
export const SAMPLE_DEFAULTS: SampleOpts = { temperature: 1, strategy: 'top-p', k: 10, p: 0.9 };

export function sampleFromLogits(logits: Float64Array, opts: SampleOpts, rng: Rng): { token: number; probs: Float64Array } {
  const V = logits.length;
  const probs = new Float64Array(V);
  let best = 0; for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
  if (opts.strategy === 'greedy' || opts.temperature <= 0) { probs[best] = 1; return { token: best, probs }; }

  const temp = opts.temperature;
  let z = 0;
  for (let i = 0; i < V; i++) { probs[i] = Math.exp((logits[i] - logits[best]) / temp); z += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= z;

  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  let keep: number;
  if (opts.strategy === 'top-k') keep = Math.max(1, Math.min(V, Math.floor(opts.k)));
  else { let cum = 0; keep = 0; for (const i of order) { cum += probs[i]; keep++; if (cum >= opts.p) break; } }
  const mask = new Set(order.slice(0, keep));
  let z2 = 0;
  for (let i = 0; i < V; i++) { if (!mask.has(i)) probs[i] = 0; z2 += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= z2;

  let r = rng.next(), token = order[keep - 1];
  for (const i of order.slice(0, keep)) { r -= probs[i]; if (r <= 0) { token = i; break; } }
  return { token, probs };
}

/** Returns prompt followed by n sampled tokens. Context is the last contextSize tokens. */
export function generate(model: GPT, prompt: number[], n: number, opts: SampleOpts, rng: Rng): number[] {
  const out = [...prompt];
  const T = model.config.contextSize;
  for (let i = 0; i < n; i++) {
    const ctxTokens = out.slice(-T);
    const ctx = model.forward(ctxTokens);
    const L = get(ctx, T_.logits); const [rows, V] = L.shape;
    const last = L.data.subarray((rows - 1) * V, rows * V);
    out.push(sampleFromLogits(Float64Array.from(last), opts, rng).token);
  }
  return out;
}
