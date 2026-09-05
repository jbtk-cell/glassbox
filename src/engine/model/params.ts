import { Tensor, tensor } from '../tensor';
import { Rng } from '../rng';

export interface GPTConfig { vocabSize: number; contextSize: number; dModel: number; dFF: number; seed: number }

export const DEFAULTS: Omit<GPTConfig, 'vocabSize'> = { contextSize: 24, dModel: 20, dFF: 30, seed: 1 };

export const PARAM_NAMES = [
  'E', 'P',
  'W_q', 'b_q', 'W_k', 'b_k', 'W_v', 'b_v', 'W_o', 'b_o',
  'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b',
  'W_1', 'b_1', 'W_2', 'b_2',
  'lnf_g', 'lnf_b',
  'U', 'b_u',
] as const;

export function paramShapes(c: GPTConfig): Record<string, number[]> {
  const { vocabSize: V, contextSize: T, dModel: d, dFF: f } = c;
  return {
    E: [V, d], P: [T, d],
    W_q: [d, d], b_q: [d], W_k: [d, d], b_k: [d], W_v: [d, d], b_v: [d], W_o: [d, d], b_o: [d],
    ln1_g: [d], ln1_b: [d], ln2_g: [d], ln2_b: [d],
    W_1: [d, f], b_1: [f], W_2: [f, d], b_2: [d],
    lnf_g: [d], lnf_b: [d],
    U: [d, V], b_u: [V],
  };
}

const INIT_STD = 0.02;

export function initParams(c: GPTConfig): Map<string, Tensor> {
  const rng = new Rng(c.seed);
  const shapes = paramShapes(c);
  const out = new Map<string, Tensor>();
  for (const name of PARAM_NAMES) {
    const t = tensor(name, shapes[name]);
    if (name.endsWith('_g')) t.data.fill(1);
    else if (name.startsWith('b_') || name.endsWith('_b')) t.data.fill(0);
    else for (let i = 0; i < t.data.length; i++) t.data[i] = rng.randn() * INIT_STD;
    out.set(name, t);
  }
  return out;
}

export function paramCount(params: Map<string, Tensor>): number {
  let n = 0; for (const t of params.values()) n += t.data.length; return n;
}
