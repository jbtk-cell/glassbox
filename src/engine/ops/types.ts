import { Tensor, tensor } from '../tensor';

export type OpKind =
  | 'embed' | 'pos_embed' | 'add' | 'layernorm' | 'linear' | 'scores' | 'mask'
  | 'softmax_rows' | 'attn_apply' | 'relu' | 'softmax_out' | 'loss';

export interface Ctx {
  tensors: Map<string, Tensor>;
  grads: Map<string, Tensor>;
  tokens: number[];
  targets?: number[];
  /** Sequence length of this pass. */
  T: number;
}

export interface OpMeta { id: string; label: string; formula: string; explain: string }

export interface Op extends OpMeta {
  kind: OpKind;
  inputs: string[];
  output: string;
  params: string[];
  forward(ctx: Ctx): void;
  /** Reads grad of `output`; accumulates into grads of `inputs` and `params`. */
  backward(ctx: Ctx): void;
}

export function newCtx(tokens: number[], params: Map<string, Tensor>, targets?: number[]): Ctx {
  const tensors = new Map<string, Tensor>();
  for (const [k, v] of params) tensors.set(k, v);
  return { tensors, grads: new Map(), tokens, targets, T: tokens.length };
}

export function get(ctx: Ctx, name: string): Tensor {
  const t = ctx.tensors.get(name);
  if (!t) throw new Error(`tensor '${name}' not in ctx`);
  return t;
}

export function put(ctx: Ctx, name: string, shape: number[]): Tensor {
  const t = tensor(name, shape);
  ctx.tensors.set(name, t);
  return t;
}

export function gradOf(ctx: Ctx, name: string): Tensor {
  let g = ctx.grads.get(name);
  if (!g) { g = tensor(name, get(ctx, name).shape); ctx.grads.set(name, g); }
  return g;
}

export function gradGet(ctx: Ctx, name: string): Tensor {
  const g = ctx.grads.get(name);
  if (!g) throw new Error(`grad of '${name}' not in ctx`);
  return g;
}
