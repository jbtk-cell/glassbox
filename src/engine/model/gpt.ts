import { Tensor, tensor } from '../tensor';
import { Op, Ctx, newCtx } from '../ops/types';
import { add, relu } from '../ops/elementwise';
import { linear } from '../ops/linear';
import { layerNorm } from '../ops/layernorm';
import { embed, posEmbed } from '../ops/embed';
import { scores, causalMask, softmaxRows, attnApply } from '../ops/attention';
import { softmaxOut, crossEntropy } from '../ops/output';
import { GPTConfig, initParams, paramCount, paramShapes, PARAM_NAMES } from './params';

/** Names of every activation tensor, in the order they are produced. */
export const T_ = {
  tok: 'tok', pos: 'pos', x0: 'x0', h1: 'h1', q: 'q', k: 'k', v: 'v', scores: 'scores', masked: 'masked',
  attn: 'attn', ctxv: 'ctxv', attn_out: 'attn_out', x1: 'x1', h2: 'h2', ff_pre: 'ff_pre', ff_act: 'ff_act',
  ff_out: 'ff_out', x2: 'x2', hf: 'hf', logits: 'logits', probs: 'probs', loss: 'loss',
} as const;

/** The whole forward pass as an ordered list of named operations. ops[21] is the loss. */
export function buildOps(_c: GPTConfig): Op[] {
  return [
    embed({ id: 'embed', label: 'Token embedding', formula: 'tok_t = E[id_t]',
      explain: 'Each word id picks one row out of the embedding table E. That row is a list of numbers the model learns to use as a description of the word. The model knows nothing about the word except what training has written into this row.' }, 'E', T_.tok),
    posEmbed({ id: 'pos_embed', label: 'Position embedding', formula: 'pos_t = P[t]',
      explain: 'Attention has no built-in sense of order, so each position t also gets its own learned row from table P. Position 0 always gets row 0, position 1 row 1, and so on.' }, 'P', T_.pos),
    add({ id: 'add_pos', label: 'Add position', formula: 'x0 = tok + pos',
      explain: 'The model adds the word vector and the position vector, one number at a time. The result x0 is the residual stream: the running record that every later step reads from and writes back into.' }, T_.tok, T_.pos, T_.x0),
    layerNorm({ id: 'ln1', label: 'Layer norm 1', formula: 'h1 = ln1_g * (x0 - mean) / sqrt(var + eps) + ln1_b',
      explain: 'Layer norm rescales each row of x0 to mean 0 and spread 1, then stretches it by ln1_g and shifts it by ln1_b. The next steps get numbers in a predictable range no matter how large the stream has grown.' }, T_.x0, 'ln1_g', 'ln1_b', T_.h1),
    linear({ id: 'q_proj', label: 'Query projection', formula: 'q = h1 W_q + b_q',
      explain: 'Every position asks a question. The query q is what this position is looking for in the positions before it, computed by multiplying its normalised vector by the learned matrix W_q.' }, T_.h1, 'W_q', 'b_q', T_.q),
    linear({ id: 'k_proj', label: 'Key projection', formula: 'k = h1 W_k + b_k',
      explain: 'Every position also advertises what it holds. The key k is that advertisement, computed with a different learned matrix W_k. The model matches queries against keys.' }, T_.h1, 'W_k', 'b_k', T_.k),
    linear({ id: 'v_proj', label: 'Value projection', formula: 'v = h1 W_v + b_v',
      explain: 'The value v is what a position hands over when another position attends to it. Queries and keys decide the matching; the value is the content that moves.' }, T_.h1, 'W_v', 'b_v', T_.v),
    scores({ id: 'scores', label: 'Attention scores', formula: 'S[i,j] = q_i . k_j / sqrt(d)',
      explain: 'The model compares each query with each key by a dot product. A large number at row i, column j means position i finds position j relevant. Dividing by the square root of d keeps the scores from growing with the vector size.' }, T_.q, T_.k, T_.scores),
    causalMask({ id: 'causal_mask', label: 'Causal mask', formula: 'S[i,j] = -inf if j > i',
      explain: 'A position may only look at itself and earlier positions, never at later ones, because at prediction time the later words do not exist yet. The mask sets scores for future positions to minus infinity, so the softmax gives them zero. This is why the tile is a triangle.' }, T_.scores, T_.masked),
    softmaxRows({ id: 'attn_softmax', label: 'Attention weights', formula: 'A[i,:] = softmax(S[i,:])',
      explain: 'Softmax turns each row of scores into positive weights that add up to 1. Row i now says how much of its attention position i spends on each earlier position.' }, T_.masked, T_.attn),
    attnApply({ id: 'attn_apply', label: 'Apply attention', formula: 'ctxv = A v',
      explain: 'Each position collects a weighted mix of the value vectors of the positions it attends to. Positions with larger weights contribute more. The result is new information gathered from context.' }, T_.attn, T_.v, T_.ctxv),
    linear({ id: 'o_proj', label: 'Output projection', formula: 'attn_out = ctxv W_o + b_o',
      explain: 'One more learned matrix, W_o, decides how the gathered information gets written back into the residual stream.' }, T_.ctxv, 'W_o', 'b_o', T_.attn_out),
    add({ id: 'residual1', label: 'Residual add 1', formula: 'x1 = x0 + attn_out',
      explain: 'The model adds the attention result onto the stream instead of replacing it. The original word and position information survive, with the new context layered on top.' }, T_.x0, T_.attn_out, T_.x1),
    layerNorm({ id: 'ln2', label: 'Layer norm 2', formula: 'h2 = ln2_g * (x1 - mean) / sqrt(var + eps) + ln2_b',
      explain: 'Normalise again before the next block, for the same reason as before: keep every row at a predictable scale.' }, T_.x1, 'ln2_g', 'ln2_b', T_.h2),
    linear({ id: 'ff_up', label: 'Feed-forward up', formula: 'ff_pre = h2 W_1 + b_1',
      explain: 'From here on, positions do not mix; each one goes through the same small network alone. W_1 expands the vector to a wider hidden layer. This is where the model stores patterns that do not depend on context.' }, T_.h2, 'W_1', 'b_1', T_.ff_pre),
    relu({ id: 'relu', label: 'ReLU', formula: 'ff_act = max(0, ff_pre)',
      explain: 'ReLU replaces every negative number in the hidden layer with zero. Without this one nonlinear step the whole network would collapse into a single matrix multiplication.' }, T_.ff_pre, T_.ff_act),
    linear({ id: 'ff_down', label: 'Feed-forward down', formula: 'ff_out = ff_act W_2 + b_2',
      explain: 'W_2 projects the hidden layer back down to the stream width so it can join the residual stream.' }, T_.ff_act, 'W_2', 'b_2', T_.ff_out),
    add({ id: 'residual2', label: 'Residual add 2', formula: 'x2 = x1 + ff_out',
      explain: 'The model adds the feed-forward result onto the stream. x2 is the final state of the residual stream for this block.' }, T_.x1, T_.ff_out, T_.x2),
    layerNorm({ id: 'ln_final', label: 'Final layer norm', formula: 'hf = lnf_g * (x2 - mean) / sqrt(var + eps) + lnf_b',
      explain: 'One last normalisation before reading the prediction out of the stream.' }, T_.x2, 'lnf_g', 'lnf_b', T_.hf),
    linear({ id: 'unembed', label: 'Unembedding', formula: 'logits = hf U + b_u',
      explain: 'Multiplying by U turns the stream back into one score per vocabulary word, for each position. These raw scores are the logits.' }, T_.hf, 'U', 'b_u', T_.logits),
    softmaxOut({ id: 'softmax_out', label: 'Next-token probabilities', formula: 'probs[t,:] = softmax(logits[t,:])',
      explain: 'Softmax turns each row of logits into probabilities that add up to 1. Row t is the model\'s prediction for the word that comes after position t. The largest entry is the model\'s best guess.' }, T_.logits, T_.probs),
    crossEntropy({ id: 'loss', label: 'Cross-entropy loss', formula: 'loss = -(1/T) sum_t log probs[t, target_t]',
      explain: 'Training scores the model on how much probability it gave to the word that came next. Minus the log makes a confident right answer cost nearly 0 and a confident wrong answer cost a lot. The average over positions is the number training pushes down.' }, T_.probs, T_.loss),
  ];
}

export class GPT {
  readonly config: GPTConfig;
  readonly params: Map<string, Tensor>;
  readonly ops: Op[];

  constructor(config: GPTConfig, params?: Map<string, Tensor>) {
    this.config = config;
    this.params = params ?? initParams(config);
    this.ops = buildOps(config);
  }

  forward(tokens: number[], targets?: number[]): Ctx {
    if (tokens.length === 0) throw new Error('forward: empty token list');
    if (tokens.length > this.config.contextSize) throw new Error(`forward: ${tokens.length} tokens exceeds context size ${this.config.contextSize}`);
    if (targets && targets.length !== tokens.length) throw new Error('forward: targets length must equal tokens length');
    const ctx = newCtx(tokens, this.params, targets);
    const n = targets ? this.ops.length : this.ops.length - 1;
    for (let i = 0; i < n; i++) this.ops[i].forward(ctx);
    return ctx;
  }

  backward(ctx: Ctx): void {
    if (!ctx.tensors.has(T_.loss)) throw new Error('backward: run forward with targets first');
    ctx.grads.set(T_.loss, tensor(T_.loss, [1], new Float64Array([1])));
    for (let i = this.ops.length - 1; i >= 0; i--) this.ops[i].backward(ctx);
  }

  paramCount(): number { return paramCount(this.params); }

  setParam(name: string, index: number, value: number): void {
    const p = this.params.get(name);
    if (!p) throw new Error(`setParam: no parameter '${name}'`);
    if (index < 0 || index >= p.data.length) throw new Error(`setParam: index ${index} out of range for '${name}'`);
    p.data[index] = value;
  }

  exportParams(): Record<string, number[]> {
    const out: Record<string, number[]> = {};
    for (const n of PARAM_NAMES) out[n] = Array.from(this.params.get(n)!.data);
    return out;
  }

  importParams(p: Record<string, number[]>): void {
    const shapes = paramShapes(this.config);
    for (const n of PARAM_NAMES) {
      const src = p[n]; if (!src) throw new Error(`importParams: missing '${n}'`);
      const t = this.params.get(n)!;
      if (src.length !== t.data.length) throw new Error(`importParams: '${n}' has ${src.length} values, expected ${t.data.length} (${shapes[n].join('x')})`);
      t.data.set(src);
    }
  }
}
