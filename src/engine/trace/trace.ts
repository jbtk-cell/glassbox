import { GPT, T_ } from '../model/gpt';
import { Adam } from '../model/adam';
import { PARAM_NAMES } from '../model/params';
import { Ctx } from '../ops/types';
import { Window } from '../corpus/windows';
import { tensor } from '../tensor';

export type Phase = 'forward' | 'backward' | 'update';
export const UPDATE_OP = 'adam_update';

export interface TraceStep { index: number; opId: string; phase: Phase; writes: string[] }
export interface Trace {
  kind: 'forward' | 'training';
  steps: TraceStep[];
  ctx: Ctx;
  deltas?: Map<string, Float64Array>;
  /** Keys that exist before step 0 (the parameters). */
  preexisting: Set<string>;
}

export const tKey = (name: string) => 't:' + name;
export const gKey = (name: string) => 'g:' + name;

function preexistingKeys(): Set<string> { return new Set(PARAM_NAMES.map(tKey)); }

/** Runs the forward pass and records one step per op. */
export function recordForward(model: GPT, tokens: number[]): Trace {
  const ctx = model.forward(tokens);
  const steps: TraceStep[] = model.ops.slice(0, -1).map((op, i) => ({ index: i, opId: op.id, phase: 'forward', writes: [tKey(op.output)] }));
  return { kind: 'forward', steps, ctx, preexisting: preexistingKeys() };
}

/** Runs forward + backward + one Adam step and records all three phases. Mutates the model's parameters. */
export function recordTrainingStep(model: GPT, adam: Adam, w: Window): Trace {
  const ctx = model.forward(w.input, w.target);
  ctx.grads.set(T_.loss, tensor(T_.loss, [1], new Float64Array([1])));
  for (let i = model.ops.length - 1; i >= 0; i--) model.ops[i].backward(ctx);
  const deltas = adam.step(ctx.grads);

  const steps: TraceStep[] = [];
  model.ops.forEach(op => steps.push({ index: steps.length, opId: op.id, phase: 'forward', writes: [tKey(op.output)] }));
  for (let i = model.ops.length - 1; i >= 0; i--) {
    const op = model.ops[i];
    const writes = [...op.inputs.map(gKey), ...op.params.map(gKey)];
    if (op.id === 'loss') writes.unshift(gKey(T_.loss));
    steps.push({ index: steps.length, opId: op.id, phase: 'backward', writes });
  }
  steps.push({ index: steps.length, opId: UPDATE_OP, phase: 'update', writes: PARAM_NAMES.map(tKey) });
  return { kind: 'training', steps, ctx, deltas, preexisting: preexistingKeys() };
}
