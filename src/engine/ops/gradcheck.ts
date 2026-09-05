import { Op, Ctx, get, gradGet } from './types';
import { tensor, size } from '../tensor';
import { Rng } from '../rng';

export interface GradFailure { name: string; index: number; analytic: number; numeric: number }

/** Tensors ~ N(0, scale). Params map is empty; everything goes in ctx.tensors directly. */
export function randomCtx(shapes: Record<string, number[]>, seed: number, scale = 1): Ctx {
  const rng = new Rng(seed);
  const tensors = new Map();
  for (const [name, shape] of Object.entries(shapes)) {
    const t = tensor(name, shape);
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.randn() * scale;
    tensors.set(name, t);
  }
  const T = shapes['__T'] ? shapes['__T'][0] : 0;
  tensors.delete('__T');
  return { tensors, grads: new Map(), tokens: [], T };
}

/**
 * Compare op.backward with central finite differences of L = sum(output * R), R fixed random.
 * Returns failures; an empty array means the op's gradient is correct for `names`.
 */
export function gradCheck(op: Op, ctx: Ctx, names: string[], opts: { eps?: number; tol?: number; seed?: number } = {}): GradFailure[] {
  const eps = opts.eps ?? 1e-6, tol = opts.tol ?? 1e-5;
  const rng = new Rng(opts.seed ?? 12345);
  op.forward(ctx);
  const out = get(ctx, op.output);
  const R = new Float64Array(size(out.shape));
  for (let i = 0; i < R.length; i++) R[i] = rng.randn();
  const objective = () => { op.forward(ctx); const o = get(ctx, op.output).data; let s = 0; for (let i = 0; i < o.length; i++) s += o[i] * R[i]; return s; };

  ctx.grads.set(op.output, tensor(op.output, out.shape, Float64Array.from(R)));
  for (const n of names) ctx.grads.delete(n);
  op.backward(ctx);

  const failures: GradFailure[] = [];
  for (const n of names) {
    const x = get(ctx, n).data;
    const g = gradGet(ctx, n).data;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      x[i] = v + eps; const lp = objective();
      x[i] = v - eps; const lm = objective();
      x[i] = v;
      const numeric = (lp - lm) / (2 * eps);
      if (Math.abs(g[i] - numeric) > tol * Math.max(1, Math.abs(numeric))) failures.push({ name: n, index: i, analytic: g[i], numeric });
    }
  }
  op.forward(ctx);
  return failures;
}
