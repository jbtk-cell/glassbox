import { Tensor } from '../tensor';

export interface AdamOpts { lr: number; beta1: number; beta2: number; eps: number }
export const ADAM_DEFAULTS: AdamOpts = { lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

export class Adam {
  opts: AdamOpts;
  t = 0;
  private m = new Map<string, Float64Array>();
  private v = new Map<string, Float64Array>();
  constructor(private params: Map<string, Tensor>, opts: Partial<AdamOpts> = {}) {
    this.opts = { ...ADAM_DEFAULTS, ...opts };
    for (const [n, p] of params) { this.m.set(n, new Float64Array(p.data.length)); this.v.set(n, new Float64Array(p.data.length)); }
  }
  /** Applies one Adam update in place. Returns the delta added to each parameter (empty map when wantDeltas is false). */
  step(grads: Map<string, Tensor>, wantDeltas = true): Map<string, Float64Array> {
    this.t++;
    const { lr, beta1, beta2, eps } = this.opts;
    const c1 = 1 - Math.pow(beta1, this.t), c2 = 1 - Math.pow(beta2, this.t);
    const deltas = new Map<string, Float64Array>();
    for (const [n, p] of this.params) {
      const g = grads.get(n)?.data; const m = this.m.get(n)!, v = this.v.get(n)!;
      const d = wantDeltas ? new Float64Array(p.data.length) : null;
      if (g) for (let i = 0; i < p.data.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        const step = -lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        if (d) d[i] = step;
        p.data[i] += step;
      }
      if (d) deltas.set(n, d);
    }
    return deltas;
  }
}
