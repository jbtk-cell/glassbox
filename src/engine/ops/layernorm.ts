import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** Per-row layer norm over the last dimension: y = gamma * (x - mean) / sqrt(var + eps) + beta. */
export function layerNorm(meta: OpMeta, x: string, gamma: string, beta: string, out: string, eps = 1e-5): Op {
  return {
    ...meta, kind: 'layernorm', inputs: [x], output: out, params: [gamma, beta],
    forward(ctx: Ctx) {
      const X = get(ctx, x), G = get(ctx, gamma).data, Bt = get(ctx, beta).data;
      const [T, D] = X.shape;
      const Y = put(ctx, out, [T, D]);
      for (let t = 0; t < T; t++) {
        let m = 0; for (let d = 0; d < D; d++) m += X.data[t * D + d]; m /= D;
        let v = 0; for (let d = 0; d < D; d++) v += (X.data[t * D + d] - m) ** 2; v /= D;
        const inv = 1 / Math.sqrt(v + eps);
        for (let d = 0; d < D; d++) Y.data[t * D + d] = G[d] * (X.data[t * D + d] - m) * inv + Bt[d];
      }
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x), G = get(ctx, gamma).data;
      const [T, D] = X.shape;
      const dY = gradGet(ctx, out).data;
      const dX = gradOf(ctx, x).data, dG = gradOf(ctx, gamma).data, dB = gradOf(ctx, beta).data;
      const xhat = new Float64Array(D), dxhat = new Float64Array(D);
      for (let t = 0; t < T; t++) {
        let m = 0; for (let d = 0; d < D; d++) m += X.data[t * D + d]; m /= D;
        let v = 0; for (let d = 0; d < D; d++) v += (X.data[t * D + d] - m) ** 2; v /= D;
        const inv = 1 / Math.sqrt(v + eps);
        let mean_dxhat = 0, mean_dxhat_xhat = 0;
        for (let d = 0; d < D; d++) {
          xhat[d] = (X.data[t * D + d] - m) * inv;
          const g = dY[t * D + d];
          dG[d] += g * xhat[d];
          dB[d] += g;
          dxhat[d] = g * G[d];
          mean_dxhat += dxhat[d] / D;
          mean_dxhat_xhat += dxhat[d] * xhat[d] / D;
        }
        for (let d = 0; d < D; d++) dX[t * D + d] += inv * (dxhat[d] - mean_dxhat - xhat[d] * mean_dxhat_xhat);
      }
    },
  };
}
