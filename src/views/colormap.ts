/** Colour maps for tensor tiles: diverging (signed values), sequential (0..1), gradientMap (signed deltas). */

export type RGB = [number, number, number];

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));
const mix = (a: RGB, b: RGB, t: number): RGB => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

const NEAR_WHITE: RGB = [247, 247, 247];
const BLUE: RGB = [33, 102, 172];
const RED: RGB = [178, 24, 43];

/** Blue (negative) - near-white (0) - red (positive). Clamps to [-1, 1] via maxAbs; maxAbs 0 -> white. */
export function diverging(v: number, maxAbs: number): RGB {
  const t = maxAbs === 0 ? 0 : clamp(v / maxAbs, -1, 1);
  return t < 0 ? mix(NEAR_WHITE, BLUE, -t) : mix(NEAR_WHITE, RED, t);
}

const WHITE: RGB = [255, 255, 255];
const DARK_TEAL: RGB = [1, 102, 94];

/** White (0) to dark teal (1), for probabilities and attention weights. Clamps to [0, 1]. */
export function sequential(v: number): RGB {
  const t = clamp(v, 0, 1);
  return mix(WHITE, DARK_TEAL, t);
}

const PURPLE: RGB = [118, 42, 131];
const GREEN: RGB = [27, 120, 55];

/** Purple (negative) - near-white (0) - green (positive), for gradients and deltas. */
export function gradientMap(v: number, maxAbs: number): RGB {
  const t = maxAbs === 0 ? 0 : clamp(v / maxAbs, -1, 1);
  return t < 0 ? mix(NEAR_WHITE, PURPLE, -t) : mix(NEAR_WHITE, GREEN, t);
}

/** Mid grey for values <= -1e8 (the sentinel for masked-out cells). */
export const MASK_COLOR: RGB = [160, 160, 160];

export function maxAbs(data: Float64Array): number {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (a > m) m = a;
  }
  return m;
}

/** Values <= -1e8 render as "-inf" and are drawn with MASK_COLOR (see Global Constraints). */
export function isMasked(v: number): boolean {
  return v <= -1e8;
}
