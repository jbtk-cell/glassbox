/** Tensor tiles: world-space rects for tensors, hit-testing, and a cache of tile images. */
import { diverging, sequential, gradientMap, robustMax, isMasked, MASK_COLOR, type RGB } from './colormap';

export type TileMode = 'value' | 'prob' | 'grad';

export interface TileRect { x: number; y: number; w: number; h: number }

/** A 1-D shape [n] is treated as 1 row of n columns. */
function rowsCols(shape: number[]): [number, number] {
  if (shape.length <= 1) return [1, shape[0] ?? 0];
  return [shape[0], shape[1]];
}

/** World units per cell so the longer side of the tile is <= maxSide, min 1, capped at 12
 *  so small tensors do not become huge. */
export function cellSize(shape: number[], maxSide = 240): number {
  const [rows, cols] = rowsCols(shape);
  const side = Math.max(1, rows, cols);
  return Math.min(12, Math.max(1, Math.floor(maxSide / side)));
}

export function tileRect(x: number, y: number, shape: number[], cs: number): TileRect {
  const [rows, cols] = rowsCols(shape);
  return { x, y, w: cols * cs, h: rows * cs };
}

/** Flat row-major index of the cell under a world point, or null when outside the rect. */
export function cellAt(rect: TileRect, shape: number[], wx: number, wy: number): number | null {
  const [rows, cols] = rowsCols(shape);
  const dx = wx - rect.x;
  const dy = wy - rect.y;
  if (dx < 0 || dy < 0 || dx >= rect.w || dy >= rect.h) return null;
  const col = Math.floor((dx / rect.w) * cols);
  const row = Math.floor((dy / rect.h) * rows);
  if (row < 0 || row >= rows || col < 0 || col >= cols) return null;
  return row * cols + col;
}

type Canvas2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;

function colorFor(name: string, mode: TileMode, v: number, m: number): RGB {
  if (isMasked(v)) return MASK_COLOR;
  if (mode === 'prob') return sequential(v);
  if (mode === 'grad') return gradientMap(v, m);
  return name === 'attn' || name === 'probs' ? sequential(v) : diverging(v, m);
}

/** Caches one canvas per (name, mode, version); regenerates when version changes. */
export class TileCache {
  private cache = new Map<string, CanvasImageSource>();

  get(name: string, shape: number[], data: Float64Array, mode: TileMode, version: number): CanvasImageSource {
    const key = `${name}|${mode}|${version}`;
    const hit = this.cache.get(key);
    if (hit) return hit;

    const [rows, cols] = rowsCols(shape);
    const canvas: HTMLCanvasElement | OffscreenCanvas =
      typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(cols, rows) : document.createElement('canvas');
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d') as Canvas2D;
    const img = ctx.createImageData(cols, rows);
    const m = robustMax(data, mode === 'grad' ? 0.95 : 0.99);
    for (let i = 0; i < rows * cols; i++) {
      const [r, g, b] = colorFor(name, mode, data[i], m);
      const o = i * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);

    this.cache.set(key, canvas);
    return canvas;
  }

  clear(): void {
    this.cache.clear();
  }
}
