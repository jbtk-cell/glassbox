/** Pan/zoom camera: world point (x, y) maps to screen ((wx - x) * zoom, (wy - y) * zoom). */
import type { TileRect } from './tiles';

export interface Camera { x: number; y: number; zoom: number }

export function toScreen(c: Camera, wx: number, wy: number): [number, number] {
  return [(wx - c.x) * c.zoom, (wy - c.y) * c.zoom];
}

export function toWorld(c: Camera, sx: number, sy: number): [number, number] {
  return [sx / c.zoom + c.x, sy / c.zoom + c.y];
}

/** Zoom by `factor`, keeping the world point under (sx, sy) fixed on screen. */
export function zoomAt(c: Camera, sx: number, sy: number, factor: number, min = 0.05, max = 40): Camera {
  const zoom = Math.min(max, Math.max(min, c.zoom * factor));
  const [wx, wy] = toWorld(c, sx, sy);
  return { x: wx - sx / zoom, y: wy - sy / zoom, zoom };
}

/** Pan by (dx, dy) screen pixels. */
export function pan(c: Camera, dx: number, dy: number): Camera {
  return { x: c.x - dx / c.zoom, y: c.y - dy / c.zoom, zoom: c.zoom };
}

/** A camera that fits `bounds` inside `viewport`, centred, with `margin` screen pixels to spare. */
export function fit(bounds: TileRect, viewport: { w: number; h: number }, margin = 40): Camera {
  const zoom = Math.min((viewport.w - 2 * margin) / bounds.w, (viewport.h - 2 * margin) / bounds.h);
  const cx = bounds.x + bounds.w / 2;
  const cy = bounds.y + bounds.h / 2;
  return { x: cx - viewport.w / 2 / zoom, y: cy - viewport.h / 2 / zoom, zoom };
}
