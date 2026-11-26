/** Pan/zoom camera: world point (x, y) maps to screen ((wx - x) * zoom, (wy - y) * zoom). */
import type { TileRect } from './tiles';

export interface Camera { x: number; y: number; zoom: number }

/** How far the views can zoom. A fitted diagram sits near 0.5. At 80 one ordinary cell of the map is 320 px wide
 *  and a row of the word table is over 40 px, so every number is readable; any closer and one cell fills the view. */
export const ZOOM_MIN = 0.05;
export const ZOOM_MAX = 80;

export function toScreen(c: Camera, wx: number, wy: number): [number, number] {
  return [(wx - c.x) * c.zoom, (wy - c.y) * c.zoom];
}

export function toWorld(c: Camera, sx: number, sy: number): [number, number] {
  return [sx / c.zoom + c.x, sy / c.zoom + c.y];
}

/** Zoom by `factor`, keeping the world point under (sx, sy) fixed on screen. */
export function zoomAt(c: Camera, sx: number, sy: number, factor: number, min = ZOOM_MIN, max = ZOOM_MAX): Camera {
  const zoom = Math.min(max, Math.max(min, c.zoom * factor));
  const [wx, wy] = toWorld(c, sx, sy);
  return { x: wx - sx / zoom, y: wy - sy / zoom, zoom };
}

/** Zoom factor for one wheel event. Scrolling up zooms in. A trackpad pinch arrives as a wheel event
 *  with ctrlKey set and small deltas, so it gets a stronger rate. The shape follows d3-zoom's wheelDelta,
 *  with the plain-scroll rate doubled: one mouse-wheel notch (100 px) is about 32%. */
export function wheelFactor(e: { deltaY: number; deltaMode: number; ctrlKey: boolean }): number {
  const unit = e.deltaMode === 1 ? 0.4 / 3 : e.deltaMode ? 2 : 0.004;   // a notch is 100 px, or 3 lines in Firefox
  return Math.pow(2, -e.deltaY * unit * (e.ctrlKey ? 5 : 1));
}

type Pt = readonly [number, number];

/** Two fingers moved from (a0, b0) to (a1, b1) in screen pixels: zoom by the change in their spread
 *  and pan by the change in their midpoint, so what was under the fingers stays under them. */
export function pinchMove(c: Camera, a0: Pt, b0: Pt, a1: Pt, b1: Pt, min = ZOOM_MIN, max = ZOOM_MAX): Camera {
  const d0 = Math.hypot(b0[0] - a0[0], b0[1] - a0[1]), d1 = Math.hypot(b1[0] - a1[0], b1[1] - a1[1]);
  const m0x = (a0[0] + b0[0]) / 2, m0y = (a0[1] + b0[1]) / 2, m1x = (a1[0] + b1[0]) / 2, m1y = (a1[1] + b1[1]) / 2;
  const zoomed = d0 > 0 && d1 > 0 ? zoomAt(c, m0x, m0y, d1 / d0, min, max) : c;
  return pan(zoomed, m1x - m0x, m1y - m0y);
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
