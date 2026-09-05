import { describe, it, expect } from 'vitest';
import { toScreen, toWorld, zoomAt, fit, type Camera } from './camera';
import type { TileRect } from './tiles';

describe('toScreen / toWorld', () => {
  it('round-trips through toWorld(toScreen(p))', () => {
    const c: Camera = { x: 12, y: -7, zoom: 2.5 };
    const [sx, sy] = toScreen(c, 40, 30);
    const [wx, wy] = toWorld(c, sx, sy);
    expect(wx).toBeCloseTo(40, 9);
    expect(wy).toBeCloseTo(30, 9);
  });
});

describe('zoomAt', () => {
  it('keeps the world point under the cursor fixed', () => {
    const c: Camera = { x: 5, y: 5, zoom: 1 };
    const sx = 120, sy = 80;
    const [wx, wy] = toWorld(c, sx, sy);
    const zoomed = zoomAt(c, sx, sy, 1.5);
    const [wx2, wy2] = toWorld(zoomed, sx, sy);
    expect(Math.abs(wx2 - wx)).toBeLessThan(1e-9);
    expect(Math.abs(wy2 - wy)).toBeLessThan(1e-9);
    expect(zoomed.zoom).toBeCloseTo(1.5, 9);
  });

  it('clamps to min/max', () => {
    const c: Camera = { x: 0, y: 0, zoom: 1 };
    expect(zoomAt(c, 0, 0, 0.0001, 0.05, 40).zoom).toBe(0.05);
    expect(zoomAt(c, 0, 0, 10000, 0.05, 40).zoom).toBe(40);
  });
});

describe('fit', () => {
  it('places the bounds fully inside the viewport with margin', () => {
    const bounds: TileRect = { x: 10, y: 20, w: 300, h: 150 };
    const viewport = { w: 800, h: 600 };
    const margin = 40;
    const camera = fit(bounds, viewport, margin);

    const corners: [number, number][] = [
      [bounds.x, bounds.y],
      [bounds.x + bounds.w, bounds.y],
      [bounds.x, bounds.y + bounds.h],
      [bounds.x + bounds.w, bounds.y + bounds.h],
    ];
    for (const [wx, wy] of corners) {
      const [sx, sy] = toScreen(camera, wx, wy);
      expect(sx).toBeGreaterThanOrEqual(margin - 1e-6);
      expect(sx).toBeLessThanOrEqual(viewport.w - margin + 1e-6);
      expect(sy).toBeGreaterThanOrEqual(margin - 1e-6);
      expect(sy).toBeLessThanOrEqual(viewport.h - margin + 1e-6);
    }
  });

  it('centres the bounds in the viewport', () => {
    const bounds: TileRect = { x: 0, y: 0, w: 100, h: 100 };
    const viewport = { w: 400, h: 400 };
    const camera = fit(bounds, viewport, 0);
    const [sx, sy] = toScreen(camera, 50, 50); // centre of bounds
    expect(sx).toBeCloseTo(200, 9);
    expect(sy).toBeCloseTo(200, 9);
  });
});
