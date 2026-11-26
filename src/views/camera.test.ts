import { describe, it, expect } from 'vitest';
import { toScreen, toWorld, zoomAt, fit, wheelFactor, pinchMove, ZOOM_MIN, ZOOM_MAX, type Camera } from './camera';
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

describe('zoom range', () => {
  it('zooms past the old limit of 40 by default, and a view can set its own limit', () => {
    const c: Camera = { x: 0, y: 0, zoom: 1 };
    expect(ZOOM_MAX).toBeGreaterThan(40);
    expect(pinchMove(c, [0, 0], [10, 0], [0, 0], [1e6, 0], 0.05, 40).zoom).toBe(40);
    expect(zoomAt(c, 0, 0, 1e9).zoom).toBe(ZOOM_MAX);
    expect(zoomAt(c, 0, 0, 1e-9).zoom).toBe(ZOOM_MIN);
  });
});

describe('wheelFactor', () => {
  it('zooms in when scrolling up and out when scrolling down', () => {
    expect(wheelFactor({ deltaY: -100, deltaMode: 0, ctrlKey: false })).toBeGreaterThan(1);
    expect(wheelFactor({ deltaY: 100, deltaMode: 0, ctrlKey: false })).toBeLessThan(1);
  });

  it('moves at least 30% per mouse-wheel notch (it used to be 10%)', () => {
    expect(wheelFactor({ deltaY: -100, deltaMode: 0, ctrlKey: false })).toBeGreaterThanOrEqual(1.3);
  });

  it('is symmetric: the same scroll up then down returns to the start', () => {
    const up = wheelFactor({ deltaY: -37, deltaMode: 0, ctrlKey: false });
    const down = wheelFactor({ deltaY: 37, deltaMode: 0, ctrlKey: false });
    expect(up * down).toBeCloseTo(1, 12);
  });

  it('treats a pinch (ctrlKey wheel, small deltas) as a strong zoom', () => {
    const pinch = wheelFactor({ deltaY: -10, deltaMode: 0, ctrlKey: true });
    const scroll = wheelFactor({ deltaY: -10, deltaMode: 0, ctrlKey: false });
    expect(pinch).toBeGreaterThan(scroll);
    expect(pinch).toBeGreaterThanOrEqual(1.1);
  });

  it('handles wheels that report lines instead of pixels', () => {
    expect(wheelFactor({ deltaY: -3, deltaMode: 1, ctrlKey: false })).toBeGreaterThanOrEqual(1.3);
  });
});

describe('pinchMove', () => {
  it('keeps the world points under both fingers under those fingers', () => {
    const c: Camera = { x: 30, y: -12, zoom: 0.8 };
    const a0: [number, number] = [100, 200], b0: [number, number] = [300, 260];
    const a1: [number, number] = [60, 150], b1: [number, number] = [420, 258];   // same angle, 1.8x apart
    const wa = toWorld(c, ...a0), wb = toWorld(c, ...b0);
    const next = pinchMove(c, a0, b0, a1, b1);
    const sa = toScreen(next, ...wa), sb = toScreen(next, ...wb);
    expect(sa[0]).toBeCloseTo(a1[0], 6); expect(sa[1]).toBeCloseTo(a1[1], 6);
    expect(sb[0]).toBeCloseTo(b1[0], 6); expect(sb[1]).toBeCloseTo(b1[1], 6);
    expect(next.zoom).toBeCloseTo(0.8 * 1.8, 9);
  });

  it('does nothing harmful when the two fingers start on the same pixel', () => {
    const c: Camera = { x: 0, y: 0, zoom: 1 };
    const next = pinchMove(c, [50, 50], [50, 50], [40, 40], [80, 80]);
    expect(Number.isFinite(next.zoom)).toBe(true);
    expect(next.zoom).toBe(1);
  });
});
