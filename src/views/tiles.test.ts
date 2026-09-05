import { describe, it, expect } from 'vitest';
import { cellSize, tileRect, cellAt } from './tiles';

describe('cellSize', () => {
  it('fits the longer side under maxSide, capped at 12', () => {
    expect(cellSize([2, 3], 240)).toBe(12);
  });
  it('shrinks for large tensors', () => {
    expect(cellSize([100, 50], 240)).toBe(2);
  });
  it('never goes below 1', () => {
    expect(cellSize([300, 10], 240)).toBe(1);
  });
});

describe('tileRect', () => {
  it('sizes w/h from cols/rows and the cell size', () => {
    expect(tileRect(0, 0, [2, 3], 10)).toEqual({ x: 0, y: 0, w: 30, h: 20 });
  });
  it('treats a 1-D shape as 1 row of n columns', () => {
    expect(tileRect(5, 7, [5], 10)).toEqual({ x: 5, y: 7, w: 50, h: 10 });
  });
});

describe('cellAt', () => {
  const rect = tileRect(0, 0, [2, 3], 10); // 2 rows x 3 cols, {x:0,y:0,w:30,h:20}
  const shape = [2, 3];

  it('returns the right flat index for several points', () => {
    expect(cellAt(rect, shape, 5, 5)).toBe(0);
    expect(cellAt(rect, shape, 15, 5)).toBe(1);
    expect(cellAt(rect, shape, 25, 5)).toBe(2);
    expect(cellAt(rect, shape, 5, 15)).toBe(3);
    expect(cellAt(rect, shape, 15, 15)).toBe(4);
    expect(cellAt(rect, shape, 29, 19)).toBe(5);
  });

  it('returns null outside the rect', () => {
    expect(cellAt(rect, shape, -1, 5)).toBeNull();
    expect(cellAt(rect, shape, 35, 5)).toBeNull();
    expect(cellAt(rect, shape, 5, -1)).toBeNull();
    expect(cellAt(rect, shape, 5, 25)).toBeNull();
  });
});
