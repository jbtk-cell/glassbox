import { describe, it, expect } from 'vitest';
import { Rng } from './rng';

describe('Rng', () => {
  it('is deterministic for a seed', () => {
    const a = new Rng(42), b = new Rng(42);
    for (let i = 0; i < 5; i++) expect(a.next()).toBe(b.next());
  });
  it('produces values in [0, 1)', () => {
    const r = new Rng(1);
    for (let i = 0; i < 1000; i++) { const x = r.next(); expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThan(1); }
  });
  it('randn has roughly zero mean and unit variance', () => {
    const r = new Rng(7); let s = 0, s2 = 0; const n = 20000;
    for (let i = 0; i < n; i++) { const x = r.randn(); s += x; s2 += x * x; }
    expect(Math.abs(s / n)).toBeLessThan(0.03);
    expect(Math.abs(s2 / n - 1)).toBeLessThan(0.05);
  });
  it('int(n) is in [0, n)', () => {
    const r = new Rng(3);
    for (let i = 0; i < 100; i++) { const k = r.int(5); expect(k).toBeGreaterThanOrEqual(0); expect(k).toBeLessThan(5); expect(Number.isInteger(k)).toBe(true); }
  });
});
