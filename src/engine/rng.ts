/** mulberry32: small, fast, seedable. Good enough for init and sampling. */
export class Rng {
  private s: number;
  constructor(seed: number) { this.s = seed >>> 0; }
  /** Uniform in [0, 1). */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** Standard normal via Box-Muller. */
  randn(): number {
    let u = 0; while (u === 0) u = this.next();
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  /** Integer in [0, n). */
  int(n: number): number { return Math.floor(this.next() * n); }
}
