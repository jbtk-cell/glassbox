import { Rng } from '../rng';

export function splitTokens(tokens: number[], ratio = 0.6): { train: number[]; test: number[] } {
  const cut = Math.floor(tokens.length * ratio);
  return { train: tokens.slice(0, cut), test: tokens.slice(cut) };
}

export interface Window { input: number[]; target: number[] }

/** Every contiguous window of T tokens; target is the same window shifted one token later. */
export function windows(tokens: number[], T: number): Window[] {
  const out: Window[] = [];
  for (let i = 0; i + T < tokens.length; i++) {
    out.push({ input: tokens.slice(i, i + T), target: tokens.slice(i + 1, i + T + 1) });
  }
  return out;
}

export function shuffled<T>(items: T[], rng: Rng): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) { const j = rng.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

/** Fewer tokens than this and there is no meaningful train/test split. */
export function minTokens(T: number): number { return 2 * T + 2; }
