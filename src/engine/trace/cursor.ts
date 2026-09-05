import { Trace, TraceStep } from './trace';

export type Status = 'pending' | 'partial' | 'active' | 'done';

export class Cursor {
  index = -1;
  private writers = new Map<string, number[]>();

  constructor(readonly trace: Trace) {
    for (const s of trace.steps) for (const k of s.writes) {
      const w = this.writers.get(k); if (w) w.push(s.index); else this.writers.set(k, [s.index]);
    }
  }

  get length(): number { return this.trace.steps.length; }
  current(): TraceStep | null { return this.index >= 0 ? this.trace.steps[this.index] : null; }

  status(key: string): Status {
    const cur = this.current();
    if (cur && cur.writes.includes(key)) return 'active';
    const ws = this.writers.get(key) ?? [];
    const done = ws.filter(i => i <= this.index).length;
    if (done === 0) return this.trace.preexisting.has(key) ? 'done' : 'pending';
    return done === ws.length ? 'done' : 'partial';
  }

  next(): void { if (this.index < this.length - 1) this.index++; }
  prev(): void { if (this.index > -1) this.index--; }
  seek(i: number): void { this.index = Math.max(-1, Math.min(this.length - 1, Math.floor(i))); }
  toStart(): void { this.index = -1; }
  toEnd(): void { this.index = this.length - 1; }
  atStart(): boolean { return this.index === -1; }
  atEnd(): boolean { return this.index === this.length - 1; }
}
