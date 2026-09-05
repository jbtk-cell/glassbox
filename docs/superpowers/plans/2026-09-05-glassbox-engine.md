# glassbox Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-TypeScript, fully gradient-checked tiny GPT engine — tokenizer, op list, model, Adam, trace recorder, cursor, and sampler — with no UI, that Plan 2 (the app) renders.

**Architecture:** The forward pass is an ordered list of named `Op` objects, each with its own `forward` and `backward`, reading and writing named tensors in a `Ctx`. Everything is computed eagerly and recorded; stepping is replay over the recorded steps. All numerics are `Float64Array`, row-major.

**Tech Stack:** TypeScript 5.9, Vitest 3 (already installed; `npx vitest run <file>`), no runtime dependencies in the engine.

## Global Constraints

- Numeric type: `Float64Array` everywhere. Row-major: element (i, j) of an R×C tensor is `data[i * C + j]`.
- Randomness: seeded mulberry32 only; never `Math.random` in the engine.
- Tokens: word-level, lowercase, punctuation `. , ? ! ; : " ' ( )` split off, `\n` is a token, contractions stay whole, vocabulary capped at 512, index 0 is `<unk>`.
- Defaults: contextSize 24, dModel 20, dFF 30, Adam β1 0.9 β2 0.999 ε 1e-8, learning rate 3e-3, init N(0, 0.02).
- Gradient checks: central finite differences, ε = 1e-6, pass if `|analytic − numeric| ≤ 1e-5 · max(1, |numeric|)`.
- No emojis anywhere. Plain prose in `explain` strings.
- Engine code lives under `src/engine/`; tests sit next to the file they test as `*.test.ts`.
- Commit after every task with the trailer line `Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC`.

---

## File structure

```
src/engine/
  rng.ts                 seeded PRNG (mulberry32), randn
  tensor.ts              Tensor type and constructors
  corpus/tokenize.ts     tokenize, Vocab, detokenize
  corpus/windows.ts      split, windows
  ops/types.ts           Op, OpKind, Ctx, get/gradOf helpers
  ops/gradcheck.ts       numerical gradient checker (used by tests only)
  ops/elementwise.ts     add, relu
  ops/linear.ts          linear (x W + b)
  ops/layernorm.ts       layer norm
  ops/embed.ts           embed, posEmbed
  ops/attention.ts       scores, causalMask, softmaxRows, attnApply
  ops/output.ts          softmaxOut, crossEntropy
  model/params.ts        parameter names, shapes, init
  model/gpt.ts           GPT class: builds the op list, forward, backward
  model/adam.ts          Adam optimizer
  model/generate.ts      greedy / top-k / top-p sampling
  trace/trace.ts         Trace, recordForward, recordTrainingStep
  trace/cursor.ts        Cursor
```

---

### Task 1: Seeded RNG and Tensor type

**Files:**
- Create: `src/engine/rng.ts`
- Create: `src/engine/tensor.ts`
- Test: `src/engine/rng.test.ts`, `src/engine/tensor.test.ts`

**Interfaces:**
- Produces: `class Rng { constructor(seed: number); next(): number; randn(): number; int(n): number }`
- Produces: `interface Tensor { name: string; shape: number[]; data: Float64Array }`, `tensor(name, shape, data?)`, `size(shape)`, `zerosLike(t, name)`

- [ ] **Step 1: Write the failing tests**

`src/engine/rng.test.ts`:
```ts
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
```

`src/engine/tensor.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { tensor, size, zerosLike } from './tensor';

describe('tensor', () => {
  it('size multiplies the shape', () => { expect(size([3, 4])).toBe(12); expect(size([])).toBe(1); });
  it('tensor() allocates zeros of the right length', () => {
    const t = tensor('x', [2, 3]);
    expect(t.data.length).toBe(6); expect(Array.from(t.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });
  it('tensor() rejects mismatched data', () => { expect(() => tensor('x', [2], new Float64Array(3))).toThrow(); });
  it('zerosLike copies shape, not data', () => {
    const t = tensor('x', [2], new Float64Array([1, 2]));
    const z = zerosLike(t, 'z');
    expect(z.shape).toEqual([2]); expect(Array.from(z.data)).toEqual([0, 0]); expect(z.name).toBe('z');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/rng.test.ts src/engine/tensor.test.ts`
Expected: FAIL — cannot resolve `./rng` / `./tensor`.

- [ ] **Step 3: Implement**

`src/engine/rng.ts`:
```ts
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
```

`src/engine/tensor.ts`:
```ts
export interface Tensor { name: string; shape: number[]; data: Float64Array }

export function size(shape: number[]): number { return shape.reduce((a, b) => a * b, 1); }

export function tensor(name: string, shape: number[], data?: Float64Array): Tensor {
  const n = size(shape);
  if (data && data.length !== n) throw new Error(`tensor ${name}: data length ${data.length} != shape size ${n}`);
  return { name, shape: [...shape], data: data ?? new Float64Array(n) };
}

export function zerosLike(t: Tensor, name: string): Tensor { return tensor(name, t.shape); }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/engine/rng.test.ts src/engine/tensor.test.ts`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/rng.ts src/engine/rng.test.ts src/engine/tensor.ts src/engine/tensor.test.ts
git commit -m "engine: seeded rng and tensor type

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 2: Tokenizer

**Files:**
- Create: `src/engine/corpus/tokenize.ts`
- Test: `src/engine/corpus/tokenize.test.ts`

**Interfaces:**
- Produces:
  ```ts
  interface Vocab { words: string[]; index: Map<string, number>; folded: number /* word types that became <unk> */ }
  const UNK = 0;
  function splitWords(text: string): string[]           // lowercase words + punctuation + '\n' tokens
  function buildVocab(words: string[], cap = 512): Vocab // by frequency desc, ties alphabetical; words[0] === '<unk>'
  function encode(words: string[], vocab: Vocab): number[]
  function tokenize(text: string, cap = 512): { tokens: number[]; vocab: Vocab; words: string[] }
  function detokenize(tokens: number[], vocab: Vocab): string  // joins with spaces, no space before punctuation, '\n' as newline
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/corpus/tokenize.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { splitWords, buildVocab, encode, tokenize, detokenize, UNK } from './tokenize';

describe('splitWords', () => {
  it('lowercases and splits on whitespace', () => {
    expect(splitWords('Hi What Is up')).toEqual(['hi', 'what', 'is', 'up']);
  });
  it('splits trailing and leading punctuation into their own tokens', () => {
    expect(splitWords("What's up? Nothing, really.")).toEqual(["what's", 'up', '?', 'nothing', ',', 'really', '.']);
    expect(splitWords('"quoted" (aside)')).toEqual(['"', 'quoted', '"', '(', 'aside', ')']);
  });
  it('keeps contractions whole', () => { expect(splitWords("don't you're")).toEqual(["don't", "you're"]); });
  it('emits newline as a token and collapses blank lines', () => {
    expect(splitWords('a b\nc\n\n\nd')).toEqual(['a', 'b', '\n', 'c', '\n', 'd']);
  });
  it('ignores leading/trailing whitespace', () => { expect(splitWords('  x  ')).toEqual(['x']); });
});

describe('buildVocab', () => {
  it('orders by frequency then alphabetically, with <unk> first', () => {
    const v = buildVocab(['b', 'a', 'b', 'c', 'a', 'b']);
    expect(v.words).toEqual(['<unk>', 'b', 'a', 'c']);
    expect(v.index.get('b')).toBe(1);
    expect(v.folded).toBe(0);
  });
  it('caps and counts folded word types', () => {
    const v = buildVocab(['a', 'a', 'b', 'c', 'd'], 3);
    expect(v.words).toEqual(['<unk>', 'a', 'b']);
    expect(v.folded).toBe(2);
  });
});

describe('encode / tokenize / detokenize', () => {
  it('maps unknown words to UNK', () => {
    const v = buildVocab(['a', 'b']);
    expect(encode(['a', 'zzz', 'b'], v)).toEqual([1, UNK, 2]);
  });
  it('round-trips simple text', () => {
    const { tokens, vocab } = tokenize("hi, what's up?\nnot much.");
    expect(detokenize(tokens, vocab)).toBe("hi, what's up?\nnot much.");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/corpus/tokenize.test.ts`
Expected: FAIL — cannot resolve `./tokenize`.

- [ ] **Step 3: Implement**

`src/engine/corpus/tokenize.ts`:
```ts
export interface Vocab { words: string[]; index: Map<string, number>; folded: number }
export const UNK = 0;

const PUNCT = new Set(['.', ',', '?', '!', ';', ':', '"', "'", '(', ')']);

/** Lowercase words, punctuation as its own tokens, '\n' as a token. Contractions stay whole. */
export function splitWords(text: string): string[] {
  const out: string[] = [];
  const lines = text.toLowerCase().split('\n');
  let pendingNewline = false;
  for (const line of lines) {
    const raw = line.trim().split(/\s+/).filter(Boolean);
    if (raw.length === 0) { continue; }
    if (pendingNewline) out.push('\n');
    for (const w of raw) pushWord(w, out);
    pendingNewline = true;
  }
  return out;
}

function pushWord(w: string, out: string[]) {
  // Peel leading punctuation.
  let start = 0;
  while (start < w.length && PUNCT.has(w[start])) start++;
  // Peel trailing punctuation.
  let end = w.length;
  while (end > start && PUNCT.has(w[end - 1])) end--;
  for (let i = 0; i < start; i++) out.push(w[i]);
  if (end > start) out.push(w.slice(start, end));
  for (let i = end; i < w.length; i++) out.push(w[i]);
}

export function buildVocab(words: string[], cap = 512): Vocab {
  const counts = new Map<string, number>();
  for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(e => e[0]);
  const kept = ranked.slice(0, Math.max(0, cap - 1));
  const list = ['<unk>', ...kept];
  const index = new Map(list.map((w, i) => [w, i] as const));
  return { words: list, index, folded: ranked.length - kept.length };
}

export function encode(words: string[], vocab: Vocab): number[] {
  return words.map(w => vocab.index.get(w) ?? UNK);
}

export function tokenize(text: string, cap = 512): { tokens: number[]; vocab: Vocab; words: string[] } {
  const words = splitWords(text);
  const vocab = buildVocab(words, cap);
  return { tokens: encode(words, vocab), vocab, words };
}

const NO_SPACE_BEFORE = new Set(['.', ',', '?', '!', ';', ':', ')']);
const NO_SPACE_AFTER = new Set(['(']);

export function detokenize(tokens: number[], vocab: Vocab): string {
  let s = '';
  let prev = '';
  for (const t of tokens) {
    const w = vocab.words[t] ?? '<unk>';
    if (w === '\n') { s += '\n'; prev = w; continue; }
    const glue = s.length === 0 || prev === '\n' || NO_SPACE_BEFORE.has(w) || NO_SPACE_AFTER.has(prev) ? '' : ' ';
    s += glue + w;
    prev = w;
  }
  return s;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/corpus/tokenize.test.ts`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/corpus/tokenize.ts src/engine/corpus/tokenize.test.ts
git commit -m "engine: word-level tokenizer with vocab cap

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 3: Train/test split and training windows

**Files:**
- Create: `src/engine/corpus/windows.ts`
- Test: `src/engine/corpus/windows.test.ts`

**Interfaces:**
- Consumes: `Rng` from Task 1.
- Produces:
  ```ts
  function splitTokens(tokens: number[], ratio = 0.6): { train: number[]; test: number[] }
  interface Window { input: number[]; target: number[] }   // both length T
  function windows(tokens: number[], T: number): Window[]  // all contiguous windows, stride 1
  function shuffled<T>(items: T[], rng: Rng): T[]           // Fisher-Yates copy
  function minTokens(T: number): number                      // 2*T + 2
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/corpus/windows.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { splitTokens, windows, shuffled, minTokens } from './windows';
import { Rng } from '../rng';

describe('splitTokens', () => {
  it('splits contiguously at the ratio', () => {
    const { train, test } = splitTokens([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.6);
    expect(train).toEqual([0, 1, 2, 3, 4, 5]);
    expect(test).toEqual([6, 7, 8, 9]);
  });
});

describe('windows', () => {
  it('produces every stride-1 window with targets shifted by one', () => {
    const w = windows([10, 11, 12, 13, 14], 3);
    expect(w).toEqual([
      { input: [10, 11, 12], target: [11, 12, 13] },
      { input: [11, 12, 13], target: [12, 13, 14] },
    ]);
  });
  it('returns nothing when the stream is too short', () => { expect(windows([1, 2, 3], 3)).toEqual([]); });
});

describe('shuffled', () => {
  it('is a permutation and deterministic for a seed', () => {
    const items = [1, 2, 3, 4, 5, 6, 7, 8];
    const a = shuffled(items, new Rng(9)), b = shuffled(items, new Rng(9));
    expect(a).toEqual(b);
    expect([...a].sort((x, y) => x - y)).toEqual(items);
    expect(items).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });
});

it('minTokens is 2T+2', () => { expect(minTokens(24)).toBe(50); });
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/corpus/windows.test.ts`
Expected: FAIL — cannot resolve `./windows`.

- [ ] **Step 3: Implement**

`src/engine/corpus/windows.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/corpus/windows.test.ts`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/corpus/windows.ts src/engine/corpus/windows.test.ts
git commit -m "engine: train/test split and training windows

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 4: Op contract, gradient checker, and elementwise ops (add, relu)

**Files:**
- Create: `src/engine/ops/types.ts`
- Create: `src/engine/ops/gradcheck.ts`
- Create: `src/engine/ops/elementwise.ts`
- Test: `src/engine/ops/elementwise.test.ts`

**Interfaces:**
- Consumes: `Tensor`, `tensor` from Task 1.
- Produces (`types.ts`):
  ```ts
  type OpKind = 'embed' | 'pos_embed' | 'add' | 'layernorm' | 'linear' | 'scores' | 'mask' | 'softmax_rows' | 'attn_apply' | 'relu' | 'softmax_out' | 'loss';
  interface Ctx { tensors: Map<string, Tensor>; grads: Map<string, Tensor>; tokens: number[]; targets?: number[]; T: number }
  interface OpMeta { id: string; label: string; formula: string; explain: string }
  interface Op extends OpMeta { kind: OpKind; inputs: string[]; output: string; params: string[]; forward(ctx: Ctx): void; backward(ctx: Ctx): void }
  function newCtx(tokens: number[], params: Map<string, Tensor>, targets?: number[]): Ctx
  function get(ctx: Ctx, name: string): Tensor          // throws if absent
  function put(ctx: Ctx, name: string, shape: number[]): Tensor  // (re)allocates zeros, stores, returns
  function gradOf(ctx: Ctx, name: string): Tensor       // existing grad, or new zeros shaped like the tensor
  function gradGet(ctx: Ctx, name: string): Tensor      // throws if absent
  ```
  **Backward convention (every op obeys this):** read `gradGet(ctx, output)`; ACCUMULATE (`+=`) into `gradOf(ctx, input)` and `gradOf(ctx, param)`. Never assign. A tensor feeding two ops must receive both contributions.
- Produces (`gradcheck.ts`, test-only helper):
  ```ts
  interface GradFailure { name: string; index: number; analytic: number; numeric: number }
  function gradCheck(op: Op, ctx: Ctx, names: string[], opts?: { eps?: number; tol?: number; seed?: number }): GradFailure[]
  function randomCtx(shapes: Record<string, number[]>, seed: number, scale = 1): Ctx   // tensors ~ N(0, scale)
  ```
- Produces (`elementwise.ts`): `add(meta: OpMeta, a: string, b: string, out: string): Op`, `relu(meta: OpMeta, x: string, out: string): Op`

- [ ] **Step 1: Write the failing test**

`src/engine/ops/elementwise.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { add, relu } from './elementwise';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'op', label: 'op', formula: '', explain: '' };

describe('add', () => {
  it('adds elementwise', () => {
    const ctx = randomCtx({ a: [2, 2], b: [2, 2] }, 1);
    const op = add(meta, 'a', 'b', 'y');
    op.forward(ctx);
    const a = get(ctx, 'a').data, b = get(ctx, 'b').data, y = get(ctx, 'y').data;
    for (let i = 0; i < 4; i++) expect(y[i]).toBeCloseTo(a[i] + b[i], 12);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ a: [3, 4], b: [3, 4] }, 2);
    expect(gradCheck(add(meta, 'a', 'b', 'y'), ctx, ['a', 'b'])).toEqual([]);
  });
});

describe('relu', () => {
  it('clamps negatives to zero', () => {
    const ctx = randomCtx({ x: [1, 3] }, 3);
    get(ctx, 'x').data.set([-1, 0.5, 2]);
    relu(meta, 'x', 'y').forward(ctx);
    expect(Array.from(get(ctx, 'y').data)).toEqual([0, 0.5, 2]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ x: [3, 5] }, 4);
    expect(gradCheck(relu(meta, 'x', 'y'), ctx, ['x'])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/ops/elementwise.test.ts`
Expected: FAIL — cannot resolve modules.

- [ ] **Step 3: Implement**

`src/engine/ops/types.ts`:
```ts
import { Tensor, tensor } from '../tensor';

export type OpKind =
  | 'embed' | 'pos_embed' | 'add' | 'layernorm' | 'linear' | 'scores' | 'mask'
  | 'softmax_rows' | 'attn_apply' | 'relu' | 'softmax_out' | 'loss';

export interface Ctx {
  tensors: Map<string, Tensor>;
  grads: Map<string, Tensor>;
  tokens: number[];
  targets?: number[];
  /** Sequence length of this pass. */
  T: number;
}

export interface OpMeta { id: string; label: string; formula: string; explain: string }

export interface Op extends OpMeta {
  kind: OpKind;
  inputs: string[];
  output: string;
  params: string[];
  forward(ctx: Ctx): void;
  /** Reads grad of `output`; accumulates into grads of `inputs` and `params`. */
  backward(ctx: Ctx): void;
}

export function newCtx(tokens: number[], params: Map<string, Tensor>, targets?: number[]): Ctx {
  const tensors = new Map<string, Tensor>();
  for (const [k, v] of params) tensors.set(k, v);
  return { tensors, grads: new Map(), tokens, targets, T: tokens.length };
}

export function get(ctx: Ctx, name: string): Tensor {
  const t = ctx.tensors.get(name);
  if (!t) throw new Error(`tensor '${name}' not in ctx`);
  return t;
}

export function put(ctx: Ctx, name: string, shape: number[]): Tensor {
  const t = tensor(name, shape);
  ctx.tensors.set(name, t);
  return t;
}

export function gradOf(ctx: Ctx, name: string): Tensor {
  let g = ctx.grads.get(name);
  if (!g) { g = tensor(name, get(ctx, name).shape); ctx.grads.set(name, g); }
  return g;
}

export function gradGet(ctx: Ctx, name: string): Tensor {
  const g = ctx.grads.get(name);
  if (!g) throw new Error(`grad of '${name}' not in ctx`);
  return g;
}
```

`src/engine/ops/gradcheck.ts`:
```ts
import { Op, Ctx, get, gradGet } from './types';
import { tensor, size } from '../tensor';
import { Rng } from '../rng';

export interface GradFailure { name: string; index: number; analytic: number; numeric: number }

/** Tensors ~ N(0, scale). Params map is empty; everything goes in ctx.tensors directly. */
export function randomCtx(shapes: Record<string, number[]>, seed: number, scale = 1): Ctx {
  const rng = new Rng(seed);
  const tensors = new Map();
  for (const [name, shape] of Object.entries(shapes)) {
    const t = tensor(name, shape);
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.randn() * scale;
    tensors.set(name, t);
  }
  const T = shapes['__T'] ? shapes['__T'][0] : 0;
  tensors.delete('__T');
  return { tensors, grads: new Map(), tokens: [], T };
}

/**
 * Compare op.backward with central finite differences of L = sum(output * R), R fixed random.
 * Returns failures; an empty array means the op's gradient is correct for `names`.
 */
export function gradCheck(op: Op, ctx: Ctx, names: string[], opts: { eps?: number; tol?: number; seed?: number } = {}): GradFailure[] {
  const eps = opts.eps ?? 1e-6, tol = opts.tol ?? 1e-5;
  const rng = new Rng(opts.seed ?? 12345);
  op.forward(ctx);
  const out = get(ctx, op.output);
  const R = new Float64Array(size(out.shape));
  for (let i = 0; i < R.length; i++) R[i] = rng.randn();
  const objective = () => { op.forward(ctx); const o = get(ctx, op.output).data; let s = 0; for (let i = 0; i < o.length; i++) s += o[i] * R[i]; return s; };

  ctx.grads.set(op.output, tensor(op.output, out.shape, Float64Array.from(R)));
  for (const n of names) ctx.grads.delete(n);
  op.backward(ctx);

  const failures: GradFailure[] = [];
  for (const n of names) {
    const x = get(ctx, n).data;
    const g = gradGet(ctx, n).data;
    for (let i = 0; i < x.length; i++) {
      const v = x[i];
      x[i] = v + eps; const lp = objective();
      x[i] = v - eps; const lm = objective();
      x[i] = v;
      const numeric = (lp - lm) / (2 * eps);
      if (Math.abs(g[i] - numeric) > tol * Math.max(1, Math.abs(numeric))) failures.push({ name: n, index: i, analytic: g[i], numeric });
    }
  }
  op.forward(ctx);
  return failures;
}
```

`src/engine/ops/elementwise.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

export function add(meta: OpMeta, a: string, b: string, out: string): Op {
  return {
    ...meta, kind: 'add', inputs: [a, b], output: out, params: [],
    forward(ctx: Ctx) {
      const A = get(ctx, a), B = get(ctx, b);
      const Y = put(ctx, out, A.shape);
      for (let i = 0; i < Y.data.length; i++) Y.data[i] = A.data[i] + B.data[i];
    },
    backward(ctx: Ctx) {
      const dY = gradGet(ctx, out).data;
      const dA = gradOf(ctx, a).data, dB = gradOf(ctx, b).data;
      for (let i = 0; i < dY.length; i++) { dA[i] += dY[i]; dB[i] += dY[i]; }
    },
  };
}

export function relu(meta: OpMeta, x: string, out: string): Op {
  return {
    ...meta, kind: 'relu', inputs: [x], output: out, params: [],
    forward(ctx: Ctx) {
      const X = get(ctx, x);
      const Y = put(ctx, out, X.shape);
      for (let i = 0; i < Y.data.length; i++) Y.data[i] = X.data[i] > 0 ? X.data[i] : 0;
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x).data, dY = gradGet(ctx, out).data, dX = gradOf(ctx, x).data;
      for (let i = 0; i < dY.length; i++) if (X[i] > 0) dX[i] += dY[i];
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/ops/elementwise.test.ts`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/ops/types.ts src/engine/ops/gradcheck.ts src/engine/ops/elementwise.ts src/engine/ops/elementwise.test.ts
git commit -m "engine: op contract, gradient checker, add and relu ops

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 5: Linear and layer norm ops

**Files:**
- Create: `src/engine/ops/linear.ts`
- Create: `src/engine/ops/layernorm.ts`
- Test: `src/engine/ops/linear.test.ts`, `src/engine/ops/layernorm.test.ts`

**Interfaces:**
- Consumes: Task 4's `Op`, `OpMeta`, `Ctx`, `get`, `put`, `gradOf`, `gradGet`, `gradCheck`, `randomCtx`.
- Produces:
  ```ts
  function linear(meta: OpMeta, x: string, W: string, b: string, out: string): Op   // y = x W + b ; x: T×In, W: In×Out, b: [Out]
  function layerNorm(meta: OpMeta, x: string, gamma: string, beta: string, out: string, eps = 1e-5): Op  // per-row over last dim
  ```

- [ ] **Step 1: Write the failing tests**

`src/engine/ops/linear.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { linear } from './linear';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'lin', label: 'lin', formula: '', explain: '' };

describe('linear', () => {
  it('computes x W + b', () => {
    const ctx = randomCtx({ x: [2, 2], W: [2, 3], b: [3] }, 1);
    get(ctx, 'x').data.set([1, 2, 3, 4]);
    get(ctx, 'W').data.set([1, 0, 1, 0, 1, 1]);
    get(ctx, 'b').data.set([10, 20, 30]);
    linear(meta, 'x', 'W', 'b', 'y').forward(ctx);
    expect(Array.from(get(ctx, 'y').data)).toEqual([11, 22, 33, 13, 24, 37]);
    expect(get(ctx, 'y').shape).toEqual([2, 3]);
  });
  it('passes the gradient check for x, W and b', () => {
    const ctx = randomCtx({ x: [3, 4], W: [4, 5], b: [5] }, 2);
    expect(gradCheck(linear(meta, 'x', 'W', 'b', 'y'), ctx, ['x', 'W', 'b'])).toEqual([]);
  });
});
```

`src/engine/ops/layernorm.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { layerNorm } from './layernorm';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'ln', label: 'ln', formula: '', explain: '' };

describe('layerNorm', () => {
  it('normalises each row to mean 0, variance 1 when gamma=1, beta=0', () => {
    const ctx = randomCtx({ x: [3, 6], g: [6], b: [6] }, 1, 3);
    get(ctx, 'g').data.fill(1); get(ctx, 'b').data.fill(0);
    layerNorm(meta, 'x', 'g', 'b', 'y').forward(ctx);
    const y = get(ctx, 'y').data;
    for (let r = 0; r < 3; r++) {
      let m = 0, v = 0;
      for (let c = 0; c < 6; c++) m += y[r * 6 + c] / 6;
      for (let c = 0; c < 6; c++) v += (y[r * 6 + c] - m) ** 2 / 6;
      expect(m).toBeCloseTo(0, 10); expect(v).toBeCloseTo(1, 4);
    }
  });
  it('passes the gradient check for x, gamma and beta', () => {
    const ctx = randomCtx({ x: [3, 5], g: [5], b: [5] }, 2);
    expect(gradCheck(layerNorm(meta, 'x', 'g', 'b', 'y'), ctx, ['x', 'g', 'b'])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/engine/ops/linear.test.ts src/engine/ops/layernorm.test.ts`
Expected: FAIL — cannot resolve modules.

- [ ] **Step 3: Implement**

`src/engine/ops/linear.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** y = x W + b.  x: T×In, W: In×Out, b: [Out]. */
export function linear(meta: OpMeta, x: string, W: string, b: string, out: string): Op {
  return {
    ...meta, kind: 'linear', inputs: [x], output: out, params: [W, b],
    forward(ctx: Ctx) {
      const X = get(ctx, x), Wt = get(ctx, W), B = get(ctx, b);
      const [T, In] = X.shape, Out = Wt.shape[1];
      if (Wt.shape[0] !== In) throw new Error(`${meta.id}: x has ${In} cols but W has ${Wt.shape[0]} rows`);
      const Y = put(ctx, out, [T, Out]);
      for (let t = 0; t < T; t++) for (let o = 0; o < Out; o++) {
        let s = B.data[o];
        for (let i = 0; i < In; i++) s += X.data[t * In + i] * Wt.data[i * Out + o];
        Y.data[t * Out + o] = s;
      }
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x), Wt = get(ctx, W);
      const [T, In] = X.shape, Out = Wt.shape[1];
      const dY = gradGet(ctx, out).data;
      const dX = gradOf(ctx, x).data, dW = gradOf(ctx, W).data, dB = gradOf(ctx, b).data;
      for (let t = 0; t < T; t++) for (let o = 0; o < Out; o++) {
        const g = dY[t * Out + o];
        dB[o] += g;
        for (let i = 0; i < In; i++) {
          dX[t * In + i] += g * Wt.data[i * Out + o];
          dW[i * Out + o] += X.data[t * In + i] * g;
        }
      }
    },
  };
}
```

`src/engine/ops/layernorm.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** Per-row layer norm over the last dimension: y = gamma * (x - mean) / sqrt(var + eps) + beta. */
export function layerNorm(meta: OpMeta, x: string, gamma: string, beta: string, out: string, eps = 1e-5): Op {
  return {
    ...meta, kind: 'layernorm', inputs: [x], output: out, params: [gamma, beta],
    forward(ctx: Ctx) {
      const X = get(ctx, x), G = get(ctx, gamma).data, Bt = get(ctx, beta).data;
      const [T, D] = X.shape;
      const Y = put(ctx, out, [T, D]);
      for (let t = 0; t < T; t++) {
        let m = 0; for (let d = 0; d < D; d++) m += X.data[t * D + d]; m /= D;
        let v = 0; for (let d = 0; d < D; d++) v += (X.data[t * D + d] - m) ** 2; v /= D;
        const inv = 1 / Math.sqrt(v + eps);
        for (let d = 0; d < D; d++) Y.data[t * D + d] = G[d] * (X.data[t * D + d] - m) * inv + Bt[d];
      }
    },
    backward(ctx: Ctx) {
      const X = get(ctx, x), G = get(ctx, gamma).data;
      const [T, D] = X.shape;
      const dY = gradGet(ctx, out).data;
      const dX = gradOf(ctx, x).data, dG = gradOf(ctx, gamma).data, dB = gradOf(ctx, beta).data;
      const xhat = new Float64Array(D), dxhat = new Float64Array(D);
      for (let t = 0; t < T; t++) {
        let m = 0; for (let d = 0; d < D; d++) m += X.data[t * D + d]; m /= D;
        let v = 0; for (let d = 0; d < D; d++) v += (X.data[t * D + d] - m) ** 2; v /= D;
        const inv = 1 / Math.sqrt(v + eps);
        let mean_dxhat = 0, mean_dxhat_xhat = 0;
        for (let d = 0; d < D; d++) {
          xhat[d] = (X.data[t * D + d] - m) * inv;
          const g = dY[t * D + d];
          dG[d] += g * xhat[d];
          dB[d] += g;
          dxhat[d] = g * G[d];
          mean_dxhat += dxhat[d] / D;
          mean_dxhat_xhat += dxhat[d] * xhat[d] / D;
        }
        for (let d = 0; d < D; d++) dX[t * D + d] += inv * (dxhat[d] - mean_dxhat - xhat[d] * mean_dxhat_xhat);
      }
    },
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/engine/ops/linear.test.ts src/engine/ops/layernorm.test.ts`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/ops/linear.ts src/engine/ops/linear.test.ts src/engine/ops/layernorm.ts src/engine/ops/layernorm.test.ts
git commit -m "engine: linear and layer norm ops with gradient checks

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 6: Embedding ops

**Files:**
- Create: `src/engine/ops/embed.ts`
- Test: `src/engine/ops/embed.test.ts`

**Interfaces:**
- Consumes: Task 4 contract. `ctx.tokens` supplies ids; `ctx.T` supplies sequence length.
- Produces:
  ```ts
  function embed(meta: OpMeta, E: string, out: string): Op      // out[t] = E[tokens[t]] ; E: V×d ; out: T×d
  function posEmbed(meta: OpMeta, P: string, out: string): Op   // out[t] = P[t] for t < T ; P: Tmax×d
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/ops/embed.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { embed, posEmbed } from './embed';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'e', label: 'e', formula: '', explain: '' };

describe('embed', () => {
  it('looks up rows of E by token id', () => {
    const ctx = randomCtx({ E: [4, 2] }, 1);
    get(ctx, 'E').data.set([0, 0, 1, 1, 2, 2, 3, 3]);
    ctx.tokens = [3, 1]; ctx.T = 2;
    embed(meta, 'E', 'x').forward(ctx);
    expect(Array.from(get(ctx, 'x').data)).toEqual([3, 3, 1, 1]);
  });
  it('passes the gradient check, including a repeated token', () => {
    const ctx = randomCtx({ E: [5, 3] }, 2);
    ctx.tokens = [2, 4, 2]; ctx.T = 3;
    expect(gradCheck(embed(meta, 'E', 'x'), ctx, ['E'])).toEqual([]);
  });
});

describe('posEmbed', () => {
  it('takes the first T rows of P', () => {
    const ctx = randomCtx({ P: [4, 2] }, 3);
    get(ctx, 'P').data.set([0, 1, 2, 3, 4, 5, 6, 7]);
    ctx.tokens = [9, 9, 9]; ctx.T = 3;
    posEmbed(meta, 'P', 'p').forward(ctx);
    expect(Array.from(get(ctx, 'p').data)).toEqual([0, 1, 2, 3, 4, 5]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ P: [6, 3] }, 4);
    ctx.tokens = [0, 0, 0, 0]; ctx.T = 4;
    expect(gradCheck(posEmbed(meta, 'P', 'p'), ctx, ['P'])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/ops/embed.test.ts`
Expected: FAIL — cannot resolve `./embed`.

- [ ] **Step 3: Implement**

`src/engine/ops/embed.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** out[t, :] = E[tokens[t], :] */
export function embed(meta: OpMeta, E: string, out: string): Op {
  return {
    ...meta, kind: 'embed', inputs: [], output: out, params: [E],
    forward(ctx: Ctx) {
      const Et = get(ctx, E); const d = Et.shape[1]; const T = ctx.T;
      const X = put(ctx, out, [T, d]);
      for (let t = 0; t < T; t++) {
        const id = ctx.tokens[t];
        if (id < 0 || id >= Et.shape[0]) throw new Error(`${meta.id}: token id ${id} out of range`);
        for (let j = 0; j < d; j++) X.data[t * d + j] = Et.data[id * d + j];
      }
    },
    backward(ctx: Ctx) {
      const Et = get(ctx, E); const d = Et.shape[1]; const T = ctx.T;
      const dX = gradGet(ctx, out).data, dE = gradOf(ctx, E).data;
      for (let t = 0; t < T; t++) { const id = ctx.tokens[t]; for (let j = 0; j < d; j++) dE[id * d + j] += dX[t * d + j]; }
    },
  };
}

/** out[t, :] = P[t, :] for t < T */
export function posEmbed(meta: OpMeta, P: string, out: string): Op {
  return {
    ...meta, kind: 'pos_embed', inputs: [], output: out, params: [P],
    forward(ctx: Ctx) {
      const Pt = get(ctx, P); const d = Pt.shape[1]; const T = ctx.T;
      if (T > Pt.shape[0]) throw new Error(`${meta.id}: sequence length ${T} exceeds context size ${Pt.shape[0]}`);
      const X = put(ctx, out, [T, d]);
      X.data.set(Pt.data.subarray(0, T * d));
    },
    backward(ctx: Ctx) {
      const Pt = get(ctx, P); const d = Pt.shape[1]; const T = ctx.T;
      const dX = gradGet(ctx, out).data, dP = gradOf(ctx, P).data;
      for (let i = 0; i < T * d; i++) dP[i] += dX[i];
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/ops/embed.test.ts`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/ops/embed.ts src/engine/ops/embed.test.ts
git commit -m "engine: token and positional embedding ops

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 7: Attention ops (scores, causal mask, row softmax, apply)

**Files:**
- Create: `src/engine/ops/attention.ts`
- Test: `src/engine/ops/attention.test.ts`

**Interfaces:**
- Consumes: Task 4 contract.
- Produces:
  ```ts
  const MASK_VALUE = -1e9;   // views render values <= -1e8 as "-inf"
  function scores(meta: OpMeta, q: string, k: string, out: string): Op        // S = q k^T / sqrt(d) ; q,k: T×d ; S: T×T
  function causalMask(meta: OpMeta, s: string, out: string): Op              // out[i,j] = j > i ? MASK_VALUE : s[i,j]
  function softmaxRows(meta: OpMeta, s: string, out: string): Op             // row-wise softmax, any R×C
  function attnApply(meta: OpMeta, a: string, v: string, out: string): Op    // O = A v ; A: T×T ; v: T×d
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/ops/attention.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { scores, causalMask, softmaxRows, attnApply, MASK_VALUE } from './attention';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const m = { id: 'a', label: 'a', formula: '', explain: '' };

describe('scores', () => {
  it('is q k^T scaled by 1/sqrt(d)', () => {
    const ctx = randomCtx({ q: [2, 4], k: [2, 4] }, 1);
    get(ctx, 'q').data.set([1, 0, 0, 0, 0, 1, 0, 0]);
    get(ctx, 'k').data.set([2, 0, 0, 0, 0, 4, 0, 0]);
    scores(m, 'q', 'k', 's').forward(ctx);
    expect(Array.from(get(ctx, 's').data)).toEqual([1, 0, 0, 2]);  // /sqrt(4) = /2
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ q: [3, 4], k: [3, 4] }, 2);
    expect(gradCheck(scores(m, 'q', 'k', 's'), ctx, ['q', 'k'])).toEqual([]);
  });
});

describe('causalMask', () => {
  it('masks strictly-future positions', () => {
    const ctx = randomCtx({ s: [3, 3] }, 3);
    get(ctx, 's').data.fill(1);
    causalMask(m, 's', 'o').forward(ctx);
    expect(Array.from(get(ctx, 'o').data)).toEqual([1, MASK_VALUE, MASK_VALUE, 1, 1, MASK_VALUE, 1, 1, 1]);
  });
  it('passes the gradient check (zero grad through masked cells)', () => {
    const ctx = randomCtx({ s: [4, 4] }, 4);
    expect(gradCheck(causalMask(m, 's', 'o'), ctx, ['s'])).toEqual([]);
  });
});

describe('softmaxRows', () => {
  it('rows sum to one and masked entries become exactly zero', () => {
    const ctx = randomCtx({ s: [2, 3] }, 5);
    get(ctx, 's').data.set([0, MASK_VALUE, MASK_VALUE, 1, 2, MASK_VALUE]);
    softmaxRows(m, 's', 'a').forward(ctx);
    const a = Array.from(get(ctx, 'a').data);
    expect(a[0]).toBe(1); expect(a[1]).toBe(0); expect(a[2]).toBe(0);
    expect(a[3] + a[4]).toBeCloseTo(1, 12); expect(a[5]).toBe(0);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ s: [3, 5] }, 6);
    expect(gradCheck(softmaxRows(m, 's', 'a'), ctx, ['s'])).toEqual([]);
  });
});

describe('attnApply', () => {
  it('mixes value rows by attention weights', () => {
    const ctx = randomCtx({ a: [2, 2], v: [2, 2] }, 7);
    get(ctx, 'a').data.set([1, 0, 0.5, 0.5]);
    get(ctx, 'v').data.set([2, 4, 6, 8]);
    attnApply(m, 'a', 'v', 'o').forward(ctx);
    expect(Array.from(get(ctx, 'o').data)).toEqual([2, 4, 4, 6]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ a: [3, 3], v: [3, 4] }, 8);
    expect(gradCheck(attnApply(m, 'a', 'v', 'o'), ctx, ['a', 'v'])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/ops/attention.test.ts`
Expected: FAIL — cannot resolve `./attention`.

- [ ] **Step 3: Implement**

`src/engine/ops/attention.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';

/** Large negative stand-in for -infinity; exp() of it underflows to exactly 0 in float64. */
export const MASK_VALUE = -1e9;

/** S[i,j] = sum_d q[i,d] k[j,d] / sqrt(d) */
export function scores(meta: OpMeta, q: string, k: string, out: string): Op {
  return {
    ...meta, kind: 'scores', inputs: [q, k], output: out, params: [],
    forward(ctx: Ctx) {
      const Q = get(ctx, q), K = get(ctx, k); const [T, d] = Q.shape; const inv = 1 / Math.sqrt(d);
      const S = put(ctx, out, [T, T]);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        let s = 0; for (let c = 0; c < d; c++) s += Q.data[i * d + c] * K.data[j * d + c];
        S.data[i * T + j] = s * inv;
      }
    },
    backward(ctx: Ctx) {
      const Q = get(ctx, q), K = get(ctx, k); const [T, d] = Q.shape; const inv = 1 / Math.sqrt(d);
      const dS = gradGet(ctx, out).data, dQ = gradOf(ctx, q).data, dK = gradOf(ctx, k).data;
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
        const g = dS[i * T + j] * inv;
        for (let c = 0; c < d; c++) { dQ[i * d + c] += g * K.data[j * d + c]; dK[j * d + c] += g * Q.data[i * d + c]; }
      }
    },
  };
}

/** out[i,j] = j > i ? MASK_VALUE : s[i,j] */
export function causalMask(meta: OpMeta, s: string, out: string): Op {
  return {
    ...meta, kind: 'mask', inputs: [s], output: out, params: [],
    forward(ctx: Ctx) {
      const S = get(ctx, s); const [T] = S.shape;
      const O = put(ctx, out, [T, T]);
      for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) O.data[i * T + j] = j > i ? MASK_VALUE : S.data[i * T + j];
    },
    backward(ctx: Ctx) {
      const [T] = get(ctx, s).shape;
      const dO = gradGet(ctx, out).data, dS = gradOf(ctx, s).data;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) dS[i * T + j] += dO[i * T + j];
    },
  };
}

/** Row-wise softmax of an R×C tensor, max-subtracted for stability. */
export function softmaxRows(meta: OpMeta, s: string, out: string): Op {
  return {
    ...meta, kind: 'softmax_rows', inputs: [s], output: out, params: [],
    forward(ctx: Ctx) {
      const S = get(ctx, s); const [R, C] = S.shape;
      const A = put(ctx, out, [R, C]);
      for (let r = 0; r < R; r++) {
        let mx = -Infinity; for (let c = 0; c < C; c++) mx = Math.max(mx, S.data[r * C + c]);
        let z = 0; for (let c = 0; c < C; c++) { const e = Math.exp(S.data[r * C + c] - mx); A.data[r * C + c] = e; z += e; }
        for (let c = 0; c < C; c++) A.data[r * C + c] /= z;
      }
    },
    backward(ctx: Ctx) {
      const A = get(ctx, out); const [R, C] = A.shape;
      const dA = gradGet(ctx, out).data, dS = gradOf(ctx, s).data;
      for (let r = 0; r < R; r++) {
        let dot = 0; for (let c = 0; c < C; c++) dot += dA[r * C + c] * A.data[r * C + c];
        for (let c = 0; c < C; c++) dS[r * C + c] += A.data[r * C + c] * (dA[r * C + c] - dot);
      }
    },
  };
}

/** O = A v.  A: T×T, v: T×d. */
export function attnApply(meta: OpMeta, a: string, v: string, out: string): Op {
  return {
    ...meta, kind: 'attn_apply', inputs: [a, v], output: out, params: [],
    forward(ctx: Ctx) {
      const A = get(ctx, a), V = get(ctx, v); const T = A.shape[0], d = V.shape[1];
      const O = put(ctx, out, [T, d]);
      for (let i = 0; i < T; i++) for (let c = 0; c < d; c++) {
        let s = 0; for (let j = 0; j < T; j++) s += A.data[i * T + j] * V.data[j * d + c];
        O.data[i * d + c] = s;
      }
    },
    backward(ctx: Ctx) {
      const A = get(ctx, a), V = get(ctx, v); const T = A.shape[0], d = V.shape[1];
      const dO = gradGet(ctx, out).data, dA = gradOf(ctx, a).data, dV = gradOf(ctx, v).data;
      for (let i = 0; i < T; i++) for (let c = 0; c < d; c++) {
        const g = dO[i * d + c];
        for (let j = 0; j < T; j++) { dA[i * T + j] += g * V.data[j * d + c]; dV[j * d + c] += A.data[i * T + j] * g; }
      }
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/ops/attention.test.ts`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/ops/attention.ts src/engine/ops/attention.test.ts
git commit -m "engine: attention ops with causal mask and gradient checks

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 8: Output ops (softmax over vocabulary, cross-entropy loss)

**Files:**
- Create: `src/engine/ops/output.ts`
- Test: `src/engine/ops/output.test.ts`

**Interfaces:**
- Consumes: `softmaxRows` from Task 7; Task 4 contract; `ctx.targets`.
- Produces:
  ```ts
  function softmaxOut(meta: OpMeta, logits: string, out: string): Op   // softmaxRows with kind 'softmax_out'
  function crossEntropy(meta: OpMeta, probs: string, out: string): Op   // out: [1] ; loss = -(1/T) sum_t log probs[t, targets[t]]
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/ops/output.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { softmaxOut, crossEntropy } from './output';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const m = { id: 'o', label: 'o', formula: '', explain: '' };

describe('softmaxOut', () => {
  it('has kind softmax_out and rows summing to one', () => {
    const ctx = randomCtx({ logits: [2, 4] }, 1);
    const op = softmaxOut(m, 'logits', 'probs');
    expect(op.kind).toBe('softmax_out');
    op.forward(ctx);
    const p = get(ctx, 'probs').data;
    expect(p[0] + p[1] + p[2] + p[3]).toBeCloseTo(1, 12);
  });
});

describe('crossEntropy', () => {
  it('is the mean negative log probability of the targets', () => {
    const ctx = randomCtx({ probs: [2, 3] }, 2);
    get(ctx, 'probs').data.set([0.5, 0.25, 0.25, 0.1, 0.8, 0.1]);
    ctx.targets = [0, 1]; ctx.T = 2;
    crossEntropy(m, 'probs', 'loss').forward(ctx);
    const L = get(ctx, 'loss');
    expect(L.shape).toEqual([1]);
    expect(L.data[0]).toBeCloseTo(-(Math.log(0.5) + Math.log(0.8)) / 2, 12);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ probs: [3, 4] }, 3);
    const p = get(ctx, 'probs').data; for (let i = 0; i < p.length; i++) p[i] = Math.abs(p[i]) + 0.1;
    ctx.targets = [1, 3, 0]; ctx.T = 3;
    expect(gradCheck(crossEntropy(m, 'probs', 'loss'), ctx, ['probs'])).toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/ops/output.test.ts`
Expected: FAIL — cannot resolve `./output`.

- [ ] **Step 3: Implement**

`src/engine/ops/output.ts`:
```ts
import { Op, OpMeta, Ctx, get, put, gradOf, gradGet } from './types';
import { softmaxRows } from './attention';

/** Same maths as softmaxRows; a distinct kind so views can treat the vocabulary distribution specially. */
export function softmaxOut(meta: OpMeta, logits: string, out: string): Op {
  return { ...softmaxRows(meta, logits, out), kind: 'softmax_out' };
}

/** loss = -(1/T) * sum_t log probs[t, targets[t]] */
export function crossEntropy(meta: OpMeta, probs: string, out: string): Op {
  return {
    ...meta, kind: 'loss', inputs: [probs], output: out, params: [],
    forward(ctx: Ctx) {
      if (!ctx.targets) throw new Error(`${meta.id}: ctx.targets required`);
      const P = get(ctx, probs); const [T, V] = P.shape;
      const L = put(ctx, out, [1]);
      let s = 0;
      for (let t = 0; t < T; t++) s += -Math.log(P.data[t * V + ctx.targets[t]]);
      L.data[0] = s / T;
    },
    backward(ctx: Ctx) {
      if (!ctx.targets) throw new Error(`${meta.id}: ctx.targets required`);
      const P = get(ctx, probs); const [T, V] = P.shape;
      const dL = gradGet(ctx, out).data[0];
      const dP = gradOf(ctx, probs).data;
      for (let t = 0; t < T; t++) { const idx = t * V + ctx.targets[t]; dP[idx] += -dL / (T * P.data[idx]); }
    },
  };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/ops/output.test.ts`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/ops/output.ts src/engine/ops/output.test.ts
git commit -m "engine: vocabulary softmax and cross-entropy loss ops

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 9: Parameters and the GPT model (with end-to-end gradient check)

**Files:**
- Create: `src/engine/model/params.ts`
- Create: `src/engine/model/gpt.ts`
- Test: `src/engine/model/gpt.test.ts`

**Interfaces:**
- Consumes: every op factory from Tasks 4–8; `Rng`, `tensor`, `newCtx`.
- Produces (`params.ts`):
  ```ts
  interface GPTConfig { vocabSize: number; contextSize: number; dModel: number; dFF: number; seed: number }
  const DEFAULTS: Omit<GPTConfig, 'vocabSize'> = { contextSize: 24, dModel: 20, dFF: 30, seed: 1 }
  const PARAM_NAMES: readonly string[]   // fixed order: E, P, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, ln1_g, ln1_b, ln2_g, ln2_b, W_1, b_1, W_2, b_2, lnf_g, lnf_b, U, b_u
  function paramShapes(cfg: GPTConfig): Record<string, number[]>
  function initParams(cfg: GPTConfig): Map<string, Tensor>   // matrices N(0, 0.02); biases 0; LN gamma 1, beta 0
  function paramCount(params: Map<string, Tensor>): number
  ```
- Produces (`gpt.ts`):
  ```ts
  const T_ = { tok, pos, x0, h1, q, k, v, scores, masked, attn, ctxv, attn_out, x1, h2, ff_pre, ff_act, ff_out, x2, hf, logits, probs, loss }  // activation tensor names (each value equals its key)
  function buildOps(cfg: GPTConfig): Op[]     // the 22 ops in forward order; ops[21] is the loss
  class GPT {
    readonly config: GPTConfig; readonly params: Map<string, Tensor>; readonly ops: Op[];
    constructor(config: GPTConfig, params?: Map<string, Tensor>)
    forward(tokens: number[], targets?: number[]): Ctx   // 21 ops, plus loss when targets given
    backward(ctx: Ctx): void                             // grads[loss] = 1, then every op in reverse
    paramCount(): number
    setParam(name: string, index: number, value: number): void
    exportParams(): Record<string, number[]>             // plain arrays, safe for postMessage
    importParams(p: Record<string, number[]>): void
  }
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/model/gpt.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { GPT, T_ } from './gpt';
import { initParams, paramCount, PARAM_NAMES } from './params';
import { get } from '../ops/types';

const cfg = { vocabSize: 7, contextSize: 5, dModel: 4, dFF: 6, seed: 3 };

describe('params', () => {
  it('creates every named parameter with the documented shapes', () => {
    const p = initParams(cfg);
    expect([...p.keys()]).toEqual([...PARAM_NAMES]);
    expect(p.get('E')!.shape).toEqual([7, 4]);
    expect(p.get('P')!.shape).toEqual([5, 4]);
    expect(p.get('W_1')!.shape).toEqual([4, 6]);
    expect(p.get('U')!.shape).toEqual([4, 7]);
    expect(Array.from(p.get('ln1_g')!.data)).toEqual([1, 1, 1, 1]);
    expect(Array.from(p.get('b_q')!.data)).toEqual([0, 0, 0, 0]);
  });
  it('counts parameters', () => {
    // E 28 + P 20 + 4*(16+4) + LN 3*8 + W_1 24 + b_1 6 + W_2 24 + b_2 4 + U 28 + b_u 7
    expect(paramCount(initParams(cfg))).toBe(28 + 20 + 80 + 24 + 24 + 6 + 24 + 4 + 28 + 7);
  });
});

describe('GPT.forward', () => {
  it('produces probabilities over the vocabulary for each position', () => {
    const m = new GPT(cfg);
    const ctx = m.forward([1, 2, 3]);
    const probs = get(ctx, T_.probs);
    expect(probs.shape).toEqual([3, 7]);
    for (let t = 0; t < 3; t++) { let s = 0; for (let v = 0; v < 7; v++) s += probs.data[t * 7 + v]; expect(s).toBeCloseTo(1, 12); }
    expect(ctx.tensors.has(T_.loss)).toBe(false);
  });
  it('starts near uniform: loss ~ ln(V) at init', () => {
    const m = new GPT(cfg);
    const ctx = m.forward([1, 2, 3, 4], [2, 3, 4, 5]);
    expect(Math.abs(get(ctx, T_.loss).data[0] - Math.log(7))).toBeLessThan(0.1);
  });
  it('is deterministic for a seed', () => {
    const a = new GPT(cfg).forward([1, 2]), b = new GPT(cfg).forward([1, 2]);
    expect(Array.from(get(a, T_.probs).data)).toEqual(Array.from(get(b, T_.probs).data));
  });
  it('rejects prompts longer than the context', () => {
    expect(() => new GPT(cfg).forward([1, 2, 3, 4, 5, 6])).toThrow();
  });
});

describe('GPT.backward (end-to-end gradient check)', () => {
  it('matches central finite differences of the loss for every parameter', () => {
    const m = new GPT(cfg);
    const tokens = [1, 2, 3, 4, 5], targets = [2, 3, 4, 5, 6];
    const ctx = m.forward(tokens, targets);
    m.backward(ctx);
    const eps = 1e-6, tol = 1e-5;
    const failures: string[] = [];
    for (const name of PARAM_NAMES) {
      const p = m.params.get(name)!; const g = ctx.grads.get(name)!;
      for (let i = 0; i < p.data.length; i++) {
        const v = p.data[i];
        p.data[i] = v + eps; const lp = get(m.forward(tokens, targets), T_.loss).data[0];
        p.data[i] = v - eps; const lm = get(m.forward(tokens, targets), T_.loss).data[0];
        p.data[i] = v;
        const num = (lp - lm) / (2 * eps);
        if (Math.abs(g.data[i] - num) > tol * Math.max(1, Math.abs(num))) failures.push(`${name}[${i}] analytic=${g.data[i]} numeric=${num}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('GPT params round trip', () => {
  it('exports and imports plain arrays', () => {
    const a = new GPT(cfg); a.setParam('W_q', 3, 0.5);
    const b = new GPT(cfg); b.importParams(a.exportParams());
    expect(b.params.get('W_q')!.data[3]).toBe(0.5);
    expect(Array.from(get(b.forward([1, 2]), T_.probs).data)).toEqual(Array.from(get(a.forward([1, 2]), T_.probs).data));
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/model/gpt.test.ts`
Expected: FAIL — cannot resolve modules.

- [ ] **Step 3: Implement**

`src/engine/model/params.ts`:
```ts
import { Tensor, tensor } from '../tensor';
import { Rng } from '../rng';

export interface GPTConfig { vocabSize: number; contextSize: number; dModel: number; dFF: number; seed: number }

export const DEFAULTS: Omit<GPTConfig, 'vocabSize'> = { contextSize: 24, dModel: 20, dFF: 30, seed: 1 };

export const PARAM_NAMES = [
  'E', 'P',
  'W_q', 'b_q', 'W_k', 'b_k', 'W_v', 'b_v', 'W_o', 'b_o',
  'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b',
  'W_1', 'b_1', 'W_2', 'b_2',
  'lnf_g', 'lnf_b',
  'U', 'b_u',
] as const;

export function paramShapes(c: GPTConfig): Record<string, number[]> {
  const { vocabSize: V, contextSize: T, dModel: d, dFF: f } = c;
  return {
    E: [V, d], P: [T, d],
    W_q: [d, d], b_q: [d], W_k: [d, d], b_k: [d], W_v: [d, d], b_v: [d], W_o: [d, d], b_o: [d],
    ln1_g: [d], ln1_b: [d], ln2_g: [d], ln2_b: [d],
    W_1: [d, f], b_1: [f], W_2: [f, d], b_2: [d],
    lnf_g: [d], lnf_b: [d],
    U: [d, V], b_u: [V],
  };
}

const INIT_STD = 0.02;

export function initParams(c: GPTConfig): Map<string, Tensor> {
  const rng = new Rng(c.seed);
  const shapes = paramShapes(c);
  const out = new Map<string, Tensor>();
  for (const name of PARAM_NAMES) {
    const t = tensor(name, shapes[name]);
    if (name.endsWith('_g')) t.data.fill(1);
    else if (name.startsWith('b_') || name.endsWith('_b')) t.data.fill(0);
    else for (let i = 0; i < t.data.length; i++) t.data[i] = rng.randn() * INIT_STD;
    out.set(name, t);
  }
  return out;
}

export function paramCount(params: Map<string, Tensor>): number {
  let n = 0; for (const t of params.values()) n += t.data.length; return n;
}
```

`src/engine/model/gpt.ts`:
```ts
import { Tensor, tensor } from '../tensor';
import { Op, Ctx, newCtx } from '../ops/types';
import { add, relu } from '../ops/elementwise';
import { linear } from '../ops/linear';
import { layerNorm } from '../ops/layernorm';
import { embed, posEmbed } from '../ops/embed';
import { scores, causalMask, softmaxRows, attnApply } from '../ops/attention';
import { softmaxOut, crossEntropy } from '../ops/output';
import { GPTConfig, initParams, paramCount, paramShapes, PARAM_NAMES } from './params';

/** Names of every activation tensor, in the order they are produced. */
export const T_ = {
  tok: 'tok', pos: 'pos', x0: 'x0', h1: 'h1', q: 'q', k: 'k', v: 'v', scores: 'scores', masked: 'masked',
  attn: 'attn', ctxv: 'ctxv', attn_out: 'attn_out', x1: 'x1', h2: 'h2', ff_pre: 'ff_pre', ff_act: 'ff_act',
  ff_out: 'ff_out', x2: 'x2', hf: 'hf', logits: 'logits', probs: 'probs', loss: 'loss',
} as const;

/** The whole forward pass as an ordered list of named operations. ops[21] is the loss. */
export function buildOps(_c: GPTConfig): Op[] {
  return [
    embed({ id: 'embed', label: 'Token embedding', formula: 'tok_t = E[id_t]',
      explain: 'Each word id picks one row out of the embedding table E. That row is a list of numbers the model learns to use as a description of the word. Nothing about the word is known yet except what training has written into this row.' }, 'E', T_.tok),
    posEmbed({ id: 'pos_embed', label: 'Position embedding', formula: 'pos_t = P[t]',
      explain: 'Attention has no built-in sense of order, so each position t also gets its own learned row from table P. Position 0 always gets row 0, position 1 row 1, and so on.' }, 'P', T_.pos),
    add({ id: 'add_pos', label: 'Add position', formula: 'x0 = tok + pos',
      explain: 'The word vector and the position vector are added, one number at a time. The result x0 is the residual stream: the running record that every later step reads from and writes back into.' }, T_.tok, T_.pos, T_.x0),
    layerNorm({ id: 'ln1', label: 'Layer norm 1', formula: 'h1 = ln1_g * (x0 - mean) / sqrt(var + eps) + ln1_b',
      explain: 'Each row of x0 is rescaled so its numbers have mean 0 and spread 1, then stretched by ln1_g and shifted by ln1_b. This keeps the numbers in a range the next steps can work with no matter how large the stream has grown.' }, T_.x0, 'ln1_g', 'ln1_b', T_.h1),
    linear({ id: 'q_proj', label: 'Query projection', formula: 'q = h1 W_q + b_q',
      explain: 'Every position asks a question. The query q is what this position is looking for in the positions before it, computed by multiplying its normalised vector by the learned matrix W_q.' }, T_.h1, 'W_q', 'b_q', T_.q),
    linear({ id: 'k_proj', label: 'Key projection', formula: 'k = h1 W_k + b_k',
      explain: 'Every position also advertises what it holds. The key k is that advertisement, computed with a different learned matrix W_k. Queries will be matched against keys.' }, T_.h1, 'W_k', 'b_k', T_.k),
    linear({ id: 'v_proj', label: 'Value projection', formula: 'v = h1 W_v + b_v',
      explain: 'The value v is what a position actually hands over if another position decides to attend to it. Matching is done with queries and keys; the content that moves is the value.' }, T_.h1, 'W_v', 'b_v', T_.v),
    scores({ id: 'scores', label: 'Attention scores', formula: 'S[i,j] = q_i . k_j / sqrt(d)',
      explain: 'Each query is compared with each key by a dot product. A large number at row i, column j means position i finds position j relevant. Dividing by the square root of d keeps the scores from growing with the vector size.' }, T_.q, T_.k, T_.scores),
    causalMask({ id: 'causal_mask', label: 'Causal mask', formula: 'S[i,j] = -inf if j > i',
      explain: 'A position may only look at itself and earlier positions, never at later ones, because at prediction time the later words do not exist yet. Scores for future positions are set to minus infinity so the softmax gives them exactly zero. This is why the tile is a triangle.' }, T_.scores, T_.masked),
    softmaxRows({ id: 'attn_softmax', label: 'Attention weights', formula: 'A[i,:] = softmax(S[i,:])',
      explain: 'Each row of scores is turned into weights that are positive and add up to 1. Row i now says how much of its attention position i spends on each earlier position.' }, T_.masked, T_.attn),
    attnApply({ id: 'attn_apply', label: 'Apply attention', formula: 'ctxv = A v',
      explain: 'Each position collects a weighted mix of the value vectors of the positions it attends to. Heavily weighted positions contribute more. The result is new information gathered from context.' }, T_.attn, T_.v, T_.ctxv),
    linear({ id: 'o_proj', label: 'Output projection', formula: 'attn_out = ctxv W_o + b_o',
      explain: 'The gathered vector is passed through one more learned matrix, W_o, which decides how the gathered information should be written back into the residual stream.' }, T_.ctxv, 'W_o', 'b_o', T_.attn_out),
    add({ id: 'residual1', label: 'Residual add 1', formula: 'x1 = x0 + attn_out',
      explain: 'The attention result is added back onto the stream rather than replacing it. The original word and position information survive, with the new context layered on top.' }, T_.x0, T_.attn_out, T_.x1),
    layerNorm({ id: 'ln2', label: 'Layer norm 2', formula: 'h2 = ln2_g * (x1 - mean) / sqrt(var + eps) + ln2_b',
      explain: 'Normalise again before the next block, for the same reason as before: keep every row at a predictable scale.' }, T_.x1, 'ln2_g', 'ln2_b', T_.h2),
    linear({ id: 'ff_up', label: 'Feed-forward up', formula: 'ff_pre = h2 W_1 + b_1',
      explain: 'Each position is processed on its own now, with no mixing between positions. The vector is expanded to a wider hidden layer by W_1. This is where the model stores facts and patterns that do not depend on context.' }, T_.h2, 'W_1', 'b_1', T_.ff_pre),
    relu({ id: 'relu', label: 'ReLU', formula: 'ff_act = max(0, ff_pre)',
      explain: 'Every negative number in the hidden layer is replaced by zero. This single nonlinear step is what lets the network compute things a chain of matrix multiplications alone never could.' }, T_.ff_pre, T_.ff_act),
    linear({ id: 'ff_down', label: 'Feed-forward down', formula: 'ff_out = ff_act W_2 + b_2',
      explain: 'The hidden layer is projected back down to the stream width by W_2 so it can be added to the residual stream.' }, T_.ff_act, 'W_2', 'b_2', T_.ff_out),
    add({ id: 'residual2', label: 'Residual add 2', formula: 'x2 = x1 + ff_out',
      explain: 'The feed-forward result is added onto the stream. x2 is the final state of the residual stream for this block.' }, T_.x1, T_.ff_out, T_.x2),
    layerNorm({ id: 'ln_final', label: 'Final layer norm', formula: 'hf = lnf_g * (x2 - mean) / sqrt(var + eps) + lnf_b',
      explain: 'One last normalisation before reading the prediction out of the stream.' }, T_.x2, 'lnf_g', 'lnf_b', T_.hf),
    linear({ id: 'unembed', label: 'Unembedding', formula: 'logits = hf U + b_u',
      explain: 'The stream is turned back into words. Multiplying by U gives one score per vocabulary word for each position. These raw scores are called logits.' }, T_.hf, 'U', 'b_u', T_.logits),
    softmaxOut({ id: 'softmax_out', label: 'Next-token probabilities', formula: 'probs[t,:] = softmax(logits[t,:])',
      explain: 'Each row of logits is turned into probabilities that add up to 1. Row t is the model\'s prediction for the word that comes after position t. The largest entry is the model\'s best guess.' }, T_.logits, T_.probs),
    crossEntropy({ id: 'loss', label: 'Cross-entropy loss', formula: 'loss = -(1/T) sum_t log probs[t, target_t]',
      explain: 'During training the model is scored on how much probability it gave to the word that actually came next. Taking minus the log makes a confident right answer cost nearly 0 and a confident wrong answer cost a lot. The average over positions is the number training pushes down.' }, T_.probs, T_.loss),
  ];
}

export class GPT {
  readonly config: GPTConfig;
  readonly params: Map<string, Tensor>;
  readonly ops: Op[];

  constructor(config: GPTConfig, params?: Map<string, Tensor>) {
    this.config = config;
    this.params = params ?? initParams(config);
    this.ops = buildOps(config);
  }

  forward(tokens: number[], targets?: number[]): Ctx {
    if (tokens.length === 0) throw new Error('forward: empty token list');
    if (tokens.length > this.config.contextSize) throw new Error(`forward: ${tokens.length} tokens exceeds context size ${this.config.contextSize}`);
    if (targets && targets.length !== tokens.length) throw new Error('forward: targets length must equal tokens length');
    const ctx = newCtx(tokens, this.params, targets);
    const n = targets ? this.ops.length : this.ops.length - 1;
    for (let i = 0; i < n; i++) this.ops[i].forward(ctx);
    return ctx;
  }

  backward(ctx: Ctx): void {
    if (!ctx.tensors.has(T_.loss)) throw new Error('backward: run forward with targets first');
    ctx.grads.set(T_.loss, tensor(T_.loss, [1], new Float64Array([1])));
    for (let i = this.ops.length - 1; i >= 0; i--) this.ops[i].backward(ctx);
  }

  paramCount(): number { return paramCount(this.params); }

  setParam(name: string, index: number, value: number): void {
    const p = this.params.get(name);
    if (!p) throw new Error(`setParam: no parameter '${name}'`);
    if (index < 0 || index >= p.data.length) throw new Error(`setParam: index ${index} out of range for '${name}'`);
    p.data[index] = value;
  }

  exportParams(): Record<string, number[]> {
    const out: Record<string, number[]> = {};
    for (const n of PARAM_NAMES) out[n] = Array.from(this.params.get(n)!.data);
    return out;
  }

  importParams(p: Record<string, number[]>): void {
    const shapes = paramShapes(this.config);
    for (const n of PARAM_NAMES) {
      const src = p[n]; if (!src) throw new Error(`importParams: missing '${n}'`);
      const t = this.params.get(n)!;
      if (src.length !== t.data.length) throw new Error(`importParams: '${n}' has ${src.length} values, expected ${t.data.length} (${shapes[n].join('x')})`);
      t.data.set(src);
    }
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/model/gpt.test.ts`
Expected: 8 passed. If the end-to-end gradient check fails, the failing parameter name tells you which op's backward is wrong; fix that op, not the test.

- [ ] **Step 5: Commit**

```bash
git add src/engine/model/params.ts src/engine/model/gpt.ts src/engine/model/gpt.test.ts
git commit -m "engine: GPT model as an ordered op list, end-to-end gradient check

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 10: Adam optimizer, training step, evaluation, overfit test

**Files:**
- Create: `src/engine/model/adam.ts`
- Create: `src/engine/model/train.ts`
- Test: `src/engine/model/train.test.ts`

**Interfaces:**
- Consumes: `GPT`, `T_` (Task 9); `Window`, `windows`, `shuffled` (Task 3); `Rng`.
- Produces (`adam.ts`):
  ```ts
  interface AdamOpts { lr: number; beta1: number; beta2: number; eps: number }
  const ADAM_DEFAULTS: AdamOpts = { lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 }
  class Adam {
    opts: AdamOpts; t: number;
    constructor(params: Map<string, Tensor>, opts?: Partial<AdamOpts>)
    step(grads: Map<string, Tensor>): Map<string, Float64Array>   // applies the update in place; returns the delta applied to each parameter
  }
  ```
- Produces (`train.ts`):
  ```ts
  function trainStep(model: GPT, adam: Adam, w: Window): { loss: number; ctx: Ctx; deltas: Map<string, Float64Array> }
  function evaluate(model: GPT, ws: Window[], limit = 64): { loss: number; accuracy: number }   // accuracy = fraction of positions where argmax(probs) == target
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/model/train.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { GPT } from './gpt';
import { Adam } from './adam';
import { trainStep, evaluate } from './train';
import { windows, shuffled } from '../corpus/windows';
import { Rng } from '../rng';

describe('Adam', () => {
  it('moves each parameter against its gradient on the first step, by about lr', () => {
    const cfg = { vocabSize: 5, contextSize: 3, dModel: 4, dFF: 4, seed: 1 };
    const m = new GPT(cfg);
    const before = m.params.get('W_q')!.data[0];
    const ctx = m.forward([1, 2, 3], [2, 3, 4]); m.backward(ctx);
    const g = ctx.grads.get('W_q')!.data[0];
    const adam = new Adam(m.params, { lr: 0.01 });
    const deltas = adam.step(ctx.grads);
    const after = m.params.get('W_q')!.data[0];
    expect(Math.sign(after - before)).toBe(-Math.sign(g));
    expect(Math.abs(after - before)).toBeCloseTo(0.01, 3);   // Adam's first step is ~lr in magnitude
    expect(deltas.get('W_q')![0]).toBeCloseTo(after - before, 12);
  });
});

describe('training', () => {
  it('overfits a tiny corpus to near-zero loss', () => {
    // 30 tokens: a repeating 9-token cycle, so every next token is fully determined and loss can reach ~0.
    const tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3];
    const cfg = { vocabSize: 10, contextSize: 6, dModel: 8, dFF: 16, seed: 2 };
    const m = new GPT(cfg);
    const adam = new Adam(m.params, { lr: 0.01 });
    const ws = windows(tokens, 6);
    const rng = new Rng(3);
    let steps = 0, loss = Infinity;
    while (steps < 3000) {
      for (const w of shuffled(ws, rng)) { loss = trainStep(m, adam, w).loss; steps++; }
      loss = evaluate(m, ws).loss;
      if (loss < 0.05) break;
    }
    expect(loss).toBeLessThan(0.05);
    expect(evaluate(m, ws).accuracy).toBeGreaterThan(0.95);
    console.log(`overfit reached loss ${loss.toFixed(4)} after ${steps} window steps`);
  });
  it('evaluate reports loss and accuracy in range', () => {
    const cfg = { vocabSize: 6, contextSize: 3, dModel: 4, dFF: 4, seed: 4 };
    const r = evaluate(new GPT(cfg), windows([1, 2, 3, 4, 5, 1, 2], 3));
    expect(r.loss).toBeGreaterThan(0); expect(r.accuracy).toBeGreaterThanOrEqual(0); expect(r.accuracy).toBeLessThanOrEqual(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/model/train.test.ts`
Expected: FAIL — cannot resolve modules.

- [ ] **Step 3: Implement**

`src/engine/model/adam.ts`:
```ts
import { Tensor } from '../tensor';

export interface AdamOpts { lr: number; beta1: number; beta2: number; eps: number }
export const ADAM_DEFAULTS: AdamOpts = { lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8 };

export class Adam {
  opts: AdamOpts;
  t = 0;
  private m = new Map<string, Float64Array>();
  private v = new Map<string, Float64Array>();
  constructor(private params: Map<string, Tensor>, opts: Partial<AdamOpts> = {}) {
    this.opts = { ...ADAM_DEFAULTS, ...opts };
    for (const [n, p] of params) { this.m.set(n, new Float64Array(p.data.length)); this.v.set(n, new Float64Array(p.data.length)); }
  }
  /** Applies one Adam update in place. Returns the delta added to each parameter. */
  step(grads: Map<string, Tensor>): Map<string, Float64Array> {
    this.t++;
    const { lr, beta1, beta2, eps } = this.opts;
    const c1 = 1 - Math.pow(beta1, this.t), c2 = 1 - Math.pow(beta2, this.t);
    const deltas = new Map<string, Float64Array>();
    for (const [n, p] of this.params) {
      const g = grads.get(n)?.data; const m = this.m.get(n)!, v = this.v.get(n)!;
      const d = new Float64Array(p.data.length);
      if (g) for (let i = 0; i < p.data.length; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
        d[i] = -lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
        p.data[i] += d[i];
      }
      deltas.set(n, d);
    }
    return deltas;
  }
}
```

`src/engine/model/train.ts`:
```ts
import { GPT, T_ } from './gpt';
import { Adam } from './adam';
import { Ctx, get } from '../ops/types';
import { Window } from '../corpus/windows';

export function trainStep(model: GPT, adam: Adam, w: Window): { loss: number; ctx: Ctx; deltas: Map<string, Float64Array> } {
  const ctx = model.forward(w.input, w.target);
  model.backward(ctx);
  const deltas = adam.step(ctx.grads);
  return { loss: get(ctx, T_.loss).data[0], ctx, deltas };
}

/** Mean loss and next-token accuracy over up to `limit` windows (evenly spaced through the list). */
export function evaluate(model: GPT, ws: Window[], limit = 64): { loss: number; accuracy: number } {
  if (ws.length === 0) return { loss: NaN, accuracy: NaN };
  const stride = Math.max(1, Math.floor(ws.length / limit));
  let lossSum = 0, correct = 0, positions = 0, n = 0;
  for (let i = 0; i < ws.length; i += stride) {
    const w = ws[i];
    const ctx = model.forward(w.input, w.target);
    lossSum += get(ctx, T_.loss).data[0]; n++;
    const P = get(ctx, T_.probs); const [T, V] = P.shape;
    for (let t = 0; t < T; t++) {
      let best = 0; for (let v = 1; v < V; v++) if (P.data[t * V + v] > P.data[t * V + best]) best = v;
      if (best === w.target[t]) correct++;
      positions++;
    }
  }
  return { loss: lossSum / n, accuracy: correct / positions };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/model/train.test.ts`
Expected: 3 passed, and a console line like `overfit reached loss 0.0xxx after NNN window steps`. Record NNN in the commit message; it is one of the honest numbers for the README.

- [ ] **Step 5: Commit**

```bash
git add src/engine/model/adam.ts src/engine/model/train.ts src/engine/model/train.test.ts
git commit -m "engine: Adam, training step, evaluation; overfit test

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 11: Generation (greedy, top-k, top-p, temperature)

**Files:**
- Create: `src/engine/model/generate.ts`
- Test: `src/engine/model/generate.test.ts`

**Interfaces:**
- Consumes: `GPT`, `T_`; `Rng`.
- Produces:
  ```ts
  type Strategy = 'greedy' | 'top-k' | 'top-p';
  interface SampleOpts { temperature: number; strategy: Strategy; k: number; p: number }
  const SAMPLE_DEFAULTS: SampleOpts = { temperature: 1, strategy: 'top-p', k: 10, p: 0.9 }
  function sampleFromLogits(logits: Float64Array, opts: SampleOpts, rng: Rng): { token: number; probs: Float64Array }  // probs after temperature and truncation, renormalised
  function generate(model: GPT, prompt: number[], n: number, opts: SampleOpts, rng: Rng): number[]   // prompt followed by n new tokens; context is the last contextSize tokens
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/model/generate.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { sampleFromLogits, generate, SAMPLE_DEFAULTS } from './generate';
import { GPT } from './gpt';
import { Rng } from '../rng';

const logits = new Float64Array([0.1, 2.0, 1.0, -1.0, 1.5]);   // argmax = 1, order 1,4,2,0,3

describe('sampleFromLogits', () => {
  it('greedy returns the argmax', () => {
    expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'greedy' }, new Rng(1)).token).toBe(1);
  });
  it('a near-zero temperature is effectively greedy', () => {
    const rng = new Rng(2);
    for (let i = 0; i < 50; i++) expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, temperature: 1e-6, strategy: 'top-p', p: 1 }, rng).token).toBe(1);
  });
  it('top-k never samples outside the top k', () => {
    const rng = new Rng(3);
    for (let i = 0; i < 200; i++) expect([1, 4]).toContain(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-k', k: 2 }, rng).token);
  });
  it('top-p with a tiny p is greedy; with p=1 every token is possible', () => {
    expect(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-p', p: 0.01 }, new Rng(4)).token).toBe(1);
    const seen = new Set<number>(); const rng = new Rng(5);
    for (let i = 0; i < 2000; i++) seen.add(sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-p', p: 1, temperature: 3 }, rng).token);
    expect(seen.size).toBe(5);
  });
  it('returned probs sum to one', () => {
    const { probs } = sampleFromLogits(logits, { ...SAMPLE_DEFAULTS, strategy: 'top-k', k: 3 }, new Rng(6));
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(probs[3]).toBe(0);
  });
});

describe('generate', () => {
  const cfg = { vocabSize: 9, contextSize: 4, dModel: 4, dFF: 4, seed: 7 };
  it('returns the prompt followed by n tokens inside the vocabulary', () => {
    const out = generate(new GPT(cfg), [1, 2], 5, SAMPLE_DEFAULTS, new Rng(8));
    expect(out.length).toBe(7); expect(out.slice(0, 2)).toEqual([1, 2]);
    for (const t of out) { expect(t).toBeGreaterThanOrEqual(0); expect(t).toBeLessThan(9); }
  });
  it('handles prompts longer than the context by using the last contextSize tokens', () => {
    const out = generate(new GPT(cfg), [1, 2, 3, 4, 5, 6], 2, { ...SAMPLE_DEFAULTS, strategy: 'greedy' }, new Rng(9));
    expect(out.length).toBe(8);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/model/generate.test.ts`
Expected: FAIL — cannot resolve `./generate`.

- [ ] **Step 3: Implement**

`src/engine/model/generate.ts`:
```ts
import { GPT, T_ } from './gpt';
import { get } from '../ops/types';
import { Rng } from '../rng';

export type Strategy = 'greedy' | 'top-k' | 'top-p';
export interface SampleOpts { temperature: number; strategy: Strategy; k: number; p: number }
export const SAMPLE_DEFAULTS: SampleOpts = { temperature: 1, strategy: 'top-p', k: 10, p: 0.9 };

export function sampleFromLogits(logits: Float64Array, opts: SampleOpts, rng: Rng): { token: number; probs: Float64Array } {
  const V = logits.length;
  const probs = new Float64Array(V);
  let best = 0; for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
  if (opts.strategy === 'greedy' || opts.temperature <= 0) { probs[best] = 1; return { token: best, probs }; }

  const temp = opts.temperature;
  let z = 0;
  for (let i = 0; i < V; i++) { probs[i] = Math.exp((logits[i] - logits[best]) / temp); z += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= z;

  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  let keep: number;
  if (opts.strategy === 'top-k') keep = Math.max(1, Math.min(V, Math.floor(opts.k)));
  else { let cum = 0; keep = 0; for (const i of order) { cum += probs[i]; keep++; if (cum >= opts.p) break; } }
  const mask = new Set(order.slice(0, keep));
  let z2 = 0;
  for (let i = 0; i < V; i++) { if (!mask.has(i)) probs[i] = 0; z2 += probs[i]; }
  for (let i = 0; i < V; i++) probs[i] /= z2;

  let r = rng.next(), token = order[keep - 1];
  for (const i of order.slice(0, keep)) { r -= probs[i]; if (r <= 0) { token = i; break; } }
  return { token, probs };
}

/** Returns prompt followed by n sampled tokens. Context is the last contextSize tokens. */
export function generate(model: GPT, prompt: number[], n: number, opts: SampleOpts, rng: Rng): number[] {
  const out = [...prompt];
  const T = model.config.contextSize;
  for (let i = 0; i < n; i++) {
    const ctxTokens = out.slice(-T);
    const ctx = model.forward(ctxTokens);
    const L = get(ctx, T_.logits); const [rows, V] = L.shape;
    const last = L.data.subarray((rows - 1) * V, rows * V);
    out.push(sampleFromLogits(Float64Array.from(last), opts, rng).token);
  }
  return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/model/generate.test.ts`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/engine/model/generate.ts src/engine/model/generate.test.ts
git commit -m "engine: sampling with temperature, top-k and top-p

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

### Task 12: Trace recorder and cursor

**Files:**
- Create: `src/engine/trace/trace.ts`
- Create: `src/engine/trace/cursor.ts`
- Test: `src/engine/trace/trace.test.ts`

**Interfaces:**
- Consumes: `GPT`, `T_` (Task 9); `Adam` (Task 10); `Window` (Task 3); `PARAM_NAMES`; `Ctx`.
- Produces (`trace.ts`):
  ```ts
  type Phase = 'forward' | 'backward' | 'update';
  const UPDATE_OP = 'adam_update';
  interface TraceStep { index: number; opId: string; phase: Phase; writes: string[] }   // keys: 't:<tensor>' values, 'g:<tensor>' gradients
  interface Trace { kind: 'forward' | 'training'; steps: TraceStep[]; ctx: Ctx; deltas?: Map<string, Float64Array>; preexisting: Set<string> }
  function tKey(name: string): string   // 't:' + name
  function gKey(name: string): string   // 'g:' + name
  function recordForward(model: GPT, tokens: number[]): Trace                 // 21 forward steps
  function recordTrainingStep(model: GPT, adam: Adam, w: Window): Trace      // 22 forward + 22 backward + 1 update = 45 steps; mutates model params
  ```
- Produces (`cursor.ts`):
  ```ts
  type Status = 'pending' | 'partial' | 'active' | 'done';
  class Cursor {
    index: number;                      // -1 before the first step
    constructor(trace: Trace)
    readonly length: number;
    current(): TraceStep | null;
    status(key: string): Status;        // 'partial' = an accumulated gradient with some but not all contributions so far
    next(): void; prev(): void; seek(i: number): void; toStart(): void; toEnd(): void;
    atStart(): boolean; atEnd(): boolean;
  }
  ```

- [ ] **Step 1: Write the failing test**

`src/engine/trace/trace.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { recordForward, recordTrainingStep, tKey, gKey, UPDATE_OP } from './trace';
import { Cursor } from './cursor';
import { GPT, T_ } from '../model/gpt';
import { Adam } from '../model/adam';
import { PARAM_NAMES } from '../model/params';
import { get } from '../ops/types';

const cfg = { vocabSize: 7, contextSize: 5, dModel: 4, dFF: 6, seed: 3 };

describe('recordForward', () => {
  it('records one forward step per op, in order, writing that op\'s output', () => {
    const m = new GPT(cfg);
    const tr = recordForward(m, [1, 2, 3]);
    expect(tr.kind).toBe('forward');
    expect(tr.steps.length).toBe(21);
    tr.steps.forEach((s, i) => {
      expect(s.index).toBe(i); expect(s.phase).toBe('forward');
      expect(s.opId).toBe(m.ops[i].id); expect(s.writes).toEqual([tKey(m.ops[i].output)]);
    });
    expect(get(tr.ctx, T_.probs).shape).toEqual([3, 7]);
    expect(tr.preexisting.has(tKey('E'))).toBe(true);
  });
  it('is byte-identical for the same seed and tokens', () => {
    const a = recordForward(new GPT(cfg), [1, 2, 3]), b = recordForward(new GPT(cfg), [1, 2, 3]);
    expect(Array.from(get(a.ctx, T_.probs).data)).toEqual(Array.from(get(b.ctx, T_.probs).data));
  });
});

describe('recordTrainingStep', () => {
  it('records forward, backward and update phases', () => {
    const m = new GPT(cfg);
    const tr = recordTrainingStep(m, new Adam(m.params), { input: [1, 2, 3], target: [2, 3, 4] });
    expect(tr.kind).toBe('training');
    expect(tr.steps.length).toBe(45);
    expect(tr.steps[21]).toMatchObject({ opId: 'loss', phase: 'forward', writes: [tKey(T_.loss)] });
    expect(tr.steps[22]).toMatchObject({ opId: 'loss', phase: 'backward', writes: [gKey(T_.loss), gKey(T_.probs)] });
    expect(tr.steps[43]).toMatchObject({ opId: 'embed', phase: 'backward', writes: [gKey('E')] });
    expect(tr.steps[44]).toMatchObject({ opId: UPDATE_OP, phase: 'update' });
    expect(tr.steps[44].writes).toEqual(PARAM_NAMES.map(tKey));
    expect(tr.deltas!.get('W_q')!.length).toBe(16);
    expect(tr.ctx.grads.has('W_q')).toBe(true);
  });
});

describe('Cursor', () => {
  it('starts before the first step with activations pending and parameters done', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1, 2]));
    expect(c.index).toBe(-1); expect(c.atStart()).toBe(true); expect(c.current()).toBeNull();
    expect(c.status(tKey(T_.tok))).toBe('pending');
    expect(c.status(tKey('E'))).toBe('done');
  });
  it('marks the current write active, earlier writes done, later writes pending', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1, 2]));
    c.seek(2);
    expect(c.status(tKey(T_.tok))).toBe('done');
    expect(c.status(tKey(T_.x0))).toBe('active');
    expect(c.status(tKey(T_.h1))).toBe('pending');
  });
  it('reports partial for an accumulated gradient with one of two contributions', () => {
    const m = new GPT(cfg);
    const tr = recordTrainingStep(m, new Adam(m.params), { input: [1, 2, 3], target: [2, 3, 4] });
    const c = new Cursor(tr);
    const residual1 = tr.steps.findIndex(s => s.opId === 'residual1' && s.phase === 'backward');
    const ln1 = tr.steps.findIndex(s => s.opId === 'ln1' && s.phase === 'backward');
    expect(residual1).toBeLessThan(ln1);
    c.seek(residual1); expect(c.status(gKey(T_.x0))).toBe('active');
    c.seek(residual1 + 1); expect(c.status(gKey(T_.x0))).toBe('partial');
    c.seek(ln1); expect(c.status(gKey(T_.x0))).toBe('active');
    c.seek(ln1 + 1); expect(c.status(gKey(T_.x0))).toBe('done');
    c.toEnd(); expect(c.status(tKey('W_q'))).toBe('active'); expect(c.atEnd()).toBe(true);
  });
  it('clamps next and prev at the ends', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1]));
    c.prev(); expect(c.index).toBe(-1);
    c.toEnd(); expect(c.index).toBe(20); c.next(); expect(c.index).toBe(20);
    c.toStart(); expect(c.index).toBe(-1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/engine/trace/trace.test.ts`
Expected: FAIL — cannot resolve modules.

- [ ] **Step 3: Implement**

`src/engine/trace/trace.ts`:
```ts
import { GPT, T_ } from '../model/gpt';
import { Adam } from '../model/adam';
import { PARAM_NAMES } from '../model/params';
import { Ctx } from '../ops/types';
import { Window } from '../corpus/windows';
import { tensor } from '../tensor';

export type Phase = 'forward' | 'backward' | 'update';
export const UPDATE_OP = 'adam_update';

export interface TraceStep { index: number; opId: string; phase: Phase; writes: string[] }
export interface Trace {
  kind: 'forward' | 'training';
  steps: TraceStep[];
  ctx: Ctx;
  deltas?: Map<string, Float64Array>;
  /** Keys that exist before step 0 (the parameters). */
  preexisting: Set<string>;
}

export const tKey = (name: string) => 't:' + name;
export const gKey = (name: string) => 'g:' + name;

function preexistingKeys(): Set<string> { return new Set(PARAM_NAMES.map(tKey)); }

/** Runs the forward pass and records one step per op. */
export function recordForward(model: GPT, tokens: number[]): Trace {
  const ctx = model.forward(tokens);
  const steps: TraceStep[] = model.ops.slice(0, -1).map((op, i) => ({ index: i, opId: op.id, phase: 'forward', writes: [tKey(op.output)] }));
  return { kind: 'forward', steps, ctx, preexisting: preexistingKeys() };
}

/** Runs forward + backward + one Adam step and records all three phases. Mutates the model's parameters. */
export function recordTrainingStep(model: GPT, adam: Adam, w: Window): Trace {
  const ctx = model.forward(w.input, w.target);
  ctx.grads.set(T_.loss, tensor(T_.loss, [1], new Float64Array([1])));
  for (let i = model.ops.length - 1; i >= 0; i--) model.ops[i].backward(ctx);
  const deltas = adam.step(ctx.grads);

  const steps: TraceStep[] = [];
  model.ops.forEach(op => steps.push({ index: steps.length, opId: op.id, phase: 'forward', writes: [tKey(op.output)] }));
  for (let i = model.ops.length - 1; i >= 0; i--) {
    const op = model.ops[i];
    const writes = [...op.inputs.map(gKey), ...op.params.map(gKey)];
    if (op.id === 'loss') writes.unshift(gKey(T_.loss));
    steps.push({ index: steps.length, opId: op.id, phase: 'backward', writes });
  }
  steps.push({ index: steps.length, opId: UPDATE_OP, phase: 'update', writes: PARAM_NAMES.map(tKey) });
  return { kind: 'training', steps, ctx, deltas, preexisting: preexistingKeys() };
}
```

`src/engine/trace/cursor.ts`:
```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/engine/trace/trace.test.ts`
Expected: 7 passed.

- [ ] **Step 5: Run the whole engine suite and typecheck, then commit**

Run: `npx vitest run && npx tsc --noEmit`
Expected: every test file passes; no type errors.

```bash
git add src/engine/trace/trace.ts src/engine/trace/cursor.ts src/engine/trace/trace.test.ts
git commit -m "engine: trace recorder and step cursor

Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC"
```

---

## Self-review against the spec

- Corpus (tokenize, split, windows, min length, cap folding): Tasks 2–3. The 50k-token truncation and user notices are UI concerns for Plan 2; `Vocab.folded` and `minTokens` give it what it needs.
- Model as an ordered op list with `formula` and `explain`: Task 9. Every op in the spec's table has a task (embed/pos Task 6; add/relu Task 4; ln/linear Task 5; attention Task 7; output Task 8).
- Backward accumulation, gradient checks per op and end-to-end: Tasks 4–9.
- Adam, evaluation, overfit criterion: Task 10 (the test measures the real step count; the spec's "300 iterations" is replaced by whatever the test reports, and the spec should be updated to that number).
- Generation with greedy / top-k / top-p / temperature: Task 11.
- Trace as eager record + replay cursor, forward/backward/update phases: Task 12. `partial` status is an addition to the spec (accumulated gradients), worth keeping.
- Determinism: Tasks 1, 9, 12.
- `setParam` / export / import for the worker and editable weights: Task 9.
- Names used across tasks were checked: `get/put/gradOf/gradGet`, `T_`, `PARAM_NAMES`, `tKey/gKey`, `Window`, `Adam.step` returning deltas.
