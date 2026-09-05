# glassbox App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. This plan assumes the engine plan (`2026-09-05-glassbox-engine.md`) is fully implemented and green.

**Goal:** The single-page site: paste text, train in a worker, scrub the recorded computation forward and backward, switch between Flow / Network / Math / Loss views, inspect and edit any number, generate text. Deployed as static files.

**Architecture:** One Zustand store owns corpus, config, model, training state, the current trace and cursor index, view, selection and hover. React renders panels and view chrome; each view is a pure renderer of `(trace, cursor, selection, hover)`. Training runs in a Web Worker with its own model; parameters sync back. All trace recording happens on the main thread.

**Tech Stack:** React 18, Vite 6, TypeScript 5.9, Zustand 5, KaTeX 0.16, Canvas2D. Vitest 3 for logic tests (add `jsdom` only if a test needs DOM).

## Global Constraints

- Views read `(trace, cursor, selection, hover)` and the model config only. They never call the model directly and never mutate the store except through the actions listed here.
- Keys for tensors follow the engine: `t:<name>` values, `g:<name>` gradients (`tKey`, `gKey` from `src/engine/trace/trace.ts`).
- Defaults: context 24, dModel 20, dFF 30, lr 3e-3 (raise if the Group-chat preset does not reach > 95 % train accuracy in under 10 s; record the value chosen), seed 1, sampling top-p 0.9 temperature 1.
- Text limits: refuse to train under `minTokens(T)` tokens; truncate above 50,000 tokens with a notice; vocabulary cap 512 with a notice of folded types.
- Values `<= -1e8` render as `-inf`.
- Keyboard: `F` = next step, `B` = previous step, `ArrowRight/ArrowLeft` same, `Home/End` = start/end, `Space` = play/pause. `T` = record one training step as the trace.
- No emojis. Plain prose. All user-facing copy gets a humanizer pass before release (Task 10).
- Commit after every task with the trailer `Claude-Session: https://claude.ai/code/session_01DrEyjJA8V7gQSexC5pq8tC`.

---

## File structure

```
src/
  app/store.ts               Zustand store: state + actions (pure logic, tested)
  app/corpusState.ts         buildCorpus(text, config) -> Corpus | notices (pure, tested)
  app/worker/trainer.worker.ts   Web Worker: owns a GPT + Adam, trains, posts progress and params
  app/worker/protocol.ts     message types shared by worker and store
  app/keys.ts                keyboard bindings hook
  presets/                   (already created) groupchat.txt, nursery.txt, alice.txt, index.ts
  views/colormap.ts          diverging / sequential colour maps (pure, tested)
  views/tiles.ts             TileImage cache: tensor -> ImageData, hit-testing (pure parts tested)
  views/camera.ts            pan/zoom transform, screen<->world (pure, tested)
  views/flow/layout.ts       fixed layout table for every tensor, param and op node (pure, tested)
  views/flow/FlowView.tsx    canvas renderer for the Flow view + semantic zoom
  views/network/NetworkView.tsx  neurons-and-wires view for one selected position
  views/math/mathFor.ts      LaTeX for the current op with real numbers substituted (pure, tested)
  views/math/MathView.tsx    KaTeX rendering
  views/loss/LossView.tsx    SVG loss/accuracy curves + gradient magnitude bars
  ui/Shell.tsx               layout: left panels, centre view tabs, bottom transport, right inspector
  ui/TextPanel.tsx           preset picker, textarea, tokenise, stats, notices, model settings
  ui/TrainPanel.tsx          train/stop, iteration, latest metrics
  ui/GeneratePanel.tsx       prompt, predict-next (records a forward trace), generate N, sampling controls
  ui/Transport.tsx           step/play controls, phase label, progress scrubber
  ui/Inspector.tsx           current op explain + formula, hovered/selected cell, dot-product breakdown, param editor
  App.tsx                    mounts Shell
  styles.css
.github/workflows/deploy.yml
README.md
```

---

### Task 1: Corpus state and the store

**Files:**
- Create: `src/app/corpusState.ts`, `src/app/store.ts`
- Test: `src/app/corpusState.test.ts`, `src/app/store.test.ts`

**Interfaces:**
```ts
// corpusState.ts
interface Corpus {
  text: string; tokens: number[]; vocab: Vocab; words: string[];
  train: number[]; test: number[]; trainWindows: Window[]; testWindows: Window[];
  notices: string[];              // human-readable, e.g. "Vocabulary capped at 512; 213 rarer words became <unk>."
  truncated: boolean;
}
const MAX_TOKENS = 50_000;
function buildCorpus(text: string, contextSize: number, cap = 512): { corpus: Corpus } | { error: string }
// error when tokens.length < minTokens(contextSize): "Need at least N words to train; this text has M."

// store.ts (Zustand, create<State>())
interface Metrics { iteration: number; trainLoss: number; testLoss: number; trainAcc: number; testAcc: number }
type View = 'flow' | 'network' | 'math' | 'loss';
interface CellRef { key: string; index: number }
interface State {
  text: string; presetId: string | null; corpus: Corpus | null; corpusError: string | null;
  config: GPTConfig; lr: number; sample: SampleOpts;
  model: GPT | null; adam: Adam | null;
  training: 'idle' | 'running'; history: Metrics[]; trainError: string | null;
  trace: Trace | null; cursorIndex: number;
  prompt: string; promptUnknown: string[]; generated: number[] | null;
  view: View; selection: CellRef | null; hover: CellRef | null; position: number;   // position = which sequence row the Network view shows
  // actions
  setText(text: string, presetId?: string | null): void;
  applyText(): void;                   // buildCorpus -> corpus or corpusError; creates a fresh model + adam; clears trace/history
  setConfig(patch: Partial<GPTConfig>): void; setLr(lr: number): void; setSample(patch: Partial<SampleOpts>): void;
  resetModel(): void;                  // new GPT(config with next seed), new Adam, clear history and trace
  recordPrompt(): void;                // tokenise prompt with corpus.vocab (unknown -> <unk>, listed in promptUnknown), recordForward, cursor to end
  recordTrainingTrace(): void;         // one training step on a seeded-random train window, recordTrainingStep, cursor to start
  seek(i: number): void; next(): void; prev(): void; toStart(): void; toEnd(): void;
  setView(v: View): void; select(c: CellRef | null): void; setHover(c: CellRef | null): void; setPosition(t: number): void;
  editParam(name: string, index: number, value: number): void;   // model.setParam, then re-record the same kind of trace at the same cursor index
  generateMore(n: number): void;       // generate() from prompt tokens; sets generated
  // worker plumbing (Task 2 fills these)
  startTraining(): void; stopTraining(): void;
  _onProgress(m: Metrics): void; _onParams(p: Record<string, number[]>): void; _onTrainError(msg: string): void;
}
```

- [ ] **Step 1: Tests** — `corpusState.test.ts`: too-short text returns the error with both numbers; a 60-word text at T=24 returns a corpus with `train.length === floor(0.6 * tokens)`, windows present, no notices; a text with 600 distinct words at cap 512 yields a notice containing "512" and `vocab.folded`; text over 50,000 tokens is truncated with `truncated === true` and a notice. `store.test.ts` (no DOM needed): `setText` + `applyText` builds a corpus and a model with `vocabSize === vocab.words.length`; `recordPrompt` with prompt "hello there" on the nursery preset produces a forward trace of 21 steps and `cursorIndex === 20`; unknown words appear in `promptUnknown`; `recordTrainingTrace` yields 45 steps and `cursorIndex === -1`; `next/prev/seek` clamp; `editParam` changes `model.params` and re-records keeping `cursorIndex`; `resetModel` changes `config.seed` and empties history.
- [ ] **Step 2: Run, see them fail.**
- [ ] **Step 3: Implement** `corpusState.ts` and `store.ts` per the interfaces. `applyText` must catch `buildCorpus` errors into `corpusError`. `recordPrompt` uses the last `contextSize` prompt tokens; an empty prompt records nothing and leaves the trace untouched.
- [ ] **Step 4: Run, green. `npx tsc --noEmit` clean.**
- [ ] **Step 5: Commit** `app: corpus state and store`.

---

### Task 2: Training worker

**Files:**
- Create: `src/app/worker/protocol.ts`, `src/app/worker/trainer.worker.ts`
- Modify: `src/app/store.ts` (startTraining / stopTraining / _on* handlers)
- Test: `src/app/worker/protocol.test.ts` (type guards), and a store test that drives `_onProgress` / `_onParams` directly.

**Interfaces:**
```ts
// protocol.ts
type ToWorker =
  | { type: 'init'; config: GPTConfig; params: Record<string, number[]>; trainWindows: Window[]; testWindows: Window[]; lr: number; seed: number }
  | { type: 'start' } | { type: 'stop' } | { type: 'setLr'; lr: number };
type FromWorker =
  | { type: 'progress'; metrics: Metrics }
  | { type: 'params'; params: Record<string, number[]>; iteration: number }
  | { type: 'error'; message: string };
```
Behaviour: one **iteration = one epoch** over `trainWindows` in seeded-shuffled order. After each epoch post `progress` with mean train loss, `evaluate(test)` loss/accuracy, `evaluate(train)` accuracy (limit 64 windows each). Post `params` every 5 epochs and on `stop`. Work in chunks via `setTimeout(0)` between windows so `stop` messages are handled promptly (check a flag). If a loss is not finite: restore the last posted params, post `error` "Training diverged at iteration N; try a lower learning rate", and stop. The store, on `params`, calls `model.importParams` and, if a trace exists, re-records it at the same cursor index so the tiles reflect the trained weights. Vite: `new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' })`. If `Worker` is undefined, `startTraining` runs 1 epoch synchronously per `requestAnimationFrame` with a notice.

- [ ] Steps 1–5 as usual. Commit `app: training worker`.

---

### Task 3: Colour maps, tile images, camera

**Files:**
- Create: `src/views/colormap.ts`, `src/views/tiles.ts`, `src/views/camera.ts`
- Test: `src/views/colormap.test.ts`, `src/views/tiles.test.ts` (hit-testing only), `src/views/camera.test.ts`

**Interfaces:**
```ts
// colormap.ts
type RGB = [number, number, number];
function diverging(v: number, maxAbs: number): RGB     // blue (neg) - near-white (0) - red (pos); clamps; maxAbs 0 -> white
function sequential(v: number): RGB                     // 0 -> white, 1 -> dark teal; for probabilities and attention weights
function gradientMap(v: number, maxAbs: number): RGB    // purple (neg) - white - green (pos); for gradients and deltas
const MASK_COLOR: RGB                                   // mid grey for values <= -1e8
function maxAbs(data: Float64Array): number

// tiles.ts
type TileMode = 'value' | 'prob' | 'grad';
interface TileRect { x: number; y: number; w: number; h: number }   // world units
function cellSize(shape: number[], maxSide = 240): number            // world units per cell so the longer side <= maxSide, min 1
function tileRect(x: number, y: number, shape: number[], cs: number): TileRect
function cellAt(rect: TileRect, shape: number[], wx: number, wy: number): number | null   // flat index or null when outside
class TileCache {
  /** Returns a canvas (OffscreenCanvas when available, else HTMLCanvasElement) of shape[1] x shape[0] pixels, one pixel per cell, regenerated when `version` changes. */
  get(name: string, shape: number[], data: Float64Array, mode: TileMode, version: number): CanvasImageSource;
  clear(): void;
}

// camera.ts
interface Camera { x: number; y: number; zoom: number }   // world point (x, y) maps to screen ((wx - x) * zoom, (wy - y) * zoom)
function toScreen(c: Camera, wx: number, wy: number): [number, number]
function toWorld(c: Camera, sx: number, sy: number): [number, number]
function zoomAt(c: Camera, sx: number, sy: number, factor: number, min = 0.05, max = 40): Camera   // keeps the world point under the cursor fixed
function pan(c: Camera, dx: number, dy: number): Camera          // dx, dy in screen pixels
function fit(bounds: TileRect, viewport: { w: number; h: number }, margin = 40): Camera
```
Tile modes: `value` uses `diverging` with the tensor's own `maxAbs`, except tensors named `attn` or `probs` which use `sequential`; `grad` uses `gradientMap`; cells `<= -1e8` use `MASK_COLOR`.

- [ ] Tests: `diverging(0, 1)` is near-white; `diverging(1, 1)` is red-dominant; `diverging(-1, 1)` blue-dominant; `maxAbs` handles empty. `cellAt` returns the right index for a 2x3 tile at several points, null outside. `toWorld(toScreen(p)) === p`; `zoomAt` keeps the cursor's world point fixed (within 1e-9); `fit` places the bounds fully inside the viewport with margin.
- [ ] Implement. `TileCache` writes RGBA into an `ImageData` then `putImageData`; keep `version` in the key so a re-recorded trace invalidates.
- [ ] Green + typecheck. Commit `views: colour maps, tile cache, camera`.

---

### Task 4: Flow layout table

**Files:**
- Create: `src/views/flow/layout.ts`
- Test: `src/views/flow/layout.test.ts`

**Interfaces:**
```ts
type NodeKind = 'tensor' | 'param' | 'op';
interface LayoutNode { id: string; kind: NodeKind; key?: string /* t:<name> for tensor/param */; opId?: string; label: string; rect: TileRect; shape?: number[]; cs?: number; group: string }
interface LayoutEdge { from: string; to: string }         // node ids; tensor/param -> op, op -> tensor
interface LayoutGroup { id: string; label: string; rect: TileRect; parent?: string }
interface FlowLayout { nodes: LayoutNode[]; edges: LayoutEdge[]; groups: LayoutGroup[]; bounds: TileRect; byId: Map<string, LayoutNode> }
function flowLayout(ops: Op[], shapes: Record<string, number[]> /* activation and param shapes for this trace */): FlowLayout
```
Layout rules (world units; y grows downward on screen, so the residual stream runs **bottom to top** as in Simbrain):
- A single vertical **spine** at x = 0 carries the residual stream tiles in order: `tok+pos` row at the bottom, then `x0`, `x1`, `x2`, `hf`, `logits`, `probs` upward. Op nodes are 120x28 pills centred on the spine between the tiles they connect.
- Groups: `Inputs` (tokens strip drawn by the view, not a tensor), `Embedding` (E, P, tok, pos, x0), `Transformer block` containing sub-groups `Attention` (h1, q, k, v, scores, masked, attn, ctxv, attn_out) and `Feed-forward` (h2, ff_pre, ff_act, ff_out), plus `Unembedding` (hf, U, logits) and `Output` (probs, loss when present).
- Inside `Attention`: `h1` sits above `x0`, then `q k v` side by side (x = -160, 0, +160), `scores` above `k`, `masked` above `scores`, `attn` above `masked`, `ctxv` above `attn` back on the spine, `attn_out` above that. `x1` sits on the spine above `attn_out`; the residual add op for `x1` has an extra edge from `x0` drawn as a straight **rail** at x = -260 (a vertical line from `x0` up to the op).
- Inside `Feed-forward`: `h2`, `ff_pre` (wider tile), `ff_act`, `ff_out`, then `x2` on the spine with a rail from `x1` at x = -260.
- Parameters sit to the **right** of the op that uses them (x = +260 for matrices, biases as a thin strip just right of their matrix), orange-bordered in the view. `E` and `U` are large: use `cellSize(shape, 240)`.
- Vertical spacing: 24 units between a tile and the next op pill; tiles never overlap; `bounds` is the union of all rects with 60 units of padding.
- Edges: for each op, `inputs` and `params` -> op node; op node -> output tensor. The two rails are ordinary edges whose `from` is `x0`/`x1` and whose routing the view chooses by node position.

- [ ] Tests: given `buildOps(cfg)` and shapes from a `recordForward` ctx, every op has a node and every tensor/param in the ops appears exactly once; no two tensor/param rects overlap (pairwise check); every edge endpoint exists in `byId`; `x0`, `x1`, `x2`, `hf`, `logits`, `probs` all have `rect.x === -rect.w / 2` (centred on the spine) and strictly decreasing `y` (stacked bottom-up); `q`, `k`, `v` share a `y`.
- [ ] Implement as a hand-written table built in code (a function that places nodes step by step and tracks a running `y`), not a general graph layout.
- [ ] Green + typecheck. Commit `views: flow layout table`.

---

### Task 5: Flow view (canvas)

**Files:**
- Create: `src/views/flow/FlowView.tsx`
- Modify: `src/App.tsx` to mount a temporary `<FlowView/>` full-screen against the nursery preset (replaced by the Shell in Task 6)

**Behaviour:**
- A `<canvas>` sized to its container with `devicePixelRatio`; redraws on store changes (`trace`, `cursorIndex`, `selection`, `hover`) and camera changes via `requestAnimationFrame` coalescing.
- Draw order: group rects (rounded, 1 px border, label top-left, `Transformer block` slightly tinted); edges as cubic curves from the top-centre of the source to the bottom-centre of the target (rails as straight vertical lines then a short horizontal into the op); op pills (label; fill by status of its output: pending grey, active accent with an outer glow, done white); tiles (image via `TileCache` scaled to `rect`, then a 1 px border: orange for params, dark for activations; alpha 0.3 when pending; hatched overlay for `partial`; 2 px accent outline + glow when active; selection outline 2 px; hover outline 1 px); tile labels (`name  rows x cols`) above each tile once `zoom >= 0.5`; cell numbers (3 significant figures, `-inf` for masked) once a cell is `>= 22` screen px.
- Semantic zoom: when `zoom < 0.22`, draw only the groups with big labels and the spine tiles as solid bars — the block reads as one box: text in, text out.
- Phase handling: if the cursor's current step is `backward` or `update`, any tile whose `g:<name>` status is not `pending` is drawn from `ctx.grads` in `grad` mode with a small "grad" badge; in the `update` step, parameter tiles draw `trace.deltas` in `grad` mode with a "delta" badge.
- Token strip: below the `Embedding` group draw the prompt tokens (or the training window input) as labelled boxes, one per position, with the selected `position` highlighted. Above `probs`, draw the predicted next token for the last position (argmax) in large text, and for a training trace the target beside it.
- Interaction: wheel = `zoomAt` cursor (factor `1.1^(-deltaY/100)`); drag = pan; move = hover cell (store `setHover`); click = select cell (`select`) or clear; double-click = `fit(bounds)`; on first mount and when a new trace kind appears, `fit`.
- Hover tooltip: a small DOM overlay near the cursor showing `name[row, col] = value` (and `grad = ...` when present).

- [ ] Implement. Manual check: `npm run dev`, open the page, confirm the diagram fits, zooming reveals numbers, stepping with the temporary keyboard hook (`F`/`B` wired in Task 7; for now a temporary `window.addEventListener('keydown')` in `App.tsx` calling `next()`/`prev()`) lights up tiles in order and dims the rest.
- [ ] Typecheck. Commit `views: flow view canvas`.

---

### Task 6: Shell, panels, transport, inspector, keyboard

**Files:**
- Create: `src/ui/Shell.tsx`, `src/ui/TextPanel.tsx`, `src/ui/TrainPanel.tsx`, `src/ui/GeneratePanel.tsx`, `src/ui/Transport.tsx`, `src/ui/Inspector.tsx`, `src/app/keys.ts`
- Modify: `src/App.tsx` (mount `<Shell/>`), `src/styles.css`
- Test: `src/ui/inspector.logic.test.ts` for `dotProductBreakdown` (pure)

**Layout:** CSS grid, `grid-template-columns: 300px 1fr 320px; grid-template-rows: 1fr 56px`. Left column scrolls; centre is the view tabs (Flow / Network / Math / Loss) over the active view; bottom row spans the centre for the transport; right column is the inspector. Under 900 px wide, stack: panels, view, transport, inspector.

**TextPanel:** preset `<select>` (Group chat / Nursery rhymes / Alice / Your own text) + `<textarea>` bound to `text`; "Use this text" -> `applyText()`; below it stats when a corpus exists: words, distinct words, `<unk>` folded, train/test windows; any `corpus.notices`; `corpusError` in red. A collapsed "Model settings" section: context size, embedding size, hidden size, learning rate, seed, and "New random model" -> `resetModel()`. Changing a size calls `setConfig` then `applyText()`.

**TrainPanel:** "Train" / "Stop" toggling `startTraining`/`stopTraining`; iteration count; latest train and test loss and accuracy (2 decimals, percentage); `trainError` if any; a tiny inline sparkline of train/test loss (last 100 points) drawn in an `<svg>`; a link "Open Loss view".

**GeneratePanel:** prompt `<textarea>` bound to `prompt`; "Predict next word" -> `recordPrompt()` and `setView('flow')`; sampling controls (strategy select, temperature 0–2 slider, k, p) bound to `setSample`; "Generate 20 words" -> `generateMore(20)`; output area rendering `detokenize(generated)` with the generated part in an accent colour; `promptUnknown` shown as "Not in vocabulary: ...".

**Transport:** buttons `|<`, `<`, play/pause, `>`, `>|`, a range input bound to `cursorIndex` over `trace.steps.length`; a label "step i / n — <phase> — <op label>"; a "Record training step" button -> `recordTrainingTrace()`; speed select (2, 5, 10 steps/s). Play advances with `setInterval`, stops at the end. Disabled when there is no trace.

**Inspector:** sections: (1) current op: label, `formula` in monospace, `explain` paragraph, phase; (2) hovered cell: `name[row, col]`, value, gradient if present, delta if present; (3) selected cell: same, plus for a `linear` op selected on its output the **dot-product breakdown** table: row of the input, column of the weight, products, bias, sum (`dotProductBreakdown(trace, opId, row, col): { terms: { xi: number; wi: number; prod: number }[]; bias: number; total: number }`); (4) if the selected cell is a parameter: a number input and `-0.1 / -0.01 / +0.01 / +0.1` buttons calling `editParam`, and a note that the diagram re-runs immediately.

**keys.ts:** `useKeys()` hook on `window` keydown: `F`/`ArrowRight` -> `next()`, `B`/`ArrowLeft` -> `prev()`, `Home` -> `toStart()`, `End` -> `toEnd()`, `Space` -> toggle play, `T` -> `recordTrainingTrace()`. Ignored when the event target is an input, textarea or select.

- [ ] Test `dotProductBreakdown` on a tiny recorded trace: the `total` equals the output cell to 1e-12 and `terms.length` equals the input width.
- [ ] Implement. Manual check: pick Nursery, Use this text, Train for ~5 s, Stop, type "twinkle twinkle" -> Predict next word -> the Flow lights up; step with F/B; hover shows values; select a `U` cell, nudge it, the probabilities tile changes; Generate 20 words returns rhyme-like text.
- [ ] Typecheck. Commit `ui: shell, panels, transport, inspector, keyboard`.

---

### Task 7: Network view (every neuron and connection, one position)

**Files:**
- Create: `src/views/network/NetworkView.tsx`, `src/views/network/networkLayout.ts`
- Test: `src/views/network/networkLayout.test.ts`

**Interfaces:**
```ts
interface NetColumn { key: string; label: string; n: number; x: number; y0: number; dy: number; r: number }   // n neurons, circle radius r
interface NetWire { from: string; to: string; param: string }   // column keys and the weight matrix name, drawn between consecutive linear stages
function networkLayout(shapes: Record<string, number[]>, viewport: { w: number; h: number }): { columns: NetColumn[]; wires: NetWire[] }
```
Columns in order: `tok`, `pos`, `x0`, `h1`, `q`, `k`, `v`, `ctxv`, `attn_out`, `x1`, `h2`, `ff_pre`, `ff_act`, `ff_out`, `x2`, `hf`, `logits`, `probs`. Wires for `W_q W_k W_v` (h1 -> q/k/v), `W_o` (ctxv -> attn_out), `W_1` (h2 -> ff_pre), `W_2` (ff_act -> ff_out), `U` (hf -> logits). Columns with more than 48 neurons (`logits`, `probs`, any V-sized) are drawn as a compact strip of 1 px rows with the top-8 by value labelled with their words.

**Behaviour:** shows the row `position` (store) of each activation tensor for the current trace: neuron fill by `diverging` (or `sequential` for `probs`); wire width `0.5 + 3 * |w| / maxAbs(W)`, colour by sign, alpha 0.6; at most 2,000 wires drawn — above that, draw only wires whose `|w|` is in the top 2,000. The `attn` row for `position` is drawn as arcs from the current position box to each earlier position box in a strip below the columns, width by weight. Hover a wire -> tooltip with `W[i, j] = value`; hover a neuron -> value; click a neuron -> highlight its fan-in and fan-out wires (others alpha 0.1) and `select` the cell; a position picker (`< position >`) calls `setPosition`. Status dimming as in Flow (pending columns at alpha 0.3, active column glowing).

- [ ] Tests: column count and order; no two columns overlap horizontally; wires reference existing columns and real parameter names.
- [ ] Implement. Manual check on the trained nursery model.
- [ ] Typecheck. Commit `views: network view`.

---

### Task 8: Math view

**Files:**
- Create: `src/views/math/mathFor.ts`, `src/views/math/MathView.tsx`
- Test: `src/views/math/mathFor.test.ts`

**Interfaces:**
```ts
interface MathBlock { title: string; latex: string; note?: string }
function mathFor(trace: Trace, step: TraceStep | null, op: Op | null, selection: CellRef | null): MathBlock[]
```
Per op `kind`, produce (a) the general formula in LaTeX, (b) the same formula with the selected cell's real numbers substituted (up to 6 terms shown, then `\cdots`, and the exact total), and (c) in the `backward` phase, the gradient rule for that op with numbers, e.g. linear: `\frac{\partial L}{\partial x_{t,i}} = \sum_o \frac{\partial L}{\partial y_{t,o}} W_{i,o}`. Kinds: embed (row lookup), pos_embed, add, layernorm (mean, variance, normalised value, output), linear (dot product + bias), scores (scaled dot), mask (case rule), softmax_rows / softmax_out (exp over sum, with the row's max subtracted, stated), attn_apply (weighted sum), relu, loss (minus log of the target probability, mean over T). When nothing is selected, show (a) only, plus a hint "Select a cell in the Flow view to see the numbers."

**MathView:** renders each block with `katex.renderToString(latex, { throwOnError: false, displayMode: true })` into `dangerouslySetInnerHTML`; the KaTeX stylesheet is imported from `katex/dist/katex.min.css`.

- [ ] Tests: for a recorded forward trace at the `q_proj` step with a selected output cell, `mathFor` returns 2 blocks and the substituted block's latex contains the cell's value formatted to 4 significant figures; at the `loss` step in backward phase, 3 blocks; with no op, 1 block hint.
- [ ] Implement. Manual check that formulas render and match the Inspector's numbers.
- [ ] Typecheck. Commit `views: math view`.

---

### Task 9: Loss view

**Files:**
- Create: `src/views/loss/LossView.tsx`

**Behaviour:** SVG, responsive. Top chart: train loss (red) and test loss (blue) vs iteration, y from 0 to the max seen (log-scale toggle), a dashed vertical marker at the current iteration and a legend with the latest values. Middle chart: train and test accuracy 0–100 %. Bottom: horizontal bar chart of `||grad||_2` per parameter tensor for the current training trace (from `trace.ctx.grads`, only when `trace.kind === 'training'`), labelled with parameter names, sorted by magnitude; empty state text otherwise. Caption under the top chart when test loss has risen for 5+ consecutive iterations while train loss fell: "The model is memorising the training text: training loss keeps falling while test loss rises. This is overfitting."

- [ ] Implement (no test; pure presentation over store data). Manual check during training.
- [ ] Typecheck. Commit `views: loss view`.

---

### Task 10: README, deploy workflow, copy pass, verification

**Files:**
- Create: `README.md`, `LICENSE` (MIT, "Copyright (c) 2026 Johnny Klaus"), `.github/workflows/deploy.yml`
- Modify: copy strings across `src/` as needed

**README** sections: what it is (two paragraphs), a screenshot placeholder line, how to use it (five steps), what you are looking at (the op list in a table: name, formula, one line each), honest numbers (parameter count at the Group-chat preset, time to > 95 % train accuracy on a MacBook Air, test accuracy at that point, bundle size gzipped, test count), how it is built (engine / trace / views in three short paragraphs), running locally (`npm install`, `npm run dev`, `npm test`), credits (Simbrain and Jeff Yoshimi as the inspiration; Transformer Explainer and bbycroft as neighbours; Alice text from Project Gutenberg), license.

**deploy.yml:** on `push` to `main`: checkout, `actions/setup-node@v4` with node 22 and npm cache, `npm ci`, `npm test`, `npm run build`, `actions/upload-pages-artifact@v3` with `dist`, `actions/deploy-pages@v4`; permissions `pages: write`, `id-token: write`; concurrency group `pages`.

**Copy pass:** run the humanizer / stop-slop skill over every `explain` string in `src/engine/model/gpt.ts`, every panel string, and the README. Fix what it flags.

**Verification (record actual numbers in the README and in the commit message):**
- `npm test` — total passing.
- `npm run build` — `dist/assets/*.js` gzipped size (`gzip -c dist/assets/*.js | wc -c`).
- Group-chat preset, default settings: time from Train to > 95 % train accuracy, and the test accuracy at that moment (from the Loss view).
- Browser check in Chrome and Safari (and Firefox if installed): load, train, predict, step, hover, edit, generate, each view.

- [ ] Commit `docs: README, MIT license, Pages deploy workflow`.

---

## Self-review against the spec

- Views: Flow (Task 5, with semantic zoom), Network (7), Math (8), Loss (9), Inspector with dot-product breakdown and param editing (6). Zoom is inside Flow, per the spec.
- Shell layout, transport, keyboard `F`/`B` (6). Worker training with param sync and NaN handling (2). Presets (already created). Failure handling: short text, truncation, cap notice (1), unknown prompt words (1, 6), diverging loss (2), no Worker (2).
- Editable weights via typed values and nudges (6). Determinism preserved (store never uses Math.random; generation uses a seeded `Rng` from `config.seed + generated.length`).
- Deploy, README, honest numbers, copy pass (10).
- Not in this plan (deferred by the spec): drag-to-edit, multi-head, second block, logit lens, saving models.
