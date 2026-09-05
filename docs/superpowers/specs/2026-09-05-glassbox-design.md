# glassbox — design

Date: 2026-09-05
Status: approved (Johnny, 2026-09-05: "Sure and whatever else you think is necessary. Now start to build")

## What it is

A single-page website where you paste your own text, train a tiny GPT on it in
front of you, and then scrub through its computation one operation at a time —
forwards through a prediction, backwards through learning — with several ways of
looking at the same computation. Every number on screen is a real number from
the model. Nothing is a cartoon.

It is a recreation, for the browser, of the "Tiny Language Model (transformer)"
simulation in Simbrain 4 (Jeff Yoshimi, UC Merced; GPL-2.0, Kotlin/Swing/
Piccolo2D). Simbrain's version is a desktop app inside a 25-year-old
general-purpose neural network lab. glassbox is one focused thing that runs from
a link.

Working name: glassbox (a glass-box model, as opposed to a black box).

## Goals

1. The hook: "I trained an AI on my group chat." Paste text, watch it learn your
   voice in seconds, watch it overfit, generate.
2. The substance: step the computation one operation at a time, forward and
   backward, and see every intermediate tensor and gradient.
3. Several views of the same computation, switchable in one click. Train-and-
   scrub is emphasised; the others are present, not deferred.
4. Runs 100% in the visitor's browser. No server, no API keys, no accounts, no
   per-visitor cost. Their text never leaves their machine.
5. Demonstrable, honest numbers Johnny can point to (see Success criteria).

## Non-goals (v1)

Multi-head attention, more than one transformer block, BPE/subword tokens,
pretrained weights, drag-to-edit weights (typed edits are in), saving or sharing
trained models, mobile-first editing.

## Decisions already made

| Decision | Choice | Why |
|---|---|---|
| Website vs desktop app | Website (static, GitHub Pages) | Friction, shareability, cost, reach. Wrap with Tauri later if ever needed. |
| Stack | React 18 + Vite + TypeScript, hand-written Canvas2D for the diagram | Half the app is panels, half is canvas; React for the former, Canvas2D is ample for a 16k-param model. |
| Tokens | Word-level, lowercase, punctuation split off, newline is a token | Readable output; vocabulary grid can be labelled with real words, as in the video. |
| Vocabulary cap | 512; rarer words fold into `<unk>` | A pasted novel would otherwise make a 10k-row unembedding tile and slow training. |
| Numeric type | `Float64Array` everywhere | Performance is irrelevant at this size; float64 makes gradient checks clean. |
| Randomness | Seeded PRNG (mulberry32) | Same seed → identical trace, or the scrubber lies. |
| Activation | ReLU | Trivial gradient, easy to explain. GELU is a later option. |
| Norm | Pre-LN (GPT-2 style), final LN before unembed | Standard, stable at small scale. |
| Positional encoding | Learned positional embedding table | Draws as a tile like everything else. |
| Embedding / unembedding | Untied | Two tiles, clearer to teach. |
| Attention output projection | Included (W_O) | Omit nothing a real GPT has. |
| License | MIT, with Simbrain credited as the inspiration in README | Maximises reuse and recognition. No Simbrain code or corpus text is copied. |
| Preset corpora | Original or public-domain text only | Simbrain's bundled texts are GPL-2.0. |

## Architecture — five layers

Dependencies point downward only. Views read from Trace + Cursor and nothing
else; that is the contract that makes additional views cheap.

```
shell (React)  -->  views (Canvas2D / DOM)  -->  trace + cursor  -->  model  -->  corpus
                                                      ^
training worker --------------------------------------+ (param sync)
```

### corpus/

- `tokenize(text): { tokens: number[], vocab: Vocab }`
  Lowercase; split on whitespace; split trailing/leading punctuation
  (`. , ? ! ; : " ' ( )`) into their own tokens; `\n` is a token; contractions
  stay whole ("what's"). Vocabulary sorted by frequency, capped at 512,
  index 0 is `<unk>`.
- `split(tokens, ratio=0.6)` → train / test token streams (contiguous, not
  shuffled — the test half is text the model never saw).
- `windows(tokens, T)` → training windows of exactly T tokens with targets
  shifted by one. Iterated in seeded-random order each epoch.
- Errors: fewer than 2·T+2 tokens → refuse with the minimum stated. More than
  50,000 tokens → truncate with a visible notice. Vocabulary over cap → notice
  listing how many word types were folded into `<unk>`.

### model/

A single-block, single-head GPT. Defaults match the video: T=24, d=20,
d_ff=30. About 16k parameters at a 326-word vocabulary.

Parameters (all learned, all drawn as orange-bordered tiles):
`E` (V×d), `P` (T×d), `W_q W_k W_v W_o` (d×d) + biases, `ln1 ln2 ln_f` (γ, β),
`W_1` (d×d_ff) + b, `W_2` (d_ff×d) + b, `U` (d×V) + b.

**The forward pass is an ordered list of named ops, not fused code.** Each op:

```ts
interface Op {
  id: string;            // "q_proj"
  label: string;         // "Query projection"
  kind: OpKind;          // 'linear' | 'add' | 'matmul' | 'softmax' | ...
  inputs: string[];      // tensor names read
  output: string;        // tensor name written
  params: string[];      // parameter names used
  formula: string;       // LaTeX-ish, e.g. "q = LN_1(x_0) W_q + b_q"
  explain: string;       // one plain-English paragraph
  forward(ctx): void;
  backward(ctx): void;   // writes grads for inputs and params
}
```

Forward op order (T = sequence length ≤ context size; no padding — prompts
shorter than T just run shorter):

| # | id | output shape | what |
|---|---|---|---|
| 1 | `embed` | T×d | E[ids] |
| 2 | `pos_embed` | T×d | P[0..T) |
| 3 | `add_pos` | T×d | x0 = tok + pos |
| 4 | `ln1` | T×d | LN(x0) |
| 5–7 | `q_proj` `k_proj` `v_proj` | T×d | linear |
| 8 | `scores` | T×T | q kᵀ / √d |
| 9 | `causal_mask` | T×T | future → −∞ (tile is visibly lower-triangular) |
| 10 | `attn_softmax` | T×T | row-wise softmax |
| 11 | `attn_apply` | T×d | A v |
| 12 | `o_proj` | T×d | linear |
| 13 | `residual1` | T×d | x1 = x0 + attn |
| 14 | `ln2` | T×d | LN(x1) |
| 15 | `ff_up` | T×d_ff | linear |
| 16 | `relu` | T×d_ff | |
| 17 | `ff_down` | T×d | linear |
| 18 | `residual2` | T×d | x2 = x1 + ff |
| 19 | `ln_final` | T×d | |
| 20 | `unembed` | T×V | logits |
| 21 | `softmax_out` | T×V | next-token probabilities |
| 22 | `loss` | scalar | mean cross-entropy over positions (training only) |

Backward runs the same list in reverse; each op's `backward` is recorded as a
step too. The optimizer step (`adam_update`) is one final recorded step whose
"output" is the set of parameter deltas.

Optimizer: Adam (β1=0.9, β2=0.999, ε=1e-8), learning rate default tuned so the
bundled preset overfits in under ten seconds on a laptop (start at 3e-3).
Weight init: N(0, 0.02) like GPT-2, biases 0, LN γ=1 β=0.

Generation: `generate(prompt, n, { temperature, strategy: 'greedy' | 'top-k' |
'top-p', k, p })`. Context is the last T tokens.

Editable parameters: `model.setParam(name, flatIndex, value)`. Any edit
invalidates the current trace; the shell re-runs the forward pass.

### trace/

Everything is computed eagerly once and *recorded*; stepping is replay. That is
what makes backward-stepping trivial and the scrubber deterministic.

```ts
interface Tensor { name: string; shape: number[]; data: Float64Array }
interface TraceStep { opId: string; phase: 'forward' | 'backward' | 'update'; writes: string[] }
interface Trace {
  steps: TraceStep[];
  tensors: Map<string, Tensor>;   // activations and parameters
  grads: Map<string, Tensor>;     // dloss/dtensor, same names
  tokens: number[]; targets?: number[];
}
class Cursor {
  index: number;                              // -1 = nothing computed yet
  status(tensor: string): 'pending' | 'active' | 'done';
  current(): TraceStep | null;
  next(); prev(); seek(i); toEnd();
}
```

`recordForward(model, tokens)` → Trace of steps 1–21.
`recordTrainingStep(model, window)` → Trace of 1–22, then 22..1 backward, then
`adam_update`.

### views/

All views take `(trace, cursor, selection)` and nothing else. Four views plus an
inspector.

- **Flow** (default; the hero). The boxed diagram from the video: Inputs →
  Embedding → Transformer block (Q/K/V, scores, MLP, two residual rails) →
  Unembedding → Softmax → Predicted next token. Tiles are heatmaps drawn to
  offscreen `ImageData`; parameters have orange borders; the active op glows;
  pending tiles are dimmed. **Semantic zoom**: at far zoom the block is a single
  box (text in → text out); zoom in and the sub-ops appear; zoom to a tile and
  cell values render as numbers. Pan/zoom with wheel/drag; no library.
- **Network**. Every neuron and every connection drawn individually, TensorFlow
  Playground style: T×d activation nodes per stage, weights as lines with width
  proportional to |w| and colour by sign. Layout is fixed columns. Hover a line
  for its weight; click a node to highlight its fan-in and fan-out.
- **Math**. The formula for the cursor's current op, rendered (KaTeX), with the
  actual numbers substituted for the selected cell. Below it, the same op's
  backward formula when the cursor is in the backward phase.
- **Loss**. Training and test cross-entropy per iteration (the overfitting
  reveal), accuracy, and a bar chart of gradient magnitude per parameter tensor
  for the current training step.
- **Inspector** (always visible, right side). What the current op is in one
  plain paragraph, the hovered tensor cell's name/index/value/gradient, and for
  a selected cell in a linear op the **dot-product breakdown** — the row, the
  column, the products, the sum. For parameter cells, an editable value field
  and ± nudge buttons (this is "play with the weights" in v1).

### shell/ (React)

Layout: left column (Text, Train, Generate panels); centre (view tabs + canvas);
bottom (transport: play/pause, step forward, step back, step training op,
speed; keyboard `F`/`B` as in Simbrain, arrows); right (Inspector). Stacks
vertically under 900 px.

Training runs in a **Web Worker** holding its own model instance. It posts
`{ iteration, trainLoss, testLoss, trainAcc, testAcc }` per iteration and the
full parameter set on stop. The main-thread model is synced from those
parameters; all trace recording happens on the main thread (a single pass is
milliseconds). "Step training one op" records one training step on the main
thread with the current parameters.

State: one store (Zustand): corpus, hyperparameters, model params, current
trace, cursor index, view, selection, training status/history.

Presets shipped: (1) "Group chat" — ~800 words of original casual two-line
exchanges; (2) "Nursery" — highly repetitive rhyming text, overfits instantly,
good for the first demo; (3) "Alice" — a public-domain excerpt from *Alice's
Adventures in Wonderland* (1865); (4) "Yours" — empty, paste your own.

### Failure handling

| Situation | Behaviour |
|---|---|
| Text shorter than 2·T+2 tokens | Refuse to train; show the minimum. |
| Text over 50k tokens | Truncate; notice with the count kept. |
| Vocabulary over 512 | Train; notice with how many types became `<unk>`. |
| Loss becomes NaN/Inf | Stop training, restore last finite params, offer a lower learning rate. |
| Prompt token not in vocabulary | Map to `<unk>`; show which words were unknown. |
| Worker unsupported | Train on main thread with a "may stutter" notice. |

## Testing

vitest, run on every push.

1. **Numerical gradient check for every op**, in isolation, against central
   finite differences (ε=1e-6, tolerance 1e-5 relative). Non-negotiable: a
   wrong gradient produces a model that trains *almost* right and teaches
   something false.
2. End-to-end gradient check of the whole model on a 5-token window.
3. The model overfits a 30-token corpus to loss < 0.05 within 300 iterations.
4. Same seed and corpus → byte-identical trace (determinism).
5. Tokenizer: round-trip on the presets; punctuation and newline cases; cap
   folding.
6. Cursor: `status()` transitions for forward, backward, update phases.
7. Generation: greedy is argmax; temperature → 0 approaches greedy; top-k never
   samples outside the top k.

## Deployment

Vite static build → GitHub Pages via a GitHub Actions workflow on push to
`main` (repo `jbtk-cell/glassbox`, base `/glassbox/`). No secrets.

## Success criteria (honest, measurable)

- Trains the ~800-word "Group chat" preset to > 95 % training accuracy in under
  10 s on a MacBook Air, and the test curve visibly diverges.
- Every op passes the gradient check.
- Any op can be stepped to, forward or backward, and every tensor it touches
  can be hovered for its value and gradient.
- Bundle under 300 KB gzipped; loads in under 2 s on a normal connection.
- Works in current Chrome, Safari and Firefox.

## Copy

All explanatory text (op `explain` strings, panel help, README) is user-facing
prose; it gets a humanizer / stop-slop pass before release, per Johnny's
standing rule.

## Roadmap after v1

- v2: drag-to-edit weights on tiles and wires; freeze and prune regions; live
  re-prediction while dragging.
- v3: gradient-flow animation through the backward walk; a 2-D slice of the
  loss surface around the current parameters.
- v4: multi-head attention with a head deck; a second block; a character-level
  tokenizer option; logit lens at each residual checkpoint.

## References

- Simbrain: https://simbrain.net, https://github.com/simbrain/simbrain,
  docs at https://docs.simbrain.net/docs/network/languageModels.html
- Neighbours: Transformer Explainer (poloclub.github.io/transformer-explainer),
  bbycroft.net/llm, TensorFlow Playground. All three are inference-only or
  non-transformer; the open ground is *trainable + steppable, in a browser*.
