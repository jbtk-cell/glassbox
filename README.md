# glassbox

A tiny GPT you can train on your own text and step through one operation at a time, in your browser.

Paste in anything: your group chat, a story, notes. Press Train and watch the loss curves as it learns your writing in about fifteen seconds, and watch it overfit. Type a prompt and it predicts the next word. Then scrub through that prediction one operation at a time, forwards through the computation or backwards through the gradients, and hover any number to see where it came from.

Nothing leaves your browser. There is no server and no account. The whole model is about fifteen thousand numbers, and your laptop trains it.

## What it is for

This is a real transformer, shrunk until every number fits on screen. The model is a one-block, one-head transformer with the same parts as GPT-2: token and position embeddings, layer norm, queries, keys and values, a causal mask, attention, an output projection, a feed-forward layer, two residual connections, an unembedding and a softmax. Every tile is a live matrix from the model.

It is a browser recreation of the "Tiny Language Model" simulation in [Simbrain](https://simbrain.net), Jeff Yoshimi's neural network lab, which does this as a desktop application. glassbox takes that one simulation and makes it run from a link.

## How to use it

The layout follows Simbrain's: a toolbar, a Text Inputs window, a Language Model Controls panel, and the Network beside them. Training and model settings live in dialogs.

1. **Model...** picks the training text (a preset, or paste your own) and the model sizes, then **Create**.
2. **Train...** opens the Train Network dialog: press **Train**, watch the loss and accuracy curves, press **Stop** when training accuracy is high and test accuracy has stalled. That gap is overfitting.
3. Click in Text Inputs and type a few words. **Step** predicts the next word and adds it; **Play** keeps going. The Network shows the computation that produced each word.
4. Under the Network, the Computation bar steps that computation one operation at a time: `F` and `B`, or the buttons. The tile being computed glows; tiles that have not been computed yet are dimmed. Scroll or use the zoom buttons to zoom into any tile until the numbers appear; hover a cell to read it; click a cell for its details.
5. **Record training step** (or `T`) records one training step. Stepping backwards through it shows the gradient flowing into every tile, then the Adam update to every weight.
6. Click a weight, type a new value in the cell panel, press **Set**. The prediction re-runs as soon as you do.

Tabs above the Network: **Network** (Simbrain's map: Inputs, Embedding, the Transformer block, Unembedding, and a circle per word for the predicted next token), **Neurons** (every neuron for one position; hover one to see its connections), **Math** (the formula for the current operation with the real numbers substituted), **Loss** (curves and gradient sizes).

## What you are looking at

| Operation | Formula | What it does |
|---|---|---|
| Token embedding | `tok = E[id]` | Each word id picks a row of numbers from a learned table. |
| Position embedding | `pos = P[t]` | Each position gets its own learned row, so order matters. |
| Add position | `x0 = tok + pos` | The residual stream starts here. |
| Layer norm 1 | `h1 = ln(x0)` | Rescale each row to a predictable range. |
| Query, key, value | `q = h1 W_q`, `k = h1 W_k`, `v = h1 W_v` | What each position asks for, advertises, and hands over. |
| Attention scores | `S = q k^T / sqrt(d)` | How relevant each earlier position is to each position. |
| Causal mask | `S[i, j] = -inf for j > i` | No looking at the future. The tile is a triangle. |
| Attention weights | `A = softmax(S)` | Scores become weights that add to one. |
| Apply attention | `ctxv = A v` | Each position gathers a weighted mix of earlier values. |
| Output projection | `attn = ctxv W_o` | Decide how to write the gathered information back. |
| Residual add 1 | `x1 = x0 + attn` | Add it to the stream instead of replacing it. |
| Layer norm 2 | `h2 = ln(x1)` | Rescale again. |
| Feed-forward up, ReLU, down | `relu(h2 W_1) W_2` | Per-position computation; the model's stored patterns. |
| Residual add 2 | `x2 = x1 + ff` | Add it to the stream. |
| Final layer norm | `hf = ln(x2)` | Rescale before reading out. |
| Unembedding | `logits = hf U` | One score per vocabulary word. |
| Softmax | `probs = softmax(logits)` | Scores become next-word probabilities. |
| Cross-entropy loss | `-mean log probs[target]` | Training only: how surprised the model was by the real next word. |

## Numbers

Measurements from a MacBook Air (Apple Silicon) in Chrome on 2026-09-05, default settings (context 24, embedding 20, hidden 30, learning rate 0.003):

- Group chat preset: 1,321 tokens, 289-word vocabulary, 768 training windows.
- Model size: 15,379 parameters.
- Training accuracy passed 95 % after 20 epochs in 15.4 seconds. Test accuracy at that point: 28 %. Test loss rose from 5.3 to 10.5 while training loss fell from 2.7 to 0.19. The model memorised the training text, which is the overfitting the tool exists to show.
- Tests check every operation's backward pass against central finite differences, and the whole model end to end. 130 tests.
- Page weight: about 175 KB gzipped (JavaScript, worker and CSS). Math fonts load on demand.

## How it is built

**Engine.** Plain TypeScript, `Float64Array`, no numerics library. The forward pass is an ordered list of named operations; each one has its own `forward` and `backward`, and a plain-English explanation. Adam optimizer. Greedy, top-k and top-p sampling with temperature. Everything is seeded, so the same text and seed give the same model.

**Trace.** Running the model records every operation and every intermediate tensor and gradient. Stepping is replay over that recording, which is why stepping backwards works the same as stepping forwards.

**Views.** React for the panels, hand-written Canvas2D for the diagram. Every view reads the same trace and cursor and nothing else, so adding a view does not touch the model. Training runs in a Web Worker so the page stays responsive.

## Running it locally

```
npm install
npm run dev      # http://localhost:5173/glassbox/
npm test
npm run build    # static files in dist/
```

## Credits

- [Simbrain](https://simbrain.net) by Jeff Yoshimi (UC Merced) is the inspiration and the original. Its transformer visualisation, and the idea of stepping training one operation at a time, are what this recreates. No Simbrain code or text is used.
- Neighbours worth knowing: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (a live GPT-2 in the browser) and [Brendan Bycroft's LLM visualisation](https://bbycroft.net/llm) (a 3D walkthrough). Both are inference-only. In glassbox you train the model and watch it learn.
- The Alice preset is the opening of *Alice's Adventures in Wonderland* (1865), from Project Gutenberg. The nursery rhymes are traditional and public domain. The group chat preset was written for this project.

## License

MIT. See LICENSE.
