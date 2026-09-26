# Play-test notes

## Existing screenshots reviewed (01–06)

- First impression from `01-untouched-first-screen.png`: this looks like a tiny next-word transformer sandbox. I expect **Step** to run one prediction and expose its internal operations, **Play** to generate repeatedly, **Clear** to erase the prompt, the four tabs to show the architecture/neuron activations/explanation/loss, and **Train** to fit on the shown training text. The map is entirely blank until Step, so the initial screen gives no visual preview of the transformer's internals.
- The copy says it reads the last 24 *words*, while the vocabulary explicitly includes punctuation and line breaks as tokens. That terminology may be misleading if punctuation consumes context slots.
- `02-neurons-before-step.png`: the empty-state wording is clear, but gives no hint that circles will be hoverable/clickable or zoomable later.
- `03-explain-before-step.png`: the beginner overview is the strongest onboarding. It says attention lets each word borrow from words "before it," but does not yet mention Q/K/V, dot products, scaling, softmax, causal masking, residual connections, normalization, or why there are multiple attention heads. Calling the MLP a "thinking step" is intuitive but imprecise.
- The explanation says training adjusts all 15,379 numbers, which I interpret as parameters, but "numbers" is too vague for someone who knows what weights are.
- `04-loss-before-training.png`: almost the whole panel is empty and provides no axes, expected baseline, or definition of training versus testing loss.
- `05-vocabulary-dialog.png`: this is an expanding panel, not a dialog. Its label says 288 words including punctuation and line breaks, reinforcing that the model tokenizes more than words. The line-break token is visible as a bent-arrow glyph, which is good.
- `06-training-text-dialog.png`: opening training text while vocabulary is open appears to push its content entirely below the 900 px viewport. The button says **Hide Training Text**, yet no training text is visible; this makes the control look broken unless the user notices they must scroll the left rail/page.
- Across 01–06 the interface uses tiny, low-contrast system UI styling. At 1440×900, the core content is readable, but the computation controls and status at the bottom are unusually small and easy to miss.

## Run A1 — own paragraph, first Step (`07`–`08`)

- I entered `you said what you did. i think you know what i mean, and you are coming.` Pressing **Step** immediately appended `quiet` to my text rather than merely calculating a prediction. That behavior is not obvious from the main instruction (“run the model”); the button tooltip (only discoverable by hovering) is more accurate.
- The toolbar changed to `20 words`, but the strip contains punctuation as separate items and counts them. This is a token count mislabeled as words.
- Comma and `and` were rendered as faded dashed boxes. Their hover titles explain that they are unknown, but there is no persistent legend, no explicit `<UNK>` replacement, and the displayed text still looks like the original token. As Dev, I cannot tell what numeric input the model actually used for them.
- The Network’s context line says the model is now reading the paragraph **plus** the newly sampled `quiet`, then predicts `wake 0.5%`. So after one Step, the visualization is not showing the calculation that chose `quiet`; it is already showing the next calculation after that choice. This off-by-one mental model is highly confusing.
- The Network map is informative at a structural level (embeddings → transformer → unembedding → probabilities), but labels inside the Transformer block are very small, lines make a circuitous path, and several miniature matrices look like indistinguishable static. I could not map Q/K/V to the familiar `softmax(QKᵀ/√d)V` flow just by looking.
- The predicted-token grid contains hundreds of unlabeled circles. Only one is outlined red; without hovering I cannot know which circle is which word or why this is more useful than a ranked list. “Most likely next words” is tiny and its pale bars are nearly indistinguishable.
- The Explain tab opens at operation 21, not operation 1. The operation list is a strong table of contents and the plain-language explanations are promising.
- Concrete math error/mislabel in `08-own-text-explain-op1.png`: the **next-token vocabulary softmax** is written as `Aᵢⱼ = exp(Sᵢⱼ − max_c Sᵢc) / Σ_c exp(...)`. `A` and two indices are attention-matrix notation; next-token probabilities should be something like `p_j = softmax(z)_j` over vocabulary logits. The page even calls these “scores,” not attention scores, so reusing `A`/`Sᵢⱼ` is misleading.
- The prose says “The largest one is the model’s guess,” but immediately says Step/Play *pick* from the distribution. Those are different: argmax versus sampling. The UI’s generated `quiet` need not be the displayed most-likely `wake`, confirming it samples.
- No console errors or warnings.

## Run A2 — walked the full forward trace with `>` (`09`–`15` plus `08`)

- `|<` goes to an unnumbered **Overview** state, so 21 presses of `>` are needed to reach operations 1–21. The bottom status correctly changes from “Start: nothing computed yet” through named forward steps. I read every operation; step 21 is shown in `08` from Run A1.
- The forward sequence matches the single-block, apparently single-head pre-norm transformer math I know: token + position embeddings; layer norm; Q/K/V linear projections; scaled dot products; causal mask; row softmax; weighted sum of V; output projection + residual; second norm; 20→30→20 ReLU MLP + residual; final norm; vocabulary logits; softmax.
- Serious conceptual error in step 3 (`10-explain-forward-04.png`, which visually shows operation 3 despite the filename): “Nothing is lost by adding: 20 slots is plenty of room for both.” Ordinary vector addition is many-to-one: from an arbitrary sum you cannot recover the two arbitrary 20-vectors. A trained model can learn embeddings that make useful combined encodings, but “nothing is lost” is not mathematically justified.
- Step 8 dynamically displays the wrong shape: despite my 20-token context, the prose says, “With 5 words that is a 5 by 5 grid of scores.” This appears to be a stale hard-coded example and contradicts the visible input/trace.
- Step 8’s actual formula and words correctly describe `qᵢ·kⱼ/√d`; steps 9–11 correctly describe causal masking, row-wise softmax, and `Σⱼ Aᵢⱼvⱼ`. This is the part I could directly reconcile with the attention equation I already knew.
- Step 11’s worked prose example jumps to words (`coming`, `to`, `the`) that are not the active query/key/value selection and partly are not even in my context. It reads like a canned example, not an explanation of the visible calculation.
- Step 15 says the MLP stores facts “that do not depend on the surrounding words.” That is misleading: it operates independently at each position, but its input already contains context from attention, so its activation/output can depend strongly on surrounding words. Position-wise is not context-free.
- The Q/K/V metaphors “question / label / content” are approachable, though “label” for K departs from the standard term **key** shown in the operation ID and may make it harder to connect the diagram to `QKᵀ` elsewhere.
- Explanations repeatedly invite me to “Select a cell in the Network tab to see the numbers,” but no numeric worked example appears by default. For a supposed glass-box view, seeing only generic formulas until I guess that matrices are clickable leaves the main educational payoff hidden.
- The highlighted operation, title, operation ID, and bottom status stay synchronized. The forward trace never stuck and emitted no console errors.
- Visual: `09`–`15` use a clear two-column Explain layout, but the tiny monospace operation label is low contrast. The generic formula sits in a large mostly blank panel, making the absence of substituted numbers conspicuous.

## Run B1 — training attempt (`16`–`17`, interrupted)

- This driver request did not complete: it produced `16-before-training-network.png` and `17-after-training-panel.png`, but both are entirely uniform light gray with no interface, and the later Loss/post-Step screenshots were never written. Because even the pre-training screenshot is blank, I treat this as a driver/service failure rather than an app rendering result.
- The request ran for about 30 seconds and returned no printed driver output. I will retry training in a shorter sequence and will not count these blank captures as product defects unless the behavior repeats in a clean request.

## Run B2 — clean-page retry (`20`)

- The failure repeated on a minimal fresh-page request. `20-training-retry-preflight.png` is entirely blank gray, `body` has no visible text, and the console reports repeated **Invalid hook call** warnings followed by `Cannot read properties of null (reading 'useEffect')` in the App component. This is now a reproducible application startup failure, not merely a long driver request.
- The site worked in Runs A1/A2 and then became unable to mount, so the failure is intermittent/state-of-dev-server rather than a permanent bad URL. A user sees no error message or recovery UI—only a blank page.

## Run B3 — cache-busted recovery attempt (`21`)

- Adding a fresh query string and starting another browser did not recover the app. `21-cachebust-recovery.png` is the same blank gray viewport, with the same invalid-hook warnings and null `useEffect` page error.
- This rules out a stale page-cache explanation. With three consecutive failed fresh-browser runs, remaining interactive scenarios are currently blocked by the running site’s startup crash.

## Run E1 — 1280×720 recovery check (`22`)

- The required laptop viewport could not reach the Network tab because the app still failed before rendering. `22-1280x720-recovery-check.png` contains only the same flat gray background at 1280×720, and the console repeats the invalid-hook/null-`useEffect` crash.
- This run does not provide evidence about responsive layout; it does show that the startup crash is viewport-independent.

