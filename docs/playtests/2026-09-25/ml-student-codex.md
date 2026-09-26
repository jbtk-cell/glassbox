# Glassbox play-test report

## 1. First 60 seconds: what you thought the site was for, and what you thought each visible control did, BEFORE clicking anything. Then what turned out to be true.

I read Glassbox as a tiny next-word transformer sandbox. The left side looked like the place to enter a prompt and train a model; the large right side looked like an empty microscope that would fill in after a prediction.

Before clicking, I expected **Step** to run one forward pass without changing my text, **Play** to generate words on a loop, and **Clear** to erase the prompt. I expected **Network** to show the transformer as blocks and arrows, **Neurons** to expose activations, **Explain** to teach the current operation, and **Loss** to plot training and testing loss. I read **Train** as “fit the model on Training Text,” **Reset weights** as “return to random parameters,” and **Record training step** as a way to capture one backward pass. The `|<`, `<`, `>`, and `>|` controls looked like trace navigation, though the app did not explain them.

Most of those guesses were close. **Step** also sampled and appended a word. After my first Step, the text gained `quiet`, while the Network already predicted the word *after* `quiet` (`wake 0.5%`) in `07-own-text-network-step.png`. The top counter said `20 words`, but it counted punctuation tokens. **Explain** gave me a 21-operation forward trace. **Show Vocabulary** and **Show Training Text** expanded sections in the left rail rather than opening dialogs.

The best onboarding was the untouched Explain overview in `03-explain-before-step.png`. The initial Network and Neurons panels in `01-untouched-first-screen.png` and `02-neurons-before-step.png` told me to press Step but did not preview the architecture, the hover/click interactions, or what I could learn from them.

## 2. Walkthrough: what you did, in order, and what you saw. Note every moment you were unsure what was happening or what to do next.

1. I inspected the untouched screen and all four tabs. Network and Neurons were empty. Explain described a 20-dimensional, 24-slot next-word model. Loss said “Train the model to see loss curves.”
2. I opened Vocabulary. It listed 288 entries and counts, including punctuation and a line-break symbol (`05-vocabulary-dialog.png`). I opened Training Text while Vocabulary stayed open. The button changed to **Hide Training Text**, but no training text appeared in the viewport (`06-training-text-dialog.png`); I had to infer that the first expanded section pushed it below the fold.
3. I entered `you said what you did. i think you know what i mean, and you are coming.` and pressed Step. The app appended `quiet`, marked comma and `and` with faded dashed boxes, and drew the full model (`07-own-text-network-step.png`). I did not know whether dashed tokens had one shared unknown embedding, no embedding, or some fallback; a hover tooltip is the sole explanation and says they are absent from the training text.
4. I opened Explain. It started at operation 21 rather than operation 1 (`08-own-text-explain-op1.png`). I used `|<`, landed on an extra Overview state (`09-explain-forward-01.png`), then pressed `>` through the trace. The operation list and bottom status stayed in sync. The sequence matched the transformer forward pass I learned in class.
5. I was most confused at operation 3. The page claimed that adding word and position rows loses nothing (`10-explain-forward-04.png`), which is false for arbitrary vectors. At operation 8 the text said a five-word prompt makes a 5×5 attention grid even though my visible context had 20 tokens. At operation 15 the page said the MLP stores facts independent of surrounding words (`14-explain-forward-16.png`), although its input already contains contextual attention output.
6. I tried to train for about 15 seconds, inspect the Training panel and Loss tab, then Step again. The app stopped mounting during that run. `16-before-training-network.png` and `17-after-training-panel.png` are blank gray frames; the request ended before it could take the Loss and post-training screenshots.
7. I retried with a minimal fresh browser, then a cache-busted URL. Both produced blank pages (`20-training-retry-preflight.png`, `21-cachebust-recovery.png`). The console repeated “Invalid hook call” and ended with a null `useEffect` page error in the App component.
8. I retried at the required 1280×720 viewport. The same startup crash produced `22-1280x720-recovery-check.png`, so I could not judge the responsive Network layout.

The startup crash blocked the requested training comparison, recorded backward-pass walkthrough, Neurons hover/click/wheel test, and break tests for empty text, one word, unknown words, punctuation, double Train, and Reset weights mid-run. I did not substitute guesses for those missing results.

## 3. Visual defects: anything overlapping, cut off, unreadable, misaligned, inconsistent, or ugly. One line each with screenshot name and location.

- `20-training-retry-preflight.png`, `21-cachebust-recovery.png`, `22-1280x720-recovery-check.png`: the whole viewport is blank light gray after the startup crash; the page gives no error or recovery control.
- `06-training-text-dialog.png`, left rail below Language Model Controls: **Hide Training Text** is visible, but the expanded training text itself sits below the viewport because the open Vocabulary section pushes it down.
- `07-own-text-network-step.png`, transformer block: labels such as `q`, `k`, `v`, `FF input`, and `Write back` are tiny, and the circuit-like route is hard to follow without frequent switches to Explain.
- `07-own-text-network-step.png`, Predicted next token panel: 288 circles have no visible labels; the grid consumes a large area but communicates less than a short ranked list.
- `07-own-text-network-step.png`, Most likely next words inset: the label and pale bars are too small and faint at 1440×900.
- `01-untouched-first-screen.png` through `15-explain-forward-21.png`, bottom computation bar: the trace buttons, slider, and monospace status use smaller text than the rest of the interface and are easy to miss.
- `08-own-text-explain-op1.png`, operation label above the heading: light gray monospace text has low contrast.
- `10-explain-forward-04.png` through `15-explain-forward-21.png`, Explain detail: a generic formula floats in a vast empty panel while the promised numeric example remains hidden until the user selects a cell elsewhere.

## 4. Functional defects: anything that did nothing, did the wrong thing, errored, got stuck, or behaved differently on repeat.

1. **The app began crashing on every fresh load.** Three clean retries, including a cache-busted URL and a different viewport, reproduced a blank page plus invalid-hook/null-`useEffect` console errors (`20`, `21`, `22`). Runs A1/A2 worked first, which makes this an intermittent startup failure rather than a bad URL.
2. **Step visualizes the prediction after the generated token, not the calculation that selected that token.** My Step appended `quiet`; the context strip then included `quiet` and showed `wake 0.5%` as the next prediction (`07`). The trace does not answer the question I had after clicking: “Why did it choose quiet?”
3. **The explanation uses stale input dimensions.** Operation 8 says “With 5 words that is a 5 by 5 grid” while the active input has 20 tokens.
4. **The toolbar mislabels tokens as words.** It reports `20 words` while the strip counts `.` and other punctuation as separate items (`07`).
5. **Opening Training Text can appear to do nothing.** The control changes state, but the content lands below the visible left rail when Vocabulary is already open (`06`).
6. **The next-token softmax uses attention notation.** In `08`, a vocabulary probability is labeled `Aᵢⱼ` and calculated from `Sᵢⱼ`, even though `A` and `S` already mean attention weights and scores. A vocabulary distribution needs one vocabulary index, such as `p_j = softmax(z)_j`.
7. **The site contradicts its sampling behavior.** Explain says the largest probability is the model’s guess, then says Step and Play pick from the probabilities. My run sampled `quiet` while the next displayed maximum was another word, so the distinction between argmax and sampling matters.

No console errors appeared during the two successful forward-trace runs.

## 5. Comprehension: after using it, explain in your own words what the model is doing when you press Step. Then say which parts of that understanding came from the site and which came from your prior knowledge. Say plainly what the site failed to explain.

Pressing Step tokenizes the last part of the text, looks up a 20-number vector for each token and position, and adds each pair. The model normalizes those vectors and makes Q, K, and V projections. It compares each query with allowed earlier keys using a scaled dot product, masks future positions, applies a row softmax, and takes weighted sums of the value vectors. An output projection and residual connection write that context back. A normalized 20→30→20 ReLU network processes each position, followed by another residual connection and final normalization. The last position becomes one logit per vocabulary item, and a softmax turns those logits into a distribution. Step samples from that distribution, appends the sampled token, then shows a trace for the following prediction.

The site taught me the model-specific sizes, order of all 21 operations, causal mask, residual paths, final unembedding, and the fact that this model uses ReLU. My machine-learning course supplied the meaning of `QKᵀ/√d`, softmax, weighted sums of V, logits, and loss curves. That prior knowledge let me catch the wrong vocabulary-softmax notation and the claim that vector addition loses nothing.

The site failed to explain whether it has one attention head or several; the diagram looks single-headed but never says so. It did not explain how unknown tokens enter the model, why Step advances past the word I want to inspect, how temperature changes sampling, how the train/test split works, or what baseline loss and accuracy I should expect. It calls learned values “numbers” instead of parameters and hides the substituted arithmetic behind cell selection. The crash prevented me from learning anything from its backward-pass trace, gradients, loss graph, or post-training comparison.

Two explanations are wrong enough to damage understanding. Adding two arbitrary vectors is not lossless, and a position-wise MLP still receives context-dependent activations. “Question / label / content” is approachable wording for Q/K/V, but calling K a label makes standard transformer material harder to recognize.

## 6. Top 5 changes, ranked, each with the one-sentence reason.

1. **Fix the invalid-hook startup crash and add an error boundary with a reload message.** A blank page ended the test and blocked every training and interaction feature.
2. **Make Step preserve and explain the prediction it sampled.** Users want to inspect why `quiet` was chosen, not jump straight to the model state that predicts the word after `quiet`.
3. **Correct the math and prose: vector addition is not lossless, attention-grid size must use the active token count, the position-wise MLP is context-dependent, and vocabulary softmax needs vocabulary notation.** These errors teach the wrong model to a learner who trusts the diagram.
4. **Show one synchronized numeric worked example by default for the active token/cell.** Generic formulas plus an instruction to hunt for a clickable matrix do not fulfill the glass-box promise.
5. **Replace or supplement the unlabeled prediction-circle wall with a readable ranked probability table, and enlarge the Network labels and trace controls.** The current view spends space on marks that a first-time user cannot decode.

## 7. Would you show this to a friend? One honest sentence.

No: the forward-pass outline has real teaching value, but I would not send a friend to a tool that teaches several false statements and can collapse into a blank page before training.
