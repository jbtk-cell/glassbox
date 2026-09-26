# Play-test notes — Maya persona (screenshots 01-20)

## 01-first-screen.png
**Action:** goto http://localhost:5173/glassbox/, wait 2500ms (page load, no clicks yet).
**What Maya thinks she's looking at:** Some kind of text-prediction tool called "glassbox." Left column has a text box with "you coming" typed in and two word-chips below it ("you", "coming"). There's a "Training" panel with a Train button, a Learning rate box, and stats all showing "–". Right side is a big empty canvas that says "The map of the model appears here after you press Step." Top has Step/Play/Clear buttons and a "Model..." button.
**Overlap/cutoff/unreadable text:** None obviously broken. Text is small at 900px width (labels like "Configure Sampling Strategy...", "Testing accuracy") but legible.
**Broken/inconsistent vs previous:** N/A, first screenshot.
**Maya's question:** "What is 'Step' going to do, and why are there already two words typed in for me?" She has no idea what "Learning rate 0.003" or "Configure Sampling Strategy" mean — no tooltip or explanation is visible for either.

## 02-after-step.png
**Action:** click "Step", wait 2500ms.
**What Maya thinks she's looking at:** The canvas has filled with a dense technical diagram: a grid of dots labeled "Predicted next token," a ranked word list ("before 0.4%", "person", "way", ...), and boxes labeled "Unembedding," "Transformer block," "Embedding," "Inputs," with sub-boxes like "FF output," "Hidden -> output," "Attention output," "Q K V." A legend at bottom right explains dot colors. The bottom bar now shows "Forward step 21 of 21: Next-token probabilities" with playback-style controls (|<, <, >, >|).
**Overlap/cutoff/unreadable text:** The diagram boxes are dense and their internal labels ("FF output", "Input -> hidden", "Attention output") are small — at this 900px width they're on the edge of legible, no zoom hint is given on this screen.
**Broken/inconsistent vs previous:** The text box at top left now reads "you coming something" (3 words, counter says "3 words") — but nothing in the action list typed "something"; it appeared purely from clicking Step. Meanwhile the model's own predicted next word, shown at top of canvas, is "before" (0.4% confidence) — a different word than what got appended to the input. This is confusing: Step seems to silently add a word to Maya's own text that doesn't match what it just told her it predicted.
**Maya's question:** "Wait — did it just add 'something' to my sentence? I didn't type that. And it says it predicted 'before,' so why isn't that the word that got added?"

## 03-hover-canvas-center.png
**Action:** hoverAt original-coords (895,482) — inside the "Predicted next token" grid area, wait 1200ms.
**What Maya thinks she's looking at:** Same diagram as 02, but now a small black tooltip reading "mother 0.32%" has popped up over one of the dots in the "Predicted next token" grid — so hovering a dot shows which word it stands for and its probability.
**Overlap/cutoff/unreadable text:** Tooltip text itself is fine, but nothing on screen told Maya that hovering the grid would do this — she'd have found it by accident.
**Broken/inconsistent vs previous:** Consistent extension of 02 (same layout, new tooltip).
**Maya's question:** "Is 'mother' a real word it could have predicted here? Why this one dot and not another?"

## 04-hover-upper-left.png
**Action:** hoverAt original-coords (600,300) — a different spot, toward the embedding/inputs area on the left side of the canvas, wait 1200ms.
**What Maya thinks she's looking at:** She expects a new tooltip relevant to whatever is at this new spot (maybe the "Embedding" or "Word table" box).
**Overlap/cutoff/unreadable text:** None new.
**Broken/inconsistent vs previous:** This screenshot looks visually identical to 03-hover-canvas-center.png — the same "mother 0.32%" tooltip is still shown in the same place, even though the mouse moved to a different part of the canvas. Either the tooltip is stuck/not clearing, or hovering elsewhere doesn't update it. This reads as broken.
**Maya's question:** "Did my hover even register? Why does it still say 'mother' when I moved my mouse somewhere else?"

## 05-zoomed-in.png
**Action:** wheel zoom in at original-coords (895,482), delta -600, wait 1200ms.
**What Maya thinks she's looking at:** The "Predicted next token" grid has zoomed in enough that individual dots now show actual words stacked in a small grid ("arm," "move," "main," "notice," "melting," "mother," "place," "quiet," "morning," etc.) — so each dot in the earlier screenshots represents one vocabulary word.
**Overlap/cutoff/unreadable text:** The "+ / − / Fit" zoom-control row sits immediately above the now-enlarged grid, crowding the first row of word cells directly beneath it. To the right of the word grid (roughly x=395-520, y=60-235 in the 900px image) there is a blank white box with no visible content — looks like the "Probabilities" ranked list panel got clipped/emptied by the zoom, leftover dead space.
**Broken/inconsistent vs previous:** New information is legible for the first time (actual words), which is good, but the blank box to its right looks unfinished/broken.
**Maya's question:** "What's that empty white box next to the words supposed to show?"

## 06-zoomed-out.png
**Action:** wheel zoom out at same point, delta +1400 (net zoom-out past the original view), wait 1200ms.
**What Maya thinks she's looking at:** The whole model diagram has shrunk to a tiny cluster of shapes near the center of a mostly blank white canvas.
**Overlap/cutoff/unreadable text:** Nothing readable at this zoom — expected for zoomed-out state.
**Broken/inconsistent vs previous:** The "+ / − / Fit" zoom-control row that was visible in every earlier canvas screenshot (02-05) is now completely gone from view. It isn't a fixed on-screen toolbar — it appears to be part of the pannable/zoomable canvas content, so once you zoom out (or pan) it scrolls away with everything else, leaving no visible way to get back to a sane view.
**Maya's question:** "I zoomed out too far — how do I get back to normal? I don't see the buttons anymore."

## 07-after-pan.png
**Action:** drag from (700,400) to (1100,600) — panning the (already zoomed-out) canvas, wait 1000ms.
**What Maya thinks she's looking at:** A mostly blank canvas with the tiny model diagram now pushed into the bottom-right corner, and a small black tooltip reading "milk 0.36%" floating alone in the middle of the empty white space.
**Overlap/cutoff/unreadable text:** The "milk 0.36%" tooltip has nothing to attach to on screen — the grid cell it refers to isn't visible here — so it reads as a stray label floating in a blank area.
**Broken/inconsistent vs previous:** Confirms 06's problem: still no zoom/fit controls visible anywhere on screen after panning. A leftover hover tooltip from before the pan/zoom is now orphaned in empty space, which looks glitchy.
**Maya's question:** "Where did everything go, and what is 'milk 0.36%' next to nothing?"

## 08-after-fit.png
**Action:** click "Fit" button, wait 1200ms.
**What Maya thinks she's looking at:** The full model diagram is back, centered and at a normal size, matching what 02 looked like — Fit clearly re-centers and re-scales the view.
**Overlap/cutoff/unreadable text:** None; this screen matches the earlier good layout.
**Broken/inconsistent vs previous:** This is the recovery from 06/07's lost state — good that it works, but the fact a user needed a "reset button" to recover from ordinary scroll-wheel zooming is itself the problem (noted above).
**Maya's question:** "Good, that fixed it — but why did I have to press something to undo just scrolling my mouse wheel?"

## 09-tab-neurons.png
**Action:** click "Neurons" tab (after the earlier Step), wait 2000ms.
**What Maya thinks she's looking at:** A much denser diagram than the Network tab: many vertical columns of small circles labeled with short codes (tok, pos, x0, h1, q, k, ctxv, attn_out, x1, h2, ff_pre, ff_act, x2, ff_out, hf, logits, probs), connected by tan hourglass-shaped bands labeled W_q, W_k, W_v, W_o, W_1, W_2, U, with little operation labels (add, norm, ReLU, softmax, "q, k, softmax attention," "mix v by attention"). A highlighted note at the bottom reads "Numbers for position 2 of 2: 'coming'. Hover a circle to see its wires."
**Overlap/cutoff/unreadable text:** Every column header (h1, ctxv, ff_pre, hf, etc.) is an unexplained abbreviation with no on-hover or on-screen definition visible here — for someone who's never heard "attention" or "embedding" this is a wall of cryptic shorthand.
**Broken/inconsistent vs previous:** Consistent with the Network tab's visual style, just far denser.
**Maya's question:** "There are like a dozen new abbreviations here I've never seen — which one do I even hover over first?"

## 10-neurons-hover.png
**Action:** hoverAt original-coords (700,400), wait 1200ms (the on-screen instruction says "Hover a circle to see its wires").
**What Maya thinks she's looking at:** She expects hovering to highlight the "wires" (connections) for whatever circle is under the cursor, per the instruction text.
**Overlap/cutoff/unreadable text:** None new.
**Broken/inconsistent vs previous:** This screenshot appears identical to 09-tab-neurons.png — no highlighted wires, no tooltip, nothing visibly changed even though the app explicitly told her to "hover a circle to see its wires." Either the hover missed every circle, or the highlight-on-hover feature isn't firing.
**Maya's question:** "I did hover — where are the wires it told me I'd see?"

## 11-tab-explain.png
**Action:** click "Explain" tab, wait 2000ms.
**What Maya thinks she's looking at:** A left-hand numbered outline (Overview; INPUTS 1-3; ATTENTION 4-13; FEED-FORWARD 14-18; PREDICTION 19-21, with "21. Turn scores into probabilities" highlighted) and a right-hand explanation panel: "Forward step 21 of 21 (softmax_out) — Turn scores into probabilities," a plain-English paragraph ("e to the power of each score, divided by the total... every word in the vocabulary has a probability... Step and Play pick a word from these probabilities"), a "Where to look" callout with "Show in Network" / "Show in Neurons" buttons, an "IN NUMBERS" section with a math formula, and Previous/Next buttons (Next greyed out since this is the last step).
**Overlap/cutoff/unreadable text:** None cut off; text is small but readable. Step name "17. Back to 20" in the outline is confusing out of context (skips backward in the numbering with no visible explanation of why).
**Broken/inconsistent vs previous:** This tab is the most legible, plain-language screen so far — first real attempt at explanation — but it assumes comfort with exponentials/"e to the power of" and words like "attention weights" and "softmax" without defining them first.
**Maya's question:** "What does 'e to the power of each score' actually mean in plain terms, and why does step 17 go 'back to 20'?"

## 12-tab-loss.png
**Action:** click "Loss" tab, wait 2000ms.
**What Maya thinks she's looking at:** A nearly blank white canvas with a single centered line of text: "Train the model to see loss curves." Clean, unambiguous empty state.
**Overlap/cutoff/unreadable text:** None — this is the clearest, least cluttered screen so far.
**Broken/inconsistent vs previous:** Consistent and appropriately minimal; no complaint here.
**Maya's question:** "Okay, so I need to press Train first — what does Train actually do to the model?"

## 13-compute-first.png
**Action:** click "Network" tab, wait 1500ms; click "|<" (rewind computation slider to start), wait 1800ms.
**What Maya thinks she's looking at:** The diagram is back, but now everything is "un-filled" — the "Predicted next token" grid is all empty circles, the "Probabilities" box says "Most likely next words: not computed yet," and the header reads "next word: not computed yet." The bottom status bar says "Start: nothing computed yet (21 steps to go)."
**Overlap/cutoff/unreadable text:** None; this is a clean, well-labeled empty/start state — genuinely one of the clearer screens.
**Broken/inconsistent vs previous:** Makes sense as the "rewind" state of a 21-step playback control that is separate from the "Step" button in the top toolbar — two different things are both called some variant of "step," which is a naming collision worth flagging (see Explain-tab notes too).
**Maya's question:** "Is this 21-step slider the same as the 'Step' button up top, or something different?"

## 14-compute-step2.png
**Action:** click ">" once, wait 1500ms.
**What Maya thinks she's looking at:** Status bar now reads "Forward step 1 of 21: Token embedding," and the "Inputs" box (bottom-left, "One dot per word") now has an orange highlight outline, showing her which part of the diagram this step corresponds to.
**Overlap/cutoff/unreadable text:** None.
**Broken/inconsistent vs previous:** Minor naming mismatch worth flagging: the file is named "...step2" but the on-screen counter reads "step 1 of 21" — only relevant if someone cross-references filenames to on-screen state, but shows how easy it is to lose count of which "step" is meant.
**Maya's question:** "What is 'Token embedding'? The word 'embedding' was never explained anywhere so far."

## 15-compute-step5.png
**Action:** three more clicks of ">" (wait 800ms each), then wait 1500ms, shot.
**What Maya thinks she's looking at:** Status bar now reads "Forward step 4 of 21: Layer norm 1." An orange highlight box has moved to sit over the "K" cell inside the Attention/Q-K-V cluster.
**Overlap/cutoff/unreadable text:** None cut off, but the highlighted box for "Layer norm 1" lands on the "K" (key) cell, not on anything visibly labeled "norm" — the Network tab never shows a box literally called "norm" (that label only appears over on the Neurons tab), so the step name and the highlighted diagram element don't obviously correspond for someone reading only this tab.
**Broken/inconsistent vs previous:** The highlight-follows-step mechanic itself is a nice touch, but here it seems to point at the wrong/an unlabeled part of the diagram relative to its step name.
**Maya's question:** "Why is 'Layer norm' highlighting the K box? Where's the actual 'norm' step?"

## 16-compute-last.png
**Action:** click ">|" (jump to last step), wait 1500ms.
**What Maya thinks she's looking at:** Same fully-computed view as 02/08 — "Forward step 21 of 21: Next-token probabilities," grid filled in, "next word: before 0.4%" header, "Predicted next token" box highlighted in orange.
**Overlap/cutoff/unreadable text:** None.
**Broken/inconsistent vs previous:** Consistent, correctly matches the earlier fully-stepped state — no complaint.
**Maya's question:** "So this is the same ending point I already saw after pressing Step — was the point of scrubbing through 21 steps just to watch it rebuild the same picture piece by piece?"

## 17-compute-past-last.png
**Action:** click ">" once more, already at the last step, wait 1200ms.
**What Maya thinks she's looking at:** Same fully-computed view as 16 — "Forward step 21 of 21: Next-token probabilities," same grid, same highlighted "Predicted next token" box.
**Overlap/cutoff/unreadable text:** None.
**Broken/inconsistent vs previous:** Good — clicking past the last step correctly stayed clamped at step 21 instead of erroring or overshooting. No complaint here, worth noting as something that works right.
**Maya's question:** None new — she'd just assume the button is now inactive at the end.

## 18-after-clear.png
**Action:** click "Clear," wait 1200ms.
**What Maya thinks she's looking at:** Everything resets to the true empty/first-load state: textarea shows greyed placeholder "Type a few words here," word count "0 words," no word chips, and the canvas shows its original instructions ("Type a few words in Text Inputs and press Step..." banner plus "The map of the model appears here after you press Step." in the middle). This matches 01-first-screen.png exactly.
**Overlap/cutoff/unreadable text:** None.
**Broken/inconsistent vs previous:** Consistent and correct — Clear fully resets the view.
**Maya's question:** "Does Clear also erase whatever the model already 'learned' from stepping through, or just the text?" (Iterations still shows 0, so unclear if anything was retained.)

## 19-step-on-empty.png
**Action:** click "Step" with an empty text box, wait 1500ms.
**What Maya thinks she's looking at:** She'd expect either an error/prompt ("type something first") or for the button to be disabled. Instead, this screenshot is pixel-for-pixel the same file size and appearance as 18-after-clear.png — clicking Step on empty input produced zero visible change: no message, no shake, no disabled state, nothing.
**Overlap/cutoff/unreadable text:** None (nothing appeared at all).
**Broken/inconsistent vs previous:** This is a real gap: Step is clickable with empty input but silently no-ops, giving a first-time user no signal about whether it worked, is broken, or requires text first.
**Maya's question:** "Did I actually click the button? Is it broken, or do I need to type something first?"

## 20-my-paragraph.png
**Action:** fill textarea with "my biology teacher said the mitochondria is the powerhouse of the cell and i still think that is funny," wait 1500ms.
**What Maya thinks she's looking at:** Her own sentence now sits in the Text Inputs box (wrapped to 3 lines), the header says "19 words," and 19 word-chips appear below it, each one matching a word from her sentence exactly, in order.
**Overlap/cutoff/unreadable text:** None — word wrap and chip-splitting both look correct even with a longer, punctuation-free sentence.
**Broken/inconsistent vs previous:** Consistent behavior, word/chip count is accurate. The right panel still shows the pre-Step instructions since Step hasn't been pressed yet on this new text — correct/expected.
**Maya's question:** "Okay, it split my sentence into words correctly — now if I press Step, will it actually try to predict what comes after MY sentence, using the words I chose?"

## Top 5 worst problems (screenshots 01-20), ranked

1. **Step silently rewrites the user's own text with a word that contradicts the model's own displayed prediction** (02-after-step.png). Clicking "Step" changed the input from "you coming" to "you coming something," but the header simultaneously says the model's actual top prediction was "before" — the appended word doesn't match what the app itself just claimed it predicted. This is actively misleading, not just confusing.

2. **Zooming/panning the canvas can strand the user with no visible way back** (06-zoomed-out.png, 07-after-pan.png). The "+ / − / Fit" controls are not a fixed toolbar — they scroll and zoom away with the canvas content, so a normal mouse-wheel zoom-out or drag-to-pan can leave the screen looking broken/empty with no visible control to recover (recovery only works if the user happens to remember "Fit" existed from an earlier screen).

3. **The explicit on-screen instruction "Hover a circle to see its wires" does nothing when followed** (09-tab-neurons.png vs. 10-neurons-hover.png). These two screenshots are visually identical even though the app told the user hovering would reveal something — this breaks trust in every other on-screen instruction.

4. **"Step" gives zero feedback when the text box is empty** (18-after-clear.png vs. 19-step-on-empty.png are effectively identical, same file size). No error, no disabled button state, no message — indistinguishable from the button being broken.

5. **Wall of unexplained jargon with no inline help**, worst on the Network tab immediately after the first Step (02-after-step.png: "Unembedding," "FF output," "Hidden -> output," "Write back," "Q K V") and the Neurons tab (09-tab-neurons.png: ff_pre, ctxv, attn_out, hf, W_q/W_k/W_v). For a user who has never had "transformer," "embedding," or "neural network" explained, this is dense technical shorthand dropped with zero definitions or tooltips right at the first moment of interaction.
# Maya play-test notes, part 2 (screenshots 21-40)

## 21-step-my-paragraph.png
Action: typed her own sentence ("my biology teacher said the mitochondria is the powerhouse of the cell and i still think that is funny") and clicked Step.
What Maya thinks she's looking at: some kind of machine that "read" her sentence and is guessing the next word, but the screen is dense with unlabeled boxes and wiring diagrams she has no context for.
Text problems: the tokenized copy of her sentence (both in the left "Text Inputs" chip list and in the "Reading" strip above the canvas) ends with a "?" that she never typed — her sentence ends in "funny", no punctuation. This looks like the app silently added a character to her input, with no explanation.
The "Predicted next token" grid (center-left) is a field of tiny gray dots with no word labels on any of them — impossible to tell which dot is which word. The small red-tinted bar chart next to it ("Not likely next words") is too small to read at this size.
Diagram labels ("Unembedding", "FF hidden", "Attention output", "Write back", "Embedding", "Position table") are all tiny (looks well under 8px effective size at this scale) and use vocabulary she's never seen defined anywhere on screen.
Broken/inconsistent vs. previous state: the model's top prediction is "didn't" shown in red at only 0.4% confidence — reasonable for an untrained model (Iterations: 0), but nothing on screen tells her that's *why* the confidence is so low.
Maya's question: "Why does it think the next word is 'didn't' after 'funny', and why is it only 0.4% sure — and where did that question mark in my sentence come from?"

## 22-vocabulary.png
Action: clicked "Show Vocabulary".
What Maya thinks she's looking at: a word-frequency list appeared at the bottom of the left sidebar, titled "Vocabulary (288 words, incl. punctuation and line breaks)". Button correctly flipped to "Hide Vocabulary" and is highlighted, so the toggle state is at least clear.
Text problems: the very first row of the list has a blank/invisible entry with count "199" — no visible character, just a number. Maya has no way to know this is a space or newline; it just looks like a missing word.
The list is cut off by the bottom edge of the browser window after about 6 rows (", / , / ? / you / i / the") with no visible scrollbar — she can't tell if there are 288 rows below that she's simply not seeing, or if the list is broken.
Nothing changed in the big canvas on the right — the vocabulary panel only affects the left sidebar, which is fine, but there was no visual cue before clicking that this is what "Show Vocabulary" would do (could just as easily have opened a popup/modal).
Maya's question: "What is the blank word worth 199, and is there more list below that I can't get to?"

## 23-training-text.png
Action: pressed Escape (after the vocabulary screenshot), then clicked "Show Training Text".
What Maya thinks she's looking at: both "Hide Vocabulary" and "Hide Training Text" buttons are now highlighted/active, implying two panels are open — but the sidebar content looks visually identical to screenshot 22 (same vocabulary list, same cut-off point). There is no distinct "training text" panel visible anywhere on screen.
Broken/inconsistent vs. previous screenshot: this is the clearest problem in this batch — clicking "Show Training Text" changed the button's own label but produced no visible new content. Either the training text was inserted below the vocabulary list and pushed further off the bottom of the window (undiscoverable, since there's no scrollbar), or the feature silently did nothing. Also notable: Escape did not close the vocabulary panel that was open in screenshot 22 (it's still open here) — so Escape doesn't dismiss these panels the way Maya would expect from any other website.
Maya's question: "I clicked 'Show Training Text' — where did it go? Did anything actually happen?"

## 24-sampling.png
Action: clicked "Configure Sampling Strategy...".
What Maya thinks she's looking at: the button was replaced in place by two new fields, "Strategy: Top-p" (dropdown) and "p: 0.9" (number box), squeezed between the Temperature slider and the Training section. "Hide Vocabulary" and "Hide Training Text" are still both shown active above it, and the vocabulary list is still visible at the very bottom, now showing only 2 rows before being clipped by the window edge (even less than before, because the new Strategy/p fields pushed everything else down).
Text problems: none of "Strategy", "Top-p", or "p" are explained anywhere — a 16-year-old with no ML background has no idea what "Top-p" sampling means or what a good value of "p" would be, and there's no tooltip or link.
Inconsistent with Vocabulary/Training Text pattern: this section has no "Hide" toggle — once opened it just stays, unlike the other two which are collapsible buttons. Three different disclosure UI patterns (collapsible-with-label-swap, collapsible-with-swap, and permanently-expanded) are used side by side in the same sidebar with no visual consistency.
Maya's question: "What does 'Top-p 0.9' actually control, and can I close this if I don't need it?"

## 25-model.png
Action: clicked "Model..." (top-right corner button).
What Maya thinks she's looking at: a centered popup titled "Tiny Language Model" with a form: Context size 24, Embedding dimension 20, Hidden size 30, Training text (dropdown: "Group chat"), Train test split 0.6, Parameters 15,379, then a preview box of sample training lines ("you coming tonight? / only if there's food. / did you finish the thing? / define finish. / where are you? / still in bed. why?"), and Cancel/Create buttons.
Text problems: "Embedding dimension" and "Hidden size" are presented as bare numeric fields with zero explanation — exactly the vocabulary the persona has never had defined, dropped into an editable settings form as if self-evident. "Context size 24" is likewise unexplained here (its meaning only becomes apparent three screenshots later in 28, and only by inference).
Visual: the background canvas behind the modal is not dimmed/darkened at all, so it doesn't read clearly as a focused, modal popup — looks almost like it could be a panel Maya could still interact with the diagram behind.
No visible close/X button on the dialog — only Cancel and Create — so the only way out without acting is a labeled button, not an obvious universal close.
Maya's question: "What is 'Embedding dimension' and what happens to my current model if I hit Create — do I lose what I've done?"

## 26-after-escape.png
Action: pressed Escape.
Broken/inconsistent: identical to screenshot 25, pixel for pixel as far as I can tell — the Escape key did not close the "Tiny Language Model" dialog. Combined with 22->23 (Escape also failed to close the Vocabulary panel), this is now a confirmed pattern: Escape does not dismiss any overlay in this app, contrary to nearly universal web convention.
Maya's question: "I hit Escape like I would anywhere else — why is this still here?"

## 27-one-unknown-word.png
Action: fresh page load, filled the textarea with "zzz" only, clicked Step.
What Maya thinks she's looking at: the Text Inputs box and the token-chip row both show "zzz carefully" — but the only text actually typed for this test was "zzz". Extra text ("carefully") appeared in the input that was never entered in this action, which would read to Maya as the site randomly inserting words into her own text box. (Likely leftover state from a prior visit, but nothing on screen explains that — it just looks like the box put words in her mouth.)
The "Reading" strip at the top of the canvas only shows "zzz" as the token consumed, with "-> next word yesterday 0.4%" — so "carefully" wasn't even used for this prediction, adding to the confusion about why it's sitting in the box at all.
The "Most likely next words" side list (right of the dot grid) is legible this time (short input) and shows a flat spread of near-identical percentages (yesterday, borrow, eat, didn't, not, me, let, does — all ~0.4%), i.e. the model has no real preference, consistent with 0 training iterations, but nothing tells Maya that's why.
Maya's question: "Wait, I only typed 'zzz' — where did 'carefully' come from, and why does a nonsense word predict 'yesterday'?"

## 28-very-long-paragraph.png
Action: pasted an intentionally long, run-on paragraph (83 words) and clicked Step.
What Maya thinks she's looking at: the header now reads "83 words, model reads the last 24" — a genuinely useful disclosure, but it's small text next to the Clear button, easy to miss, and the concept of a fixed reading window is never explained in place.
The token-chip list on the left shows all 83 words wrapped over many rows, with no visual distinction (graying-out, dimming, a divider line) between the ~24 words actually used by the model and the ~59 earlier words that are being silently ignored — so even a careful reader can't tell from the chip list alone which words "count."
Layout: the long chip list has pushed the sidebar content noticeably taller than in earlier screenshots — the "Language Model Controls" section is visible but everything below it (Training panel, etc.) is presumably pushed further down/off-screen (not confirmed in this shot, but the growth pattern matches the vocabulary-list cut-off problem seen earlier).
The Reading strip confirms only the last 24 words are shown ("words and keeps going on and on past the point where it is reasonable honestly i am just curious about whether the layout breaks") predicting "book" at 0.4% — so the paragraph's whole first half (the part explaining *why* she's testing this) is invisible to the model with only a one-line hint as to why.
Maya's question: "Did it actually read my whole paragraph, or only part of it — and which part?"

## 29-punctuation-only.png
Action: filled the textarea with "!!! ??? ... ,,, ;;; @@@ ### $$$ %%%" (punctuation stress-test) and clicked Step.
What Maya thinks she's looking at: the token-chip row breaks the punctuation into separate chips ("!!!", "???", "...", ",,,", ";;;", "@@@", "###", "$$$", "%%%") — reasonable — but there's an extra final chip reading "saw", a real word that was never part of this typed input. This is the same kind of ghost-text problem as screenshot 27's "carefully": text she didn't type is appearing in the box.
Text problems: the header word counter still reads "20 words" — the same number shown for the entirely different biology sentence back in screenshots 20-26 — even though the visible input here is 9 punctuation groups plus one stray word. The counter looks like it isn't updating with the actual content, or is reading stale state.
Nothing crashed or visually overflowed — the punctuation chips render fine, no clipping — so the app tolerates weird characters gracefully; the problem is specifically the mismatched/stuck text and counter, not a layout break.
Maya's question: "Why does it say 20 words when I only typed symbols, and where did 'saw' come from?"

## 30-1280-first.png
Action: resized to a 1280x720 laptop viewport and loaded the site fresh.
What Maya thinks she's looking at: a clean first screen with a default example already sitting in the box ("you coming", 2 words) and, importantly, actual plain-English instructions in the canvas area: "Type a few words in Text Inputs and press Step. The model reads the last 24 words and guesses the next one." and "The map of the model appears here after you press Step." This is the clearest, most helpful piece of copy seen in the whole set — it's the one place the 24-word context limit is stated in plain terms.
No visual problems at this narrower width — header, buttons, and tabs all fit without overlap or clipping.
Note relevant to earlier confusion: since a fresh load here starts with "you coming" (not "zzz" or anything with "carefully"), the default starter text is not fixed/constant across loads — it's a different snippet than the one implied by screenshot 27's "zzz carefully", which supports something changing the box's contents beyond what the user typed.
Maya's question: (none — this screen is actually clear) though she'd wonder "why does it already have someone else's words in the box?"

## 31-1280-after-step.png
Action: typed the biology paragraph and clicked Step, still at 1280x720.
What Maya thinks she's looking at: same network diagram as the 1440px view, now slightly more cramped. The trailing auto-added "?" is present again in the token chips ("funny] ?]") even though she typed no question mark — confirming this happens consistently, not a one-off from screenshot 21.
Visual defect: the small "Not likely next words" bar-list that sat to the right of the "Predicted next token" dot grid at 1440px width appears to be missing/blank at this narrower width — just empty space where it used to be. Content that exists at one viewport size seems to disappear at another rather than reflowing.
Maya's question: "Is that side panel supposed to be empty, or did something break because the window got smaller?"

## 32-1280-neurons.png
Action: clicked the "Neurons" tab.
What Maya thinks she's looking at: a completely different, denser diagram of small columns and wires with a caption "Numbers for position 19 of 19: 'funny'. Hover a circle to see its wires." — at least this caption tells her what to do next (hover a circle).
Text problems: virtually every label in this diagram is raw ML shorthand with no explanation anywhere on screen: "W_q", "W_1", "W_2", "q . k, softmax", "attn_o", "ff_pre", "ff_act", "ReLU", "logits", "probs", "mix v by attention", "x0 carried forward". These are far beyond "transformer/embedding/neural network" in obscurity for a first-time visitor — this is effectively a research-notation cheat sheet with no glossary or hover-defined terms visible.
The labels are also extremely small — at this reduced viewing size several (e.g. "ff_pre", "ff_act", "attn_o") are only legible by guessing from context, not by actually reading the letters clearly.
Maya's question: "None of these labels mean anything to me — what am I even supposed to be looking at?"

## 33-1280-explain.png
Action: clicked the "Explain" tab.
What Maya thinks she's looking at: this is by far the most accessible screen so far — a left-hand numbered outline (Overview; INPUTS: 1-3; ATTENTION: 4-13; FEED-FORWARD: 14-18; PREDICTION: 19-20) and a right-hand write-up for the selected step, "Turn scores into probabilities," in actual sentences: "The same trick as the attention weights: e to the power of each score, divided by the total. Now every word in the vocabulary has a probability and they add up to 1. The largest one is the model's guess for the next word." Plus a "Where to look" callout pointing to the Network/Neurons tabs, and "Show in Network" / "Show in Neurons" buttons.
Text problems: it still drops into a raw math formula ("IN NUMBERS", softmax equation with exp/max/sigma notation) with no option to skip it, and phrases like "the attention weights" and "softmax_out" (in the small "Forward step 21 of 21 (softmax_out)" header) assume the reader already knows earlier jargon — fine as a recap, unusable as a first introduction.
Numbering oddity: the outline's last item is "20. Score every word" but the header says "step 21 of 21" — a one-off numbering mismatch between the outline (1-20) and the computation step counter (1-21) that isn't explained.
Maya's question: "This is the first thing that's read like actual English — but what's 'e to the power of' doing in a biology-adjacent tool, and do I need to know that formula or can I skip it?"

## 34-1280-vocab.png
Action: clicked "Show Vocabulary" while the Explain tab (from 33) was open.
Broken: this is the single biggest problem found in this batch. Instead of a vocabulary panel appearing, the entire app reverted to its pristine startup state — Text Inputs back to the default "you coming" (her typed biology paragraph is gone), the tab bar back on "Network" (not "Explain"), and the canvas back to the "Type a few words in Text Inputs and press Step..." placeholder with no diagram at all. The "Show Vocabulary" button itself is not even toggled/highlighted, so it's not clear the vocabulary feature engaged at all — it looks exactly like screenshot 30 (the very first load), as if the whole session silently reset.
From Maya's seat this reads as: she did several minutes of exploring (typed a sentence, stepped the model, opened Neurons, opened Explain) and one click on an unrelated sidebar button wiped everything back to square one with no warning, no confirmation, and no way to undo it.
Maya's question: "What just happened — did I lose everything I did? Did I do something wrong?"

## 35-training-running.png
Action: fresh reload, clicked "Train".
What Maya thinks she's looking at: the button correctly flips to "Stop" and "Reset weights" grays out while training runs — good, clear affordance. The Training stats update live: Iterations 1, Training loss 2.3143, Testing loss 5.2730, Training accuracy 59.2%, Testing accuracy 30.8%. A small loss-graph placeholder with a "train loss"/"test loss" color key sits below but has basically nothing plotted yet (only one data point).
Mild issue: the big canvas on the right still shows the static "Type a few words... The map of the model appears here after you press Step" placeholder the entire time training runs — nothing in the main visual area indicates training is happening, so a user watching the big panel (which is most of the screen) would see no activity at all and might think Train did nothing.
Maya's question: "It says 'Stop' so something's running, but nothing on the big part of the screen is moving — is it actually doing anything?"

## 36-train-pressed-twice.png
Action: the script tried to click "Train" a second time to test double-clicking it, but by this point the button had already changed its label to "Stop" (per screenshot 35), so there was no button reading "Train" left to click — the deliberate "press Train twice" stress test couldn't actually land on the button.
What's on screen: training simply continued on its own — Iterations now 3, Training loss down sharply to 0.4699, Training accuracy up to 90.1%, but Testing loss *up* to 7.7835 and Testing accuracy *down* to 27.8%. A small red/blue loss-curve graph is now visibly plotted (red training line dropping, blue testing line roughly flat/higher).
Comprehension gap: training accuracy rocketing to 90% while testing accuracy falls to 27.8% after only 3 iterations is a striking, screenshot-worthy result (classic overfitting on a tiny toy dataset) — but nothing on screen names or explains this divergence; a curious user has no in-app way to learn why the two numbers are moving in opposite directions.
Maya's question: "Why did the training score jump to 90% but the testing score went down — isn't it supposed to be getting better overall?"

## 37-after-training.png
Action: waited 6 seconds with training still running (button never clicked to Stop).
What Maya thinks she's looking at: iterations climbed on their own to 8 (Training loss 0.2493, Testing loss 9.4060, Training accuracy 93.6%, Testing accuracy 28.7%) — the loss graph now clearly shows the red (train) line flattening near zero while the blue (test) line keeps climbing, a widening gap.
Same issue as 35/36: the entire right-hand canvas (the majority of the screen) is still just the static "Type a few words... The map of the model appears here" placeholder the whole time training runs — no visual feedback there at all that anything is happening.
Maya's question: "Is it going to just keep running forever until I tell it to stop? How would I know when to stop it?"

## 38-loss-curves.png
Action: clicked the "Loss" tab.
What Maya thinks she's looking at: two clean, readable charts — "Loss" (red train line dropping to ~0, blue test line rising to 9.44 by iteration 9, with a "log scale" checkbox) and "Accuracy" (red train line climbing to ~93%, blue test line flat around 27-30%), each with a color-coded numeric legend underneath. This is one of the clearest, best-labeled screens in the whole set.
Text problems: a third section header, "Gradient sizes," appears at the very bottom of the screen with no chart or data visible under it at all — either it hasn't rendered yet or it's cut off by the window edge; either way there's no scrollbar to check, so it just looks unfinished/broken.
The "log scale" checkbox and the very concept of two diverging curves are not explained here — a curious student has no signpost that this shape (train down, test up) has a name (overfitting) or is worth worrying about.
Maya's question: "What's supposed to be under 'Gradient sizes' — is that section broken?"

## 39-step-after-training.png
Action: clicked "Network" tab, then clicked "Step" (training still running in the background).
Broken/inconsistent: the Text Inputs box now reads "you coming back" — three words — but the only text ever placed there in this run was the default "you coming" (2 words); "back" was never typed by anyone. This confirms what screenshots 27 and 29 only hinted at: clicking "Step" appears to silently append the model's own predicted next word onto the actual Text Inputs content, mixing "what I wrote" with "what the model guessed" in the same box, in the same font, with no label or distinction. That's a real, reproducible behavior, not a one-off glitch — and it's never explained anywhere on screen.
What's genuinely good here: the "Most likely next words" list is legible and shows a real, strong preference now ("back" at 95.2%, everything else under 1%) versus the near-uniform ~0.4% spread seen before training — a nice, correct demonstration that training changed the model's confidence, though the app never calls this contrast out explicitly for the user to notice.
Bottom-left corner shows new orange text "11 iterations (training)" confirming training is still silently running while she's looking at something else entirely.
Maya's question: "Wait, I didn't type 'back' — did the box just add its own guess to my sentence?"

## 40-after-reset.png
Action: clicked "Reset weights".
Broken: nothing changed except the numbers that were already drifting from ongoing background training — Iterations kept climbing (11 to 16), Training accuracy stayed high (94.4%), Testing accuracy stayed poor (27.2%), and the Text Inputs/prediction ("back" 95.8%) are essentially unchanged from screenshot 39. "Reset weights" is visibly grayed out in both this and the previous screenshot (disabled while training is active/"Stop" is showing), so the click did nothing — but there is no tooltip, message, or any visible feedback telling the user *why* the button had no effect.
Maya's question: "I clicked Reset weights and literally nothing happened — is that button broken, or do I have to do something else first?"

---

## Top 5 worst problems (screenshots 21-40)

1. **Clicking "Show Vocabulary" wiped the entire session.** After typing a real sentence, stepping the model, and switching to the Explain tab, one click on an unrelated sidebar toggle (34-1280-vocab.png) reset the whole app back to its default startup state — default text, default tab, no diagram, no warning, no undo. This is the single most damaging bug found: it destroys user work with no explanation.
2. **Escape never closes anything, twice confirmed.** Pressing Escape failed to dismiss the open Vocabulary panel (22-vocabulary.png -> 23-training-text.png) and failed to dismiss the "Tiny Language Model" dialog (25-model.png -> 26-after-escape.png, pixel-identical before and after). This breaks a convention every web user relies on, and it's consistent, not a fluke.
3. **"Step" silently appends the model's own guess into the user's typed text.** First seen as unexplained ghost text in 27-one-unknown-word.png ("zzz" became "zzz carefully") and 29-punctuation-only.png (a stray "saw" appeared), then confirmed directly in 39-step-after-training.png where "you coming" became "you coming back" after one Step click with no new typing. There's no visual distinction between user-typed and model-appended words, so it looks exactly like a text-corruption bug rather than a feature.
4. **Sidebar disclosure panels stack, clip, and sometimes show nothing.** Vocabulary, Training Text, and Sampling Strategy all live in the same left column with three different, inconsistent expand/collapse patterns; the vocabulary list is clipped by the bottom of the window with no scrollbar (22-vocabulary.png, 24-sampling.png), and clicking "Show Training Text" produced literally zero visible new content anywhere on screen (23-training-text.png).
5. **"Reset weights" fails silently, and training never stops itself.** The button is disabled the entire time training is running, with no tooltip explaining why (39-step-after-training.png, 40-after-reset.png show a click having zero effect), and nothing in the large canvas panel shows any sign that training is active while it silently runs in the background for the rest of the session (35-training-running.png through 38-loss-curves.png).
