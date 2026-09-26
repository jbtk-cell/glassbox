# Play-test round 2 report — glassbox (Maya, 16)

Site tested: http://localhost:5174/glassbox/ at 1440x900. Three short driver runs, 9 screenshots.

## The 8 complaints, re-checked

1. **Step-added word didn't match the strip's predicted word, no explanation.** — **FIXED.** The strip now reads "Reading the cat sat on the mat quickly → most likely `not` 0.5% · Step picked `finished` 0.3%", and the word actually appended to Text Inputs was "finished" — they match, and the strip explains there's a "most likely" word versus the word Step actually picked. (screenshot: `r2-01-after-step.png`)

2. **Added words looked identical to typed words.** — **FIXED.** In the Text Inputs token strip, the model-added word ("finished") is rendered in blue text with a blue outline, clearly different from the plain black-bordered boxes around the words I typed. (screenshot: `r2-01-after-step.png`)

3. **Hovering on Neurons tab showed no visible change.** — **FIXED.** Hovering near (not directly on) a circle in a tall column instantly popped a tooltip ("word row tok[11] = -0.04508") with a line pointing to the nearest circle, which lit up. (screenshot: `r3-01-neurons-hover.png`)

4. **Empty-textbox Step press gave no feedback.** — **FIXED.** After Clear, the Step and Play buttons visibly grey out; confirmed disabled with tooltip text "Type a few words in Text Inputs first". (screenshot: `r3-03-clear-hover-step.png`)

5. **Jargon labels (x0, ff_pre, ctxv, W_q) unexplained.** — **FIXED.** Every Neurons-tab column now has a plain-English caption above its short code (e.g. "tidied / h1", "hidden / ff_pre", "mix / ctxv"), and Explain → Overview has a full glossary table spelling out every one of these codes in plain English. (screenshots: `r3-01-neurons-hover.png`, `r3-02-explain-overview.png`)

6. **Show Vocabulary did nothing / panel landed below the window.** — **FIXED**, with a caveat. Clicking it immediately shows a "Vocabulary (288 words...)" list right in the sidebar (button also relabels to "Hide Vocabulary"), fully visible without scrolling. (screenshot: `r3-04-vocabulary.png`) Caveat: if Vocabulary is left open while Training is running, the Training panel's bottom rows (learning rate, loss chart, explanatory note) get pushed below the 900px fold — see New Problems below.

7. **Neurons-tab column captions overlapped.** — **FIXED.** All ~20 column headers are legible and cleanly separated even packed into one screen; no overlapping text seen. (screenshots: `r2-02-neurons-initial.png`, `r3-01-neurons-hover.png`)

8. **Nothing said what training does or why Reset weights was greyed out.** — **FIXED.** During training the Train button becomes "Stop", Reset weights visibly greys out and carries the tooltip "Press Stop first", and the panel text notes "Training runs until you press Stop. Press Step at any time to see a prediction with the current numbers." (screenshot: `r3-05-training.png`; note text confirmed in DOM dump, though scrolled off-screen in that particular screenshot — see New Problems)

## New problems noticed

- With the Vocabulary panel open during active training, the sidebar's Training section (Learning rate box, loss sparkline, and the "Training runs until you press Stop" note) is pushed below the 900px viewport with no visible scrollbar — you'd need to guess to scroll the sidebar to see it (`r3-05-training.png`).
- Minor/cosmetic: in the Text Inputs token strip, some typed words ("the", "quickly") get a dashed border instead of the plain solid one others get, with no visible label saying why — a small new "what does that mean" moment, though nowhere near as confusing as the old jargon issue.

## Verdict (Maya)

Okay, this actually explains itself now — I can see which word Step picked, why the button won't let me press it on nothing, and what all those weird little codes mean, so I'd probably let a friend poke at it for a few minutes without them going "wait, what?" every ten seconds.
