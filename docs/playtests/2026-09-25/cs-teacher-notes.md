# Play-test notes — Mr. Ortega persona (screenshots 01-21)

Persona: high-school CS teacher, used Simbrain's Tiny LM desktop demo, deciding if this
site could replace it for a 50-minute lesson with 25 Python-basics students (no linear
algebra). Judging only what is on screen.

---

## 01-first-screen.png
**Action:** goto page, wait 2.5s (fresh load, nothing clicked).
**What it shows:** A control-panel-plus-canvas layout. Left column: "Text Inputs" box
pre-filled with "you coming" and a row of two token chips [you][coming]; "Language Model
Controls" (Show Vocabulary, Show Training Text, Temperature slider at 1.00, "Configure
Sampling Strategy..."); "Training" block with Train/Reset weights buttons and a stats
table (Iterations 0, Training loss –, Testing loss –, Training accuracy –, Testing
accuracy –, Learning rate 0.003). Top bar: glassbox wordmark, Step/Play/Clear buttons, a
"2 words" counter, and a "Model..." button top-right. Right canvas has tabs
Network/Neurons/Explain/Loss and one line of instructions: "Type a few words in Text
Inputs and press Step. The model reads the last 24 words and guesses the next one." plus
placeholder text "The map of the model appears here after you press Step." Bottom transport
bar has |< < ▶ >| step controls (all look disabled), a slider, "Press Step to run the
model", and a "Record training step" button on the far right.
**Text problems:** none cut off here, but the whole left column of controls is dense
8-9pt-equivalent text at 900px width — on a projector this column would be the first
thing to become illegible.
**Inconsistent/broken:** N/A (first screen).
**Ortega would have to say out loud:** what "Model..." does, what "Record training step"
does (it sits far bottom-right, easy to miss and its purpose is not explained anywhere on
this screen), and why the model is pre-loaded with the odd two-word phrase "you coming"
rather than something a lesson would choose.
**Projector readability:** Top control column text is small but probably legible if
projected large; the canvas placeholder text is comfortably sized.

---

## 02-show-vocabulary.png
**Action:** click "Show Vocabulary".
**What it shows:** Button relabels itself "Hide Vocabulary" and a new panel appears below
the Training stats: "Vocabulary (288 words, incl. punctuation and line breaks)" followed
by a two-column scrolling list of token/count pairs: the first row's token glyph is not
legible at this size (looks like a blank/quote-like mark), then "," 156, "?" 86, "you" 61,
"i" 33, "the" 24, "did" 22, "it" 21, "what" 21, "is" 19, and the list is cut off by the
panel's bottom edge with a partial row visible.
**Text problems:** first vocabulary row's token is unreadable — cannot tell if it's a
space, a newline marker, or a stray character (top of vocabulary list, left column).
The list is truncated mid-row at the bottom of its visible box with no visible
scrollbar affordance to signal there's more below.
**Inconsistent/broken:** N/A vs 01, this is the vocabulary panel appearing correctly.
**Ortega would have to say out loud:** what token #1 in the list actually is (he'd have
to guess or scroll to check), and that 288 "words" includes punctuation — the panel title
does say this, so that part is at least explained on-screen.
**Projector readability:** the two-column word/count list is small monospace-ish text;
from the back of a classroom this would be very hard to read, though the concept (a
frequency table) would still come across.

---

## 03-after-escape.png
**Action:** press Escape (intended to test whether it closes the vocabulary panel).
**What it shows:** Visually identical to 02 — the vocabulary panel is still open. The only
difference is the "Hide Vocabulary" button now has a blue focus ring around it.
**Text problems:** none new.
**Inconsistent/broken:** Escape did nothing except leave a focus outline on the last-clicked
button; it did not close/collapse the vocabulary panel. Not necessarily a bug (many sites
don't bind Escape to panel toggles), but it is a dead end for a keyboard-habituated user.
**Ortega would have to say out loud:** "click Hide Vocabulary again to close it, Escape
doesn't do it" — otherwise a student would sit there pressing Escape repeatedly.
**Projector readability:** same as 02.

---

## 04-after-step.png
**Action:** fresh reload, click "Step" once (no vocabulary shown this time).
**What it shows:** The full network diagram now renders in the canvas. Header line reads
"Reading you coming → next word before 0.4%" with "coming" shown in an orange chip and
"before" in a boxed label. Below: a "Predicted next token" grid of small circles, a
"Probabilities" ranked list (before 0.4%, person 0.4%, way 0.4%, weird 0.4%, social 0.4%,
silent 0.4%, noise 0.4%, out 0.4%), an "Unembedding" box, a large "Transformer block" box
containing Output/FF output/Hidden→output/FF hidden/Input→hidden/Attention
output/Write back/Attention/Q,K,V/Input sub-boxes, an "Embedding" box with Word table +
Position table, and an "Inputs" box. A legend bottom-right explains colors: "negative 0
positive", "0 to 1: attention, probabilities", "learned numbers (training changes them)",
"numbers computed from the words". Bottom bar now reads "0 iterations ... Forward step 21
of 21: Next-token probabilities" and the transport arrows are active. The Text Inputs box
and word count at top have also changed to "you coming something" / [you][coming]
[something] chips / "3 words" — Step evidently appended a newly-sampled word to the input.
**Text problems:** the eight probability labels are all rounded to the same "0.4%",
which for an untrained model (~1/288 ≈ 0.35% uniform) is mathematically expected but
visually reads as "every option is equally likely" with no way to see the tiny real
differences — the display doesn't give enough decimal precision to show that they aren't
exactly tied (right-center of canvas, "Probabilities" list). Small print throughout the
diagram (box labels like "FF hidden", "Input -> hidden") would be very hard to read
projected from the back of a room.
**Inconsistent/broken:** the header explicitly says "next word before 0.4%", but the word
actually appended to the Text Inputs box is "something", not "before" — the headline
"predicted" word and the word the model actually used to extend the sentence disagree.
This is presumably because of temperature-based sampling (before = highest-probability
argmax, something = the word actually drawn), but the site never says this, so it looks
like the tool computed one answer and then silently typed in a different one.
**Ortega would have to say out loud:** why the diagram exploded from one line of
instructions into ~15 labeled boxes with no walkthrough of what to look at first; why the
top-line predicted word and the appended word don't match; what "learned numbers" vs
"computed from the words" means in practice (the legend states it but doesn't point to
which boxes are which type).
**Projector readability:** poor — this is the densest screen so far, most box labels
inside the Transformer block are small and would not be legible from typical classroom
projector distance without zooming in first.

---

## 05-after-fit.png
**Action:** (fresh reload) click "Step", then click "Fit".
**What it shows:** Pixel-for-pixel the same layout as 04-after-step.png — same header
("Reading you coming → next word before 0.4%"), same red-marked circle in the "Predicted
next token" grid, same diagram positions.
**Text problems:** none new.
**Inconsistent/broken:** "Fit" produced no visible change at all. It presumably re-centers
/ re-scales the diagram to the viewport, but since the diagram was already fully visible
and un-panned, there is nothing on screen to prove the button did anything. A first-time
user who had panned or zoomed away and hit Fit would have no other feedback than "the
diagram snapped back" — here, with nothing to snap back from, the button looks inert.
**Ortega would have to say out loud:** "Fit re-centers the diagram if you get lost while
zooming — trust me, it works" since screenshot evidence alone doesn't show it doing
anything.
**Projector readability:** same as 04.

---

## 06-fwdstep-1.png
**Action:** (fresh reload) click "Step", then click "|<" (rewind to start of the forward
pass).
**What it shows:** Same "you coming something" input state, but the diagram resets to its
pre-computation state: header now reads "Reading you coming → next word: not computed
yet", the "Predicted next token" grid is all plain grey circles (no highlighted answer),
and the "Probabilities" box shows only its "Most likely next words" caption over an empty
grey placeholder bar. Bottom transport bar now reads "Start: nothing computed yet (21
steps to go)" with the slider at the far left.
**Text problems:** in the Probabilities box, the grey placeholder bar and the "not
computed yet"-style empty state sit very close together with little visual separation
(center-right of canvas, inside the "Probabilities" box) — at a glance it reads as
clutter rather than a clean "nothing here yet" state.
**Inconsistent/broken:** none — this is a legitimate and useful "rewind to start" state
that clearly demonstrates the forward pass hasn't run yet, which is exactly what a
teacher would want to show students before stepping forward one operation at a time.
**Ortega would have to say out loud:** that "|<" means "rewind to before the first
computation," since the button itself is only labeled with the ASCII glyph "|<" with no
tooltip text visible on screen.
**Projector readability:** the "21 steps to go" bottom-bar text and grey circles are
readable; fine for projection.

---

## 07-fwdstep-2.png
**Action:** click ">" once (advance one micro-step from the rewound state).
**What it shows:** Bottom bar now reads "Forward step 1 of 21: Token embedding" and the
"Word table" box inside "Embedding" plus the "Inputs" box are outlined in orange to show
which part of the network just ran. Top header is unchanged ("next word: not computed
yet") since the full pass hasn't finished.
**Text problems:** none new.
**Inconsistent/broken:** none — this is the core value of the tool working correctly:
one click, one labeled operation, one highlighted box. This is the closest analog to
Simbrain's tile-and-arrow stepping and is the strongest single feature seen so far.
**Ortega would have to say out loud:** what "Token embedding" means conceptually (the
site does label the step by name, which helps, but doesn't define the term on screen).
**Projector readability:** good — the orange highlight is a strong, high-contrast cue
that would read fine from the back of a room even if the fine print doesn't.

---

## 08-fwdstep-3.png
**Action:** click ">" again (second micro-step).
**What it shows:** Bottom bar reads "Forward step 2 of 21: Position embedding"; now the
"Position table" box (instead of "Word table") plus "Inputs" are outlined in orange.
**Text problems:** none new.
**Inconsistent/broken:** none — consistent, incremental progression from 07, exactly the
step-by-step behavior a teacher would want.
**Ortega would have to say out loud:** why token embedding and position embedding are two
separate operations (a real linear-algebra concept the site names but does not explain
for a no-linear-algebra audience).
**Projector readability:** good, same as 07.

---

## 09-fwdstep-5.png
**Action:** two more clicks on ">" from 08 (site's own counter lands on "Forward step 4 of
21: Layer norm 1" — one intermediate step, step 3, was skipped over without a screenshot).
**What it shows:** Bottom bar: "Forward step 4 of 21: Layer norm 1". Highlight has moved
to a small "norm" label near the bottom of the Embedding column and the "Inputs" box.
Everything else (Text Inputs, chips, stats) unchanged from 06-08.
**Text problems:** the "norm" / "add x0" micro-labels this step highlights are extremely
small even at full resolution (bottom-center of the Transformer block) — a student
looking for what changed between this frame and the last would have trouble spotting it.
**Inconsistent/broken:** none functionally; the step counter and step name both advanced
correctly and match the transport bar's own numbering.
**Ortega would have to say out loud:** what "Layer norm" means and why it happens right
after adding position info — another un-glossed technical term.
**Projector readability:** weak — the specific box that changed is a small label easy to
lose from a few rows back in a classroom.

---

## 10-back-to-4.png
**Action:** click "<" once (step back from step 4 to step 3).
**What it shows:** Bottom bar: "Forward step 3 of 21: Add position" — correctly one step
earlier than 09, with the highlight moved back to the "add"/"Inputs" area.
**Text problems:** none new.
**Inconsistent/broken:** none — stepping backward works and the step name/number pair
stays internally consistent with stepping forward. This is a genuine strength: the
transport control behaves like a scrubber, which is more flexible than Simbrain's
presumably-forward-only stepping.
**Ortega would have to say out loud:** nothing new here beyond what's already been said
about unexplained jargon; this screen itself is fine.
**Projector readability:** same limitation as 09 — the exact highlighted sub-box is small.

---

## 11-tab-neurons.png
**Action:** (after running the full forward pass via console script) click the "Neurons"
tab.
**What it shows:** A completely different, much denser diagram: vertical columns of small
circles labeled tok, pos, x0, h1, q/k/v (question/label/content), ctxv, attn_out, x1, h2,
ff_pre, ff_act, ff_out, x2, hf, logits, probs, connected by faint lines, with two dashed
arcs at the top labeled "x0 carried forward" / "x1 carried forward" and a shaded fan
region labeled "q, k, softmax attention" / "mix v by attention" in the middle. On the far
right a ranked word list (cat, way, before, day, holding, minute, person, weekend) sits
next to the "probs" column. A caption below reads: "Numbers for position 2 of 2: "coming".
Hover a circle to see its wires."
**Text problems:** the column headers (tok, pos, x0, h1, q, k, ctxv, attn_out, x1, h2,
ff_pre, ff_act, ff_out, x2, hf, logits, probs) are all single-line abbreviations packed
along the top edge of the canvas and are close to the smallest text seen in the whole
site so far — several are only 2-4 characters and visually run together with the circle
columns beneath them.
**Inconsistent/broken:** none — the header line ("Reading you coming → next word before
0.4%") correctly carried over from the Network tab, so the two tabs agree on state.
**Ortega would have to say out loud:** what every one of those ~16 abbreviated column
labels stands for (tok/pos/x0/h1/q/k/v/ctxv/attn_out/x1/h2/ff_pre/ff_act/ff_out/x2/hf/
logits/probs) — none are spelled out on this screen itself, only implied by position.
**Projector readability:** the worst of any screen so far — this is Simbrain's core
strength (neuron-level tiles) reproduced here, but at a density and abbreviation level
that would be unreadable from more than the front row unless heavily zoomed in first.

---

## 12-tab-explain.png
**Action:** click the "Explain" tab.
**What it shows:** A two-pane reading view. Left: a numbered table of contents grouped
under headers "INPUTS" (1-3), "ATTENTION" (4-13), "FEED-FORWARD" (14-18), "PREDICTION"
(19-21), with step 21 "Turn scores into probabilities" highlighted (matching the current
forward-step position). Right: a plain-English paragraph — "The same trick as the
attention weights: e to the power of each score, divided by the total. Now every word in
the vocabulary has a probability and they add up to 1. The largest one is the model's
guess for the next word. Step and Play pick a word from these probabilities." — followed
by a "Where to look." callout naming the Network and Neurons tab locations, two jump
buttons ("Show in Network" / "Show in Neurons"), an "IN NUMBERS" section with a softmax
formula, and "Previous"/"Next" buttons (Next correctly disabled since 21 is the last
step).
**Text problems:** item "17. Back to 20" in the left-hand list reads like a typo/broken
label out of context (top-left list, FEED-FORWARD group) — it likely means "project back
to the model's 20-dimensional embedding," but standing alone as "Back to 20" it looks like
a mistake or a stray placeholder number.
**Inconsistent/broken:** none — this tab is the most pedagogically complete screen in the
set: it names the step, explains it in plain English, shows the formula for those who
want it, and points back to where to look in the other two tabs.
**Ortega would have to say out loud:** whether to have students read the "IN NUMBERS"
softmax formula at all, since the class has no linear algebra — the site doesn't offer a
way to hide/collapse that section for a non-math audience, it's just there on the page.
**Projector readability:** good — this is plain paragraph text at a normal reading size,
by far the most projector-friendly screen so far.

---

## 13-tab-loss.png
**Action:** click the "Loss" tab (having just viewed Neurons then Explain, forward pass
already fully stepped).
**What it shows:** An almost entirely blank white canvas with a single sentence: "Train
the model to see loss curves." Nothing else in the main pane.
**Text problems:** none (there's barely any content to misplace).
**Inconsistent/broken:** the bottom "Computation" transport bar still shows "Forward step
21 of 21: Next-token probabilities" and active |</</>>| controls even though this tab has
nothing to do with forward-pass stepping and displays no chart — the per-step scrubber
staying live under an unrelated, empty tab is a small state-leak that could confuse a
student into thinking those buttons do something here.
**Ortega would have to say out loud:** "we'll come back to this tab after we train" — the
tab itself gives no preview of what a loss curve will look like, so there's nothing to
anchor the concept to yet.
**Projector readability:** trivially fine, it's one sentence — but that's also the problem;
there is nothing here yet to hold 25 students' attention.

---

## 14-show-training-text.png
**Action:** (fresh reload) click "Show Training Text".
**What it shows:** Button relabels "Hide Training Text" (highlighted) and a new panel
appears below Training stats: "Training Text (1,321 tokens)" showing the default built-in
corpus, a casual group-chat-style dialogue: "you coming tonight? / only if there's food. /
did you finish the thing? / define finish. / where are you? / still in bed. why? / late
again? / no. everyone else is early. / what time is it? / too late to ask." (list is cut
off by the panel's bottom edge, same truncation style as the vocabulary list in 02).
**Text problems:** the training-text panel is truncated at the same fixed height as the
vocabulary panel, cutting off mid-list with no scrollbar visibly indicated (bottom edge of
"Training Text" box).
**Inconsistent/broken:** none — consistent with 02's vocabulary-panel behavior.
**Ortega would have to say out loud:** that this "Group chat" text is just a stock demo
corpus, not something he chose, and that he'll need to swap it out via "Model..." before
using his own lesson sentences (confirmed in 15/16 below).
**Projector readability:** fine, normal paragraph-sized text.

---

## 15-model-dialog.png
**Action:** click "Model...".
**What it shows:** A modal titled "Tiny Language Model" with fields: Context size (24),
Embedding dimension (20), Hidden size (30), Training text (dropdown, currently "Group
chat"), Train test split (0.6), Parameters (15,379, presumably read-only/computed), and a
scrollable preview of the current training text below, with Cancel/Create buttons. This is
a genuinely good screen for the target audience — it surfaces "how big is this model"
(15,379 parameters) as a single concrete number, which is a strong discussion hook for a
no-linear-algebra class.
**Text problems:** none — labels and values are clearly aligned in a two-column form.
**Inconsistent/broken:** none.
**Ortega would have to say out loud:** what "Embedding dimension," "Hidden size," and
"Train test split" mean — none are defined in this dialog itself (no tooltips or info
icons visible), only "Context size" is implicitly explained by that first-screen sentence
("reads the last 24 words").
**Projector readability:** good — clean modal, readable form labels.

---

## 16-model-dialog-mytext.png
**Action:** with the Model dialog open, fill the training-text box with the teacher's own
14-line class.txt lesson text (the bell/students sentences).
**What it shows:** Same modal, but "Training text" dropdown now auto-switched from "Group
chat" to "Your own text", and the preview textarea (now focused, blue outline) shows the
tail of the pasted lesson text ending "...the teacher reads the sentence. / the bell rings
and the students go out." Context size/Embedding dimension/Hidden size (24/20/30) and
Parameters (15,379) are unchanged from 15.
**Text problems:** the very top line of the pasted text is sliced off by the textarea's
top edge (textarea auto-scrolled to show the bottom of the pasted content) — a partial,
clipped line is visible just under the field label with no ellipsis or fade to signal it's
cut off (top edge of the training-text box, inside the Model dialog).
**Inconsistent/broken:** the "Parameters" count still reads 15,379, identical to the
Group-chat text in 15, even though a different training text (with a different, smaller
vocabulary) was just typed in — if this number depends on vocabulary size, it appears not
to recompute live as you edit the text, which would be misleading if Ortega tells students
"watch the parameter count change" before clicking Create.
**Ortega would have to say out loud:** to trust that swapping the text worked (the dropdown
label is the only confirmation) since the visible preview is mid-scroll and doesn't
obviously show the start of the pasted text.
**Projector readability:** fine except for the clipped top line noted above.

---

## 17-after-create.png
**Action:** click "Create" in the Model dialog (model rebuilt on the class's own
bell/students text).
**What it shows:** Dialog closes, back to the main screen. Iterations/Training loss/
Testing loss/accuracy fields are all reset to 0/– as expected for a freshly created model.
But the "Text Inputs" box and word counter still show the old default "you coming" / "2
words" — the prompt was not cleared or updated to match the newly-loaded lesson text.
**Text problems:** none new.
**Inconsistent/broken:** creating a model trained on the class's own sentences leaves an
unrelated leftover prompt ("you coming") sitting in the box — if Ortega pressed Step right
now without first retyping a prompt, he'd be feeding the model two words that may not even
share much vocabulary overlap with the new corpus, which would look like a broken demo to
students for reasons that have nothing to do with the model.
**Ortega would have to say out loud:** "wait, don't press Step yet, I need to change the
prompt text first" — a step the site doesn't prompt him to do.
**Projector readability:** fine, this is just the base layout again.

---

## 18-my-prompt.png
**Action:** fill the prompt box (`.prompt-box`) with "the bell rings and the students".
**What it shows:** Text Inputs textarea (focused, blue outline) now contains "the bell
rings and the students"; the chip row below shows six matching chips [the][bell][rings]
[and][the][students]; the top counter reads "6 words".
**Text problems:** none.
**Inconsistent/broken:** none — straightforward, correct tokenization feedback.
**Ortega would have to say out loud:** nothing extra here; this screen does what it looks
like it does.
**Projector readability:** good.

---

## 19-step-untrained-mytext.png
**Action:** click "Step" (model still untrained, 0 iterations, now on the class's own
text and prompt).
**What it shows:** Full diagram again. Header: "Reading the bell rings and the students →
next word laptops 4.3%". Text Inputs box now reads "the bell rings and the students
students" (a duplicate "students" appended on its own wrapped line), counter "7 words".
4.3% for the top guess is close to what pure chance would give with this much smaller,
custom vocabulary — a nice small-numbers moment if Ortega chooses to point it out (the
site itself doesn't explain the connection between vocab size and chance-level
probability).
**Text problems:** the probability list to the right of "Predicted next token" is dense
small text and, same as screenshot 04, several candidate words round to visually similar
percentages, making the ranking hard to read at a glance (right side of canvas).
**Inconsistent/broken:** same discrepancy as 04-after-step.png — the headline predicted
word is "laptops" (4.3%), but the word actually appended to the Text Inputs box is
"students" (a repeat of the prompt's last word), not "laptops". This is now the SECOND
time in this play-test that the "next word" shown at the top and the word actually typed
into the input disagree, with no on-screen note about temperature/sampling explaining why
— this looks like a real, reproducible pattern, not a one-off glitch.
**Ortega would have to say out loud:** why the model's answer at the top of the screen
and the word that lands in the text box are different words, twice in a row, with no
built-in explanation — he would have to already know about sampling temperature to explain
this to students, and even then the site gives him no on-screen hook to point to.
**Projector readability:** same density issues as 04.

---

## 20-training-1.5s.png
**Action:** click "Train" and wait 1.5 seconds.
**What it shows:** "Train" button becomes "Stop" (Reset weights greys out). Stats update
live: Iterations 61, Training loss 0.0586, Testing loss 2.9165, Training accuracy 97.6%,
Testing accuracy 63.3%, Learning rate 0.003 (editable). A small red/blue loss-curve chart
appears under the stats (red = train loss, blue = test loss), with train loss already
flat near the bottom and test loss elevated and roughly flat. The right-hand canvas is
still just the placeholder text (training runs independently of the step-through view).
**Text problems:** none — chart is small but legend-labeled clearly at bottom-left.
**Inconsistent/broken:** none functionally, but the training on this 14-line custom
corpus reached 97.6% training accuracy vs. only 63.3% testing accuracy in just 1.5 seconds
/ 61 iterations — a textbook overfitting gap, appearing almost instantly. The site does
not label this as overfitting or explain the train/test accuracy gap anywhere on screen.
**Ortega would have to say out loud:** both (a) that training happened almost too fast to
watch anything change, since a class-sized corpus this small is memorized almost
immediately, and (b) what the train-vs-test accuracy gap means (overfitting) — a genuinely
great teaching moment the site hands him for free but never names.
**Projector readability:** the numeric stats are legible; the loss-chart lines are thin
and would be hard to distinguish (red vs. blue, both near the bottom) from the back of a
room without zooming in.

---

## 21-training-5.5s.png
**Action:** wait 4 more seconds (5.5s total since clicking "Train").
**What it shows:** Iterations now 145 (up from 61). Training loss 0.0541 (basically flat
vs. 0.0586), Training accuracy still 97.6% (unchanged). Testing loss has gone UP to
3.1570 (from 2.9165) and Testing accuracy has barely moved (63.6% vs. 63.3%). The loss
chart shows the red (train) line flat near the bottom and the blue (test) line slightly
rising/noisy above it. Text Inputs/chips/word count are unchanged from 18/19/20.
**Text problems:** none new.
**Inconsistent/broken:** nothing broken, but this is a clear, worsening overfitting curve
(more training time made testing loss worse, not better) and the site gives no label,
callout, or tooltip anywhere on this screen calling that out as "overfitting" or
explaining why train and test numbers are diverging.
**Ortega would have to say out loud:** "see how the blue line is going up while the red
line stays flat — that's the model memorizing our 14 sentences instead of learning the
pattern" — entirely his own narration, the site shows the data but not the concept.
**Projector readability:** same chart-legibility issue as 20; the direction of the blue
line (the whole point of this screen) is a subtle slope that would be very hard to see
from a distance.

---

## Top 5 worst problems (ranked)

1. **The headline "next word" prediction and the word actually appended to the Text
   Inputs box disagree, twice, with no explanation** (04-after-step.png: header says
   "before" but "something" gets typed in; 19-step-untrained-mytext.png: header says
   "laptops" but "students" gets typed in). This happens on the single most important
   interaction in the whole tool — pressing Step — and looks like the tool contradicts
   itself in front of the class, with temperature-based sampling never mentioned on
   screen to explain the gap.
2. **Core diagram text is too small/dense to read on a projector**, especially the ~16
   abbreviated column headers on the Neurons tab (tok/pos/x0/h1/q/k/ctxv/attn_out/x1/h2/
   ff_pre/ff_act/ff_out/x2/hf/logits/probs in 11-tab-neurons.png) and the nested box
   labels inside the Transformer block (04 through 10). This is the site's equivalent of
   Simbrain's tiles-and-arrows view — if it can't be read from the back of a classroom
   without constant zooming, the core value proposition doesn't survive a 50-minute
   group lesson.
3. **Training on a class-sized corpus "finishes" (overfits) almost instantly** — 97.6%
   training accuracy after 1.5 seconds / 61 iterations (20-training-1.5s.png), with
   testing loss then getting worse, not better, over the next 4 seconds
   (21-training-5.5s.png) — and none of it is labeled as overfitting anywhere on screen.
   There is no slow, watchable "learning arc" for 25 students to observe together with a
   short lesson-length text.
4. **Heavy unglossed jargon throughout** — Layer norm, Token/Position embedding, ReLU,
   ctxv, attn_out, hf, ff_pre/ff_act — introduced by label only (06-11) for an audience
   explicitly said to have no linear algebra; only the Explain tab (12) defines terms,
   and it doesn't cover the Neurons-tab abbreviations at all, so Ortega has to supply
   nearly all vocabulary support live.
5. **Small trust-eroding inconsistencies right at lesson setup time**: creating a new
   model from the class's own text leaves a stale, unrelated prompt ("you coming") in
   the Text Inputs box (17-after-create.png), and the "Parameters" count in the Model
   dialog doesn't visibly update when the training text is swapped to the class's own
   sentences (16-model-dialog-mytext.png) — both are the kind of "wait, did that actually
   work?" moments that are worst right before 25 students are watching.
# Ortega play-test notes, screenshots 22-41

## 22-training-13s.png
Action: fresh load, opened "Model..." dialog, pasted a 14-line custom training text (bell/students/teacher classroom lines), clicked Create, typed prompt "the bell rings and the students", clicked Train, then just waited (1.5s, then +4s, then +8s = ~13.5s of training) before this shot.
What Ortega would say it shows: the left "Training" panel counting up — Iterations 300, Training loss 0.0536, Testing loss 3.0009, Training accuracy 97.6%, Testing accuracy 66.1%, Learning rate 0.003 — plus a small red/blue loss-vs-iteration graph. Top bar shows "6 words" and the tokenized prompt as tiles: the/bell/rings/and/the/students.
Overlap/cut-off/unreadable/misaligned: nothing overlapping; all text in the numbers panel is legible at this size. The loss graph's two-line legend ("train loss" / "test loss", bottom-left of the little chart) is very small (~7px) but not overlapping anything.
Broken/inconsistent vs. previous shot: N/A (first in this group), but flag for the record: this run is training on the same 14-line class corpus used in every other steps file in this batch — each file is a brand-new page load, so "Iterations" resets and drifts differently from file to file (300 here, later 114, 129, 11...). A teacher watching one continuous class session wouldn't hit this, but it means none of these screenshots are literally sequential frames of one run.
What he'd have to say out loud: the entire right two-thirds of the screen is blank except the sentence "The map of the model appears here after you press Step." While Train is actively running (13+ seconds), there is nothing to point at on the projector except a small numbers table on the far left — no live visualization of training happening. He'd have to say "trust me, it's training" to the class. Also nothing on screen explains what "Testing loss"/"Testing accuracy" data actually is — there's only one text box of input, so where does a train/test split come from? He'd have to invent an explanation.
Projector readability: the numbers panel is fine (14-16px equivalent); the blank right pane is trivially "readable" but pedagogically dead air for 13+ seconds — bad for keeping 25 students' attention.

## 23-trained-step.png
Action: fresh load, same Model dialog + custom text + Create, prompt "the bell rings and the students", Train (3s), Stop, Step (one single forward step), wait 1.8s.
What Ortega would say it shows: the full network map has now appeared. Top of the canvas reads "Reading the bell rings and the the students → next word come 99.8%" with "students" highlighted orange. Below: boxes labeled Predicted next token, Probabilities, Unembedding, Transformer block (Output, FF output, Hidden→output, FF hidden, Input→hidden, Attention output, Write back, Attention, Input), Embedding (Word table, Position table), Inputs ("One dot per word"). A legend bottom-right explains the color code: red/blue = negative/positive, teal = "0 to 1: attention, probabilities", orange = "learned numbers (training changes them)", white/gray = "numbers computed from the words." Bottom transport bar now reads "Forward step 21 of 21: Next-token probabilities," and the Text Inputs box has grown a 7th tile ("come") appended to the prompt automatically.
Overlap/cut-off/unreadable/misaligned: the "Predicted next token" word-probability list ("Most likely next words: come 99.8%, and 0.1%, event 0.1%, bell 0.1%, laptops 0.1%, go 0.1% ...") is printed in genuinely tiny type crammed next to a dot-grid — at 900px wide this text is barely legible even zoomed in; at classroom-projector scale it would not be readable past the second row of desks. Every box inside the Transformer block (FF output, Hidden→output, Attention output, Write back, etc.) is a small colored dot-grid with an orange label at ~8-9px — same problem, worse, since there are 8+ of these packed into one column.
Broken/inconsistent vs. previous shot: Iterations dropped from 300 (shot 22) to 114 here — expected, since this is a separate fresh run with a shorter Train wait, but juxtaposed in a slide deck it would look like the model "forgot" training.
What he'd have to say out loud: why the top sentence literally reads "the bell rings and the the students" (a doubled "the" — see below, this is really the tokenization including a duplicate article from wrap, needs checking against input) — he'd have to narrate that this echoes the prompt tokens, not a typo the class typed. Also nothing labels which arrow order to read the diagram in (top-to-bottom? bottom-to-top? there are arrows both directions) — he'd have to explain the data flow verbally since the diagram doesn't number the steps.
Projector readability: the top banner line ("Reading ... → next word ...") is fine at a distance; the entire matrix/tile diagram is not — too much small text and small colored dot-grids for anyone past the front row.

## 24-record-training-step.png
Action: fresh load, same setup, Train (3s), Stop, then clicked "Record training step" (a different button from "Step"), wait 2.5s.
What Ortega would say it shows: a different mode. The canvas header now reads "Training on: the question . ... the students write an answer . ... the teacher reads the answer . ... the bell rings and the students" (last "students" highlighted orange) with "→ next word: not computed yet" underneath. The Transformer block boxes are all outlined in orange now (Embedding, both Attention boxes, Write back, FF hidden, FF output, Hidden→output, Output). Bottom bar: "129 iterations" and "Start: nothing computed yet (45 steps to go)".
Overlap/cut-off/unreadable/misaligned: same small-text problem as shot 23 throughout the diagram; the long "Training on: the question . ... the students write an answer ..." header line is set in small type and runs the width of the canvas — legible on a laptop, marginal on a projector.
Broken/inconsistent vs. previous shot: this is the one real red flag in this group. The left "Text Inputs" box still shows the user's own typed prompt ("the bell rings and the students," 6 tiles, "6 words" in the top bar), but the big canvas is now visualizing a completely different sentence pulled from the training corpus ("the question . ... the students write an answer ..."). Nothing on screen explains that "Record training step" samples a random training example instead of using whatever is in the Text Inputs box. A first-time viewer would reasonably think the app is showing the wrong sentence or is broken.
What he'd have to say out loud: he would have to explicitly tell the class "ignore the text box on the left, it's now training on a different sentence it picked from the training data" — the UI gives him zero help with that, no caption, no arrow, no tooltip.
Projector readability: the "45 steps to go" figure is legible and important; if he wants to click through all 45 sub-steps of one training pass live in front of the class, that alone eats a large chunk of the 50-minute period.

## 25-record-end-adam.png
Action: fresh load, same setup, Train (3s), Stop, "Record training step", wait 2s, then clicked ">|" (jump to end of the recorded playback), wait 1.8s.
What Ortega would say it shows: the last of the 45 recorded steps. Bottom bar: "Update step 45 of 45: Adam update, nudge every parameter." A new banner appears above the iteration count: "Update: tiles show how far each learned number just moved," and the previously-orange boxes are now outlined in magenta/pink (Embedding's Word table & Position table, both Attention boxes, Write back, FF hidden, FF output, Hidden→output, Output) to show which weights just changed. Canvas header now reads "Training on [same corpus sentence as shot 24] → guessed go 83.5%, actual next word go" both in green (a correct prediction), with a loss value "0.110" shown net to "go" in the predicted-token box.
Overlap/cut-off/unreadable/misaligned: same small-font problem in the matrix boxes; the new "Update: tiles show how far..." banner text is small (~9-10px) and easy to miss against the pale background — on the earlier shots there was no such banner, so a viewer's eye has no established place to look for it.
Broken/inconsistent vs. previous shot: the Text Inputs box now shows "you coming" (2 words, tile row [you][coming]) — completely different from every other shot in this batch (which showed the user's typed 6-word prompt). This happened because this particular steps file never re-typed the prompt box, so the app's own leftover/default prompt text ("you coming") is what's displayed. It is jarring next to shot 24's "the bell rings and the students" if viewed back-to-back, and "you coming" by itself reads as an odd, ungrammatical fragment (no question mark, no context) for a default example a teacher might leave on screen by accident.
What he'd have to say out loud: he'd need to explain that "Adam" is the name of an optimizer algorithm — the screen just names it, it doesn't say what it does or why it's called that, so he supplies the "nudging every number a little bit based on the error" explanation himself. He'd also have to explain, again, why the Text Inputs box doesn't match the sentence being visualized (see shot 24's issue, still unresolved here, now compounded by the odd default text).
Projector readability: the magenta-highlighted boxes read reasonably well as a group (color coding is high-contrast enough to see from a distance), but the actual per-cell numeric detail inside them is still too small to be useful at range — fine for "these parts changed," useless for "here's exactly what changed."

## 26-backward-attn-weights.png
Action: same run as shot 25 (still "you coming" / 11 iterations), script pressed "|<" to reset the 45-step playback to the start, then clicked the ">" (single-step) button 35 times.
What Ortega would say it shows: bottom bar reads "Backward step 35 of 45: Attention weights." A new orange banner reads "Backward pass: tiles show gradients, how the loss reacts to each number," and the color legend bottom-right has swapped entirely — instead of shot 24/25's "negative 0 positive / 0 to 1: attention, probabilities" legend, it now reads "gradient: raise it lowers loss / raises loss" over a purple/pink swatch, plus "learned numbers" and "numbers computed from the words." The rest of the diagram (matrix tiles, box layout) looks nearly identical to shots 24/25 apart from color values inside the tiles.
Overlap/cut-off/unreadable/misaligned: the legend swap is a good idea in principle but the new banner text and the legend re-label are both small type in the same visual "furniture" spot the eye has already learned to ignore from earlier screens — easy to miss that the meaning of the same colors just inverted (orange/red used to mean "learned number," now the legend redefines part of that space for "gradient").
Broken/inconsistent vs. previous shot: nothing broken, but note this is a real functional gap: the on-screen diagram between "step 35" here and neighboring steps looks visually almost the same as the surrounding frames (same boxes, same rough coloring) — the only way to know which of the 45 sub-steps you're looking at is to read the small caption at the bottom. A student who glances at the picture without reading the caption text would not be able to tell backward step 35 apart from, say, backward step 30.
What he'd have to say out loud: he'd have to explain what "gradient" means and why "raise it lowers loss / raises loss" is written as a single ambiguous line — that phrase alone doesn't parse as a sentence on first read (there's no "if" or "vs." separating the two halves); he'd need to rewrite it verbally as "if you increase this number, does the error go up or down."
Projector readability: the banner and caption are legible; the actual tile-by-tile gradient values (which is the entire pedagogical point of this screen) are, again, too small at any distance to see which specific numbers are highlighted.

## 27-cross-entropy-loss.png
Action: same run, script pressed "|<" to reset again, then clicked ">" 22 times.
What Ortega would say it shows: bottom bar now reads "Forward step 22 of 45: Cross-entropy loss." The special orange "Backward pass" banner from shot 26 is gone, and the legend has reverted to the plain forward-pass version (negative/positive, attention/probabilities, learned numbers, computed numbers) — because step 22 is technically still inside the "forward" portion of the 45-step recording (loss is computed right after the forward pass, before backward propagation begins at higher step numbers).
Overlap/cut-off/unreadable/misaligned: same recurring problem — the "Probabilities" box shows a small "loss 0.110" figure next to a tiny colored square that's easy to overlook; no callout or highlight box draws the eye to it as "here is the number this step computed."
Broken/inconsistent vs. previous shot: visually this frame is nearly indistinguishable from shots 25/26 at a glance (same box layout, same rough tile coloring) — again, only the bottom caption tells you it's a different, non-adjacent step (22 vs. 35, and the transport was reset to the start both times, so the two shots are not simply "one step apart" even though they appear back to back in this write-up).
What he'd have to say out loud: he'd have to define "cross-entropy loss" by name for the class since nothing on screen expands the term or ties it back to the "loss" number they've already seen in the corner numbers table — the site never spells out that this step is literally the formula that produces that Training-loss/Testing-loss figure.
Projector readability: fine for the caption text; the diagram itself gives no visual cue distinguishing "loss computed here" from any neighboring step, so it wouldn't read as a distinct, memorable frame on a projector.

## 28-hover-wordtable.png
Action: fresh load, same setup, Train (3s), Stop, Step (single forward step), then hovered the mouse over a cell inside the Embedding "Word table" box, wait 1.2s.
What Ortega would say it shows: same "Reading the bell rings and the students → next word come 99.2%" banner as the plain Step view (shot 23's layout), plus a small black tooltip reading "E[13, 10] = -0.132" appearing right over the Word table matrix, showing the exact numeric value under the cursor.
Overlap/cut-off/unreadable/misaligned: the tooltip itself is legible (dark background, white text) and doesn't overlap other text. However, the "Predicted next token" word list in this shot includes a word that does not appear anywhere in the 14-line training text Ortega typed (class.txt) — the list reads roughly "come 99.2%, patron 0.2%, and 0.2%, out 0.1%, reads 0.1%, bell 0.1%" — "patron" is not one of the words the teacher entered. If confirmed, that's a real head-scratcher for a classroom demo built around the idea that the model can only know the words you fed it.
Broken/inconsistent vs. previous shot: iteration count is 49 here (yet another fresh run) — training/testing accuracy numbers again land at slightly different values (97.2%/59.7%) than every other shot, reinforcing that none of these screenshots share one continuous training run.
What he'd have to say out loud: "E[13, 10]" is unexplained notation — no on-screen key says row 13 corresponds to which word or column 10 to which embedding dimension. He'd have to tell students "13 is the row for [whichever token], 10 is just one of 32 hidden dimensions we don't have a name for," which is exactly the kind of linear-algebra-flavored explanation the class (no linear algebra background) is least equipped to absorb without more scaffolding than the tooltip gives.
Projector readability: the tooltip text is small (looks ~9-10px) and appears at the exact mouse position, meaning on a shared projector only the person holding the mouse can reliably read it before it moves — a real problem for a whole-class demo where only the teacher's laptop cursor is visible.

## 29-hover-attention.png
Action: same run as 28, hovered a different point over the Attention box inside the Transformer block, wait 1.2s.
What Ortega would say it shows: same base screen as 28 (identical "come 99.2%" prediction, same 49 iterations), now with a tooltip "attn[2, 2] = 0.413" over the Attention box, and that box highlighted in blue.
Overlap/cut-off/unreadable/misaligned: tooltip readable in isolation; same small-font issue as shot 28's tooltip.
Broken/inconsistent vs. previous shot: consistent with 28 apart from the hover target — no problems introduced.
What he'd have to say out loud: "attn[2, 2] = 0.413" needs translation — which two words does index "2, 2" refer to? The diagonal (word attending to itself) isn't called out as a special case, so he'd have to explain both the indexing and why a word attending to itself is a meaningful example to show first (or pick a better example himself).
Projector readability: same limitation as 28 — a mouse-position tooltip only the presenter can chase down in real time; 25 students watching a projection would have no independent way to explore these values themselves without also having a working link at their own seats.

## 30-hover-probs-list.png
Action: same run as 28/29, hovered over the "Most likely next words" probability list this time, wait 1.2s.
What Ortega would say it shows: a tooltip reading "come 99.15%" pinned next to the word "come" in the predicted-word list.
Overlap/cut-off/unreadable/misaligned: none new. But note a small inconsistency: the banner at the very top of the canvas has said "next word come 99.2%" throughout this run, while this hover tooltip gives a more precise "99.15%" for the same value — same number, different rounding, shown two different ways on the same screen with no explanation. A sharp student would ask "wait, which one is it?"
Broken/inconsistent vs. previous shot: nothing broken, consistent with 28/29 (same 49-iteration run).
What he'd have to say out loud: he'd have to pre-empt the 99.2% vs 99.15% rounding question himself, since the site gives no indication these are the same number at different precision.
Projector readability: same as 28/29 — tooltip text is small and tied to the mouse position, not something the whole room can read independently.

## 31-wheel-zoom-in.png
Action: fresh load, same setup, Train (3s), Stop, Step, wait 1.8s, then a mouse-wheel zoom-in gesture centered around (1060, 570) on the canvas, wait 1.2s.
What Ortega would say it shows: the canvas has zoomed in on the right/lower portion of the diagram — Output, FF output/Hidden→output, FF hidden/Input→hidden, Embedding (Word table, Position table), Inputs, and the Attention area, with a tooltip "attn[1, 1] = 0.326" sitting over the Attention box even though nothing in the action list explicitly hovered it — the zoom gesture apparently left the cursor resting over that cell and triggered the tooltip as a side effect. The "+ − Fit" zoom control row is still visible near the top-left of the canvas at this point, so a way back to the default view is still on screen.
Overlap/cut-off/unreadable/misaligned: the row of tiles at the very bottom of the diagram (near "Input"/Q-K-V) sits right at the bottom edge of the canvas viewport, effectively cropped — a user would need to pan or zoom out further to see it in full.
Broken/inconsistent vs. previous shot: the top "Reading the bell rings and the students → next word come 99.8%" banner is unaffected by zoom, as expected — that part behaves correctly and stays put.
What he'd have to say out loud: nothing new to explain content-wise, but he'd have to narrate that zooming is mouse-wheel-driven, since there's no on-screen hint (no scroll icon, no instructional text) telling first-time users that the canvas responds to the wheel at all.
Projector readability: at this zoom level individual matrix cells and the "attn[1,1]" tooltip are more legible than the full-diagram view in earlier shots — zooming in is clearly the right move for showing details to a room, which makes it worse that nothing on screen tells a new user to do it.

## 32-drag-pan.png
Action: same run, immediately after the wheel-zoom, dragged from (1100, 600) to (700, 400) to pan the canvas, wait 1.2s.
What Ortega would say it shows: the view has panned up/left — now showing Position table, FF hidden/Input→hidden, Attention output/Write back, the Attention box (now displaying as a small teal/white grid), and the Q/K/V row with dimension labels like "attn_out 6x20" and "attn 6x6" in tiny orange text.
Overlap/cut-off/unreadable/misaligned: there is a tooltip, "attn[4, 5] = 0.00", floating by itself in blank white canvas space to the right of the diagram, not attached to any box or cell. This looks like a stale tooltip left over from before the drag — the cursor used to be over an Attention cell, the diagram then moved under the drag, and the tooltip stayed anchored to the screen position rather than to the (now-moved) cell or disappearing. That's a genuine rendering bug: a floating number label detached from anything it labels.
Broken/inconsistent vs. previous shot: on close inspection the "+ − Fit" zoom-control row is still present, fixed at the same top-left position of the canvas — but it renders in such light, low-contrast gray text on the white background that it is very easy to lose track of once the diagram itself has panned away underneath it; a first-time user who isn't specifically looking for it could easily believe (as I first did on a quick pass) that it had disappeared.
What he'd have to say out loud: he'd have to tell students to ignore the stray "attn[4, 5] = 0.00" floating in empty space, since the app itself doesn't clear it.
Projector readability: the dimension labels ("6x20", "6x6", "20x20") are a genuinely useful teaching detail (ties the picture to matrix-shape concepts) but are so small they would not be readable from more than a few feet away from any screen, let alone a projector.

## 33-plus-plus.png
Action: same run, clicked the "+" zoom button twice in a row, wait 1.2s.
What Ortega would say it shows: further zoomed-in view — FF hidden/Input→hidden, Attention output/Write back, a larger and clearer Attention grid (now legibly showing a teal/white checkerboard pattern), K/V boxes labeled "h_k 20x20"/"h_v 20x20", an Input box labeled "h1 6x20", and the curved orange flow-arrow from Input into K/V. Only a sliver of the "Position table" box peeks in at the top-left edge.
Overlap/cut-off/unreadable/misaligned: the "Position table" box is now mostly cropped off-screen at the top-left, showing only its bottom edge — a user who double-clicked "+" without first checking their pan position would lose track of where they are in the larger diagram.
Broken/inconsistent vs. previous shot: the stray floating tooltip from shot 32 is gone here, which is good. The "+ − Fit" row is (on closer look) still present in its fixed top-left spot, in the same low-contrast gray — easy to overlook but not actually missing.
What he'd have to say out loud: nothing content-wise new; the low-contrast "Fit" label is something he'd want to point out to students explicitly ("that gray text top-left gets you back to normal") since its styling doesn't call attention to itself as a functioning button.
Projector readability: this is the single most legible close-up of the Attention grid in the whole batch — genuinely good for showing "this is what attention weights look like" to a room, if the teacher can reliably get back to this exact framing on demand.

## 34-fit-after-zoom.png
Action: same run, clicked "Fit," wait 1.2s.
What Ortega would say it shows: the view snaps back to the original full-diagram framing, identical in layout to shots 23/25 (all boxes visible: Predicted next token, Probabilities, Unembedding, full Transformer block column, Embedding, Inputs), with the same 128-iteration numbers as shots 31-33.
Overlap/cut-off/unreadable/misaligned: none — this is the cleanest, most "reset" state in the batch.
Broken/inconsistent vs. previous shot: nothing broken — "Fit" does what it says, confirming the earlier concern about the button was about its visibility, not its function.
What he'd have to say out loud: nothing new — this is a good recovery moment he could demo once to reassure the class that they can't "lose" the diagram permanently.
Projector readability: good — this is the same clear top-level framing as the earlier full-diagram shots.

## 35-neurons-trained.png
Action: same setup as always (custom text, Create, prompt, Train 3s, Stop, Step), then clicked the "Neurons" tab, wait 1.8s.
What Ortega would say it shows: a completely different, much denser diagram style — a column-by-column node graph resembling a classic neural-network picture, with labeled columns tok (word row), pos (position row), x0 (sum), q/k/v (question/label/content), a shaded "q, k, softmax" attention region, ctxv (mix by attention), attn_out (written back after attention), x1, h2, ff_pre (hidden), ff_act (after ReLU), ff_out (block output), x2, logits (scores), probs (probabilities), with orange weight-matrix boxes (W_q, W_k, W_v, W_0, W_1, W_2, U) connecting them, and two dashed arcs across the top labeled "x0 carried forward" / "x1 carried forward" for the residual connections. A caption at the bottom reads "Numbers for position 6 of 6: 'students'. Hover a circle to see its wires." This is, on content alone, the single most useful screen in the whole batch for a CS class — actual neurons-as-circles, actual weight matrices named and positioned, an explicit softmax and ReLU labeled in place.
Overlap/cut-off/unreadable/misaligned: the entire diagram is extremely dense — dozens of small circles (a few pixels each) and 8-9px column labels packed across the full canvas width. Nothing overlaps, but at normal viewing distance most of the individual circles and their inter-column wires are indistinguishable from visual noise.
Broken/inconsistent vs. previous shot: this is a different steps file (steps14.json), so Iterations reads 127 here versus 128 in shot 34 — coincidentally close, but again a reminder these aren't one continuous run.
What he'd have to say out loud: he would need to pre-teach residual connections ("carried forward"), Q/K/V naming, and softmax/ReLU by name before this screen would mean anything — none of those terms are defined on screen, only labeled.
Projector readability: this is the worst screen in the batch for a projector specifically because it is the richest one on a laptop — the sheer density of circles and 8-9px labels that would be legible up close on a personal screen would almost certainly be an illegible smear from the back of a classroom.

## 36-neurons-hover.png
Action: same run, hovered at pixel (606, 400) — over one of the neuron circles in the diagram — wait 1.2s.
What Ortega would say it shows: visually, nothing distinguishable from shot 35 — same caption ("Numbers for position 6 of 6: 'students'. Hover a circle to see its wires."), same diagram, no visible tooltip, no highlighted "wires" connecting the hovered circle to its inputs/outputs that I can detect at this resolution.
Overlap/cut-off/unreadable/misaligned: n/a — nothing changed to check.
Broken/inconsistent vs. previous shot: this is the concerning part — the caption explicitly instructs "Hover a circle to see its wires," which promises a reactive highlight, but hovering produced no visible difference from the unhovered state. Either this specific pixel missed the actual clickable circle (targets may be only a couple of pixels wide, easy to miss), or the hover feedback is real but too subtle to read in a screenshot — either way, it's exactly the kind of feature a first-time user (or 25 students, mouse-less, watching a projector) would fail to discover or see.
What he'd have to say out loud: he'd have to tell the class "trust me, hovering highlights connections" even though the screen in front of them shows no evidence of it happening.
Projector readability: if the effect is real but subtle, it fails completely at classroom-projector distance; if the effect requires pixel-precise targeting, it's also impractical to demo live.

## 37-neurons-click.png
Action: same run, clicked at the same pixel (606, 400), wait 1.5s.
What Ortega would say it shows: again, visually indistinguishable from shots 35 and 36 — same caption, same diagram, no visible selection highlight, no side panel, no change in color or emphasis anywhere I can detect.
Overlap/cut-off/unreadable/misaligned: n/a.
Broken/inconsistent vs. previous shot: three consecutive actions in a row — plain load, hover, and click, all targeting the same spot — produced three screenshots with no visible differences between them. That's a real functional concern: if clicking a neuron is supposed to do something (select it, pin its wires, show its value), nothing in the screenshot evidence confirms it did.
What he'd have to say out loud: he would not be able to show the class "watch what happens when I click a neuron" with any confidence, because nothing observably happens at this zoom level.
Projector readability: fails outright — a class watching this sequence projected would see three identical-looking frames and conclude correctly that nothing happened.

