# Play-test round 1, 2026-09-25

Three testers with no knowledge of the code drove the running site black-box, following
`brief.md`. Each had a persona and its own headless Chromium; none saw the others' reports
or anything the builder thought was wrong.

| Tester | Persona | Model | Evidence |
|---|---|---|---|
| novice teen | 16-year-old who uses ChatGPT, never had "transformer" explained | Claude (Opus drove, Sonnet judged the 40 screenshots) | `novice-teen-notes.md` |
| CS teacher | has used Simbrain's Tiny LM in class, judging fit for a 50-minute lesson | Claude (Opus drove, Sonnet judged the 41 screenshots) | `cs-teacher-notes.md` |
| ML student | took one ML course, checks the drawing against the maths | OpenAI Codex (gpt-5.6) | `ml-student-codex.md`, `ml-student-codex-notes.md` |

The two Claude drivers stalled repeatedly (the machine slept for two hours, then the API
degraded), so their notes were written by fresh judges from the screenshots and step files
they left. Codex ran inside its sandbox, which cannot launch Chromium, so it drove the
browser through a tiny local HTTP service that ran the same driver script outside the sandbox.

## What every tester hit

1. The strip's predicted word and the word Step appended did not match, and nothing said
   why. Cause: the strip showed the most likely word while Step samples at temperature 1.
2. Labels such as `x0`, `ff_pre`, `ctxv`, `W_q` appeared with no explanation.
3. Words Step appended looked identical to words the user typed.

## What at least one tester hit

- Neurons captions overlapped at 1280 wide; strip word labels ran into the next strip.
- Hovering on the Neurons tab showed nothing unless exactly on a circle.
- Step on empty text gave no feedback (the button was disabled with no reason).
- Show Vocabulary opened a panel below the fold, so it seemed to do nothing.
- Training: nothing on the map moved, Reset weights was greyed with no reason, the test
  split was unexplained, instant overfitting on a 14-line corpus went unlabelled.
- Explain: vocabulary softmax used attention notation; "nothing is lost by adding";
  "with 5 words that is a 5 by 5 grid" with 20 words in the window; the MLP described as
  independent of context; the walkthrough opened at step 21.
- Model dialog: Escape did not close it; the parameter count did not follow the text.
- Codex's late runs hit a blank page with "Invalid hook call". That was the builder's
  fault, not the app's: a second dev server shared the main checkout's `node_modules/.vite`
  cache through a symlink and rewrote it, so the first server shipped two copies of React.

Claims that did not reproduce: "Show Vocabulary reset the app" (a misread of a 1280x720
screenshot where the panel was below the fold) and "the zoom buttons scroll away" (they
are fixed; the map shrank to a dot and its group tags piled up, which was the real defect).

All of the above was addressed in the commit that added this folder, except the "worked
numeric example by default" and "watchable learning arc" requests, which are design work
for another round.

## Round 2: re-check of the fixed build

A fresh novice-persona tester (Sonnet) drove the fixed build against the eight main
complaints: all eight came back FIXED (`round2-recheck.md`). It raised two small new
points: the dashed border on unknown words had no legend (a legend line now appears under
the word strip when it applies), and with Vocabulary open during training the bottom rows
of the Training panel sit below the fold at 900 px; the column scrolls, but nothing says so.
