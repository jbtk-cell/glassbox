# Play-test brief

You are play-testing a website, as a first-time user. Treat it as a black box:
do NOT read its source code, README, docs, or git history. Judge only what you
can see and do in the browser. Nobody who built it will see your report before
it is used, so be blunt and specific. Vague praise is useless; "the word
'Probabilities' overlaps the orange tile at the bottom right of screenshot
06.png" is useful.

Site: http://localhost:5173/glassbox/   (a local dev server; it is already running)

## How to drive it

A headless browser driver is at /Users/johnnyklaus/.claude/jobs/9b256b46/tmp/drive.mjs.
Read the comment at the top of that file for the action list. Write a JSON
list of actions to a file in YOUR OWN output folder, then run:

    node /Users/johnnyklaus/.claude/jobs/9b256b46/tmp/drive.mjs <steps.json> <your-outdir>

Screenshots land in your outdir; open every one with the Read tool and look at
it properly (the whole picture, then the corners and small text). The viewport
is 1440x900 by default. Run as many rounds as you need; each run starts a
fresh browser, so put the whole sequence you want in one steps file (goto,
then actions, with "shot" wherever you want to look). Waits of 1000-2000 ms
after clicks are usually needed. The canvas responds to mouse: hover, click,
drag to pan, wheel to zoom. Use {"text": "body"} or {"html": "..."} to read the
DOM when a screenshot is unclear, and {"console": true} at the end of every run.

Do what a real curious user would do: read the screen, guess what things are
for, click them, type your own sentences, train, step through, switch every
tab, hover things, zoom, and try to make it break (empty text, one word, a very
long paragraph, unknown words, punctuation, fast repeated clicks, pressing
Train twice, Reset in the middle, resizing to a laptop window 1280x720).

## Your report

Write it to the path given in your task, in Markdown, under these headings.
Rank items by how much they would confuse or annoy a real user. Reference
screenshot filenames (they will be read alongside your report).

1. First 60 seconds: what you thought the site was for, and what you thought
   each visible control did, BEFORE clicking anything. Then what turned out to
   be true.
2. Walkthrough: what you did, in order, and what you saw. Note every moment
   you were unsure what was happening or what to do next.
3. Visual defects: anything overlapping, cut off, unreadable, misaligned,
   inconsistent, or ugly. One line each with screenshot name and location.
4. Functional defects: anything that did nothing, did the wrong thing, errored,
   got stuck, or behaved differently on repeat.
5. Comprehension: after using it, explain in your own words what the model is
   doing when you press Step. Then say which parts of that understanding came
   from the site and which came from your prior knowledge. Say plainly what
   the site failed to explain.
6. Top 5 changes, ranked, each with the one-sentence reason.
7. Would you show this to a friend? One honest sentence.
