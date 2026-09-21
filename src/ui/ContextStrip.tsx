/** The sentence the model is reading and the word it guessed, shown above the Network and
 *  Neurons views so the picture underneath always has a subject. Click a word to pick the
 *  position the Neurons view shows. */
import { useStore } from '../app/store';
import { Cursor } from '../engine/trace/cursor';
import { tKey } from '../engine/trace/trace';
import { T_ } from '../engine/model/gpt';
import { contextWords, displayWord } from './words';

export function ContextStrip() {
  const trace = useStore(s => s.trace), cursorIndex = useStore(s => s.cursorIndex), position = useStore(s => s.position);
  const setPosition = useStore(s => s.setPosition), corpus = useStore(s => s.corpus), prompt = useStore(s => s.prompt);
  const contextSize = useStore(s => s.config.contextSize);
  if (!trace || !corpus) {
    return <div className="context-strip empty">Type a few words in Text Inputs and press <b>Step</b>. The model reads the last {contextSize} words and guesses the next one.</div>;
  }
  const { words, known } = contextWords(trace, prompt, corpus.vocab);
  const T = trace.ctx.T;
  const cursor = new Cursor(trace); cursor.index = cursorIndex;
  const ready = cursor.status(tKey(T_.probs)) === 'done' || cursor.status(tKey(T_.probs)) === 'active';
  const P = trace.ctx.tensors.get(T_.probs)!; const V = P.shape[1]; const row = (T - 1) * V;
  let best = 1; for (let v = 2; v < V; v++) if (P.data[row + v] > P.data[row + best]) best = v;
  const pct = (P.data[row + best] * 100).toFixed(1);
  const target = trace.ctx.targets ? displayWord(corpus.vocab.words[trace.ctx.targets[T - 1]]) : null;
  return (
    <div className="context-strip">
      <span className="strip-label">{trace.kind === 'training' ? 'Training on' : 'Reading'}</span>
      <span className="strip-words">
        {words.map((w, t) => (
          <button key={t} type="button" className={'tok' + (t === position ? ' on' : '') + (known[t] ? '' : ' unk')}
            title={(known[t] ? '' : 'Not in Training Text, so the model sees it as an unknown word. ') + `Position ${t + 1} of ${T}. Click to show this position in Neurons.`}
            onClick={() => setPosition(t)}>{w}</button>
        ))}
      </span>
      <span className="strip-result">
        <span className="strip-arrow">&rarr;</span>
        {ready ? (
          <span className="strip-guess">
            {trace.kind === 'training' ? 'guessed' : 'next word'} <b>{displayWord(corpus.vocab.words[best])}</b> <span className="pct">{pct}%</span>
            {target !== null && <span className="target">actual next word <b>{target}</b></span>}
          </span>
        ) : <span className="strip-guess pending">next word: not computed yet</span>}
      </span>
    </div>
  );
}
