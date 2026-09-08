import { useStore } from '../app/store';
import { splitWords } from '../engine/corpus/tokenize';

export function TextInputs() {
  const prompt = useStore(s => s.prompt), setPrompt = useStore(s => s.setPrompt);
  const corpus = useStore(s => s.corpus);
  const words = splitWords(prompt);
  const known = (w: string) => !corpus || corpus.vocab.index.has(w);

  return (
    <section className="frame">
      <div className="titlebar">Text Inputs</div>
      <div className="frame-body">
        <textarea className="prompt-box" value={prompt} rows={5} spellCheck={false} placeholder="Type a few words" onChange={e => setPrompt(e.target.value)} />
        {words.length > 0 && (
          <div className="tokens strip">
            {words.map((w, i) => w === '\n' ? <br key={i} /> : <span key={i} className={'tok' + (known(w) ? '' : ' unk')}>{w}</span>)}
          </div>
        )}
      </div>
    </section>
  );
}

export function VocabularyPanel() {
  const corpus = useStore(s => s.corpus);
  if (!corpus) return null;
  const counts = new Map<number, number>();
  for (const t of corpus.tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
  return (
    <section className="frame">
      <div className="titlebar">Vocabulary ({corpus.vocab.words.length - 1} words)</div>
      <div className="frame-body list">
        {corpus.vocab.words.map((w, i) => i === 0 ? null : <div key={i} className="list-row"><span>{w === '\n' ? '\\n' : w}</span><span>{counts.get(i) ?? 0}</span></div>)}
      </div>
    </section>
  );
}

export function TrainingTextPanel() {
  const corpus = useStore(s => s.corpus);
  if (!corpus) return null;
  return (
    <section className="frame">
      <div className="titlebar">Training Text ({corpus.tokens.length.toLocaleString()} tokens)</div>
      <div className="frame-body"><pre className="training-text">{corpus.text}</pre></div>
    </section>
  );
}
