import { useStore } from '../app/store';
import { splitWords } from '../engine/corpus/tokenize';

const NO_SPACE_BEFORE = new Set(['.', ',', '?', '!', ';', ':', ')']);

export function GeneratePanel() {
  const prompt = useStore(s => s.prompt), setPrompt = useStore(s => s.setPrompt);
  const recordPrompt = useStore(s => s.recordPrompt), generateMore = useStore(s => s.generateMore), setView = useStore(s => s.setView);
  const sample = useStore(s => s.sample), setSample = useStore(s => s.setSample);
  const corpus = useStore(s => s.corpus), generated = useStore(s => s.generated), unknown = useStore(s => s.promptUnknown);
  const words = corpus?.vocab.words ?? [];
  const promptLen = splitWords(prompt).length;
  const ready = !!corpus && useStore.getState().model !== null;

  const rendered = generated?.map((t, i) => {
    const w = words[t] ?? '<unk>';
    const glue = i === 0 || w === '\n' || NO_SPACE_BEFORE.has(w) || words[generated[i - 1]] === '\n' ? '' : ' ';
    return <span key={i} className={i >= promptLen ? 'gen' : undefined}>{glue}{w === '\n' ? '\n' : w}</span>;
  });

  return (
    <section className="panel">
      <h2>Predict</h2>
      <textarea value={prompt} rows={2} spellCheck={false} placeholder="type a few words the model has seen" onChange={e => setPrompt(e.target.value)} />
      {unknown.length > 0 && <p className="notice">Not in vocabulary: {unknown.join(', ')}</p>}
      <div className="row">
        <button className="primary" onClick={() => { recordPrompt(); setView('flow'); }} disabled={!ready || promptLen === 0}>Predict next word</button>
        <button onClick={() => generateMore(20)} disabled={!ready || promptLen === 0}>Generate 20 words</button>
      </div>
      <details>
        <summary>Sampling</summary>
        <label className="row"><span>Strategy</span>
          <select value={sample.strategy} onChange={e => setSample({ strategy: e.target.value as typeof sample.strategy })}>
            <option value="greedy">Greedy (always the top word)</option>
            <option value="top-k">Top-k</option>
            <option value="top-p">Top-p (nucleus)</option>
          </select>
        </label>
        <label className="row"><span>Temperature {sample.temperature.toFixed(2)}</span>
          <input type="range" min={0} max={2} step={0.05} value={sample.temperature} onChange={e => setSample({ temperature: Number(e.target.value) })} />
        </label>
        {sample.strategy === 'top-k' && <label className="row"><span>k</span><input type="number" min={1} max={50} value={sample.k} onChange={e => setSample({ k: Number(e.target.value) || 1 })} /></label>}
        {sample.strategy === 'top-p' && <label className="row"><span>p</span><input type="number" min={0.05} max={1} step={0.05} value={sample.p} onChange={e => setSample({ p: Number(e.target.value) || 0.9 })} /></label>}
      </details>
      {generated && <p className="output">{rendered}</p>}
    </section>
  );
}
