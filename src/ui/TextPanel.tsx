import { useShallow } from 'zustand/react/shallow';
import { useStore } from '../app/store';
import { PRESETS } from '../presets';

export function TextPanel() {
  const { text, presetId, corpus, corpusError, config, lr } = useStore(useShallow(s => ({ text: s.text, presetId: s.presetId, corpus: s.corpus, corpusError: s.corpusError, config: s.config, lr: s.lr })));
  const setText = useStore(s => s.setText), applyText = useStore(s => s.applyText);
  const setConfig = useStore(s => s.setConfig), setLr = useStore(s => s.setLr), resetModel = useStore(s => s.resetModel);
  const training = useStore(s => s.training);

  const num = (label: string, key: 'contextSize' | 'dModel' | 'dFF', min: number, max: number) => (
    <label className="row"><span>{label}</span>
      <input type="number" min={min} max={max} value={config[key]} disabled={training === 'running'}
        onChange={e => { setConfig({ [key]: Math.max(min, Math.min(max, Number(e.target.value) || min)) }); }} onBlur={() => applyText()} />
    </label>
  );

  return (
    <section className="panel">
      <h2>Text</h2>
      <label className="row"><span>Preset</span>
        <select value={presetId ?? 'custom'} onChange={e => { const p = PRESETS.find(x => x.id === e.target.value); if (p) setText(p.text, p.id); else setText('', null); }}>
          {PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          <option value="custom">Your own text</option>
        </select>
      </label>
      {presetId && <p className="hint">{PRESETS.find(p => p.id === presetId)?.blurb}</p>}
      <textarea value={text} rows={8} spellCheck={false} placeholder="Paste anything: messages, notes, a story. It stays in your browser." onChange={e => setText(e.target.value, null)} />
      <button className="primary" onClick={applyText} disabled={training === 'running'}>Use this text</button>
      {corpusError && <p className="error">{corpusError}</p>}
      {corpus && (
        <dl className="stats">
          <dt>Words</dt><dd>{corpus.tokens.length.toLocaleString()}</dd>
          <dt>Distinct</dt><dd>{corpus.vocab.words.length - 1}{corpus.vocab.folded > 0 && ` (+${corpus.vocab.folded} folded)`}</dd>
          <dt>Train / test windows</dt><dd>{corpus.trainWindows.length} / {corpus.testWindows.length}</dd>
        </dl>
      )}
      {corpus?.notices.map((n, i) => <p key={i} className="notice">{n}</p>)}
      <details>
        <summary>Model settings</summary>
        {num('Context size', 'contextSize', 4, 64)}
        {num('Embedding size', 'dModel', 4, 64)}
        {num('Hidden size', 'dFF', 4, 128)}
        <label className="row"><span>Learning rate</span>
          <input type="number" step="0.001" min={0.0001} max={0.1} value={lr} onChange={e => setLr(Number(e.target.value) || 0.001)} />
        </label>
        <p className="hint">Seed {config.seed}. Parameters: {useStore.getState().model?.paramCount().toLocaleString() ?? '-'}</p>
        <button onClick={resetModel} disabled={!corpus || training === 'running'}>New random model</button>
      </details>
    </section>
  );
}
