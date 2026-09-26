import { useEffect, useMemo, useState } from 'react';
import { useStore } from '../app/store';
import { PRESETS } from '../presets';
import { buildCorpus, TRAIN_RATIO } from '../app/corpusState';
import { GPT } from '../engine/model/gpt';

export function ModelDialog({ onClose }: { onClose: () => void }) {
  const s = useStore.getState();
  const [contextSize, setContextSize] = useState(s.config.contextSize);
  const [dModel, setDModel] = useState(s.config.dModel);
  const [dFF, setDFF] = useState(s.config.dFF);
  const [presetId, setPresetId] = useState(s.presetId ?? 'custom');
  const [text, setText] = useState(s.text);
  const corpusError = useStore(x => x.corpusError);
  // The count follows the dialog's own settings and text, so it updates before Create is pressed.
  const paramCount = useMemo(() => {
    const r = buildCorpus(text, contextSize);
    if ('error' in r) return null;
    return new GPT({ ...s.config, contextSize, dModel, dFF, vocabSize: r.corpus.vocab.words.length }).paramCount();
  }, [text, contextSize, dModel, dFF, s.config]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  const create = () => {
    const st = useStore.getState();
    st.setConfig({ contextSize, dModel, dFF });
    st.setText(text, presetId === 'custom' ? null : presetId);
    st.applyText();
    if (!useStore.getState().corpusError) onClose();
  };
  const num = (label: string, v: number, set: (n: number) => void, min: number, max: number) => (
    <tr><td>{label}</td><td><input type="number" min={min} max={max} value={v} onChange={e => set(Math.max(min, Math.min(max, Number(e.target.value) || min)))} /></td></tr>
  );
  return (
    <div className="backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog">
        <div className="titlebar">Tiny Language Model</div>
        <div className="dialog-body">
          <table className="props">
            <tbody>
              {num('Context size', contextSize, setContextSize, 4, 64)}
              {num('Embedding dimension', dModel, setDModel, 4, 64)}
              {num('Hidden size', dFF, setDFF, 4, 128)}
              <tr><td>Training text</td><td>
                <select value={presetId} onChange={e => { const id = e.target.value; setPresetId(id); const p = PRESETS.find(x => x.id === id); if (p) setText(p.text); }}>
                  {PRESETS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                  <option value="custom">Your own text</option>
                </select>
              </td></tr>
              <tr title="The first part of the text is for learning; the rest is held back to test whether what it learned carries over"><td>Train test split</td><td>{TRAIN_RATIO}</td></tr>
              <tr title="Every learned number in the model; the total grows with the vocabulary"><td>Parameters</td><td>{paramCount?.toLocaleString() ?? '-'}</td></tr>
            </tbody>
          </table>
          <textarea value={text} rows={9} spellCheck={false} onChange={e => { setText(e.target.value); setPresetId('custom'); }} />
          {corpusError && <p className="error">{corpusError}</p>}
        </div>
        <div className="buttons">
          <span className="spacer" />
          <button onClick={onClose}>Cancel</button>
          <button onClick={create}>Create</button>
        </div>
      </div>
    </div>
  );
}
