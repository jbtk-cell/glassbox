import { useState } from 'react';
import { useStore } from '../app/store';

export function LMControls({ showVocab, setShowVocab, showText, setShowText }: { showVocab: boolean; setShowVocab: (b: boolean) => void; showText: boolean; setShowText: (b: boolean) => void }) {
  const sample = useStore(s => s.sample), setSample = useStore(s => s.setSample);
  const [strategyOpen, setStrategyOpen] = useState(false);
  return (
    <section className="frame">
      <div className="titlebar">Language Model Controls</div>
      <div className="frame-body controls">
        <button className={showVocab ? 'on' : ''} onClick={() => setShowVocab(!showVocab)}>{showVocab ? 'Hide' : 'Show'} Vocabulary</button>
        <button className={showText ? 'on' : ''} onClick={() => setShowText(!showText)}>{showText ? 'Hide' : 'Show'} Training Text</button>
        <label className="slider"><span>Temperature</span>
          <input type="range" min={0} max={2} step={0.05} value={sample.temperature} onChange={e => setSample({ temperature: Number(e.target.value) })} />
          <span className="value">{sample.temperature.toFixed(2)}</span>
        </label>
        <button onClick={() => setStrategyOpen(!strategyOpen)}>Configure Sampling Strategy...</button>
        {strategyOpen && (
          <div className="form">
            <label><span>Strategy</span>
              <select value={sample.strategy} onChange={e => setSample({ strategy: e.target.value as typeof sample.strategy })}>
                <option value="greedy">Greedy</option><option value="top-k">Top-k</option><option value="top-p">Top-p</option>
              </select>
            </label>
            {sample.strategy === 'top-k' && <label><span>k</span><input type="number" min={1} max={50} value={sample.k} onChange={e => setSample({ k: Number(e.target.value) || 1 })} /></label>}
            {sample.strategy === 'top-p' && <label><span>p</span><input type="number" min={0.05} max={1} step={0.05} value={sample.p} onChange={e => setSample({ p: Number(e.target.value) || 0.9 })} /></label>}
          </div>
        )}
      </div>
    </section>
  );
}
