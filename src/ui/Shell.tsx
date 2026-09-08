import { useCallback, useEffect, useState } from 'react';
import { useStore, type View } from '../app/store';
import { useKeys } from '../app/keys';
import { TextInputs, VocabularyPanel, TrainingTextPanel } from './TextInputs';
import { LMControls } from './LMControls';
import { Transport } from './Transport';
import { CellPanel } from './CellPanel';
import { TrainingPanel } from './TrainingPanel';
import { ModelDialog } from './ModelDialog';
import { MapView } from '../views/map/MapView';
import { NetworkView } from '../views/network/NetworkView';
import { ExplainView } from '../views/explain/ExplainView';
import { LossView } from '../views/loss/LossView';
import { splitWords } from '../engine/corpus/tokenize';

const VIEWS: { id: View; label: string }[] = [{ id: 'flow', label: 'Network' }, { id: 'network', label: 'Neurons' }, { id: 'math', label: 'Explain' }, { id: 'loss', label: 'Loss' }];

export function Shell() {
  const view = useStore(s => s.view), setView = useStore(s => s.setView);
  const prompt = useStore(s => s.prompt), setPrompt = useStore(s => s.setPrompt), stepGenerate = useStore(s => s.stepGenerate);
  const model = useStore(s => s.model);
  const [playing, setPlaying] = useState(false);
  const [dialog, setDialog] = useState<'model' | null>(null);
  const [showVocab, setShowVocab] = useState(false), [showText, setShowText] = useState(false);
  const togglePlay = useCallback(() => setPlaying(p => !p), []);
  useKeys(togglePlay);
  const tokens = splitWords(prompt).length;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => { const s = useStore.getState(); if (!s.model || splitWords(s.prompt).length === 0 || splitWords(s.prompt).length > 400) setPlaying(false); else s.stepGenerate(); }, 350);
    return () => clearInterval(id);
  }, [playing]);

  return (
    <div className="workspace">
      <div className="toolbar">
        <span className="app">glassbox</span>
        <button onClick={stepGenerate} disabled={!model || tokens === 0} title="Predict the next word and add it">Step</button>
        <button onClick={togglePlay} disabled={!model || tokens === 0} title="Keep predicting">{playing ? 'Stop' : 'Play'}</button>
        <button onClick={() => { setPlaying(false); setPrompt(''); }} disabled={tokens === 0}>Clear</button>
        <span className="readout">Tokens: {tokens}</span>
        <span className="spacer" />
        <button onClick={() => setDialog('model')}>Model...</button>
      </div>
      <div className="desk">
        <div className="column">
          <TextInputs />
          <LMControls showVocab={showVocab} setShowVocab={setShowVocab} showText={showText} setShowText={setShowText} />
          <TrainingPanel />
          {showVocab && <VocabularyPanel />}
          {showText && <TrainingTextPanel />}
        </div>
        <section className="frame network">
          <div className="titlebar">
            <span className="tabs">{VIEWS.map(v => <button key={v.id} className={view === v.id ? 'active' : ''} onClick={() => setView(v.id)}>{v.label}</button>)}</span>
          </div>
          <div className="view">
            {view === 'flow' && <MapView />}
            {view === 'network' && <NetworkView />}
            {view === 'math' && <ExplainView />}
            {view === 'loss' && <div className="loss-wrap"><LossView /></div>}
            <CellPanel />
          </div>
          <Transport />
        </section>
      </div>
      {dialog === 'model' && <ModelDialog onClose={() => setDialog(null)} />}
    </div>
  );
}
