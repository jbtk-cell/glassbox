import { useCallback, useState } from 'react';
import { useStore, type View } from '../app/store';
import { useKeys } from '../app/keys';
import { TextPanel } from './TextPanel';
import { TrainPanel } from './TrainPanel';
import { GeneratePanel } from './GeneratePanel';
import { Transport } from './Transport';
import { Inspector } from './Inspector';
import { FlowView } from '../views/flow/FlowView';
import { NetworkView } from '../views/network/NetworkView';
import { MathView } from '../views/math/MathView';
import { LossView } from '../views/loss/LossView';

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'flow', label: 'Flow', hint: 'The computation as tiles and wires. Scroll to zoom, drag to pan.' },
  { id: 'network', label: 'Network', hint: 'Every neuron and every connection for one position.' },
  { id: 'math', label: 'Math', hint: 'The formula for the current operation, with the real numbers.' },
  { id: 'loss', label: 'Loss', hint: 'Training and test loss, accuracy, and gradient sizes.' },
];

export function Shell() {
  const view = useStore(s => s.view), setView = useStore(s => s.setView);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(5);
  const togglePlay = useCallback(() => setPlaying(p => !p), []);
  useKeys(togglePlay);

  return (
    <div className="shell">
      <header className="top">
        <h1>glassbox</h1>
        <p>A tiny GPT you can train on your own text and step through one operation at a time. Nothing leaves your browser.</p>
        <a href="https://github.com/jbtk-cell/glassbox" target="_blank" rel="noreferrer">Source</a>
      </header>
      <div className="left">
        <TextPanel />
        <TrainPanel />
        <GeneratePanel />
      </div>
      <main className="centre">
        <nav className="tabs">
          {VIEWS.map(v => <button key={v.id} className={view === v.id ? 'active' : ''} title={v.hint} onClick={() => setView(v.id)}>{v.label}</button>)}
          <span className="hint">{VIEWS.find(v => v.id === view)?.hint}</span>
        </nav>
        <div className="view">
          {view === 'flow' && <FlowView />}
          {view === 'network' && <NetworkView />}
          {view === 'math' && <MathView />}
          {view === 'loss' && <LossView />}
        </div>
      </main>
      <Transport playing={playing} setPlaying={setPlaying} speed={speed} setSpeed={setSpeed} />
      <Inspector />
    </div>
  );
}
