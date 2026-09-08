import { useStore } from '../app/store';
import type { Metrics } from '../app/worker/protocol';
import './training.css';

const pct = (v?: number) => v !== undefined && Number.isFinite(v) ? (v * 100).toFixed(1) + ' %' : '-';
const f3 = (v?: number) => v !== undefined && Number.isFinite(v) ? v.toFixed(4) : '-';

function Sparkline({ history }: { history: Metrics[] }) {
  if (history.length === 0) return null;
  let max = 0;
  for (const m of history) {
    if (Number.isFinite(m.trainLoss)) max = Math.max(max, m.trainLoss);
    if (Number.isFinite(m.testLoss)) max = Math.max(max, m.testLoss);
  }
  const w = 300, h = 56;
  const x = (i: number) => history.length > 1 ? (i / (history.length - 1)) * w : 0;
  const y = (v: number) => max > 0 ? h - (v / max) * h : h;
  const points = (key: 'trainLoss' | 'testLoss') =>
    history.map((m, i) => Number.isFinite(m[key]) ? `${x(i)},${y(m[key])}` : null).filter((p): p is string => p !== null).join(' ');
  return (
    <>
      <svg className="sparkline" viewBox={`0 0 ${w} ${h}`}>
        <polyline points={points('trainLoss')} fill="none" stroke="#b2182b" strokeWidth={1.5} />
        <polyline points={points('testLoss')} fill="none" stroke="#2166ac" strokeWidth={1.5} />
      </svg>
      <div className="legend">
        <span><i style={{ background: '#b2182b' }} />train loss</span>
        <span><i style={{ background: '#2166ac' }} />test loss</span>
      </div>
    </>
  );
}

export function TrainingPanel() {
  const training = useStore(s => s.training), history = useStore(s => s.history), trainError = useStore(s => s.trainError);
  const corpus = useStore(s => s.corpus), lr = useStore(s => s.lr), setLr = useStore(s => s.setLr);
  const start = useStore(s => s.startTraining), stop = useStore(s => s.stopTraining), resetModel = useStore(s => s.resetModel);
  const last = history[history.length - 1];
  return (
    <section className="frame training">
      <div className="titlebar">Training</div>
      <div className="frame-body">
        <div className="train-buttons">
          {training === 'running' ? <button onClick={stop}>Stop</button> : <button onClick={start} disabled={!corpus || corpus.trainWindows.length === 0}>Train</button>}
          <button onClick={resetModel} disabled={!corpus || training === 'running'}>Reset weights</button>
        </div>
        <table className="props">
          <tbody>
            <tr><td>Iterations</td><td>{last?.iteration ?? 0}</td></tr>
            <tr><td>Training loss</td><td>{f3(last?.trainLoss)}</td></tr>
            <tr><td>Testing loss</td><td>{f3(last?.testLoss)}</td></tr>
            <tr><td>Training accuracy</td><td>{pct(last?.trainAcc)}</td></tr>
            <tr><td>Testing accuracy</td><td>{pct(last?.testAcc)}</td></tr>
            <tr><td>Learning rate</td><td><input type="number" step="0.001" min={0.0001} max={0.1} value={lr} onChange={e => setLr(Number(e.target.value) || 0.001)} /></td></tr>
          </tbody>
        </table>
        <Sparkline history={history} />
        {trainError && <p className="error">{trainError}</p>}
      </div>
    </section>
  );
}
