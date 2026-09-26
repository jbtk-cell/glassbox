import { TRAIN_RATIO } from '../app/corpusState';
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

/** True once testing loss has climbed well above its best value while training loss went on falling. */
export function isOverfitting(history: Metrics[]): boolean {
  if (history.length < 30) return false;
  let bestI = 0;
  for (let i = 1; i < history.length; i++) if (history[i].testLoss < history[bestI].testLoss) bestI = i;
  const best = history[bestI], last = history[history.length - 1];
  if (!Number.isFinite(best.testLoss) || !Number.isFinite(last.testLoss)) return false;
  return bestI < history.length - 10 && last.testLoss > best.testLoss * 1.08 && last.trainLoss < best.trainLoss;
}

export function TrainingPanel() {
  const training = useStore(s => s.training), history = useStore(s => s.history), trainError = useStore(s => s.trainError);
  const corpus = useStore(s => s.corpus), lr = useStore(s => s.lr), setLr = useStore(s => s.setLr);
  const start = useStore(s => s.startTraining), stop = useStore(s => s.stopTraining), resetModel = useStore(s => s.resetModel);
  const last = history[history.length - 1];
  const overfit = isOverfitting(history);
  const trainPct = Math.round(TRAIN_RATIO * 100), testPct = 100 - trainPct;
  return (
    <section className="frame training">
      <div className="titlebar">Training</div>
      <div className="frame-body">
        <div className="train-buttons">
          {training === 'running' ? <button onClick={stop}>Stop</button> : <button onClick={start} disabled={!corpus || corpus.trainWindows.length === 0}>Train</button>}
          <button onClick={resetModel} disabled={!corpus || training === 'running'} title={training === 'running' ? 'Press Stop first' : 'Start again from random numbers (a new seed)'}>Reset weights</button>
        </div>
        <table className="props">
          <tbody>
            <tr title="One iteration: the model reads one stretch of Training Text, guesses each next word, and every learned number is nudged once"><td>Iterations</td><td>{last?.iteration ?? 0}</td></tr>
            <tr title={`How wrong the guesses are on the first ${trainPct}% of Training Text, the part the model learns from. Lower is better; 0 would be perfect`}><td>Training loss</td><td>{f3(last?.trainLoss)}</td></tr>
            <tr title={`The same measure on the last ${testPct}% of Training Text, which the model never learns from. If this rises while training loss falls, the model is memorising`}><td>Testing loss</td><td>{f3(last?.testLoss)}</td></tr>
            <tr title={`How often the most likely word is the right one, on the ${trainPct}% it learns from`}><td>Training accuracy</td><td>{pct(last?.trainAcc)}</td></tr>
            <tr title={`How often the most likely word is the right one, on the held-out ${testPct}%`}><td>Testing accuracy</td><td>{pct(last?.testAcc)}</td></tr>
            <tr><td>Learning rate</td><td><input type="number" step="0.001" min={0.0001} max={0.1} value={lr} onChange={e => setLr(Number(e.target.value) || 0.001)} /></td></tr>
          </tbody>
        </table>
        <Sparkline history={history} />
        {training === 'running' && !overfit && <p className="note quiet">Training runs until you press Stop. Press Step at any time to see a prediction with the current numbers.</p>}
        {overfit && <p className="note">Overfitting: testing loss is rising while training loss keeps falling. The model is memorising Training Text rather than learning patterns that carry over. Stop here, or add more text.</p>}
        {trainError && <p className="error">{trainError}</p>}
      </div>
    </section>
  );
}
