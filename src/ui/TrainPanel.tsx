import { useStore } from '../app/store';

const pct = (v: number) => Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : 'n/a';
const f2 = (v: number) => Number.isFinite(v) ? v.toFixed(3) : 'n/a';

export function TrainPanel() {
  const training = useStore(s => s.training), history = useStore(s => s.history), trainError = useStore(s => s.trainError);
  const corpus = useStore(s => s.corpus), fallback = useStore(s => s.trainerFallback);
  const start = useStore(s => s.startTraining), stop = useStore(s => s.stopTraining), setView = useStore(s => s.setView);
  const last = history[history.length - 1];
  const recent = history.slice(-100);
  const maxLoss = Math.max(0.01, ...recent.flatMap(m => [m.trainLoss, m.testLoss].filter(Number.isFinite)));
  const path = (key: 'trainLoss' | 'testLoss') => recent.map((m, i) => Number.isFinite(m[key]) ? `${i === 0 ? 'M' : 'L'}${(i / Math.max(1, recent.length - 1)) * 100},${40 - (m[key] / maxLoss) * 38}` : '').join(' ');

  return (
    <section className="panel">
      <h2>Train</h2>
      <div className="row">
        {training === 'running'
          ? <button className="primary" onClick={stop}>Stop</button>
          : <button className="primary" onClick={start} disabled={!corpus || corpus.trainWindows.length === 0}>Train</button>}
        <span className="hint">{last ? `iteration ${last.iteration}` : 'not trained yet'}</span>
      </div>
      {last && (
        <dl className="stats">
          <dt>Train loss</dt><dd>{f2(last.trainLoss)}</dd>
          <dt>Test loss</dt><dd>{f2(last.testLoss)}</dd>
          <dt>Train accuracy</dt><dd>{pct(last.trainAcc)}</dd>
          <dt>Test accuracy</dt><dd>{pct(last.testAcc)}</dd>
        </dl>
      )}
      {recent.length > 1 && (
        <svg className="spark" viewBox="0 0 100 40" preserveAspectRatio="none">
          <path d={path('testLoss')} fill="none" stroke="#2166ac" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          <path d={path('trainLoss')} fill="none" stroke="#b2182b" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        </svg>
      )}
      {trainError && <p className="error">{trainError}</p>}
      {fallback && training === 'running' && <p className="notice">Training on the main thread; the page may stutter.</p>}
      <button className="link" onClick={() => setView('loss')}>Open Loss view</button>
    </section>
  );
}
