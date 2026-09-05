import { useStore } from '../app/store';
import { LossView } from '../views/loss/LossView';

const pct = (v?: number) => v !== undefined && Number.isFinite(v) ? (v * 100).toFixed(1) + ' %' : '-';
const f3 = (v?: number) => v !== undefined && Number.isFinite(v) ? v.toFixed(4) : '-';

export function TrainDialog({ onClose }: { onClose: () => void }) {
  const training = useStore(s => s.training), history = useStore(s => s.history), trainError = useStore(s => s.trainError);
  const corpus = useStore(s => s.corpus), lr = useStore(s => s.lr), setLr = useStore(s => s.setLr);
  const start = useStore(s => s.startTraining), stop = useStore(s => s.stopTraining), resetModel = useStore(s => s.resetModel);
  const last = history[history.length - 1];
  return (
    <div className="backdrop" onMouseDown={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="dialog wide">
        <div className="titlebar">Train Network</div>
        <div className="dialog-body train">
          <table className="props">
            <tbody>
              <tr><td>Iterations</td><td>{last?.iteration ?? 0}</td></tr>
              <tr><td>Training loss</td><td>{f3(last?.trainLoss)}</td></tr>
              <tr><td>Testing loss</td><td>{f3(last?.testLoss)}</td></tr>
              <tr><td>Training accuracy</td><td>{pct(last?.trainAcc)}</td></tr>
              <tr><td>Testing accuracy</td><td>{pct(last?.testAcc)}</td></tr>
              <tr><td>Training windows</td><td>{corpus?.trainWindows.length ?? 0}</td></tr>
              <tr><td>Testing windows</td><td>{corpus?.testWindows.length ?? 0}</td></tr>
              <tr><td>Learning rate</td><td><input type="number" step="0.001" min={0.0001} max={0.1} value={lr} onChange={e => setLr(Number(e.target.value) || 0.001)} /></td></tr>
            </tbody>
          </table>
          <div className="plot"><LossView /></div>
        </div>
        {trainError && <p className="error">{trainError}</p>}
        <div className="buttons">
          {training === 'running' ? <button onClick={stop}>Stop</button> : <button onClick={start} disabled={!corpus || corpus.trainWindows.length === 0}>Train</button>}
          <button onClick={resetModel} disabled={!corpus || training === 'running'}>Randomize</button>
          <span className="spacer" />
          <button onClick={onClose}>Done</button>
        </div>
      </div>
    </div>
  );
}
