import { useStore } from '../app/store';

/** Steps the recorded computation one operation at a time. */
export function Transport() {
  const trace = useStore(s => s.trace), i = useStore(s => s.cursorIndex), model = useStore(s => s.model);
  const seek = useStore(s => s.seek), next = useStore(s => s.next), prev = useStore(s => s.prev), toStart = useStore(s => s.toStart), toEnd = useStore(s => s.toEnd);
  const recordTrainingTrace = useStore(s => s.recordTrainingTrace);
  const n = trace?.steps.length ?? 0;
  const step = trace && i >= 0 ? trace.steps[i] : null;
  const op = step ? model?.ops.find(o => o.id === step.opId) : null;
  const phase = step?.phase === 'backward' ? 'Backward' : step?.phase === 'update' ? 'Update' : 'Forward';
  const text = !trace ? 'Press Step to run the model' : !step ? `Start: nothing computed yet (${n} steps to go)` : `${phase} step ${i + 1} of ${n}: ${step.opId === 'adam_update' ? 'Adam update, nudge every parameter' : op?.label ?? step.opId}`;
  return (
    <div className="op-bar">
      <span className="op-label">Computation</span>
      <button onClick={toStart} disabled={!trace} title="Back to the start (Home)">|&lt;</button>
      <button onClick={prev} disabled={!trace} title="Previous operation (B)">&lt;</button>
      <button onClick={next} disabled={!trace} title="Next operation (F)">&gt;</button>
      <button onClick={toEnd} disabled={!trace} title="Jump to the end (End)">&gt;|</button>
      <input type="range" min={-1} max={Math.max(0, n - 1)} value={i} disabled={!trace} onChange={e => seek(Number(e.target.value))} />
      <span className="op-status">{text}</span>
      <button onClick={recordTrainingTrace} disabled={!model} title="Record one training step: forward, backward and the parameter update (T)">Record training step</button>
    </div>
  );
}
