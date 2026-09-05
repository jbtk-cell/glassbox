import { useStore } from '../app/store';

/** Steps the recorded computation one operation at a time. */
export function Transport() {
  const trace = useStore(s => s.trace), i = useStore(s => s.cursorIndex), model = useStore(s => s.model);
  const seek = useStore(s => s.seek), next = useStore(s => s.next), prev = useStore(s => s.prev), toStart = useStore(s => s.toStart), toEnd = useStore(s => s.toEnd);
  const recordTrainingTrace = useStore(s => s.recordTrainingTrace);
  const n = trace?.steps.length ?? 0;
  const step = trace && i >= 0 ? trace.steps[i] : null;
  const op = step ? model?.ops.find(o => o.id === step.opId) : null;
  const what = !trace ? '' : !step ? 'start' : step.opId === 'adam_update' ? 'Adam update' : `${op?.label ?? step.opId}   ${op?.formula ?? ''}`;
  return (
    <div className="op-bar">
      <span className="op-label">Operation</span>
      <button onClick={toStart} disabled={!trace} title="Home">|&lt;</button>
      <button onClick={prev} disabled={!trace} title="B">&lt;</button>
      <button onClick={next} disabled={!trace} title="F">&gt;</button>
      <button onClick={toEnd} disabled={!trace} title="End">&gt;|</button>
      <input type="range" min={-1} max={Math.max(0, n - 1)} value={i} disabled={!trace} onChange={e => seek(Number(e.target.value))} />
      <span className="op-status">{trace ? `${Math.max(0, i + 1)} / ${n}` : ''}  {step?.phase === 'backward' ? 'backward  ' : step?.phase === 'update' ? 'update  ' : ''}{what}</span>
      <button onClick={recordTrainingTrace} disabled={!model} title="T">Training step</button>
    </div>
  );
}
