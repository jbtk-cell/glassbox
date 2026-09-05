import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useStore } from '../app/store';

export function Transport({ playing, setPlaying, speed, setSpeed }: { playing: boolean; setPlaying: (p: boolean) => void; speed: number; setSpeed: (s: number) => void }) {
  const trace = useStore(s => s.trace), i = useStore(s => s.cursorIndex), model = useStore(s => s.model);
  const { seek, next, prev, toStart, toEnd, recordTrainingTrace } = useStore(useShallow(s => ({ seek: s.seek, next: s.next, prev: s.prev, toStart: s.toStart, toEnd: s.toEnd, recordTrainingTrace: s.recordTrainingTrace })));
  const n = trace?.steps.length ?? 0;
  const step = trace && i >= 0 ? trace.steps[i] : null;
  const op = step ? model?.ops.find(o => o.id === step.opId) : null;
  const label = !trace ? 'no computation recorded' : step ? `step ${i + 1} / ${n}  -  ${step.phase}  -  ${step.opId === 'adam_update' ? 'Adam update' : op?.label ?? step.opId}` : `start  -  0 / ${n}`;

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => { const s = useStore.getState(); if (!s.trace || s.cursorIndex >= s.trace.steps.length - 1) setPlaying(false); else s.next(); }, 1000 / speed);
    return () => clearInterval(id);
  }, [playing, speed, setPlaying]);

  return (
    <div className="transport">
      <button onClick={toStart} disabled={!trace} title="Start (Home)">|&lt;</button>
      <button onClick={prev} disabled={!trace} title="Previous step (B)">&lt;</button>
      <button className="primary" onClick={() => setPlaying(!playing)} disabled={!trace} title="Play / pause (Space)">{playing ? 'Pause' : 'Play'}</button>
      <button onClick={next} disabled={!trace} title="Next step (F)">&gt;</button>
      <button onClick={toEnd} disabled={!trace} title="End (End)">&gt;|</button>
      <input type="range" min={-1} max={Math.max(0, n - 1)} value={i} disabled={!trace} onChange={e => seek(Number(e.target.value))} />
      <span className="label">{label}</span>
      <select value={speed} onChange={e => setSpeed(Number(e.target.value))} title="Steps per second">
        <option value={1}>1/s</option><option value={2}>2/s</option><option value={5}>5/s</option><option value={10}>10/s</option>
      </select>
      <button onClick={recordTrainingTrace} disabled={!model} title="Record one training step (T)">Record training step</button>
    </div>
  );
}
