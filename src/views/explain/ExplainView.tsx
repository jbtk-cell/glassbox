/** Explain: a guided walk through the recorded computation. The list on the left is every step;
 *  the pane on the right explains the step at the cursor in plain words, shows its formula with
 *  the real numbers, and says where to look for it in the other tabs. */
import 'katex/dist/katex.min.css';
import katex from 'katex';
import { useStore } from '../../app/store';
import { mathFor } from '../math/mathFor';
import { STEPS, GROUPS, OVERVIEW, BACKWARD_NOTE, GLOSSARY } from './steps';
import type { TraceStep } from '../../engine/trace/trace';

export function ExplainView() {
  const trace = useStore(s => s.trace), cursorIndex = useStore(s => s.cursorIndex), selection = useStore(s => s.selection);
  const model = useStore(s => s.model), seek = useStore(s => s.seek), next = useStore(s => s.next), prev = useStore(s => s.prev);
  const setView = useStore(s => s.setView);
  const paramCount = model ? [...model.params.values()].reduce((a, t) => a + t.data.length, 0) : 0;

  if (!trace || !model) {
    return <div className="explain-view"><div className="explain-pane">
      <h2>What this model does</h2>
      {OVERVIEW.map((p, i) => <p key={i}>{p.replace('15,379', paramCount ? paramCount.toLocaleString() : '15,379')}</p>)}
      <p className="hint">Press Step in the toolbar to run the model on the words in Text Inputs. Then this tab walks through every operation.</p>
    </div></div>;
  }

  const step: TraceStep | null = cursorIndex >= 0 ? trace.steps[cursorIndex] : null;
  const op = step ? model.ops.find(o => o.id === step.opId) ?? null : null;
  const forward = trace.steps.filter(s => s.phase === 'forward');
  const backward = trace.steps.filter(s => s.phase !== 'forward');
  const titleOf = (s: TraceStep) => STEPS[s.opId]?.title ?? model.ops.find(o => o.id === s.opId)?.label ?? s.opId;

  const list = (
    <div className="explain-list">
      <button className={'explain-item' + (cursorIndex < 0 ? ' on' : '')} onClick={() => seek(-1)}>Overview</button>
      {GROUPS.map(gname => {
        const items = forward.filter(s => (STEPS[s.opId]?.group ?? 'Training') === gname);
        if (items.length === 0) return null;
        return <div key={gname}>
          <div className="explain-group">{gname}</div>
          {items.map(s => <button key={s.index} className={'explain-item' + (s.index === cursorIndex ? ' on' : '')} onClick={() => seek(s.index)}>{s.index + 1}. {titleOf(s)}</button>)}
        </div>;
      })}
      {backward.length > 0 && <div>
        <div className="explain-group">Backward pass and update</div>
        {backward.map(s => <button key={s.index} className={'explain-item' + (s.index === cursorIndex ? ' on' : '')} onClick={() => seek(s.index)}>{s.index + 1}. {s.phase === 'update' ? 'Nudge every number' : titleOf(s)}</button>)}
      </div>}
    </div>
  );

  let pane: React.ReactNode;
  if (!step) {
    pane = <>
      <h2>What this model does</h2>
      {OVERVIEW.map((p, i) => <p key={i}>{p.replace('15,379', paramCount.toLocaleString())}</p>)}
      <p className="hint">Press Next to walk through the {forward.length} steps that produced the current prediction{backward.length ? `, then the ${backward.length} steps of the training update` : ''}.</p>
      <h3>Short names on the Neurons tab</h3>
      <table className="glossary"><tbody>{GLOSSARY.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>
    </>;
  } else {
    const info = STEPS[step.opId];
    const blocks = mathFor(trace, step, op, selection);
    pane = <>
      <div className="explain-kicker">{step.phase === 'forward' ? 'Forward' : step.phase === 'backward' ? 'Backward' : 'Update'} step {step.index + 1} of {trace.steps.length}{op ? ` (${op.id})` : ''}</div>
      <h2>{step.phase === 'update' ? STEPS.adam_update.title : info?.title ?? op?.label ?? step.opId}</h2>
      {step.phase === 'backward' && <p className="note">{BACKWARD_NOTE}</p>}
      <p>{step.phase === 'update' ? STEPS.adam_update.text : info?.text ?? op?.explain}</p>
      {(info || step.phase === 'update') && <p className="where"><b>Where to look.</b> {step.phase === 'update' ? STEPS.adam_update.where : info!.where}</p>}
      <div className="explain-links"><button onClick={() => setView('flow')}>Show in Network</button><button onClick={() => setView('network')}>Show in Neurons</button></div>
      {blocks.length > 0 && <h3>In numbers</h3>}
      {blocks.map((b, i) => <section key={i}>
        <div className="explain-block-title">{b.title}</div>
        <div dangerouslySetInnerHTML={{ __html: katex.renderToString(b.latex, { throwOnError: false, displayMode: true }) }} />
        {b.note && <p className="note">{b.note}</p>}
      </section>)}
      {!selection && <p className="hint">Click a cell in the Network tab, or a circle in the Neurons tab, to see this step worked out for that one number.</p>}
    </>;
  }

  return (
    <div className="explain-view">
      {list}
      <div className="explain-pane">
        {pane}
        <div className="explain-nav">
          {cursorIndex >= trace.steps.length - 1 && <button onClick={() => seek(-1)} title="Back to the overview, then Next walks through every step">Start from the beginning</button>}
          <button onClick={prev} disabled={cursorIndex < 0}>Previous</button>
          <button onClick={next} disabled={cursorIndex >= trace.steps.length - 1}>Next</button>
        </div>
      </div>
    </div>
  );
}
