import { useState } from 'react';
import { useStore } from '../app/store';
import { dotProductBreakdown, rowCol, fmtNum } from './inspectorLogic';
import { PARAM_NAMES } from '../engine/model/params';
import { UPDATE_OP } from '../engine/trace/trace';

const PARAMS = new Set<string>(PARAM_NAMES);

function CellInfo({ title, cell }: { title: string; cell: { key: string; index: number } }) {
  const trace = useStore(s => s.trace);
  if (!trace) return null;
  const name = cell.key.slice(2);
  const t = trace.ctx.tensors.get(name); if (!t) return null;
  const { row, col } = rowCol(t.shape, cell.index);
  const g = trace.ctx.grads.get(name)?.data[cell.index];
  const d = trace.deltas?.get(name)?.[cell.index];
  return (
    <div className="cell">
      <h3>{title}</h3>
      <code>{name}[{row}, {col}]</code>
      <dl className="stats">
        <dt>Value</dt><dd>{fmtNum(t.data[cell.index])}</dd>
        {g !== undefined && <><dt>Gradient</dt><dd>{fmtNum(g)}</dd></>}
        {d !== undefined && <><dt>Update</dt><dd>{fmtNum(d)}</dd></>}
      </dl>
    </div>
  );
}

function ParamEditor({ cell }: { cell: { key: string; index: number } }) {
  const model = useStore(s => s.model), editParam = useStore(s => s.editParam);
  const name = cell.key.slice(2);
  const value = model?.params.get(name)?.data[cell.index];
  const [text, setText] = useState<string | null>(null);
  if (value === undefined) return null;
  const apply = (v: number) => { if (Number.isFinite(v)) editParam(name, cell.index, v); setText(null); };
  return (
    <div className="editor">
      <h3>Edit this weight</h3>
      <div className="row">
        <input type="number" step="0.01" value={text ?? String(Number(value.toPrecision(6)))} onChange={e => setText(e.target.value)} onBlur={() => text !== null && apply(Number(text))} onKeyDown={e => { if (e.key === 'Enter') apply(Number((e.target as HTMLInputElement).value)); }} />
      </div>
      <div className="row nudge">
        {[-0.1, -0.01, 0.01, 0.1].map(dv => <button key={dv} onClick={() => apply(value + dv)}>{dv > 0 ? '+' : ''}{dv}</button>)}
      </div>
      <p className="hint">The diagram re-runs with the new value immediately.</p>
    </div>
  );
}

export function Inspector() {
  const trace = useStore(s => s.trace), i = useStore(s => s.cursorIndex), model = useStore(s => s.model);
  const selection = useStore(s => s.selection), hover = useStore(s => s.hover);
  const step = trace && i >= 0 ? trace.steps[i] : null;
  const op = step && step.opId !== UPDATE_OP ? model?.ops.find(o => o.id === step.opId) ?? null : null;
  const selName = selection?.key.slice(2);
  const selOp = selName && model ? model.ops.find(o => o.output === selName) ?? null : null;
  const breakdown = trace && selection && selOp && selOp.kind === 'linear' ? (() => { const t = trace.ctx.tensors.get(selName!)!; const { row, col } = rowCol(t.shape, selection.index); return dotProductBreakdown(trace, selOp, row, col); })() : null;

  return (
    <aside className="inspector">
      <section>
        <h2>{step ? (step.opId === UPDATE_OP ? 'Adam update' : op?.label) : trace ? 'Before the first step' : 'Nothing recorded yet'}</h2>
        {step && <p className="phase">{step.phase === 'forward' ? 'Forward pass' : step.phase === 'backward' ? 'Backward pass: gradients flowing back' : 'Parameter update'}</p>}
        {op && <code className="formula">{op.formula}</code>}
        {op && <p>{op.explain}</p>}
        {step?.opId === UPDATE_OP && <p>Adam moves every parameter a small step against its gradient, scaled by the learning rate and by running averages of past gradients. The orange tiles show the change applied to each weight.</p>}
        {!trace && <p>Type a prompt and press Predict next word, or press T to record one training step, then step through it with F and B.</p>}
        {trace && !step && <p>Press F or the play button to run the first operation.</p>}
      </section>
      {hover && (!selection || hover.key !== selection.key || hover.index !== selection.index) && <CellInfo title="Under the cursor" cell={hover} />}
      {selection && <CellInfo title="Selected" cell={selection} />}
      {breakdown && (
        <section className="breakdown">
          <h3>How this number was made</h3>
          <p className="hint">{breakdown.inputName}[{breakdown.row}, :] times {breakdown.weightName}[:, {breakdown.col}], plus the bias.</p>
          <table>
            <thead><tr><th>i</th><th>input</th><th>weight</th><th>product</th></tr></thead>
            <tbody>
              {breakdown.terms.map(t => <tr key={t.i}><td>{t.i}</td><td>{fmtNum(t.xi)}</td><td>{fmtNum(t.wi)}</td><td>{fmtNum(t.prod)}</td></tr>)}
              <tr><td colSpan={3}>bias</td><td>{fmtNum(breakdown.bias)}</td></tr>
              <tr className="total"><td colSpan={3}>sum</td><td>{fmtNum(breakdown.total)}</td></tr>
            </tbody>
          </table>
        </section>
      )}
      {selection && PARAMS.has(selection.key.slice(2)) && <ParamEditor cell={selection} />}
    </aside>
  );
}
