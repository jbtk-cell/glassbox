import { useState } from 'react';
import { useStore } from '../app/store';
import { rowCol, fmtNum, cellName } from './inspectorLogic';
import { PARAM_NAMES } from '../engine/model/params';

const PARAMS = new Set<string>(PARAM_NAMES);

/** Small floating panel for the selected cell: name, index, value, gradient, and an edit field for weights. */
export function CellPanel() {
  const trace = useStore(s => s.trace), selection = useStore(s => s.selection), select = useStore(s => s.select);
  const model = useStore(s => s.model), editParam = useStore(s => s.editParam);
  const words = useStore(s => s.corpus?.vocab.words);
  const [text, setText] = useState<string | null>(null);
  if (!trace || !selection) return null;
  const name = selection.key.slice(2);
  const t = trace.ctx.tensors.get(name); if (!t) return null;
  const { row, col } = rowCol(t.shape, selection.index);
  const desc = cellName(name, row, col, words);
  const g = trace.ctx.grads.get(name)?.data[selection.index];
  const d = trace.deltas?.get(name)?.[selection.index];
  const isParam = PARAMS.has(name);
  const current = model?.params.get(name)?.data[selection.index] ?? t.data[selection.index];
  const apply = () => { if (text !== null && Number.isFinite(Number(text))) editParam(name, selection.index, Number(text)); setText(null); };
  return (
    <div className="cell-panel">
      <div className="titlebar">{name}<button className="close" onClick={() => { select(null); setText(null); }}>×</button></div>
      <table>
        <tbody>
          <tr><td>Shape</td><td>{t.shape.join(' x ')}</td></tr>
          <tr><td>Index</td><td>[{row}, {col}]</td></tr>
          {desc !== null && <tr><td>Means</td><td className="means">{desc}</td></tr>}
          <tr><td>Value</td><td>{fmtNum(t.data[selection.index])}</td></tr>
          {g !== undefined && <tr><td>Gradient</td><td>{fmtNum(g)}</td></tr>}
          {d !== undefined && <tr><td>Update</td><td>{fmtNum(d)}</td></tr>}
        </tbody>
      </table>
      {isParam && (
        <div className="edit">
          <input type="number" step="0.01" value={text ?? String(Number(current.toPrecision(6)))} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') apply(); }} />
          <button onClick={apply}>Set</button>
        </div>
      )}
    </div>
  );
}
