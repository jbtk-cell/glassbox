/** Renders the maths for the op at the cursor: the general formula, the selected cell's real
 *  numbers, and (in the backward phase) the gradient rule. Pure presentation over the store. */
import 'katex/dist/katex.min.css';
import katex from 'katex';
import { useStore } from '../../app/store';
import { mathFor } from './mathFor';

export function MathView() {
  const trace = useStore(s => s.trace);
  const cursorIndex = useStore(s => s.cursorIndex);
  const selection = useStore(s => s.selection);
  const model = useStore(s => s.model);

  const step = trace?.steps[cursorIndex] ?? null;
  const op = model?.ops.find(o => o.id === step?.opId) ?? null;
  // `op` is only non-null when `trace` is non-null (op is derived from a step of that trace), so
  // this call is safe even though `mathFor` itself does not accept a nullable trace.
  const blocks = mathFor(trace!, step, op, selection);

  return (
    <div className="math-view" style={{ overflow: 'auto' }}>
      {blocks.map((b, i) => (
        <section key={i}>
          <h3>{b.title}</h3>
          <div dangerouslySetInnerHTML={{ __html: katex.renderToString(b.latex, { throwOnError: false, displayMode: true }) }} />
          {b.note && <p>{b.note}</p>}
        </section>
      ))}
    </div>
  );
}
