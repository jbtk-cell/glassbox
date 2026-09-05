/** SVG loss/accuracy curves plus a gradient-magnitude bar chart. Pure presentation over the
 *  store's training history and the current trace; no chart library. */
import { useState } from 'react';
import { useStore } from '../../app/store';
import { PARAM_NAMES } from '../../engine/model/params';

const TRAIN_COLOR = '#b2182b';
const TEST_COLOR = '#2166ac';

const W = 640, H = 170;
const PAD_L = 46, PAD_R = 14, PAD_T = 14, PAD_B = 26;
const PLOT_W = W - PAD_L - PAD_R, PLOT_H = H - PAD_T - PAD_B;
const Y_TICKS = 5, X_TICKS = 8;

const isNum = (v: number): boolean => Number.isFinite(v);
const fmt2 = (v: number): string => (isNum(v) ? v.toFixed(2) : 'n/a');

/** `n` evenly spaced values across [0, max], including both ends. */
function linTicks(max: number, n: number): number[] {
  if (!(max > 0)) return [0];
  return Array.from({ length: n }, (_, i) => (max * i) / (n - 1));
}

/** Up to `n` of the iteration values present, evenly spaced through the run. */
function xTicks(iterations: number[], n: number): number[] {
  const uniq = [...new Set(iterations)];
  if (uniq.length <= n) return uniq;
  const picked = new Set<number>();
  for (let i = 0; i < n; i++) picked.add(uniq[Math.round((i * (uniq.length - 1)) / (n - 1))]);
  return [...picked];
}

/** SVG path data for a line through `values` at the given iterations; null values break the line. */
function pathFor(iterations: number[], values: (number | null)[], xOf: (it: number) => number, yOf: (v: number) => number | null): string {
  let d = '', started = false;
  for (let i = 0; i < iterations.length; i++) {
    const v = values[i];
    const y = v === null ? null : yOf(v);
    if (y === null) { started = false; continue; }
    d += `${started ? 'L' : 'M'}${xOf(iterations[i]).toFixed(2)},${y.toFixed(2)} `;
    started = true;
  }
  return d.trim();
}

function l2norm(data: Float64Array | undefined): number {
  if (!data) return 0;
  let s = 0; for (let i = 0; i < data.length; i++) s += data[i] * data[i];
  return Math.sqrt(s);
}

/** True when test loss rose and train loss fell on every step of the last 5 history entries. */

function Axes({ yTicks, xTicks: xt, xOf, yOf }: {
  yTicks: { y: number; label: string }[]; xTicks: { x: number; label: string }[];
  xOf: (it: number) => number; yOf: (v: number) => number;
}) {
  return (
    <g fontSize={10} fill="#666">
      <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + PLOT_H} stroke="#ccc" />
      <line x1={PAD_L} y1={PAD_T + PLOT_H} x2={PAD_L + PLOT_W} y2={PAD_T + PLOT_H} stroke="#ccc" />
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={PAD_L - 3} y1={yOf(t.y)} x2={PAD_L + PLOT_W} y2={yOf(t.y)} stroke="#eee" />
          <text x={PAD_L - 6} y={yOf(t.y)} textAnchor="end" dominantBaseline="middle">{t.label}</text>
        </g>
      ))}
      {xt.map((t, i) => (
        <text key={i} x={xOf(t.x)} y={PAD_T + PLOT_H + 14} textAnchor="middle">{t.label}</text>
      ))}
    </g>
  );
}

export function LossView() {
  const history = useStore(s => s.history);
  const trace = useStore(s => s.trace);
  const [logScale, setLogScale] = useState(false);

  if (history.length === 0) {
    return <div className="loss-view"><p>Train the model to see loss curves.</p></div>;
  }

  const iterations = history.map(m => m.iteration);
  const minIter = iterations[0], maxIter = iterations[iterations.length - 1];
  const iterSpan = Math.max(1, maxIter - minIter);
  const xOf = (it: number) => PAD_L + ((it - minIter) / iterSpan) * PLOT_W;
  const latest = history[history.length - 1];

  // ---- Loss chart ----
  const finiteLosses = history.flatMap(m => [m.trainLoss, m.testLoss]).filter(isNum);
  const maxLoss = finiteLosses.length ? Math.max(...finiteLosses) : 1;
  const positiveLosses = finiteLosses.filter(v => v > 0);
  const logMin = positiveLosses.length ? Math.log10(Math.min(...positiveLosses)) : 0;
  const logMax = positiveLosses.length ? Math.log10(Math.max(...positiveLosses)) : 1;
  const logSpan = logMax - logMin || 1;

  const lossY = (v: number): number | null => {
    if (logScale) return v > 0 ? PAD_T + PLOT_H - ((Math.log10(v) - logMin) / logSpan) * PLOT_H : null;
    return PAD_T + PLOT_H - (v / (maxLoss || 1)) * PLOT_H;
  };
  const trainLossVals = history.map(m => (isNum(m.trainLoss) ? m.trainLoss : null));
  const testLossVals = history.map(m => (isNum(m.testLoss) ? m.testLoss : null));

  const lossYTicks = logScale
    ? linTicks(logSpan, Y_TICKS).map(t => ({ y: 10 ** (logMin + t), label: (10 ** (logMin + t)).toPrecision(2) }))
    : linTicks(maxLoss, Y_TICKS).map(t => ({ y: t, label: t.toFixed(2) }));
  const lossXTicks = xTicks(iterations, X_TICKS).map(it => ({ x: it, label: String(it) }));
  const lossYOf = (v: number) => lossY(v) ?? PAD_T + PLOT_H;


  // ---- Accuracy chart ----
  const accY = (v: number): number | null => (isNum(v) ? PAD_T + PLOT_H - (v / 100) * PLOT_H : null);
  const trainAccVals = history.map(m => (isNum(m.trainAcc) ? m.trainAcc * 100 : null));
  const testAccVals = history.map(m => (isNum(m.testAcc) ? m.testAcc * 100 : null));
  const accYTicks = linTicks(100, Y_TICKS).map(t => ({ y: t, label: `${t.toFixed(0)}%` }));
  const accXTicks = lossXTicks;
  const accYOf = (v: number) => accY(v) ?? PAD_T + PLOT_H;

  // ---- Gradient magnitudes ----
  const grads = trace?.kind === 'training'
    ? [...PARAM_NAMES].map(name => ({ name, norm: l2norm(trace.ctx.grads.get(name)?.data) })).sort((a, b) => b.norm - a.norm)
    : null;
  const maxNorm = grads ? Math.max(...grads.map(g => g.norm), 1e-12) : 1;
  const ROW_H = 18, LABEL_W = 70, BAR_MAX_W = W - LABEL_W - 70;
  const gradH = grads ? grads.length * ROW_H + 10 : 0;

  return (
    <div className="math-view loss-view" style={{ overflow: 'auto' }}>
      <section>
        <h3>Loss</h3>
        <label style={{ fontSize: 12 }}>
          <input type="checkbox" checked={logScale} onChange={e => setLogScale(e.target.checked)} /> log scale
        </label>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
          <Axes yTicks={lossYTicks} xTicks={lossXTicks} xOf={xOf} yOf={lossYOf} />
          <path d={pathFor(iterations, trainLossVals, xOf, lossY)} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.5} />
          <path d={pathFor(iterations, testLossVals, xOf, lossY)} fill="none" stroke={TEST_COLOR} strokeWidth={1.5} />
        </svg>
        <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
          <span style={{ color: TRAIN_COLOR }}>train loss {fmt2(latest.trainLoss)}</span>
          <span style={{ color: TEST_COLOR }}>test loss {fmt2(latest.testLoss)}</span>
        </div>
              </section>

      <section>
        <h3>Accuracy</h3>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img">
          <Axes yTicks={accYTicks} xTicks={accXTicks} xOf={xOf} yOf={accYOf} />
          <path d={pathFor(iterations, trainAccVals, xOf, accY)} fill="none" stroke={TRAIN_COLOR} strokeWidth={1.5} />
          <path d={pathFor(iterations, testAccVals, xOf, accY)} fill="none" stroke={TEST_COLOR} strokeWidth={1.5} />
        </svg>
        <div style={{ display: 'flex', gap: 16, fontSize: 12 }}>
          <span style={{ color: TRAIN_COLOR }}>train accuracy {fmt2(latest.trainAcc * 100)}%</span>
          <span style={{ color: TEST_COLOR }}>test accuracy {fmt2(latest.testAcc * 100)}%</span>
        </div>
      </section>

      <section>
        <h3>Gradient sizes</h3>
        {grads === null ? (
          <p>Record a training step (press T) to see gradient sizes.</p>
        ) : (
          <svg viewBox={`0 0 ${W} ${gradH}`} width="100%" role="img" fontSize={10}>
            {grads.map((g, i) => {
              const y = i * ROW_H;
              const w = (g.norm / maxNorm) * BAR_MAX_W;
              return (
                <g key={g.name}>
                  <text x={0} y={y + ROW_H / 2} dominantBaseline="middle">{g.name}</text>
                  <rect x={LABEL_W} y={y + 3} width={Math.max(0, w)} height={ROW_H - 6} fill="#4d4d4d" />
                  <text x={LABEL_W + w + 4} y={y + ROW_H / 2} dominantBaseline="middle">{g.norm.toPrecision(3)}</text>
                </g>
              );
            })}
          </svg>
        )}
      </section>
    </div>
  );
}
