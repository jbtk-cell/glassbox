import { GPT } from '../../engine/model/gpt';
import { Adam } from '../../engine/model/adam';
import { Rng } from '../../engine/rng';
import { EpochRunner } from './epoch';
import { PARAMS_EVERY, type FromWorker, type ToWorker } from './protocol';

export interface TrainerHandlers { onMessage(m: FromWorker): void }
export interface Trainer { send(m: ToWorker): void; dispose(): void; readonly fallback: boolean }

/** Wraps the training worker; falls back to a main-thread loop when Workers are unavailable. */
export function createTrainer(h: TrainerHandlers): Trainer {
  if (typeof Worker !== 'undefined') {
    const w = new Worker(new URL('./trainer.worker.ts', import.meta.url), { type: 'module' });
    w.onmessage = (e: MessageEvent<FromWorker>) => h.onMessage(e.data);
    w.onerror = (e) => h.onMessage({ type: 'error', message: `Training worker failed: ${e.message}` });
    return { send: m => w.postMessage(m), dispose: () => w.terminate(), fallback: false };
  }
  // Main-thread fallback: same protocol, driven by setTimeout.
  let runner: EpochRunner | null = null, model: GPT | null = null, running = false, lastGood: Record<string, number[]> | null = null, timer: ReturnType<typeof setTimeout> | null = null;
  const tick = () => {
    if (!running || !runner || !model) return;
    const out = runner.step(32);
    if (out && 'diverged' in out) {
      running = false; if (lastGood) model.importParams(lastGood);
      h.onMessage({ type: 'error', message: `Training diverged at iteration ${runner.iteration + 1}; try a lower learning rate.` });
      h.onMessage({ type: 'params', params: model.exportParams(), iteration: runner.iteration }); return;
    }
    if (out && 'metrics' in out) {
      h.onMessage({ type: 'progress', metrics: out.metrics });
      if (out.metrics.iteration % PARAMS_EVERY === 0) { lastGood = model.exportParams(); h.onMessage({ type: 'params', params: lastGood, iteration: out.metrics.iteration }); }
    }
    timer = setTimeout(tick, 0);
  };
  return {
    fallback: true,
    dispose: () => { running = false; if (timer) clearTimeout(timer); },
    send: (m) => {
      switch (m.type) {
        case 'init': model = new GPT(m.config); model.importParams(m.params); lastGood = m.params;
          runner = new EpochRunner(model, new Adam(model.params, { lr: m.lr }), m.trainWindows, m.testWindows, new Rng(m.seed)); runner.iteration = m.iteration ?? 0; break;
        case 'start': if (!running && runner) { running = true; timer = setTimeout(tick, 0); } break;
        case 'stop': running = false; if (model && runner) { lastGood = model.exportParams(); h.onMessage({ type: 'params', params: lastGood, iteration: runner.iteration }); } break;
        case 'setLr': if (runner) runner.adam.opts.lr = m.lr; break;
      }
    },
  };
}
