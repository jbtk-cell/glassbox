/// <reference lib="webworker" />
import { GPT } from '../../engine/model/gpt';
import { Adam } from '../../engine/model/adam';
import { Rng } from '../../engine/rng';
import { EpochRunner } from './epoch';
import { PARAMS_EVERY, type FromWorker, type ToWorker } from './protocol';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: FromWorker) => ctx.postMessage(m);

let runner: EpochRunner | null = null;
let model: GPT | null = null;
let running = false;
let lastGood: Record<string, number[]> | null = null;
const CHUNK = 64;

function tick() {
  if (!running || !runner || !model) return;
  const out = runner.step(CHUNK);
  if (out && 'diverged' in out) {
    running = false;
    if (lastGood) model.importParams(lastGood);
    post({ type: 'error', message: `Training diverged at iteration ${runner.iteration + 1}; try a lower learning rate.` });
    post({ type: 'params', params: model.exportParams(), iteration: runner.iteration });
    return;
  }
  if (out && 'metrics' in out) {
    post({ type: 'progress', metrics: out.metrics });
    if (out.metrics.iteration % PARAMS_EVERY === 0) { lastGood = model.exportParams(); post({ type: 'params', params: lastGood, iteration: out.metrics.iteration }); }
  }
  setTimeout(tick, 0);
}

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init': {
      model = new GPT(msg.config);
      model.importParams(msg.params);
      lastGood = msg.params;
      const adam = new Adam(model.params, { lr: msg.lr });
      runner = new EpochRunner(model, adam, msg.trainWindows, msg.testWindows, new Rng(msg.seed));
      runner.iteration = msg.iteration ?? 0;
      break;
    }
    case 'start': if (!running && runner) { running = true; setTimeout(tick, 0); } break;
    case 'stop': {
      running = false;
      if (model && runner) { lastGood = model.exportParams(); post({ type: 'params', params: lastGood, iteration: runner.iteration }); }
      break;
    }
    case 'setLr': if (runner) runner.adam.opts.lr = msg.lr; break;
  }
};
