import { create } from 'zustand';
import { GPT } from '../engine/model/gpt';
import { DEFAULTS, type GPTConfig } from '../engine/model/params';
import { Adam, ADAM_DEFAULTS } from '../engine/model/adam';
import { generate, SAMPLE_DEFAULTS, type SampleOpts } from '../engine/model/generate';
import { recordForward, recordTrainingStep, type Trace } from '../engine/trace/trace';
import { splitWords, encode } from '../engine/corpus/tokenize';
import { Rng } from '../engine/rng';
import { buildCorpus, type Corpus } from './corpusState';
import { PRESETS } from '../presets';
import { createTrainer, type Trainer } from './worker/bridge';
import type { Metrics, FromWorker } from './worker/protocol';

export type View = 'flow' | 'network' | 'math' | 'loss';
export interface CellRef { key: string; index: number }
export const MAX_HISTORY = 5000;

export interface Data {
  text: string; presetId: string | null; corpus: Corpus | null; corpusError: string | null;
  config: GPTConfig; lr: number; sample: SampleOpts;
  model: GPT | null; adam: Adam | null;
  training: 'idle' | 'running'; history: Metrics[]; trainError: string | null; trainerFallback: boolean;
  trace: Trace | null; cursorIndex: number; nonce: number;
  prompt: string; promptUnknown: string[]; generated: number[] | null;
  view: View; selection: CellRef | null; hover: CellRef | null; position: number;
}

export interface Actions {
  setText(text: string, presetId?: string | null): void;
  applyText(): void;
  setConfig(patch: Partial<GPTConfig>): void; setLr(lr: number): void; setSample(patch: Partial<SampleOpts>): void;
  resetModel(): void;
  setPrompt(p: string): void; recordPrompt(): void; recordTrainingTrace(): void;
  seek(i: number): void; next(): void; prev(): void; toStart(): void; toEnd(): void;
  setView(v: View): void; select(c: CellRef | null): void; setHover(c: CellRef | null): void; setPosition(t: number): void;
  editParam(name: string, index: number, value: number): void;
  generateMore(n: number): void;
  startTraining(): void; stopTraining(): void;
  _onProgress(m: Metrics): void; _onParams(p: Record<string, number[]>): void; _onTrainError(msg: string): void;
}
export type State = Data & Actions;

export const initialData: Data = {
  text: PRESETS[0].text, presetId: PRESETS[0].id, corpus: null, corpusError: null,
  config: { ...DEFAULTS, vocabSize: 0 }, lr: ADAM_DEFAULTS.lr, sample: { ...SAMPLE_DEFAULTS },
  model: null, adam: null, training: 'idle', history: [], trainError: null, trainerFallback: false,
  trace: null, cursorIndex: -1, nonce: 0,
  prompt: '', promptUnknown: [], generated: null,
  view: 'flow', selection: null, hover: null, position: 0,
};

let trainer: Trainer | null = null;
let trainerStale = true;   // main-thread params changed since the worker last saw them

function promptTokens(prompt: string, corpus: Corpus): { tokens: number[]; unknown: string[] } {
  const words = splitWords(prompt);
  const unknown = [...new Set(words.filter(w => !corpus.vocab.index.has(w)))];
  return { tokens: encode(words, corpus.vocab), unknown };
}

/** Re-run the current trace's computation with the model's current parameters. Training traces do not move the parameters. */
function rerecord(model: GPT, trace: Trace, lr: number): Trace {
  if (trace.kind === 'forward') return recordForward(model, trace.ctx.tokens);
  const snap = model.exportParams();
  const t = recordTrainingStep(model, new Adam(model.params, { lr }), { input: trace.ctx.tokens, target: trace.ctx.targets! });
  model.importParams(snap);
  return t;
}

export const useStore = create<State>()((set, get) => {
  const clampCursor = (i: number) => { const t = get().trace; return t ? Math.max(-1, Math.min(t.steps.length - 1, i)) : -1; };
  const freshModel = (config: GPTConfig, lr: number) => { const model = new GPT(config); return { model, adam: new Adam(model.params, { lr }) }; };
  const dropTrainer = () => { trainer?.dispose(); trainer = null; trainerStale = true; };
  const handle = (m: FromWorker) => {
    if (m.type === 'progress') get()._onProgress(m.metrics);
    else if (m.type === 'params') get()._onParams(m.params);
    else get()._onTrainError(m.message);
  };

  return {
    ...initialData,

    setText: (text, presetId = null) => set({ text, presetId }),
    applyText: () => {
      const { text, config, lr } = get();
      dropTrainer();
      const r = buildCorpus(text, config.contextSize);
      if ('error' in r) { set({ corpus: null, corpusError: r.error, model: null, adam: null, trace: null, cursorIndex: -1, history: [], generated: null, selection: null, hover: null }); return; }
      const cfg = { ...config, vocabSize: r.corpus.vocab.words.length };
      set({ corpus: r.corpus, corpusError: null, config: cfg, ...freshModel(cfg, lr), trace: null, cursorIndex: -1, history: [], trainError: null, generated: null, promptUnknown: [], selection: null, hover: null, position: 0 });
    },
    setConfig: patch => set(s => ({ config: { ...s.config, ...patch } })),
    setLr: lr => { const { adam } = get(); if (adam) adam.opts.lr = lr; trainer?.send({ type: 'setLr', lr }); set({ lr }); },
    setSample: patch => set(s => ({ sample: { ...s.sample, ...patch } })),
    resetModel: () => {
      const { config, lr, corpus } = get();
      if (!corpus) return;
      dropTrainer();
      const cfg = { ...config, seed: config.seed + 1 };
      set({ config: cfg, ...freshModel(cfg, lr), history: [], trainError: null, trace: null, cursorIndex: -1, generated: null, selection: null, hover: null });
    },

    setPrompt: prompt => set({ prompt }),
    recordPrompt: () => {
      const { model, corpus, prompt, config } = get();
      if (!model || !corpus) return;
      const { tokens, unknown } = promptTokens(prompt, corpus);
      if (tokens.length === 0) { set({ promptUnknown: unknown }); return; }
      const trace = recordForward(model, tokens.slice(-config.contextSize));
      set({ trace, cursorIndex: trace.steps.length - 1, promptUnknown: unknown, selection: null, hover: null, position: trace.ctx.T - 1 });
    },
    recordTrainingTrace: () => {
      const { model, adam, corpus, config, nonce } = get();
      if (!model || !adam || !corpus || corpus.trainWindows.length === 0) return;
      const rng = new Rng(config.seed * 31 + nonce);
      const w = corpus.trainWindows[rng.int(corpus.trainWindows.length)];
      const trace = recordTrainingStep(model, adam, w);
      trainerStale = true;
      set({ trace, cursorIndex: -1, nonce: nonce + 1, selection: null, hover: null, position: trace.ctx.T - 1 });
    },

    seek: i => set({ cursorIndex: clampCursor(i) }),
    next: () => set(s => ({ cursorIndex: clampCursor(s.cursorIndex + 1) })),
    prev: () => set(s => ({ cursorIndex: clampCursor(s.cursorIndex - 1) })),
    toStart: () => set({ cursorIndex: -1 }),
    toEnd: () => set(s => ({ cursorIndex: s.trace ? s.trace.steps.length - 1 : -1 })),

    setView: view => set({ view }),
    select: selection => set({ selection }),
    setHover: hover => set({ hover }),
    setPosition: t => set(s => ({ position: s.trace ? Math.max(0, Math.min(s.trace.ctx.T - 1, t)) : 0 })),

    editParam: (name, index, value) => {
      const { model, trace, lr } = get();
      if (!model) return;
      model.setParam(name, index, value);
      trainerStale = true;
      if (trace) set({ trace: rerecord(model, trace, lr) });
      else set({});
    },
    generateMore: n => {
      const { model, corpus, prompt, sample, config, nonce } = get();
      if (!model || !corpus) return;
      const { tokens, unknown } = promptTokens(prompt, corpus);
      const rng = new Rng(config.seed * 7919 + nonce);
      set({ generated: generate(model, tokens, n, sample, rng), promptUnknown: unknown, nonce: nonce + 1 });
    },

    startTraining: () => {
      const { model, corpus, config, lr, training, history } = get();
      if (!model || !corpus || training === 'running' || corpus.trainWindows.length === 0) return;
      if (!trainer) { trainer = createTrainer({ onMessage: handle }); trainerStale = true; }
      if (trainerStale) {
        trainer.send({ type: 'init', config, params: model.exportParams(), trainWindows: corpus.trainWindows, testWindows: corpus.testWindows, lr, seed: config.seed + 1000, iteration: history.length ? history[history.length - 1].iteration : 0 });
        trainerStale = false;
      }
      trainer.send({ type: 'start' });
      set({ training: 'running', trainError: null, trainerFallback: trainer.fallback });
    },
    stopTraining: () => { trainer?.send({ type: 'stop' }); set({ training: 'idle' }); },

    _onProgress: m => set(s => ({ history: s.history.length >= MAX_HISTORY ? [...s.history.slice(1), m] : [...s.history, m] })),
    _onParams: p => {
      const { model, trace, lr } = get();
      if (!model) return;
      model.importParams(p);
      if (trace) set({ trace: rerecord(model, trace, lr) }); else set({});
    },
    _onTrainError: msg => set({ trainError: msg, training: 'idle' }),
  };
});
