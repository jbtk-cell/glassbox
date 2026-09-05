import type { GPTConfig } from '../../engine/model/params';
import type { Window } from '../../engine/corpus/windows';

export interface Metrics { iteration: number; trainLoss: number; testLoss: number; trainAcc: number; testAcc: number }

export type ToWorker =
  | { type: 'init'; config: GPTConfig; params: Record<string, number[]>; trainWindows: Window[]; testWindows: Window[]; lr: number; seed: number; iteration?: number }
  | { type: 'start' }
  | { type: 'stop' }
  | { type: 'setLr'; lr: number };

export type FromWorker =
  | { type: 'progress'; metrics: Metrics }
  | { type: 'params'; params: Record<string, number[]>; iteration: number }
  | { type: 'error'; message: string };

export const PARAMS_EVERY = 5;   // epochs between parameter syncs
