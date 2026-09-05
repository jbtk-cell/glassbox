import { GPT } from '../../engine/model/gpt';
import { Adam } from '../../engine/model/adam';
import { trainStep, evaluate } from '../../engine/model/train';
import { Window, shuffled } from '../../engine/corpus/windows';
import { Rng } from '../../engine/rng';
import type { Metrics } from './protocol';

/**
 * Runs epochs in resumable chunks so a worker can check for messages between chunks.
 * One iteration = one epoch = one pass over the training windows in seeded-shuffled order.
 */
export class EpochRunner {
  iteration = 0;
  private order: Window[] = [];
  private pos = 0;
  private lossSum = 0;
  constructor(readonly model: GPT, readonly adam: Adam, readonly trainWindows: Window[], readonly testWindows: Window[], readonly rng: Rng) {
    if (trainWindows.length === 0) throw new Error('EpochRunner: no training windows');
    this.startEpoch();
  }
  private startEpoch() { this.order = shuffled(this.trainWindows, this.rng); this.pos = 0; this.lossSum = 0; }

  /** Trains on up to `maxWindows` windows. Returns metrics when an epoch completes, `{ diverged: true }` on a non-finite loss, else null. */
  step(maxWindows: number): { metrics: Metrics } | { diverged: true } | null {
    const end = Math.min(this.order.length, this.pos + maxWindows);
    for (; this.pos < end; this.pos++) {
      const { loss } = trainStep(this.model, this.adam, this.order[this.pos]);
      if (!Number.isFinite(loss)) return { diverged: true };
      this.lossSum += loss;
    }
    if (this.pos < this.order.length) return null;
    this.iteration++;
    const trainLoss = this.lossSum / this.order.length;
    const tr = evaluate(this.model, this.trainWindows);
    const te = this.testWindows.length ? evaluate(this.model, this.testWindows) : { loss: NaN, accuracy: NaN };
    this.startEpoch();
    return { metrics: { iteration: this.iteration, trainLoss, testLoss: te.loss, trainAcc: tr.accuracy, testAcc: te.accuracy } };
  }
}
