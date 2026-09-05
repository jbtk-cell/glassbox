import { describe, it, expect } from 'vitest';
import { mathFor } from './mathFor';
import { GPT } from '../../engine/model/gpt';
import { Adam } from '../../engine/model/adam';
import { recordForward, recordTrainingStep, tKey } from '../../engine/trace/trace';
import { get } from '../../engine/ops/types';

const cfg = { vocabSize: 9, contextSize: 5, dModel: 4, dFF: 6, seed: 1 };

describe('mathFor', () => {
  it('shows the formula plus substituted numbers for a selected output cell', () => {
    const model = new GPT(cfg);
    const trace = recordForward(model, [1, 2, 3]);
    const qi = trace.steps.findIndex(s => s.opId === 'q_proj');
    const op = model.ops.find(o => o.id === 'q_proj')!;
    const selection = { key: tKey('q'), index: 5 };
    const blocks = mathFor(trace, trace.steps[qi], op, selection);
    expect(blocks).toHaveLength(2);
    const want = get(trace.ctx, 'q').data[5].toPrecision(4);
    expect(blocks[1].latex).toContain(want);
  });

  it('adds a gradient block at a backward step, when the loss cell is selected', () => {
    const model = new GPT(cfg);
    const adam = new Adam(model.params, { lr: 3e-3 });
    const trace = recordTrainingStep(model, adam, { input: [1, 2, 3], target: [2, 3, 1] });
    const li = trace.steps.findIndex(s => s.opId === 'loss' && s.phase === 'backward');
    const op = model.ops.find(o => o.id === 'loss')!;
    const selection = { key: tKey('loss'), index: 0 };
    const blocks = mathFor(trace, trace.steps[li], op, selection);
    expect(blocks).toHaveLength(3);
  });

  it('shows only a hint when nothing is selected', () => {
    const model = new GPT(cfg);
    const trace = recordForward(model, [1, 2, 3]);
    const blocks = mathFor(trace, null, null, null);
    expect(blocks).toHaveLength(1);
    expect(JSON.stringify(blocks)).toContain('Select a cell');
  });
});
