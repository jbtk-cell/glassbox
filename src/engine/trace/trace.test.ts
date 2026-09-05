import { describe, it, expect } from 'vitest';
import { recordForward, recordTrainingStep, tKey, gKey, UPDATE_OP } from './trace';
import { Cursor } from './cursor';
import { GPT, T_ } from '../model/gpt';
import { Adam } from '../model/adam';
import { PARAM_NAMES } from '../model/params';
import { get } from '../ops/types';

const cfg = { vocabSize: 7, contextSize: 5, dModel: 4, dFF: 6, seed: 3 };

describe('recordForward', () => {
  it('records one forward step per op, in order, writing that op\'s output', () => {
    const m = new GPT(cfg);
    const tr = recordForward(m, [1, 2, 3]);
    expect(tr.kind).toBe('forward');
    expect(tr.steps.length).toBe(21);
    tr.steps.forEach((s, i) => {
      expect(s.index).toBe(i); expect(s.phase).toBe('forward');
      expect(s.opId).toBe(m.ops[i].id); expect(s.writes).toEqual([tKey(m.ops[i].output)]);
    });
    expect(get(tr.ctx, T_.probs).shape).toEqual([3, 7]);
    expect(tr.preexisting.has(tKey('E'))).toBe(true);
  });
  it('is byte-identical for the same seed and tokens', () => {
    const a = recordForward(new GPT(cfg), [1, 2, 3]), b = recordForward(new GPT(cfg), [1, 2, 3]);
    expect(Array.from(get(a.ctx, T_.probs).data)).toEqual(Array.from(get(b.ctx, T_.probs).data));
  });
});

describe('recordTrainingStep', () => {
  it('records forward, backward and update phases', () => {
    const m = new GPT(cfg);
    const tr = recordTrainingStep(m, new Adam(m.params), { input: [1, 2, 3], target: [2, 3, 4] });
    expect(tr.kind).toBe('training');
    expect(tr.steps.length).toBe(45);
    expect(tr.steps[21]).toMatchObject({ opId: 'loss', phase: 'forward', writes: [tKey(T_.loss)] });
    expect(tr.steps[22]).toMatchObject({ opId: 'loss', phase: 'backward', writes: [gKey(T_.loss), gKey(T_.probs)] });
    expect(tr.steps[43]).toMatchObject({ opId: 'embed', phase: 'backward', writes: [gKey('E')] });
    expect(tr.steps[44]).toMatchObject({ opId: UPDATE_OP, phase: 'update' });
    expect(tr.steps[44].writes).toEqual(PARAM_NAMES.map(tKey));
    expect(tr.deltas!.get('W_q')!.length).toBe(16);
    expect(tr.ctx.grads.has('W_q')).toBe(true);
  });
});

describe('Cursor', () => {
  it('starts before the first step with activations pending and parameters done', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1, 2]));
    expect(c.index).toBe(-1); expect(c.atStart()).toBe(true); expect(c.current()).toBeNull();
    expect(c.status(tKey(T_.tok))).toBe('pending');
    expect(c.status(tKey('E'))).toBe('done');
  });
  it('marks the current write active, earlier writes done, later writes pending', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1, 2]));
    c.seek(2);
    expect(c.status(tKey(T_.tok))).toBe('done');
    expect(c.status(tKey(T_.x0))).toBe('active');
    expect(c.status(tKey(T_.h1))).toBe('pending');
  });
  it('reports partial for an accumulated gradient with one of two contributions', () => {
    const m = new GPT(cfg);
    const tr = recordTrainingStep(m, new Adam(m.params), { input: [1, 2, 3], target: [2, 3, 4] });
    const c = new Cursor(tr);
    const residual1 = tr.steps.findIndex(s => s.opId === 'residual1' && s.phase === 'backward');
    const ln1 = tr.steps.findIndex(s => s.opId === 'ln1' && s.phase === 'backward');
    expect(residual1).toBeLessThan(ln1);
    c.seek(residual1); expect(c.status(gKey(T_.x0))).toBe('active');
    c.seek(residual1 + 1); expect(c.status(gKey(T_.x0))).toBe('partial');
    c.seek(ln1); expect(c.status(gKey(T_.x0))).toBe('active');
    c.seek(ln1 + 1); expect(c.status(gKey(T_.x0))).toBe('done');
    c.toEnd(); expect(c.status(tKey('W_q'))).toBe('active'); expect(c.atEnd()).toBe(true);
  });
  it('clamps next and prev at the ends', () => {
    const c = new Cursor(recordForward(new GPT(cfg), [1]));
    c.prev(); expect(c.index).toBe(-1);
    c.toEnd(); expect(c.index).toBe(20); c.next(); expect(c.index).toBe(20);
    c.toStart(); expect(c.index).toBe(-1);
  });
});
