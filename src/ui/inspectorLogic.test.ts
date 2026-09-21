import { describe, it, expect } from 'vitest';
import { dotProductBreakdown, rowCol, fmtNum, cellName } from './inspectorLogic';
import { GPT, T_ } from '../engine/model/gpt';
import { recordForward } from '../engine/trace/trace';
import { get } from '../engine/ops/types';

describe('dotProductBreakdown', () => {
  const m = new GPT({ vocabSize: 9, contextSize: 5, dModel: 4, dFF: 6, seed: 5 });
  const tr = recordForward(m, [1, 2, 3]);
  it('sums to the output cell for a linear op', () => {
    const op = m.ops.find(o => o.id === 'ff_up')!;
    const b = dotProductBreakdown(tr, op, 2, 5)!;
    expect(b.terms.length).toBe(4);
    expect(b.total).toBeCloseTo(get(tr.ctx, T_.ff_pre).data[2 * 6 + 5], 12);
    expect(b.inputName).toBe(T_.h2); expect(b.weightName).toBe('W_1');
  });
  it('returns null for non-linear ops and out-of-range cells', () => {
    expect(dotProductBreakdown(tr, m.ops.find(o => o.id === 'relu')!, 0, 0)).toBeNull();
    expect(dotProductBreakdown(tr, m.ops.find(o => o.id === 'ff_up')!, 9, 0)).toBeNull();
  });
});

it('rowCol and fmtNum', () => {
  expect(rowCol([3, 4], 7)).toEqual({ row: 1, col: 3, cols: 4 });
  expect(rowCol([5], 2)).toEqual({ row: 0, col: 2, cols: 5 });
  expect(fmtNum(-1e9)).toBe('-inf'); expect(fmtNum(0)).toBe('0'); expect(fmtNum(0.123456)).toBe('0.1235'); expect(fmtNum(12345)).toBe('1.235e+4');
});

describe('cellName', () => {
  const words = ['<unk>', 'the', 'cat'];
  it('describes an E cell', () => {
    expect(cellName('E', 2, 3, words)).toBe('Row of word "cat", column 4');
  });
  it('describes a probs cell', () => {
    expect(cellName('probs', 0, 1, words)).toBe('Probability of word "the" at position 1');
  });
  it('returns null for an unknown name', () => {
    expect(cellName('bogus', 0, 0, words)).toBeNull();
  });
});
