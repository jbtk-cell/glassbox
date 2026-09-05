import { describe, it, expect, beforeEach } from 'vitest';
import { useStore, initialData } from './store';
import { get } from '../engine/ops/types';
import { T_ } from '../engine/model/gpt';

const text = Array.from({ length: 200 }, (_, i) => `w${i % 12}`).join(' ');
const S = () => useStore.getState();

beforeEach(() => { useStore.setState({ ...initialData }); S().setText(text, null); S().applyText(); });

describe('store: corpus and model', () => {
  it('builds a corpus and a model sized to the vocabulary', () => {
    expect(S().corpusError).toBeNull();
    expect(S().corpus!.vocab.words.length).toBe(13);
    expect(S().config.vocabSize).toBe(13);
    expect(S().model!.paramCount()).toBeGreaterThan(1000);
  });
  it('reports an error for text that is too short and drops the model', () => {
    S().setText('too short'); S().applyText();
    expect(S().corpusError).toMatch(/Need at least/); expect(S().model).toBeNull();
  });
  it('resetModel bumps the seed and clears history', () => {
    S()._onProgress({ iteration: 1, trainLoss: 1, testLoss: 1, trainAcc: 0, testAcc: 0 });
    const seed = S().config.seed; const before = S().model;
    S().resetModel();
    expect(S().config.seed).toBe(seed + 1); expect(S().history).toEqual([]); expect(S().model).not.toBe(before);
  });
});

describe('store: traces and cursor', () => {
  it('recordPrompt records a forward trace at its end and lists unknown words', () => {
    S().setPrompt('w1 w2 zzz'); S().recordPrompt();
    expect(S().trace!.kind).toBe('forward'); expect(S().trace!.steps.length).toBe(21);
    expect(S().cursorIndex).toBe(20); expect(S().promptUnknown).toEqual(['zzz']); expect(S().position).toBe(2);
  });
  it('ignores an empty prompt', () => { S().setPrompt('   '); S().recordPrompt(); expect(S().trace).toBeNull(); });
  it('recordTrainingTrace records 45 steps starting before the first', () => {
    S().recordTrainingTrace();
    expect(S().trace!.kind).toBe('training'); expect(S().trace!.steps.length).toBe(45); expect(S().cursorIndex).toBe(-1);
  });
  it('next, prev and seek clamp to the trace', () => {
    S().setPrompt('w1 w2'); S().recordPrompt();
    S().next(); expect(S().cursorIndex).toBe(20);
    S().seek(500); expect(S().cursorIndex).toBe(20);
    S().seek(-9); expect(S().cursorIndex).toBe(-1); S().prev(); expect(S().cursorIndex).toBe(-1);
    S().seek(3); S().next(); expect(S().cursorIndex).toBe(4);
  });
  it('editParam changes the model and re-records at the same cursor index', () => {
    S().setPrompt('w1 w2'); S().recordPrompt(); S().seek(5);
    const before = Array.from(get(S().trace!.ctx, T_.probs).data);
    S().editParam('U', 0, 5);
    expect(S().model!.params.get('U')!.data[0]).toBe(5);
    expect(S().cursorIndex).toBe(5);
    expect(Array.from(get(S().trace!.ctx, T_.probs).data)).not.toEqual(before);
  });
  it('editParam on a training trace keeps the edited value (no extra optimizer step)', () => {
    S().recordTrainingTrace(); S().editParam('W_q', 2, 0.25);
    expect(S().model!.params.get('W_q')!.data[2]).toBe(0.25); expect(S().trace!.steps.length).toBe(45);
  });
});

describe('store: worker messages and generation', () => {
  it('_onProgress appends and _onParams imports into the model', () => {
    S()._onProgress({ iteration: 1, trainLoss: 2, testLoss: 2.5, trainAcc: 0.1, testAcc: 0.05 });
    expect(S().history.length).toBe(1);
    const p = S().model!.exportParams(); p['b_q'][0] = 42;
    S()._onParams(p);
    expect(S().model!.params.get('b_q')!.data[0]).toBe(42);
  });
  it('stepGenerate appends one word to the context and records that pass', () => {
    S().setPrompt('w1 w2'); S().stepGenerate();
    expect(S().prompt.split(/\s+/).length).toBe(3);
    expect(S().trace!.kind).toBe('forward'); expect(S().cursorIndex).toBe(20);
    S().stepGenerate(); expect(S().prompt.split(/\s+/).length).toBe(4);
  });
  it('generateMore returns the prompt followed by n tokens', () => {
    S().setPrompt('w1 w2'); S().generateMore(5);
    expect(S().generated!.length).toBe(7); expect(S().generated!.slice(0, 2)).toEqual([1 + 0, 1 + 1].map(() => expect.any(Number)));
  });
});
