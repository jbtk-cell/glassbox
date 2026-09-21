import { describe, it, expect } from 'vitest';
import { displayWord, contextWords } from './words';
import { buildVocab, splitWords, encode } from '../engine/corpus/tokenize';
import type { Trace } from '../engine/trace/trace';

const vocab = buildVocab(splitWords('are you coming to the party\nyes'));
const fakeTrace = (tokens: number[], kind: 'forward' | 'training' = 'forward'): Trace =>
  ({ kind, steps: [], ctx: { tokens, T: tokens.length, tensors: new Map(), grads: new Map() }, preexisting: new Set() } as unknown as Trace);

describe('displayWord', () => {
  it('gives line breaks and unknown tokens a visible form', () => {
    expect(displayWord('\n')).toBe('↵');
    expect(displayWord('<unk>')).toBe('?');
    expect(displayWord(undefined)).toBe('?');
    expect(displayWord('party')).toBe('party');
  });
});

describe('contextWords', () => {
  it("recovers the user's spelling of unknown words from the prompt", () => {
    const prompt = 'are you coming to the zorb';
    const tokens = encode(splitWords(prompt), vocab);
    const { words, known } = contextWords(fakeTrace(tokens), prompt, vocab);
    expect(words).toEqual(['are', 'you', 'coming', 'to', 'the', 'zorb']);
    expect(known[5]).toBe(false);
  });
  it('ignores the word Step appended after the context', () => {
    const ctx = 'are you coming';
    const tokens = encode(splitWords(ctx), vocab);
    expect(contextWords(fakeTrace(tokens), ctx + ' party', vocab).words).toEqual(['are', 'you', 'coming']);
  });
  it('falls back to vocabulary words for training traces', () => {
    const tokens = encode(['party', '\n', 'yes'], vocab);
    expect(contextWords(fakeTrace(tokens, 'training'), 'unrelated text', vocab).words).toEqual(['party', '↵', 'yes']);
  });
});
