import { describe, it, expect } from 'vitest';
import { splitWords, buildVocab, encode, tokenize, detokenize, UNK } from './tokenize';

describe('splitWords', () => {
  it('lowercases and splits on whitespace', () => {
    expect(splitWords('Hi What Is up')).toEqual(['hi', 'what', 'is', 'up']);
  });
  it('splits trailing and leading punctuation into their own tokens', () => {
    expect(splitWords("What's up? Nothing, really.")).toEqual(["what's", 'up', '?', 'nothing', ',', 'really', '.']);
    expect(splitWords('"quoted" (aside)')).toEqual(['"', 'quoted', '"', '(', 'aside', ')']);
  });
  it('keeps contractions whole', () => { expect(splitWords("don't you're")).toEqual(["don't", "you're"]); });
  it('emits newline as a token and collapses blank lines', () => {
    expect(splitWords('a b\nc\n\n\nd')).toEqual(['a', 'b', '\n', 'c', '\n', 'd']);
  });
  it('ignores leading/trailing whitespace', () => { expect(splitWords('  x  ')).toEqual(['x']); });
});

describe('buildVocab', () => {
  it('orders by frequency then alphabetically, with <unk> first', () => {
    const v = buildVocab(['b', 'a', 'b', 'c', 'a', 'b']);
    expect(v.words).toEqual(['<unk>', 'b', 'a', 'c']);
    expect(v.index.get('b')).toBe(1);
    expect(v.folded).toBe(0);
  });
  it('caps and counts folded word types', () => {
    const v = buildVocab(['a', 'a', 'b', 'c', 'd'], 3);
    expect(v.words).toEqual(['<unk>', 'a', 'b']);
    expect(v.folded).toBe(2);
  });
});

describe('encode / tokenize / detokenize', () => {
  it('maps unknown words to UNK', () => {
    const v = buildVocab(['a', 'b']);
    expect(encode(['a', 'zzz', 'b'], v)).toEqual([1, UNK, 2]);
  });
  it('round-trips simple text', () => {
    const { tokens, vocab } = tokenize("hi, what's up?\nnot much.");
    expect(detokenize(tokens, vocab)).toBe("hi, what's up?\nnot much.");
  });
});
