import { describe, it, expect } from 'vitest';
import { buildCorpus, MAX_TOKENS } from './corpusState';

const wordsN = (n: number, distinct: number) => Array.from({ length: n }, (_, i) => `w${i % distinct}`).join(' ');

describe('buildCorpus', () => {
  it('refuses text that is too short, naming both numbers', () => {
    const r = buildCorpus('one two three', 24);
    expect('error' in r && r.error).toBe('Need at least 50 words to train; this text has 3.');
  });
  it('splits 60/40 contiguously and builds windows', () => {
    const r = buildCorpus(wordsN(100, 10), 24);
    if ('error' in r) throw new Error(r.error);
    const c = r.corpus;
    expect(c.tokens.length).toBe(100);
    expect(c.train.length).toBe(60);
    expect(c.test.length).toBe(40);
    expect(c.trainWindows.length).toBe(60 - 24);
    expect(c.testWindows.length).toBe(40 - 24);
    expect(c.notices).toEqual([]);
    expect(c.truncated).toBe(false);
    expect(c.vocab.words.length).toBe(11);   // <unk> + 10
  });
  it('warns when there is no room for a test set', () => {
    const r = buildCorpus(wordsN(52, 5), 24);
    if ('error' in r) throw new Error(r.error);
    expect(r.corpus.testWindows.length).toBe(0);
    expect(r.corpus.notices.join(' ')).toContain('test set');
  });
  it('caps the vocabulary and says how many words were folded', () => {
    const r = buildCorpus(wordsN(1200, 600), 24);
    if ('error' in r) throw new Error(r.error);
    expect(r.corpus.vocab.words.length).toBe(512);
    expect(r.corpus.vocab.folded).toBe(600 - 511);
    expect(r.corpus.notices.join(' ')).toContain('512');
    expect(r.corpus.notices.join(' ')).toContain(String(600 - 511));
  });
  it('truncates very long text with a notice', () => {
    const r = buildCorpus(wordsN(MAX_TOKENS + 10, 20), 24);
    if ('error' in r) throw new Error(r.error);
    expect(r.corpus.truncated).toBe(true);
    expect(r.corpus.tokens.length).toBe(MAX_TOKENS);
    expect(r.corpus.notices[0]).toContain('truncated');
  });
});
