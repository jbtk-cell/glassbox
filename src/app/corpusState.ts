import { splitWords, buildVocab, encode, Vocab } from '../engine/corpus/tokenize';
import { splitTokens, windows, minTokens, Window } from '../engine/corpus/windows';

export interface Corpus {
  text: string;
  tokens: number[];
  vocab: Vocab;
  words: string[];
  train: number[];
  test: number[];
  trainWindows: Window[];
  testWindows: Window[];
  notices: string[];
  truncated: boolean;
}

export const MAX_TOKENS = 50_000;
export const VOCAB_CAP = 512;
export const TRAIN_RATIO = 0.6;

/** Tokenise text and prepare train/test windows, or explain why it cannot be trained on. */
export function buildCorpus(text: string, contextSize: number, cap = VOCAB_CAP): { corpus: Corpus } | { error: string } {
  let words = splitWords(text);
  const notices: string[] = [];
  let truncated = false;
  if (words.length > MAX_TOKENS) {
    notices.push(`Text truncated to ${MAX_TOKENS.toLocaleString()} words (it had ${words.length.toLocaleString()}).`);
    words = words.slice(0, MAX_TOKENS);
    truncated = true;
  }
  const min = minTokens(contextSize);
  if (words.length < min) return { error: `Need at least ${min} words to train; this text has ${words.length}.` };

  const vocab = buildVocab(words, cap);
  if (vocab.folded > 0) notices.push(`Vocabulary capped at ${cap} word types; ${vocab.folded} rarer words became <unk>.`);
  const tokens = encode(words, vocab);
  const { train, test } = splitTokens(tokens, TRAIN_RATIO);
  const trainWindows = windows(train, contextSize);
  const testWindows = windows(test, contextSize);
  if (testWindows.length === 0) notices.push('Too little text for a test set; test metrics will be blank until you add more.');

  return { corpus: { text, tokens, vocab, words, train, test, trainWindows, testWindows, notices, truncated } };
}
