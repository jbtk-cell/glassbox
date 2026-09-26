/** How a vocabulary token reads on screen. The tokenizer keeps line breaks and punctuation as
 *  tokens, so a few of them need a visible form. */
import { splitWords, encode, type Vocab } from '../engine/corpus/tokenize';
import type { Trace } from '../engine/trace/trace';

export const NEWLINE_GLYPH = '↵';           // a line break
export const UNKNOWN_GLYPH = '?';

export function displayWord(w: string | undefined): string {
  if (w === undefined) return UNKNOWN_GLYPH;
  if (w === '\n') return NEWLINE_GLYPH;
  if (w === '<unk>') return UNKNOWN_GLYPH;
  return w;
}

/** The words behind a trace's tokens. For a forward trace recorded from the current prompt, the
 *  user's own spelling of unknown words is recovered from the prompt; otherwise the vocabulary
 *  word is used and unknown tokens read as "?". `known[i]` is false for an unknown token. */
export function contextWords(trace: Trace, prompt: string, vocab: Vocab, stored?: string[] | null): { words: string[]; known: boolean[] } {
  const T = trace.ctx.T;
  const known = trace.ctx.tokens.map(t => t !== 0);
  const fromVocab = trace.ctx.tokens.map(t => displayWord(vocab.words[t]));
  if (trace.kind !== 'forward') return { words: fromVocab, known };
  // The store remembers the exact words a Step or recordPrompt ran on; that beats guessing from the prompt.
  if (stored && stored.length === T) return { words: stored.map(displayWord), known };
  const pw = splitWords(prompt);
  // Step appends the guessed word to the prompt, so the context may end one word before the end.
  for (const drop of [0, 1]) {
    const cand = pw.slice(0, pw.length - drop).slice(-T);
    if (cand.length === T && encode(cand, vocab).every((id, i) => id === trace.ctx.tokens[i])) return { words: cand.map(displayWord), known };
  }
  return { words: fromVocab, known };
}
