export interface Vocab { words: string[]; index: Map<string, number>; folded: number }
export const UNK = 0;

const PUNCT = new Set(['.', ',', '?', '!', ';', ':', '"', "'", '(', ')']);

/** Lowercase words, punctuation as its own tokens, '\n' as a token. Contractions stay whole. */
export function splitWords(text: string): string[] {
  const out: string[] = [];
  const lines = text.toLowerCase().split('\n');
  let pendingNewline = false;
  for (const line of lines) {
    const raw = line.trim().split(/\s+/).filter(Boolean);
    if (raw.length === 0) { continue; }
    if (pendingNewline) out.push('\n');
    for (const w of raw) pushWord(w, out);
    pendingNewline = true;
  }
  return out;
}

function pushWord(w: string, out: string[]) {
  // Peel leading punctuation.
  let start = 0;
  while (start < w.length && PUNCT.has(w[start])) start++;
  // Peel trailing punctuation.
  let end = w.length;
  while (end > start && PUNCT.has(w[end - 1])) end--;
  for (let i = 0; i < start; i++) out.push(w[i]);
  if (end > start) out.push(w.slice(start, end));
  for (let i = end; i < w.length; i++) out.push(w[i]);
}

export function buildVocab(words: string[], cap = 512): Vocab {
  const counts = new Map<string, number>();
  for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1);
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(e => e[0]);
  const kept = ranked.slice(0, Math.max(0, cap - 1));
  const list = ['<unk>', ...kept];
  const index = new Map(list.map((w, i) => [w, i] as const));
  return { words: list, index, folded: ranked.length - kept.length };
}

export function encode(words: string[], vocab: Vocab): number[] {
  return words.map(w => vocab.index.get(w) ?? UNK);
}

export function tokenize(text: string, cap = 512): { tokens: number[]; vocab: Vocab; words: string[] } {
  const words = splitWords(text);
  const vocab = buildVocab(words, cap);
  return { tokens: encode(words, vocab), vocab, words };
}

const NO_SPACE_BEFORE = new Set(['.', ',', '?', '!', ';', ':', ')']);
const NO_SPACE_AFTER = new Set(['(']);

export function detokenize(tokens: number[], vocab: Vocab): string {
  let s = '';
  let prev = '';
  for (const t of tokens) {
    const w = vocab.words[t] ?? '<unk>';
    if (w === '\n') { s += '\n'; prev = w; continue; }
    const glue = s.length === 0 || prev === '\n' || NO_SPACE_BEFORE.has(w) || NO_SPACE_AFTER.has(prev) ? '' : ' ';
    s += glue + w;
    prev = w;
  }
  return s;
}
