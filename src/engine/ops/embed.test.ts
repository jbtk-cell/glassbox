import { describe, it, expect } from 'vitest';
import { embed, posEmbed } from './embed';
import { gradCheck, randomCtx } from './gradcheck';
import { get } from './types';

const meta = { id: 'e', label: 'e', formula: '', explain: '' };

describe('embed', () => {
  it('looks up rows of E by token id', () => {
    const ctx = randomCtx({ E: [4, 2] }, 1);
    get(ctx, 'E').data.set([0, 0, 1, 1, 2, 2, 3, 3]);
    ctx.tokens = [3, 1]; ctx.T = 2;
    embed(meta, 'E', 'x').forward(ctx);
    expect(Array.from(get(ctx, 'x').data)).toEqual([3, 3, 1, 1]);
  });
  it('passes the gradient check, including a repeated token', () => {
    const ctx = randomCtx({ E: [5, 3] }, 2);
    ctx.tokens = [2, 4, 2]; ctx.T = 3;
    expect(gradCheck(embed(meta, 'E', 'x'), ctx, ['E'])).toEqual([]);
  });
});

describe('posEmbed', () => {
  it('takes the first T rows of P', () => {
    const ctx = randomCtx({ P: [4, 2] }, 3);
    get(ctx, 'P').data.set([0, 1, 2, 3, 4, 5, 6, 7]);
    ctx.tokens = [9, 9, 9]; ctx.T = 3;
    posEmbed(meta, 'P', 'p').forward(ctx);
    expect(Array.from(get(ctx, 'p').data)).toEqual([0, 1, 2, 3, 4, 5]);
  });
  it('passes the gradient check', () => {
    const ctx = randomCtx({ P: [6, 3] }, 4);
    ctx.tokens = [0, 0, 0, 0]; ctx.T = 4;
    expect(gradCheck(posEmbed(meta, 'P', 'p'), ctx, ['P'])).toEqual([]);
  });
});
