/** Plain-language text for the Explain tab, one entry per operation. Written for someone who
 *  has never seen a neural network: no jargon without a definition, no formulas here (those
 *  come from mathFor). `where` says what lights up in the other tabs. */

export interface StepText { title: string; group: string; text: string; where: string }

export const GROUPS = ['Inputs', 'Attention', 'Feed-forward', 'Prediction', 'Training'] as const;

export const STEPS: Record<string, StepText> = {
  embed: {
    title: 'Word table lookup', group: 'Inputs',
    text: 'Every word in Text Inputs picks out its own row of the word table. A row is a list of 20 numbers, and that list is everything the model knows about the word. The rows start out random when the model is created; training nudges them.',
    where: 'Network tab: the Word table tile in Embedding. Neurons tab: the tok column.',
  },
  pos_embed: {
    title: 'Position table lookup', group: 'Inputs',
    text: 'Each slot in Text Inputs (first word, second word, and so on up to 24) also picks a row of 20 numbers from the position table. This stamps "where I was" onto each word. It is needed because the mixing step later on treats the words as a pile and would otherwise not know their order.',
    where: 'Network tab: the Position table tile in Embedding. Neurons tab: the pos column.',
  },
  add_pos: {
    title: 'Add the two rows', group: 'Inputs',
    text: 'The word row and the position row are added together, number by number, giving one list of 20 numbers per word. Nothing is lost by adding: 20 slots is plenty of room for both. This list is a notepad that every later step writes onto.',
    where: 'Network tab: the Input tile at the bottom of the Transformer block. Neurons tab: the x0 column.',
  },
  ln1: {
    title: 'Tidy the numbers', group: 'Attention',
    text: 'Housekeeping. Each word\'s 20 numbers are rescaled so they average 0 and have a typical size of 1, then stretched and shifted by 40 learned numbers. It keeps everything in a sensible range so the next steps behave the same way for every word.',
    where: 'Network tab: the Input tile. Neurons tab: the h1 column.',
  },
  q_proj: {
    title: 'Question (q)', group: 'Attention',
    text: 'Each word\'s list is pushed through the W_q grid: 20 weighted sums, each with its own 20 weights, so 400 numbers plus 20 offsets. Out comes a new list of 20 numbers, the word\'s question: what it is looking for in the words before it. The names "question", "label" and "content" are just a way to talk about the three lists; training decides what they really do.',
    where: 'Network tab: the Q weight tile and the q tile above it. Neurons tab: the h1 to q fan of wires.',
  },
  k_proj: {
    title: 'Label (k)', group: 'Attention',
    text: 'The same move with a different grid, W_k. The result is the word\'s label: what it has to offer to the words that come after it.',
    where: 'Network tab: the K weight tile and the k tile. Neurons tab: the h1 to k wires.',
  },
  v_proj: {
    title: 'Content (v)', group: 'Attention',
    text: 'The same move again with a third grid, W_v. The content is what a word hands over when another word decides to borrow from it.',
    where: 'Network tab: the V weight tile and the v tile. Neurons tab: the h1 to v wires.',
  },
  scores: {
    title: 'Score every pair', group: 'Attention',
    text: 'For every pair of words, the question of one is compared with the label of the other: multiply matching numbers, add them up, divide by the square root of 20. A large score means "you matter to me right now". With 5 words that is a 5 by 5 grid of scores.',
    where: 'Network tab: the Attention tile. Neurons tab: the wires from q and k into the attention column.',
  },
  causal_mask: {
    title: 'Hide the future', group: 'Attention',
    text: 'A word may only look at itself and the words before it, because at guessing time the later words do not exist yet. Scores that point at later words are replaced by a huge negative number, which the next step turns into zero. That is why the tile is a triangle.',
    where: 'Network tab: the Attention tile. Neurons tab: hollow dashed circles in the attention column.',
  },
  attn_softmax: {
    title: 'Turn scores into weights', group: 'Attention',
    text: 'Each row of scores becomes a row of weights that are all positive and add up to 1: take e to the power of each score, then divide by the row\'s total. Big scores get much bigger weights than small ones. These weights are "how much of each word to borrow".',
    where: 'Network tab: the Attention tile. Neurons tab: the attention column, filled by weight.',
  },
  attn_apply: {
    title: 'Mix', group: 'Attention',
    text: 'Each word\'s new list is a weighted average of the content (v) of the words it is allowed to look at, using those weights. If "coming" got 0.5 and "to" got 0.3, the list for "the" is now half "coming" plus a third "to". After this step a word\'s list means the word in its context.',
    where: 'Network tab: the Attention output tile. Neurons tab: the teal wires from the attention column into ctxv.',
  },
  o_proj: {
    title: 'Write back', group: 'Attention',
    text: 'The mixed list goes through one more grid, W_o, which decides how the borrowed information gets written onto the notepad.',
    where: 'Network tab: the Write back weight tile. Neurons tab: the ctxv to attn_out wires.',
  },
  residual1: {
    title: 'Add onto the notepad', group: 'Attention',
    text: 'The result is added onto the word\'s original list instead of replacing it. The word and its position survive, and the context is layered on top. The dashed arc in the Neurons tab shows the original list being carried forward to here.',
    where: 'Network tab: the FF Input tile. Neurons tab: the x1 column and the dashed arc from x0.',
  },
  ln2: {
    title: 'Tidy again', group: 'Feed-forward',
    text: 'The same housekeeping as before: rescale each word\'s 20 numbers, then stretch and shift them by learned amounts.',
    where: 'Network tab: the FF Input tile. Neurons tab: the h2 column.',
  },
  ff_up: {
    title: 'Hidden layer', group: 'Feed-forward',
    text: 'From here on each word is processed alone; there is no more mixing. Its 20 numbers go through the W_1 grid into 30 hidden numbers. This is where the model stores facts that do not depend on the surrounding words, such as which words tend to follow which.',
    where: 'Network tab: the Input -> Hidden weight tile and the FF Hidden tile. Neurons tab: the h2 to ff_pre wires.',
  },
  relu: {
    title: 'ReLU', group: 'Feed-forward',
    text: 'Every negative hidden number is replaced by 0; positive ones are left alone. This one small step is what lets the model do more than draw straight lines. Without it, all the grids in a row would collapse into a single grid.',
    where: 'Network tab: the FF Hidden tile. Neurons tab: the ff_act column, where many circles turn white.',
  },
  ff_down: {
    title: 'Back to 20', group: 'Feed-forward',
    text: 'The 30 hidden numbers go through the W_2 grid back into 20 numbers, the right size to be added onto the notepad.',
    where: 'Network tab: the Hidden -> Output weight tile and the FF Output tile. Neurons tab: the ff_act to ff_out wires.',
  },
  residual2: {
    title: 'Add onto the notepad', group: 'Feed-forward',
    text: 'Added onto the list once more. This is the final version of each word\'s 20 numbers.',
    where: 'Network tab: the Output tile at the top of the block. Neurons tab: the x2 column and the dashed arc from x1.',
  },
  ln_final: {
    title: 'Tidy one last time', group: 'Prediction',
    text: 'One last rescale before the scoring, for the same reason as before.',
    where: 'Network tab: the Output tile. Neurons tab: the hf column.',
  },
  unembed: {
    title: 'Score every word', group: 'Prediction',
    text: 'The last word\'s tidied list goes through the unembedding grid U: one weighted sum per word in the vocabulary, so one raw score for each of the words in Training Text. These raw scores are called logits.',
    where: 'Network tab: the Unembedding tile. Neurons tab: the wide fan of wires from hf into the logits strip.',
  },
  softmax_out: {
    title: 'Turn scores into probabilities', group: 'Prediction',
    text: 'The same trick as the attention weights: e to the power of each score, divided by the total. Now every word in the vocabulary has a probability and they add up to 1. The largest one is the model\'s guess for the next word. Step and Play pick a word from these probabilities.',
    where: 'Network tab: the Softmax sequence tile and the Predicted next token circles. Neurons tab: the probs strip.',
  },
  loss: {
    title: 'How wrong was it?', group: 'Training',
    text: 'Training looks at the probability the model gave to the word that actually came next, and takes minus its log. Confident and right costs almost nothing; confident and wrong costs a lot. Averaged over every position in the window, this is the loss, the one number training pushes down.',
    where: 'Network tab: the Loss tile under Softmax sequence.',
  },
  adam_update: {
    title: 'Nudge every number', group: 'Training',
    text: 'Every one of the model\'s numbers is moved a little in the direction that lowers the loss, using the gradients from the backward pass. Adam, the update rule, makes each nudge bigger when a number\'s gradients have been steady and smaller when they have been jumping around. The learning rate sets the overall size.',
    where: 'Network tab: every weight tile shows its update in purple and green.',
  },
};

export const BACKWARD_NOTE = 'Backward pass: the same step run in reverse. Starting from the loss, the model works out how much each number that went into this step would change the loss if it were nudged. Those numbers are the gradients, shown in purple and green in the other tabs.';

export const OVERVIEW = [
  'This model guesses the next word. That is the whole job: given the words in Text Inputs, give every word in the vocabulary a probability of coming next, then pick one.',
  'It only knows the words that appear in Training Text. Each of those words has a row of 20 numbers in the word table, and each of the 24 slots in Text Inputs has a row of 20 numbers in the position table. Adding a word\'s row to its slot\'s row gives one list of 20 numbers per word.',
  'Those lists go through the Transformer block. First the mixing step, called attention, where each word borrows from the words before it. Then the thinking step, a small network that each word goes through alone. Both write their results back onto the same list.',
  'The last word\'s list is then scored against every word in the vocabulary, and the scores become probabilities. Training compares the probabilities with the word that really came next and nudges all 15,379 numbers to make that word more likely next time.',
];
