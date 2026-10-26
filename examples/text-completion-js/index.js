// Greedy text completion, device-carried: the JavaScript twin of `text-completion`.

import { chat, eta, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, WorkingSet, channelCapacity, dtype, intrinsics, reduceArgmax, reshape } = eta;

/** The greedy pick over a logits row, as a one-lane `[1]` i32 cell. */
function greedy(logits) {
  return reshape(reduceArgmax(logits), [1]);
}

export function main(input) {
  const promptText = input.prompt ?? 'The capital of France is';
  const maxTokens = Number(input.max_tokens ?? 8);

  const ws = new WorkingSet();

  if (maxTokens === 0) return { text: '', count: 0, tokens: [] };

  // The model's opening (`<bos>` where it has one) before the raw text: a
  // gemma without it answers noise.
  let prompt = [...chat.prefix(), ...model.encode(promptText)];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;

  const pipe = new Pipeline();
  const generated = [];

  const toks = Channel.from(prompt, dtype.i32).named('toks_p');
  const embedIndptr = Channel.from([0, n], dtype.u32).named('embed_indptr_p');
  const tokOut = new Channel([1], dtype.i32).named('tok_out_p');

  const prefill = new ForwardPass();
  prefill.embed(toks, embedIndptr);
  prefill.state([ws], { fold: 'all' });
  prefill.epilogue(() => {
    tokOut.put(greedy(intrinsics.output(0, 'logits')));
  });
  pipe.submit(prefill);
  const first = tokOut.takeScalar();
  generated.push(first);

  // ── DECODE (1-wide, device loop-carried token) ─────────────────────────
  if (generated.length < maxTokens) {
    const tokIn = Channel.from([first], dtype.i32).named('tok_in');
    const tokOut = new Channel([1], dtype.i32).capacity(channelCapacity()).named('tok_out');
    const embedIndptr = Channel.from([0, 1], dtype.u32).named('embed_indptr');

    const fwd = new ForwardPass();
    fwd.embed(tokIn, embedIndptr);
    fwd.state([ws], { fold: 'all' });
    fwd.epilogue(() => {
      const token = greedy(intrinsics.output(0, 'logits'));
      tokIn.put(token);
      tokOut.put(token);
    });

    const budget = maxTokens - 1;
    pipe.runAhead(fwd, budget, () => {
      generated.push(tokOut.takeScalar());
      return true;
    });
  }
  pipe.close();

  return {
    text: model.decode(generated),
    count: generated.length,
    tokens: generated,
  };
}
