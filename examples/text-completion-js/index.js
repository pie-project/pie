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
  const pipe = new Pipeline();

  // The model's opening (`<bos>` where it has one) before the raw text: a
  // gemma without it answers noise.
  const head = [...chat.prefix(), ...model.encode(promptText)];
  const last = head.pop();
  if (last === undefined) throw new Error('empty prompt');
  if (head.length > 0) {
    const fill = new ForwardPass();
    fill.embed(Channel.from(head, dtype.i32), Channel.from([0, head.length], dtype.u32));
    fill.state([ws], { fold: 'all' });
    pipe.submit(fill);
  }

  const tokIn = Channel.from([last], dtype.i32);
  const tokOut = new Channel([1], dtype.i32).capacity(channelCapacity());
  const fwd = new ForwardPass();
  fwd.embed(tokIn, Channel.from([0, 1], dtype.u32));
  fwd.state([ws], { fold: 'all' });
  fwd.epilogue(() => {
    const token = greedy(intrinsics.output(0, 'logits'));
    tokIn.put(token);
    tokOut.put(token);
  });

  const generated = [];
  pipe.runAhead(fwd, maxTokens, () => {
    generated.push(tokOut.takeScalar());
    return true;
  });
  pipe.close();

  return {
    text: model.decode(generated),
    count: generated.length,
    tokens: generated,
  };
}
