// Prefix-tree KV-cache sharing — the JavaScript twin of `prefix-tree-kv-cache`.
//
// The common prompt is prefilled once. Two first-level branches fork that
// working set, append distinct text, and are each forked again into two
// leaves; generation then continues independently from all four
// shared-prefix leaves, one decode lane per leaf.

import { chat, eta, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, WorkingSet, channelCapacity, dtype, intrinsics, reduceArgmax, reshape } = eta;

/** Freezes what it appends, so a later fork shares it instead of copying. */
function appendTokens(ws, pipeline, tokens) {
  if (tokens.length === 0) throw new Error('cannot append an empty token sequence');
  const fwd = new ForwardPass();
  fwd.embed(Channel.from(tokens, dtype.i32), Channel.from([0, tokens.length], dtype.u32));
  fwd.state([ws], { fold: 'all' });
  pipeline.submit(fwd);
}

/** Greedy decode, one lane per leaf, each seeded with its suffix's last token. */
function generate(leaves, pipeline, last, maxTokens) {
  const B = leaves.length;
  const vocab = model.outputVocabSize();
  const stop = new Set(chat.stopTokens());
  const tokenIn = Channel.from(last, dtype.i32);
  const tokenOut = new Channel([B], dtype.i32).capacity(channelCapacity());

  const fwd = new ForwardPass();
  fwd.embed(tokenIn, Channel.from(Array.from({ length: B + 1 }, (_, i) => i), dtype.u32));
  fwd.state(leaves, { fold: 'all' });
  fwd.epilogue(() => {
    const tokens = reshape(reduceArgmax(reshape(intrinsics.output(0, 'logits'), [B, vocab])), [B]);
    tokenIn.put(tokens);
    tokenOut.put(tokens);
  });

  const generated = leaves.map(() => []);
  const done = leaves.map(() => false);
  pipeline.runAhead(fwd, maxTokens, () => {
    tokenOut.takeHost().forEach((token, lane) => {
      if (done[lane]) return;
      if (stop.has(token)) done[lane] = true;
      else generated[lane].push(token);
    });
    return !done.every((d) => d);
  });
  return generated;
}

export function main(input) {
  const numTokens = Number(input.num_tokens ?? 32);
  const root = new WorkingSet();
  const pipeline = new Pipeline();
  appendTokens(root, pipeline, [...model.encode('Write a short scene set')]);

  const firstLevel = [];
  for (const suffix of [' in a city', ' in a forest']) {
    const child = root.fork(pipeline);
    appendTokens(child, pipeline, [...model.encode(suffix)]);
    firstLevel.push({ label: suffix.trim(), ws: child });
  }

  const labels = [];
  const leaves = [];
  const last = [];
  for (const parent of firstLevel) {
    for (const suffix of [' at dawn', ' at night']) {
      const leaf = parent.ws.fork(pipeline);
      const head = [...model.encode(suffix)];
      const tail = head.pop();
      if (tail === undefined) throw new Error('empty leaf suffix');
      if (head.length > 0) appendTokens(leaf, pipeline, head);
      labels.push(`${parent.label} ${suffix.trim()}`);
      leaves.push(leaf);
      last.push(tail);
    }
  }

  const generated = generate(leaves, pipeline, last, numTokens);
  pipeline.close();
  return labels.map((label, i) => `${label}: ${model.decode(generated[i])}`).join('\n');
}
