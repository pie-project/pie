// Prefix-tree KV-cache sharing — the JavaScript twin of `prefix-tree-kv-cache`.
//
// The common prompt is prefilled once. Two first-level branches fork that
// working set, append distinct text, and are each forked again into two
// leaves; generation then continues independently from all four
// shared-prefix leaves, one pipeline per leaf.

import { chat, eta, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, WorkingSet, channelCapacity, dtype, intrinsics, reduceArgmax, reshape } = eta;

/** Freezes what it appends, so a later fork shares it instead of copying. */
function appendTokens(ws, pipeline, tokens) {
  if (tokens.length === 0) throw new Error('cannot append an empty token sequence');
  const n = tokens.length;
  const tokenInput = Channel.from(tokens, dtype.i32);
  const embedIndptr = Channel.from([0, n], dtype.u32);
  const nextToken = new Channel([1], dtype.i32);

  const fwd = new ForwardPass();
  fwd.embed(tokenInput, embedIndptr);
  fwd.state([ws], { fold: 'all' });
  fwd.epilogue(() => {
    nextToken.put(reshape(reduceArgmax(intrinsics.output(0, 'logits')), [1]));
  });
  pipeline.submit(fwd);
  return nextToken.takeScalar();
}

function generate(ws, pipeline, firstToken, maxTokens) {
  if (maxTokens === 0) return [];
  const stop = new Set(chat.stopTokens());
  const generated = [];
  if (!stop.has(firstToken)) generated.push(firstToken);
  if (generated.length >= maxTokens || stop.has(firstToken)) return generated;

  const tokenIn = Channel.from([firstToken], dtype.i32);
  const embedIndptr = Channel.from([0, 1], dtype.u32);
  const tokenOut = new Channel([1], dtype.i32).capacity(channelCapacity());

  const fwd = new ForwardPass();
  fwd.embed(tokenIn, embedIndptr);
  fwd.state([ws], { fold: 'all' });
  fwd.epilogue(() => {
    const token = reshape(reduceArgmax(intrinsics.output(0, 'logits')), [1]);
    tokenIn.put(token);
    tokenOut.put(token);
  });

  pipeline.runAhead(fwd, maxTokens - generated.length, () => {
    const token = tokenOut.takeScalar();
    if (stop.has(token)) return false;
    generated.push(token);
    return true;
  });
  return generated;
}

export function main(input) {
  const numTokens = Number(input.num_tokens ?? 32);
  const root = new WorkingSet();

  const rootTokens = [...model.encode('Write a short scene set')];
  if (rootTokens.length === 0) throw new Error('tokenizer produced an empty root prompt');

  const treePipeline = new Pipeline();
  appendTokens(root, treePipeline, rootTokens);

  const firstLevel = [];
  for (const suffix of [' in a city', ' in a forest']) {
    const child = root.fork(treePipeline);
    appendTokens(child, treePipeline, [...model.encode(suffix)]);
    firstLevel.push({ label: suffix.trim(), ws: child });
  }

  const leaves = [];
  for (const parent of firstLevel) {
    for (const suffix of [' at dawn', ' at night']) {
      const leaf = parent.ws.fork(treePipeline);
      const first = appendTokens(leaf, treePipeline, [...model.encode(suffix)]);
      leaves.push({ label: `${parent.label} ${suffix.trim()}`, ws: leaf, first });
    }
  }
  treePipeline.close();

  const outputs = [];
  for (const leaf of leaves) {
    const generated = generate(leaf.ws, new Pipeline(), leaf.first, numTokens);
    outputs.push(`${leaf.label}: ${model.decode(generated)}`);
  }
  return outputs.join('\n');
}
