// Prefix-tree KV-cache sharing — the JavaScript twin of `prefix-tree-kv-cache`.
//
// The common prompt is prefilled once. Two first-level branches fork that
// working set (copy-on-write), append distinct text, and are each forked
// again into two leaves; generation then continues independently from all
// four shared-prefix leaves, one pipeline per leaf.

import { chat, eta, model } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, RsWorkingSet, WorkingSet, channelCapacity, dtype, indptr, intrinsics, kvPageSize, reduceArgmax, reshape, runAhead } = eta;

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const divCeil = (a, b) => Math.floor((a + b - 1) / b);

function appendTokens(ws, rs, pipeline, start, tokens) {
  if (tokens.length === 0) throw new Error('cannot append an empty token sequence');
  const n = tokens.length;
  const total = start + n;
  const pageSize = kvPageSize();
  const maxPages = Math.max(divCeil(total, pageSize), 1);
  const have = ws.pageLen();
  if (maxPages > have) ws.reserve(maxPages - have);
  const tokenInput = Channel.from(tokens, dtype.i32);
  const embedIndptr = Channel.from([0, n], dtype.u32).named('embed_indptr');
  const positions = Channel.from(range(start, total), dtype.u32).named('positions');
  const pages = Channel.from(range(0, ws.pageLen()), dtype.u32).named('pages');
  const pageIndptr = Channel.from([0, divCeil(total, pageSize)], dtype.u32).named('page_indptr');
  const wSlot = Channel.from(range(start, total).map((p) => Math.floor(p / pageSize)), dtype.u32).named('w_slot');
  const wOff = Channel.from(range(start, total).map((p) => p % pageSize), dtype.u32).named('w_off');
  const nextToken = new Channel([1], dtype.i32).named('next_token');
  const kvLen = Channel.from([total], dtype.u32).named('kv_len');

  const fwd = new ForwardPass();
  fwd.embed(tokenInput, embedIndptr);
  fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions, writablePages: Math.floor(start / pageSize) }, rs);
  fwd.epilogue(() => {
    nextToken.put(reshape(reduceArgmax(intrinsics.logits()), [1]));
  });
  fwd.submit(pipeline);
  return nextToken.takeScalar();
}

function generate(ws, rs, pipeline, seqLen, firstToken, maxTokens) {
  if (maxTokens === 0) return [];
  const stop = new Set(chat.stopTokens());
  const generated = [];
  if (!stop.has(firstToken)) generated.push(firstToken);
  if (generated.length >= maxTokens || stop.has(firstToken)) return generated;

  const pageSize = kvPageSize();
  const maxPages = Math.max(divCeil(seqLen + maxTokens + 1, pageSize), 1);
  const have = ws.pageLen();
  if (maxPages > have) ws.reserve(maxPages - have);
  const tokenIn = Channel.from([firstToken], dtype.i32).named('token_in');
  const embedIndptr = Channel.from([0, 1], dtype.u32).named('embed_indptr');
  const positions = Channel.from([seqLen], dtype.u32).named('positions');
  const pages = Channel.from(range(0, maxPages), dtype.u32).named('pages');
  const pageIndptr = Channel.from([0, divCeil(seqLen + 1, pageSize)], dtype.u32).named('page_indptr');
  const wSlot = Channel.from([Math.floor(seqLen / pageSize)], dtype.u32).named('w_slot');
  const wOff = Channel.from([seqLen % pageSize], dtype.u32).named('w_off');
  const tokenOut = new Channel([1], dtype.i32).capacity(channelCapacity()).named('token_out');
  const kvLen = Channel.from([seqLen + 1], dtype.u32).named('kv_len');

  const fwd = new ForwardPass();
  fwd.embed(tokenIn, embedIndptr);
  fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions, writablePages: Math.floor(seqLen / pageSize) }, rs);
  fwd.epilogue(() => {
    const length = kvLen.take();
    const token = reshape(reduceArgmax(intrinsics.logits()), [1]);
    const nextLength = length.add(1);
    const pageCount = nextLength.divCeil(pageSize);
    tokenIn.put(token);
    kvLen.put(nextLength);
    positions.put(length);
    wSlot.put(length.div(pageSize));
    wOff.put(length.rem(pageSize));
    pageIndptr.put(indptr(1, pageCount));
    tokenOut.put(token);
  });

  runAhead(pipeline, fwd, maxTokens - generated.length, () => {
    const token = tokenOut.takeScalar();
    if (stop.has(token)) return false;
    generated.push(token);
    return true;
  });
  return generated;
}

export function main(input) {
  const numTokens = Number(input.num_tokens ?? 32);
  const hybrid = model.passKind() !== 'attention';
  const root = new WorkingSet();
  const rootRs = hybrid ? [new RsWorkingSet()] : [];

  const rootTokens = [...model.encode('Write a short scene set')];
  if (rootTokens.length === 0) throw new Error('tokenizer produced an empty root prompt');

  const treePipeline = new Pipeline();
  appendTokens(root, rootRs, treePipeline, 0, rootTokens);
  const rootLen = rootTokens.length;
  const fork = (ws, rs) => [ws.fork(treePipeline), rs.map((r) => r.fork(treePipeline))];

  const firstLevel = [];
  for (const suffix of [' in a city', ' in a forest']) {
    const [child, childRs] = fork(root, rootRs);
    const tokens = [...model.encode(suffix)];
    appendTokens(child, childRs, treePipeline, rootLen, tokens);
    firstLevel.push({ label: suffix.trim(), ws: child, rs: childRs, seqLen: rootLen + tokens.length });
  }

  const leaves = [];
  for (const parent of firstLevel) {
    for (const suffix of [' at dawn', ' at night']) {
      const [leaf, leafRs] = fork(parent.ws, parent.rs);
      const tokens = [...model.encode(suffix)];
      const first = appendTokens(leaf, leafRs, treePipeline, parent.seqLen, tokens);
      leaves.push({ label: `${parent.label} ${suffix.trim()}`, ws: leaf, rs: leafRs, seqLen: parent.seqLen + tokens.length, first });
    }
  }
  treePipeline.close();

  const outputs = [];
  for (const leaf of leaves) {
    const generated = generate(leaf.ws, leaf.rs, new Pipeline(), leaf.seqLen, leaf.first, numTokens);
    outputs.push(`${leaf.label}: ${model.decode(generated)}`);
  }
  return outputs.join('\n');
}
