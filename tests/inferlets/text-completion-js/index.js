// Greedy text completion, host-driven — the JavaScript twin of `text-completion`.
//
// Same program as `tests/inferlets/text-completion/src/lib.rs`, traced from
// JavaScript: one chunked prefill, then a 1-wide decode loop whose ONE
// host-driven channel is the token. The traced container is byte-identical
// to the Rust inferlet's, so both share one program-cache entry. Host reads
// block (`takeScalar`), since a JS guest cannot lower `async func` imports.

import { model, eta } from '@pie-project/inferlet';

const { Channel, ForwardPass, Pipeline, RsWorkingSet, WorkingSet, dtype, indptr, intrinsics, kvPageSize, prefillChunks, reduceArgmax, reshape } = eta;

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const divCeil = (a, b) => Math.floor((a + b - 1) / b);

/** The greedy pick over a logits row, as a one-lane `[1]` i32 cell. */
function greedy(logits) {
  return reshape(reduceArgmax(logits), [1]);
}

export function main(input) {
  const promptText = input.prompt ?? 'The capital of France is';
  const maxTokens = Number(input.max_tokens ?? 8);

  const kind = model.passKind();
  if (kind === 'recurrent') throw new Error('this program has no recurrent-only path');

  const ws = new WorkingSet();
  const rsWs = kind !== 'attention' ? [new RsWorkingSet()] : [];
  const pageSize = kvPageSize();

  if (maxTokens === 0) return { text: '', count: 0, tokens: [] };

  let prompt = [...model.encode(promptText)];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;
  const maxPages = Math.max(divCeil(n + maxTokens + 1, pageSize), 1);
  ws.reserve(maxPages);

  const pipe = new Pipeline();
  const generated = [];

  // ── PREFILL (chunked, C-wide) ─────────────────────────────────────────
  let first = 0;
  for (const [base, end] of prefillChunks(n)) {
    const length = end - base;
    const toks = Channel.from(prompt.slice(base, end), dtype.i32).named('toks_p');
    const embedIndptr = Channel.from([0, length], dtype.u32).named('embed_indptr_p');
    const positions = Channel.from(range(base, end), dtype.u32).named('positions_p');
    const pages = Channel.from(range(0, maxPages), dtype.u32).named('pages_p');
    const pageIndptr = Channel.from([0, divCeil(end, pageSize)], dtype.u32).named('page_indptr_p');
    const wSlot = Channel.from(range(base, end).map((p) => Math.floor(p / pageSize)), dtype.u32).named('w_slot_p');
    const wOff = Channel.from(range(base, end).map((p) => p % pageSize), dtype.u32).named('w_off_p');
    const kvLen = Channel.from([end], dtype.u32).named('kv_len_p');
    const tokOut = new Channel([1], dtype.i32).named('tok_out_p');

    const fwd = new ForwardPass(kind);
    fwd.embed(toks, embedIndptr);
    fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions }, rsWs);
    fwd.epilogue(() => {
      tokOut.put(greedy(intrinsics.logits()));
    });
    fwd.submit(pipe);
    // Every chunk samples and every sample must be drained.
    first = tokOut.takeScalar();
  }
  generated.push(first);

  // ── DECODE (1-wide, host-driven token) ────────────────────────────────
  if (generated.length < maxTokens) {
    const tokIn = Channel.from([first], dtype.i32).named('tok_in');
    const embedIndptr = Channel.from([0, 1], dtype.u32).named('embed_indptr');
    const positions = Channel.from([n], dtype.u32).named('positions');
    const pages = Channel.from(range(0, maxPages), dtype.u32).named('pages');
    const pageIndptr = Channel.from([0, divCeil(n + 1, pageSize)], dtype.u32).named('page_indptr');
    const wSlot = Channel.from([Math.floor(n / pageSize)], dtype.u32).named('w_slot');
    const wOff = Channel.from([n % pageSize], dtype.u32).named('w_off');
    const kvLen = Channel.from([n + 1], dtype.u32).named('kv_len');
    const tokOut = new Channel([1], dtype.i32).named('tok_out');

    const fwd = new ForwardPass(kind);
    fwd.embed(tokIn, embedIndptr);
    fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions }, rsWs);
    fwd.epilogue(() => {
      // `length` is the readable extent this fire runs at, so it is also
      // the position the NEXT fire's token sits at.
      const length = kvLen.take();
      const nextLength = length.add(1);
      const pageCount = nextLength.divCeil(pageSize);
      kvLen.put(nextLength);
      positions.put(length);
      wSlot.put(length.div(pageSize));
      wOff.put(length.rem(pageSize));
      pageIndptr.put(indptr(1, pageCount));
      tokOut.put(greedy(intrinsics.logits()));
    });

    for (;;) {
      fwd.submit(pipe);
      const token = tokOut.takeScalar();
      generated.push(token);
      if (generated.length >= maxTokens) break;
      tokIn.put([token]);
    }
  }
  pipe.close();

  return {
    text: model.decode(generated),
    count: generated.length,
    tokens: generated,
  };
}
