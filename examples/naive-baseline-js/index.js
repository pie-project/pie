// Naive text completion — the JavaScript twin of `naive-baseline`.
//
// One N-wide prefill fire, then a device-carried decode loop driven by
// `runAhead`, which keeps the runtime's run-ahead window full ahead of the
// host drain. The epilogue temperature-scales the logits and draws a
// Gumbel-max sample; `stats` adds the two extra `[1]` f32 drains. Traces to
// the same container bytes as the Rust inferlet.

import { chat, model, eta } from '@pie-project/inferlet';

const {
  Channel,
  ForwardPass,
  Pipeline,
  RsWorkingSet,
  WorkingSet,
  cast,
  channelCapacity,
  dtype,
  gumbelMax,
  indptr,
  intrinsics,
  iota,
  kvPageSize,
  prefillChunks,
  reshape,
  runAhead,
} = eta;

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);
const divCeil = (a, b) => Math.floor((a + b - 1) / b);

/** One sampling step: temperature, then a Gumbel-max draw over the vocab. */
function step(logits, temperature, rngState) {
  const scaled = temperature === 1.0 ? logits : logits.div(temperature);
  return gumbelMax(scaled, rngState);
}

export function main(input) {
  const promptText = input.prompt ?? 'Write a short paragraph about naive sampling.';
  const temperature = Number(input.temperature ?? 1.0);
  const maxTokens = Number(input.max_tokens ?? 32);
  const seed = Number(input.seed ?? 0x7ce1);
  const wantStats = Boolean(input.stats ?? false);
  const maxLayers = input.max_layers;

  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('temperature must be finite and greater than 0');

  const kind = model.passKind();
  const ws = new WorkingSet();
  const rsWs = kind !== 'attention' ? [new RsWorkingSet()] : [];
  const pageSize = kvPageSize();

  if (maxTokens === 0) return { sampler: 'naive-baseline-js', text: '', tokens: [], count: 0, stats: wantStats };

  let prompt = [...chat.prefix(), ...model.encode(promptText)];
  if (prompt.length === 0) prompt = [0];
  const n = prompt.length;
  const maxPages = Math.max(divCeil(n + maxTokens + 1, pageSize), 1);
  ws.reserve(maxPages);

  const generated = [];
  const pipe = new Pipeline();

  // ── PREFILL (chunked, C-wide): first sampled token comes off the prompt.
  let g0 = 0;
  for (const [base, end] of prefillChunks(n)) {
    const length = end - base;
    const toksP = Channel.from(prompt.slice(base, end), dtype.i32).named('toks_p');
    const embedIndptrP = Channel.from([0, length], dtype.u32).named('embed_indptr_p');
    const positionsP = Channel.from(range(base, end), dtype.u32).named('positions_p');
    const pagesP = Channel.from(range(0, maxPages), dtype.u32).named('pages_p');
    const pageIndptrP = Channel.from([0, divCeil(end, pageSize)], dtype.u32).named('page_indptr_p');
    const wSlotP = Channel.from(range(base, end).map((p) => Math.floor(p / pageSize)), dtype.u32).named('w_slot_p');
    const wOffP = Channel.from(range(base, end).map((p) => p % pageSize), dtype.u32).named('w_off_p');
    const kvLenP = Channel.from([end], dtype.u32).named('kv_len_p');
    const rngP = Channel.from([seed, 0], dtype.u32).named('rng_p');
    const tokOutP = new Channel([1], dtype.i32).named('tok_out_p');
    const s1OutP = new Channel([1], dtype.f32).named('s1_out_p');
    const s2OutP = new Channel([1], dtype.f32).named('s2_out_p');

    const fwdP = new ForwardPass(kind);
    if (maxLayers != null) fwdP.setMaxLayers(Number(maxLayers));
    fwdP.embed(toksP, embedIndptrP);
    fwdP.bindState(ws, { kvLen: kvLenP, pages: pagesP, pageIndptr: pageIndptrP, wSlot: wSlotP, wOff: wOffP, positions: positionsP }, rsWs);
    fwdP.epilogue(() => {
      const r = rngP.take();
      const logits = intrinsics.logits();
      const token = step(logits, temperature, r);
      const rNext = r.add(iota(2));
      tokOutP.put(token);
      if (wantStats) {
        const mirror = reshape(cast(token, dtype.f32), [1]);
        s1OutP.put(mirror);
        s2OutP.put(mirror);
      }
      rngP.put(rNext);
    });
    fwdP.submit(pipe);
    g0 = tokOutP.takeScalar();
    if (wantStats) {
      s1OutP.takeHost();
      s2OutP.takeHost();
    }
  }
  generated.push(g0);

  // ── DECODE LOOP (1-wide, run-ahead). ──
  if (generated.length < maxTokens) {
    const cap = channelCapacity();
    const tokIn = Channel.from([g0], dtype.i32).named('tok_in');
    const rng = Channel.from([seed ^ 0x5bd1, 0], dtype.u32).named('rng');
    const tokOut = new Channel([1], dtype.i32).capacity(cap).named('tok_out');
    const s1Out = new Channel([1], dtype.f32).capacity(cap).named('s1_out');
    const s2Out = new Channel([1], dtype.f32).capacity(cap).named('s2_out');
    const lane1 = Channel.from([0, 1], dtype.u32).named('embed_indptr');
    const positions = Channel.from([n], dtype.u32).named('positions');
    const pages = Channel.from(range(0, maxPages), dtype.u32).named('pages');
    const pageIndptr = Channel.from([0, divCeil(n + 1, pageSize)], dtype.u32).named('page_indptr');
    const wSlot = Channel.from([Math.floor(n / pageSize)], dtype.u32).named('w_slot');
    const wOff = Channel.from([n % pageSize], dtype.u32).named('w_off');
    const kvLen = Channel.from([n + 1], dtype.u32).named('kv_len');

    const fwd = new ForwardPass(kind);
    if (maxLayers != null) fwd.setMaxLayers(Number(maxLayers));
    fwd.embed(tokIn, lane1);
    fwd.bindState(ws, { kvLen, pages, pageIndptr, wSlot, wOff, positions, writablePages: Math.floor(n / pageSize) }, rsWs);
    fwd.epilogue(() => {
      const length = kvLen.take();
      const r = rng.take();
      const logits = intrinsics.logits();
      const token = step(logits, temperature, r);

      const rNext = r.add(iota(2));
      const nextLength = length.add(1);
      const pageCount = nextLength.divCeil(pageSize);

      tokIn.put(token);
      kvLen.put(nextLength);
      positions.put(length);
      wSlot.put(length.div(pageSize));
      wOff.put(length.rem(pageSize));
      pageIndptr.put(indptr(1, pageCount));
      tokOut.put(token);
      if (wantStats) {
        const mirror = reshape(cast(token, dtype.f32), [1]);
        s1Out.put(mirror);
        s2Out.put(mirror);
      }
      rng.put(rNext);
    });

    const budget = maxTokens - 1;
    runAhead(pipe, fwd, budget, () => {
      const t = tokOut.takeScalar();
      if (wantStats) {
        s1Out.takeHost();
        s2Out.takeHost();
      }
      generated.push(t);
      return true;
    });
  }
  pipe.close();

  return {
    sampler: 'naive-baseline-js',
    text: model.decode(generated),
    tokens: generated,
    count: generated.length,
    stats: wantStats,
  };
}
