// On-device beam search with logical ancestry masks — the JavaScript twin of
// `beam-search`.
//
// The chat-templated prompt (or `prompt_tokens`) is prefilled once; that fire
// picks the first `beams` tokens, and the beams then share one working set,
// each reading the pool up to its own slot under a mask that admits its own
// ancestry.

import { chat, eta, model } from '@pie-project/inferlet';

const {
  Channel, ForwardPass, Pipeline, WorkingSet, broadcast, cast, channelCapacity, dtype, eq, gather, intrinsics,
  iota, logSoftmax, or, reduceArgmax, reshape, topK,
} = eta;

const DEFAULT_PROMPT = 'Write a short paragraph about beam search.';
const DEFAULT_SYSTEM = 'You are a helpful, respectful and honest assistant.';

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

function countGreedyMismatches(picked, greedy, beams) {
  let n = 0;
  for (let lane = 0; lane < beams; lane++) if (picked[lane] !== greedy[lane]) n++;
  return n;
}

function advanceHypotheses(hypotheses, picked, parents, beams) {
  return range(0, beams).map((lane) => [...hypotheses[parents[lane]], picked[lane]]);
}

export function main(input) {
  const maxSteps = Number(input.max_tokens ?? 16);
  const B = Number(input.beams ?? 2);
  if (B === 0) throw new Error('beams must be at least 1');
  if (maxSteps === 0) return '';

  const v = model.outputVocabSize();
  if (B > v) throw new Error(`beams (${B}) exceeds the vocabulary (${v})`);

  let prompt;
  if (input.prompt_tokens != null) {
    prompt = input.prompt_tokens.map(Number);
  } else {
    prompt = [
      ...chat.systemUser(input.system ?? DEFAULT_SYSTEM, input.prompt ?? DEFAULT_PROMPT),
      ...chat.cueThinking(!(input.no_think ?? false)),
    ];
  }
  if (prompt.length === 0) throw new Error('empty prompt');
  const n = prompt.length;
  // The prompt's slots, then one slot per beam per decode step.
  const decodes = maxSteps - 1;
  const poolLen = n + decodes * B;
  if (poolLen > 0xffff_ffff) throw new Error('beams * max_tokens exceeds the u32 range');
  const ws = new WorkingSet();
  // The device chooses the beams' slots, so the pool is reserved up front.
  ws.reserve(poolLen);

  const pipeline = new Pipeline();

  // Step 0: the prompt's own next-token distribution; its top B are the first beams.
  const toksP = Channel.from(prompt, dtype.i32).named('toks_p');
  const embedIndptrP = Channel.from([0, n], dtype.u32).named('embed_indptr_p');
  const outP = new Channel([B], dtype.i32).named('out_p');
  const scrP = new Channel([B], dtype.f32).named('scr_p');
  const greedyP = new Channel([1], dtype.i32).named('greedy_p');
  const fwdP = new ForwardPass();
  fwdP.embed(toksP, embedIndptrP);
  fwdP.state([ws], { fold: 'all' });
  fwdP.epilogue(() => {
    const logits = reshape(intrinsics.logits(), [v]);
    const [s, i] = topK(logSoftmax(logits), B);
    outP.put(cast(i, dtype.i32));
    scrP.put(s);
    greedyP.put(reshape(reduceArgmax(logits), [1]));
  });
  pipeline.submit(fwdP);
  const first = Array.from(outP.takeHost());
  let finalScores = Array.from(scrP.takeHost());
  const greedy0 = greedyP.takeHost();
  let greedyMismatches = first[0] !== greedy0[0] ? 1 : 0;
  let hypotheses = first.map((t) => [t]);

  if (decodes > 0) {
    const scores = Channel.from(finalScores, dtype.f32).named('scores');
    const toks = Channel.from(first, dtype.i32).named('toks');
    const cap = channelCapacity();
    const out = new Channel([B], dtype.i32).capacity(cap).named('out');
    const outPar = new Channel([B], dtype.u32).capacity(cap).named('out_par');
    const outScr = new Channel([B], dtype.f32).capacity(cap).named('out_scr');
    // Independent per-lane greedy argmax over the RAW logits, beside the beam
    // pick: at `beams == 1` the two must agree on every step.
    const outGreedy = new Channel([B], dtype.i32).capacity(cap).named('out_greedy');
    const lanesB = Channel.from(range(0, B + 1), dtype.u32).named('embed_indptr');

    const step = () => {
      // `logits()` squeezes to `[v]` for a single read-out row, so reshape back.
      const logits = reshape(intrinsics.logits(), [B, v]);
      const cand = broadcast(reshape(scores.take(), [B, 1]), [B, v]).add(logSoftmax(logits));
      const [s, i] = topK(reshape(cand, [B * v]), B);
      const parent = i.div(v);
      const tokI = cast(i.rem(v), dtype.i32);
      scores.put(s);
      toks.put(tokI);
      out.put(tokI);
      outPar.put(parent);
      outScr.put(s);
      outGreedy.put(reshape(reduceArgmax(logits), [B]));
      return parent;
    };

    const fwd = new ForwardPass();
    fwd.embed(toks, lanesB);
    const pool = new Array(B).fill(ws);

    // The first decode writes beam i's token at slot n + i; it sees the prompt and its own cell.
    const initMask = [];
    for (let lane = 0; lane < B; lane++) for (let p = 0; p < poolLen; p++) initMask.push(p < n || p === n + lane);
    const mask = Channel.fromShaped([B, poolLen], initMask).named('mask');
    const slots = Channel.from(range(n, n + B), dtype.u32).named('slots');
    // A beam's RoPE position is its depth in the sequence, not its pool slot.
    const pos = Channel.from(new Array(B).fill(n), dtype.u32).named('pos');
    // Where the fire after this one writes its first beam.
    const fill = Channel.from([n + B], dtype.u32).named('fill');
    // Each beam reads up to its own slot.
    const kvLen = Channel.from(range(n + 1, n + B + 1), dtype.u32).named('kv_len');
    fwd.state(pool, { positions: pos, slots, kvLen });
    fwd.mask(mask);
    fwd.epilogue(() => {
      const parent = step();

      const base = fill.take();
      const baseB = broadcast(reshape(base, [1]), [B]);
      const wpos = baseB.add(iota(B));

      const inherited = gather(mask.take(), parent);
      const col = broadcast(reshape(iota(poolLen), [1, poolLen]), [B, poolLen]);
      const wposB = broadcast(reshape(wpos, [B, 1]), [B, poolLen]);
      const newpos = eq(col, wposB);
      mask.put(or(inherited, newpos));

      // Loop-carried: the host never drains these rings, so every fire re-puts them.
      // Every beam reads the pool up to its own slot.
      const filled = base.add(B);
      slots.put(wpos);
      kvLen.put(wpos.add(1));
      pos.put(pos.take().add(1));
      fill.put(filled);
    });

    pipeline.runAhead(fwd, decodes, () => {
      const picked = out.takeHost();
      const parents = outPar.takeHost();
      finalScores = outScr.takeHost();
      const greedy = outGreedy.takeHost();
      greedyMismatches += countGreedyMismatches(picked, greedy, B);
      hypotheses = advanceHypotheses(hypotheses, picked, parents, B);
      return true;
    });
  }
  pipeline.close();

  // The last of the maximal lanes, as Rust's `Iterator::max_by` picks it —
  // a tie between beams is a real outcome (two hypotheses of equal mass).
  let bestLane = 0;
  for (let lane = 1; lane < B; lane++) if (finalScores[lane] >= finalScores[bestLane]) bestLane = lane;
  if (B === 1 && greedyMismatches !== 0) {
    throw new Error(`beam identity violated: width-1 beam search disagreed with greedy argmax on ${greedyMismatches} of ${maxSteps} steps`);
  }
  const text = model.decode(hypotheses[bestLane]);
  return `${text}\n[beam] width=${B} steps=${maxSteps} best_score=${finalScores[bestLane].toFixed(4)} greedy_mismatches=${greedyMismatches}`;
}
