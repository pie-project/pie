// On-device beam search with logical ancestry masks — the JavaScript twin of
// `beam-search`.
//
// IT DOES NOT READ `prompt`: every beam is seeded with `BOS` at slot 0.

import { eta, model } from '@pie-project/inferlet';

const {
  Channel, ForwardPass, Pipeline, WorkingSet, broadcast, cast, channelCapacity, dtype, eq, gather, intrinsics,
  iota, logSoftmax, or, reduceArgmax, reshape, topK,
} = eta;

const BOS = 1;

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

  const initialScores = new Array(B).fill(-Infinity);
  initialScores[0] = 0.0;
  const scores = Channel.from(initialScores, dtype.f32).named('scores');
  const toks = Channel.from(new Array(B).fill(BOS), dtype.i32).named('toks');
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
    const logits = reshape(intrinsics.output(0, 'logits'), [B, v]);
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

  const pipeline = new Pipeline();
  const fwd = new ForwardPass();
  fwd.embed(toks, lanesB);

  let hypotheses = range(0, B).map(() => []);
  let finalScores = new Array(B).fill(-Infinity);
  let greedyMismatches = 0;

  /** Take this step's four rows off the device; returns the parents. */
  const drain = () => {
    const picked = out.takeHost();
    const parents = outPar.takeHost();
    finalScores = outScr.takeHost();
    const greedy = outGreedy.takeHost();
    greedyMismatches += countGreedyMismatches(picked, greedy, B);
    hypotheses = advanceHypotheses(hypotheses, picked, parents, B);
    return parents;
  };

  const poolLen = maxSteps * B + 1;
  if (poolLen > 0xffff_ffff) throw new Error('beams * max_tokens exceeds the u32 range');
  const ws = new WorkingSet();
  // The device chooses the slots, so the pool is reserved up front.
  ws.reserve(poolLen);
  const pool = new Array(B).fill(ws);

  const initMask = [];
  for (let b = 0; b < B; b++) for (let p = 0; p < poolLen; p++) initMask.push(p === 0);
  const mask = Channel.fromShaped([B, poolLen], initMask).named('mask');
  const slots = Channel.from(new Array(B).fill(0), dtype.u32).named('slots');
  // A beam's RoPE position is its depth, not its pool slot.
  const pos = Channel.from(new Array(B).fill(0), dtype.u32).named('pos');
  const fill = Channel.from([1], dtype.u32).named('fill');
  const kvLen = Channel.from(new Array(B).fill(1), dtype.u32).named('kv_len');
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
    const filled = base.add(B);
    slots.put(wpos);
    kvLen.put(broadcast(reshape(filled, [1]), [B]));
    pos.put(pos.take().add(1));
    fill.put(filled);
  });

  pipeline.runAhead(fwd, maxSteps, () => {
    drain();
    return true;
  });
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
