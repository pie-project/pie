// On-device beam search with logical ancestry masks — the JavaScript twin of
// `beam-search`.
//
// The KV cache is a prefix tree over a shared page pool. Each surviving beam
// appends its new token at the next free FLAT pool position (`wpos = fill +
// lane`), and its ancestry is encoded in the per-beam attention mask —
// inherit the parent's mask (`gather(mask, parent)`) then set the one new
// cell (`eq(col, wpos)`). That is the entire fork/prune mechanism.
//
// IT DOES NOT READ `prompt`: every beam is seeded with `BOS` at pool position
// 0 and decodes from there, as the Rust program does.

import { eta, model } from '@pie-project/inferlet';

const {
  Channel, ForwardPass, Pipeline, RsWorkingSet, WorkingSet, broadcast, cast, channelCapacity, dtype, eq, gather, intrinsics,
  iota, logSoftmax, or, reduceArgmax, reshape, runAhead, topK,
} = eta;

const POOL_PAGES = 8; // shared pool pages (over-allocated; compaction bounds this)
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
  const pageT = model.kvPageSize();
  const poolLen = POOL_PAGES * pageT;
  const maxSteps = Number(input.max_tokens ?? 16);
  const B = Number(input.beams ?? 2);
  if (B === 0) throw new Error('beams must be at least 1');
  if (B > poolLen - 1) throw new Error(`beams exceeds the fixed pool (${poolLen - 1} positions)`);
  const capacity = Math.floor((poolLen - 1) / B);
  if (maxSteps > capacity) throw new Error(`max_tokens exceeds fixed beam pool capacity (${capacity})`);
  if (maxSteps === 0) return '';

  const v = model.outputVocabSize();

  // A fixed logical page pool: flat position `wpos` maps to
  // `poolIds[wpos / pageT]` at offset `wpos % pageT`.
  const ws = new WorkingSet();
  const poolIds = ws.reserve(POOL_PAGES).ids;
  const tiled = new Array(B * POOL_PAGES).fill(poolIds[0]);
  const pool0 = poolIds[0];

  // Shared BOS at pool position 0: every beam attends it; fill = 1.
  const initMask = [];
  for (let b = 0; b < B; b++) for (let p = 0; p < poolLen; p++) initMask.push(p === 0);

  // Loop-carried search and page geometry.
  const mask = Channel.fromShaped([B, poolLen], initMask).named('mask');
  const initialScores = new Array(B).fill(-Infinity);
  initialScores[0] = 0.0;
  const scores = Channel.from(initialScores, dtype.f32).named('scores');
  const toks = Channel.from(new Array(B).fill(BOS), dtype.i32).named('toks');
  const pos = Channel.from(new Array(B).fill(0), dtype.u32).named('pos');
  const fill = Channel.from([1], dtype.u32).named('fill');
  const klen = Channel.from(new Array(B).fill(1), dtype.u32).named('klen');
  const wSlot = Channel.from(new Array(B).fill(pool0), dtype.u32).named('w_slot');
  const wOff = Channel.from(new Array(B).fill(0), dtype.u32).named('w_off');
  const pages = Channel.from(tiled, dtype.u32).named('pages');
  const pageIndptr = Channel.fromShaped([B + 1], range(0, B + 1), dtype.u32).named('page_indptr');
  const lanesB = Channel.from(range(0, B + 1), dtype.u32).named('embed_indptr');
  const poolIdsCh = Channel.from(poolIds, dtype.u32).named('pool_ids');
  const cap = channelCapacity();
  const out = new Channel([B], dtype.i32).capacity(cap).named('out');
  const outPar = new Channel([B], dtype.u32).capacity(cap).named('out_par');
  const outScr = new Channel([B], dtype.f32).capacity(cap).named('out_scr');
  // Independent per-lane greedy argmax over the RAW logits, beside the beam
  // pick: at `beams == 1` the two must agree on every step.
  const outGreedy = new Channel([B], dtype.i32).capacity(cap).named('out_greedy');

  const pipeline = new Pipeline();
  const kind = model.passKind();
  let rsWorkingSets;
  if (kind === 'attention') rsWorkingSets = [];
  else if (kind === 'hybrid') rsWorkingSets = range(0, B).map(() => new RsWorkingSet());
  else if (kind === 'recurrent') throw new Error('beam-search has no recurrent-only path (no registered model reports that kind)');
  else throw new Error('this program decodes a token at a time; a diffusion model wants a canvas loop');

  const fwd = new ForwardPass(kind);
  const geometry = { kvLen: klen, pages, pageIndptr, wSlot, wOff, positions: pos, mask };
  // Beams never buffer: every fire folds its one token straight into the
  // recurrence, which is what makes a fork a plain state copy.
  const bindState = (rs) => fwd.bindState(ws, geometry, rs);
  bindState(rsWorkingSets);
  fwd.embed(toks, lanesB);

  fwd.epilogue(() => {
    // 1. top-B over the flattened [B, V] candidate block. `logits()` squeezes
    // to `[v]` for a single read-out row, so reshape back.
    const logits = reshape(intrinsics.logits(), [B, v]);
    const cand = broadcast(reshape(scores.take(), [B, 1]), [B, v]).add(logSoftmax(logits));
    const [s, i] = topK(reshape(cand, [B * v]), B);
    const parent = i.div(v);
    const tokI = cast(i.rem(v), dtype.i32);

    // 2. flat tail-append positions: wpos = fill + lane.
    const base = fill.take();
    const lane = iota(B);
    const baseB = broadcast(reshape(base, [1]), [B]);
    const wpos = baseB.add(lane);

    // 3. mask evolution: inherit the parent's ancestry, OR the new position.
    const inherited = gather(mask.take(), parent);
    const col = broadcast(reshape(iota(poolLen), [1, poolLen]), [B, poolLen]);
    const wposB = broadcast(reshape(wpos, [B, 1]), [B, poolLen]);
    const newpos = eq(col, wposB);
    mask.put(or(inherited, newpos));

    // 4. Explicit write descriptor for each surviving beam.
    const pids = poolIdsCh.take();
    const logicalSlot = wpos.div(pageT);
    const wSlotV = gather(pids, logicalSlot);
    const wOffV = wpos.rem(pageT);
    wSlot.put(wSlotV);
    wOff.put(wOffV);

    // KV span after this step's appends (the mask restricts attention).
    const filled = base.add(B);
    klen.put(broadcast(reshape(filled, [1]), [B]));

    pos.put(pos.take().add(1));
    fill.put(filled);
    scores.put(s);
    toks.put(tokI);
    // Live page count for the NEXT fire, from that fire's klen; the pool ids
    // tiled B times at that stride, and the constant CSR re-emitted.
    const pageCount = filled.divCeil(pageT);
    pages.put(gather(pids, iota(B * POOL_PAGES).rem(broadcast(pageCount, [B * POOL_PAGES]))));
    pageIndptr.put(iota(B + 1).mul(broadcast(pageCount, [B + 1])));

    out.put(tokI);
    outPar.put(parent);
    outScr.put(s);
    outGreedy.put(reshape(reduceArgmax(logits), [B]));
    poolIdsCh.put(pids);
  });

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

  if (rsWorkingSets.length === 0) {
    runAhead(pipeline, fwd, maxSteps, () => {
      drain();
      return true;
    });
  } else {
    for (let step = 0; step < maxSteps; step++) {
      fwd.submit(pipeline);
      const parents = drain();
      const nextRs = parents.map((p) => rsWorkingSets[p].fork(pipeline));
      bindState(nextRs);
      rsWorkingSets = nextRs;
    }
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
