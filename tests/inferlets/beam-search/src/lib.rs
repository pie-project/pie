//! On-device beam search with logical ancestry masks.
//!
//! The KV cache is a prefix tree over a shared page pool. Each surviving beam
//! appends its new token
//! at the next free FLAT pool position (`wpos = fill + lane`), and its ancestry
//! is encoded in the per-beam `AttnMask` — inherit the parent's mask
//! (`gather(mask, parent)`) then set the one new cell (`eq(col, wpos)`). That is
//! the entire fork/prune mechanism.
//!
//! This compact example uses a fixed pool and therefore bounds generation
//! instead of compacting dead cells.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};
use serde::Deserialize;

const PAGE_T: u32 = 16; // tokens per pool page
const POOL_PAGES: u32 = 8; // shared pool pages (over-allocated; compaction bounds this)
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions
// Qwen3-0.6B
const BOS: i32 = 1;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// Beam width. `1` degenerates to greedy decoding (the beam identity).
    #[serde(default = "default_beams")]
    beams: u32,
    /// Bind the per-beam ancestry mask to the attention port. Default on.
    ///
    /// Setting this false leaves the whole program identical — the mask is
    /// still evolved in the epilogue — and only stops attention from reading
    /// it, so every beam attends the whole filled pool instead of its own
    /// ancestry. That is a deliberately broken beam, and it exists as a
    /// CONTROL: a driver that silently ignores `custom_mask_d` produces the
    /// mask-off token stream while reporting success, so a benchmark can only
    /// trust a model family whose mask-on and mask-off runs actually differ.
    /// `ModelCapabilities` carries no `supports_custom_mask` flag to check
    /// instead, which is why this has to be established by differencing.
    #[serde(default = "default_mask")]
    mask: bool,
}

fn default_max_tokens() -> usize {
    16
}

fn default_beams() -> u32 {
    2
}

fn default_mask() -> bool {
    true
}

/// Per-step disagreements between the beam pick and the raw-logit argmax.
fn count_greedy_mismatches(picked: &[i32], greedy: &[i32], beams: u32) -> usize {
    (0..beams as usize)
        .filter(|&lane| picked.get(lane) != greedy.get(lane))
        .count()
}

fn advance_hypotheses(
    hypotheses: &[Vec<u32>],
    picked: &[i32],
    parents: &[u32],
    beams: u32,
) -> Result<Vec<Vec<u32>>> {
    let mut next = Vec::with_capacity(beams as usize);
    for lane in 0..beams as usize {
        let parent = *parents
            .get(lane)
            .ok_or_else(|| format!("missing parent for beam {lane}"))?
            as usize;
        let token = *picked
            .get(lane)
            .ok_or_else(|| format!("missing token for beam {lane}"))? as u32;
        let mut hypothesis = hypotheses
            .get(parent)
            .ok_or_else(|| format!("invalid parent beam {parent}"))?
            .clone();
        hypothesis.push(token);
        next.push(hypothesis);
    }
    Ok(next)
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_steps = input.max_tokens;
    let b = input.beams;
    let use_mask = input.mask;
    if b == 0 {
        return Err("beams must be at least 1".into());
    }
    if b > POOL - 1 {
        return Err(format!(
            "beams exceeds the fixed pool ({} positions)",
            POOL - 1
        ));
    }
    // Bind the width once so the epilogue closure captures a plain `u32`
    // exactly where the old `const B` was substituted.
    #[allow(non_snake_case)]
    let B = b;
    let capacity = ((POOL - 1) / B) as usize;
    if max_steps > capacity {
        return Err(format!(
            "max_tokens exceeds fixed beam pool capacity ({capacity})"
        ));
    }
    if max_steps == 0 {
        return Ok(String::new());
    }

    let vocab = wit_model::output_vocab_size();
    let v = vocab;

    // Allocate a fixed logical page pool. Flat position `wpos` maps to
    // `pool_ids[wpos / PAGE_T]` at offset `wpos % PAGE_T`.
    let ws = WorkingSet::new();
    let pool = ws
        .reserve(POOL_PAGES)
        .map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids = pool.ids().to_vec();
    // Seeded at klen = 1 ⇒ one live page per lane, so lane b's single page sits at
    // flat slot b. `pages` keeps its [B*POOL_PAGES] capacity; only the first
    // `B * page_count` entries are read.
    let tiled: Vec<u32> = (0..B * POOL_PAGES)
        .map(|i| pool_ids[(i % 1) as usize])
        .collect(); // [B*POOL_PAGES]
    let pool0 = pool_ids[0];

    // Shared BOS prompt at pool position 0: both beams attend it (mask), and the
    // fire-0 write descriptor lands both BOS at (page pool_ids[0], off 0) — the
    // shared prefix cell. fill = 1 (position 0 filled).
    let init_mask: Vec<bool> = (0..B).flat_map(|_| (0..POOL).map(|p| p == 0)).collect();

    // Loop-carried search and page geometry.
    let mask = Channel::from_shaped([B, POOL], init_mask).named("mask"); // [B, POOL] bool
    let mut initial_scores = vec![f32::NEG_INFINITY; B as usize];
    initial_scores[0] = 0.0;
    let scores = Channel::from(initial_scores).named("scores");
    let toks = Channel::from(vec![BOS; B as usize]).named("toks");
    let pos = Channel::from(vec![0u32; B as usize]).named("pos");
    let fill = Channel::from(vec![1u32; 1]).named("fill"); // next free flat position
    let klen = Channel::from(vec![1u32; B as usize]).named("klen");
    let w_slot = Channel::from(vec![pool0; B as usize]).named("w_slot");
    let w_off = Channel::from(vec![0u32; B as usize]).named("w_off");
    let pages = Channel::from(tiled.clone()).named("pages");
    // Constant pool geometry: page_indptr = [0, POOL_PAGES, 2*POOL_PAGES] (each
    // beam references all pool pages). Bound via a CHANNEL (not the sugar's const
    // PageIndptr): a fire that binds ANY descriptor port to a channel is a
    // device-geometry fire, and the driver's device-geometry resolver skips const
    // ports (they never populate the wire) — so a mixed const-PageIndptr /
    // channel-Pages fire ships an EMPTY page_indptr and the driver reads a null
    // kv_page_indptr. Feeding page_indptr through a channel (re-put each fire with
    // the same constant) keeps every descriptor port channel-bound (the
    // device-geometry-fire wire-form).
    // The page CSR is the wire's source of truth for kv_len: the driver derives
    // `last_page_len = ((kv_len-1) % PAGE_T) + 1` and then reads back a span of
    // `(page_count-1)*PAGE_T + last_page_len` cells per lane. Declaring all
    // POOL_PAGES here would inflate that span past the live prefix and attend
    // uninitialized KV, so the per-lane page count must track `klen` exactly and
    // `pages` must be tiled at THAT stride, not at POOL_PAGES.
    let pidx_const: Vec<u32> = (0..=B).map(|b| b).collect();
    let page_indptr = Channel::from_shaped([B + 1], pidx_const.clone()).named("page_indptr");
    let lanes_b = Channel::from((0u32..=B).collect::<Vec<_>>()).named("embed_indptr");

    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
    let out = Channel::new([B], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out");
    let out_par = Channel::new([B], dtype::u32)
        .capacity(channel_capacity() as u32)
        .named("out_par");
    let out_scr = Channel::new([B], dtype::f32)
        .capacity(channel_capacity() as u32)
        .named("out_scr");
    // Independent per-lane greedy argmax over the RAW logits, published beside
    // the beam pick. At `beams == 1` the beam identity says the two must agree
    // on every step: top-1 over the flattened [1*V] candidate block is
    // `argmax(log_softmax(logits) + score)`, and both `log_softmax` and adding
    // a per-row constant are monotone, so it reduces to `argmax(logits)`. The
    // comparison therefore exercises log_softmax, the score accumulator, the
    // [B*V] flatten, `top_k`, and the `idx / v` / `idx % v` decomposition
    // against an operator that shares none of that machinery.
    let out_greedy = Channel::new([B], dtype::i32)
        .capacity(channel_capacity() as u32)
        .named("out_greedy");

    let pipeline = Pipeline::new();
    let mut rs_working_sets = if wit_model::rs_state_size() > 0 {
        (0..B).map(|_| RsWorkingSet::new()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let fwd = ForwardPass::new();
    fwd.rs_working_sets(&rs_working_sets)
        .map_err(|e| format!("bind initial recurrent states: {e}"))?;
    fwd.embed(&toks, &lanes_b)?;
    // All descriptor ports channel-bound (device-geometry fire wire-form):
    // Pages ← pages, PageIndptr ← page_indptr, KvLen ← klen, WSlot/WOff ← the
    // explicit write descriptor. The pool is fixed so these carry constant values.
    // The AttnMask port is the ONE thing `input.mask` changes: everything else
    // about the fire, including the epilogue's mask evolution, is identical in
    // both arms, so a mask-on/mask-off token-stream difference isolates whether
    // attention honoured the mask and nothing else.
    fwd.attention(
        &ws,
        ..,
        ..,
        &klen,
        &pages,
        &page_indptr,
        &w_slot,
        &w_off,
        &pos,
        if use_mask { Some(&mask) } else { None },
    )?;
    fwd.epilogue(move || {
        // 1. top-B over the flattened [B,V] cand block.
        // `intrinsics::logits()` squeezes to rank-1 `[v]` when the fire has a
        // single read-out row, so a width-1 beam has to be reshaped back to
        // `[B, v]` before it can meet the `[B, 1]`-broadcast score column.
        let logits = reshape(intrinsics::logits(), [B, v]);
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, v]),
            log_softmax(&logits),
        );
        let (s, i) = top_k(reshape(cand, [B * v]), B);
        let parent = div(&i, v);
        let tok_i = cast(rem(&i, v), DType::I32);

        // 2. flat tail-append positions: wpos = fill + lane.
        let base = fill.take(); // [1]
        let lane = iota(B); // [B]
        let base_b = broadcast(reshape(&base, [1]), [B]); // [1] -> [B]
        let wpos = add(&base_b, &lane); // [B]

        // 3. mask evolution: inherit parent's ancestry, OR the new position.
        let inherited = gather(mask.take(), &parent); // bool [B,POOL]
        let col = broadcast(reshape(iota(POOL), [1, POOL]), [B, POOL]);
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, POOL]);
        let newpos = eq(col, wpos_b); // bool [B,POOL]
        let new_mask = or(inherited, &newpos);
        mask.put(&new_mask);

        // 4. Explicit write descriptor for each surviving beam.
        let pids = pool_ids_ch.take().tensor();
        let logical_slot = div(&wpos, PAGE_T); // [B] index into the pool
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&wpos, PAGE_T);
        // Device-resolved geometry is loop-carried: the host never drains
        // these rings, so the graph has to take before it puts or the
        // readiness check sees a full ring and refuses to commit the pass.
        w_slot.take();
        w_slot.put(&w_slot_v);
        w_off.take();
        w_off.put(&w_off_v);

        // KV span after this step's appends (the mask restricts attention).
        let filled = add(&base, B); // [1]
        klen.take();
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.take();
        toks.put(&tok_i);
        // Re-emit the fixed Pages port each fire: the pool ids tiled B
        // times (every beam references all POOL_PAGES pool pages; the mask does
        // the per-beam selection). Built in-graph from the host-fed pids.
        // Live page count for the NEXT fire, from that fire's klen.
        let page_count = div(add(&filled, PAGE_T - 1), PAGE_T);
        let pages_ig = gather(
            &pids,
            rem(
                iota(B * POOL_PAGES),
                broadcast(&page_count, [B * POOL_PAGES]),
            ),
        );
        pages.take();
        pages.put(&pages_ig);
        // Re-emit the constant page_indptr each fire (channel-bound; peeked ports
        // still want a fresh value each pass). [0, POOL_PAGES, 2*POOL_PAGES].
        page_indptr.take();
        page_indptr.put(mul(iota(B + 1), broadcast(&page_count, [B + 1])));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        out_greedy.put(&reshape(cast(reduce_argmax(&logits), DType::I32), [B]));
        pool_ids_ch.put(&pids);
    });

    // Beam decode loop: feed the fixed pool ids and reconstruct each surviving
    // hypothesis from the parent permutation emitted by the device.
    let mut hypotheses = vec![Vec::<u32>::new(); B as usize];
    let mut final_scores = vec![f32::NEG_INFINITY; B as usize];
    let mut greedy_mismatches = 0usize;
    if rs_working_sets.is_empty() {
        let mut step = 0usize;
        run_ahead(&pipeline, &fwd, max_steps, async || {
            let picked = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let parents = out_par
                .take()
                .get::<u32>()
                .await
                .map_err(|e| format!("out_par.take @{step}: {e}"))?;
            final_scores = out_scr
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("out_scr.take @{step}: {e}"))?;
            let greedy = out_greedy
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out_greedy.take @{step}: {e}"))?;
            greedy_mismatches += count_greedy_mismatches(&picked, &greedy, B);
            hypotheses = advance_hypotheses(&hypotheses, &picked, &parents, B)?;
            step += 1;
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    } else {
        for step in 0..max_steps {
            fwd.submit(&pipeline)
                .map_err(|e| format!("submit @{step}: {e}"))?;
            let picked = out
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out.take @{step}: {e}"))?;
            let parents = out_par
                .take()
                .get::<u32>()
                .await
                .map_err(|e| format!("out_par.take @{step}: {e}"))?;
            final_scores = out_scr
                .take()
                .get::<f32>()
                .await
                .map_err(|e| format!("out_scr.take @{step}: {e}"))?;
            let greedy = out_greedy
                .take()
                .get::<i32>()
                .await
                .map_err(|e| format!("out_greedy.take @{step}: {e}"))?;
            greedy_mismatches += count_greedy_mismatches(&picked, &greedy, B);
            let mut next_rs = Vec::with_capacity(B as usize);
            for lane in 0..B as usize {
                let parent = *parents
                    .get(lane)
                    .ok_or_else(|| format!("missing parent for beam {lane}"))?
                    as usize;
                let parent_rs = rs_working_sets
                    .get(parent)
                    .ok_or_else(|| format!("invalid parent beam {parent}"))?;
                next_rs.push(
                    parent_rs
                        .fork(&pipeline)
                        .map_err(|e| format!("rs fork beam {lane} from parent {parent}: {e}"))?,
                );
            }
            hypotheses = advance_hypotheses(&hypotheses, &picked, &parents, B)?;
            fwd.rs_working_sets(&next_rs)
                .map_err(|e| format!("rebind recurrent states @{step}: {e}"))?;
            rs_working_sets = next_rs;
        }
    }
    pipeline.close();

    let best_lane = final_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .ok_or("beam search produced no hypotheses")?;
    // The beam identity is only an identity at width 1; at B > 1 the top-B
    // block legitimately picks non-argmax continuations for the lower lanes.
    if B == 1 && greedy_mismatches != 0 {
        return Err(format!(
            "beam identity violated: width-1 beam search disagreed with greedy \
             argmax on {greedy_mismatches} of {max_steps} steps"
        ));
    }
    eprintln!(
        "beam-search: width={B} steps={max_steps} best_score={:.4}",
        final_scores[best_lane]
    );
    let text = wit_model::decode(&hypotheses[best_lane])?;
    // Emit the returned beam's TOKEN IDS beside the text. A benchmark digests
    // these directly; re-tokenizing `text` would not be equivalent, because
    // detokenize→tokenize is not the identity, so a digest taken that way could
    // agree across two genuinely different token streams.
    let returned_tokens = hypotheses[best_lane]
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    // Shared-pool occupancy after the run. `fill` starts at 1 (the BOS cell at
    // pool position 0) and every step appends exactly one cell per lane, so it
    // is monotonic and its peak is its final value. Reported by the inferlet
    // rather than re-derived host-side so the number comes from the algorithm.
    let kv_cells_occupied_peak = 1 + (B as usize) * max_steps;
    Ok(format!(
        "{text}\n[beam] width={B} steps={max_steps} best_score={:.4} greedy_mismatches={greedy_mismatches}\n\
         [beam] mask={use_mask} kv_cells_occupied_peak={kv_cells_occupied_peak} returned_tokens={returned_tokens}",
        final_scores[best_lane]
    ))
}
