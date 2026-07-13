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

const B: u32 = 2; // beams
const PAGE_T: u32 = 16; // tokens per pool page
const POOL_PAGES: u32 = 8; // shared pool pages (over-allocated; compaction bounds this)
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions
const NUM_LAYERS: u32 = 28; // Qwen3-0.6B
const BOS: i32 = 1;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
}

fn default_max_tokens() -> usize {
    16
}

fn advance_hypotheses(
    hypotheses: &[Vec<u32>],
    picked: &[i32],
    parents: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let mut next = Vec::with_capacity(B as usize);
    for lane in 0..B as usize {
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
    model::configure(vocab, PAGE_T, NUM_LAYERS);

    // Allocate a fixed logical page pool. Flat position `wpos` maps to
    // `pool_ids[wpos / PAGE_T]` at offset `wpos % PAGE_T`.
    let ws = WorkingSet::new();
    let pool = ws
        .reserve(POOL_PAGES)
        .map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids = pool.ids().to_vec();
    let tiled: Vec<u32> = (0..B).flat_map(|_| pool_ids.iter().copied()).collect(); // [B*POOL_PAGES]
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
    let pidx_const: Vec<u32> = (0..=B).map(|b| b * POOL_PAGES).collect();
    let page_indptr = Channel::from_shaped([B + 1], pidx_const.clone()).named("page_indptr");
    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>());

    let pool_ids_ch = Channel::from(pool_ids.clone()).named("pool_ids");
    let out = Channel::new([B], dtype::i32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("out");
    let out_par = Channel::new([B], dtype::u32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("out_par");
    let out_scr = Channel::new([B], dtype::f32)
        .capacity(DEFAULT_RUNAHEAD_DEPTH as u32)
        .named("out_scr");

    let pipeline = Pipeline::new();
    let mut rs_working_sets = if wit_model::rs_state_size() > 0 {
        (0..B).map(|_| RsWorkingSet::new()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let fwd = ForwardPass::new();
    fwd.set_rs_working_sets(&rs_working_sets)
        .map_err(|e| format!("bind initial recurrent states: {e}"))?;
    fwd.embed(&toks, lanes_b);
    fwd.positions(&pos);
    // All descriptor ports channel-bound (device-geometry fire wire-form):
    // Pages ← pages, PageIndptr ← page_indptr, KvLen ← klen, WSlot/WOff ← the
    // explicit write descriptor. The pool is fixed so these carry constant values.
    fwd.attn_working_set(&ws, &klen);
    fwd.port_channel(Port::Pages, &pages);
    fwd.port_channel(Port::PageIndptr, &page_indptr);
    fwd.port_channel(Port::WSlot, &w_slot);
    fwd.port_channel(Port::WOff, &w_off);
    fwd.attn_mask(&mask);
    fwd.epilogue(|| {
        // 1. top-B over the flattened [B,V] cand block.
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, v]),
            log_softmax(intrinsics::logits()),
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
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        // KV span after this step's appends (the mask restricts attention).
        let filled = add(&base, B); // [1]
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);
        // Re-emit the fixed Pages port each fire: the pool ids tiled B
        // times (every beam references all POOL_PAGES pool pages; the mask does
        // the per-beam selection). Built in-graph from the host-fed pids.
        let pages_ig = reshape(
            broadcast(reshape(&pids, [1, POOL_PAGES]), [B, POOL_PAGES]),
            [B * POOL_PAGES],
        );
        pages.put(&pages_ig);
        // Re-emit the constant page_indptr each fire (channel-bound; peeked ports
        // still want a fresh value each pass). [0, POOL_PAGES, 2*POOL_PAGES].
        page_indptr.put(Tensor::constant(
            (0..=B).map(|b| b * POOL_PAGES).collect::<Vec<_>>(),
        ));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        pool_ids_ch.put(&pids);
    });

    // Beam decode loop: feed the fixed pool ids and reconstruct each surviving
    // hypothesis from the parent permutation emitted by the device.
    let mut hypotheses = vec![Vec::<u32>::new(); B as usize];
    let mut final_scores = vec![f32::NEG_INFINITY; B as usize];
    if rs_working_sets.is_empty() {
        let mut submitted = 0usize;
        let mut in_flight = 0usize;
        while in_flight < DEFAULT_RUNAHEAD_DEPTH && submitted < max_steps {
            fwd.submit(&pipeline)
                .map_err(|e| format!("submit @{submitted}: {e}"))?;
            submitted += 1;
            in_flight += 1;
        }
        for step in 0..max_steps {
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
            in_flight -= 1;
            hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
            if submitted < max_steps {
                fwd.submit(&pipeline)
                    .map_err(|e| format!("submit @{submitted}: {e}"))?;
                submitted += 1;
                in_flight += 1;
            }
        }
        debug_assert_eq!(in_flight, 0);
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
            hypotheses = advance_hypotheses(&hypotheses, &picked, &parents)?;
            fwd.set_rs_working_sets(&next_rs)
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
    eprintln!(
        "beam-search: width={B} steps={max_steps} best_score={:.4}",
        final_scores[best_lane]
    );
    wit_model::decode(&hypotheses[best_lane])
}
