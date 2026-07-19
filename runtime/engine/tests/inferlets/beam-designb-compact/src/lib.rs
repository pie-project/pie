//! **Design B compaction device-proof — `pipeline.copy_into` (KV cell move).**
//!
//! This is the exact steady-state mask-out beam decode of the device-proven
//! `beam-designb` inferlet, plus ONE mid-run `pipeline.copy_into`: at step
//! `COMPACT_AT` the guest physically relocates the shared BOS KV cell from flat
//! pool position 0 to a fresh flat position `DST_FLAT`, and remaps the per-beam
//! `AttnMask` in-graph (clear column `src`, set column `dst`) so every beam
//! attends the moved cell at its new home.
//!
//! **Why the tokens must be IDENTICAL to `beam-designb`.** KV is stored
//! POST-RoPE, so a physical slot is pure storage — the token's position lives in
//! the attention mask, not the slot. Moving BOS's all-layers K/V from one
//! physical cell to another and pointing the mask at the new column yields the
//! same query attending the same stored K/V; the softmax, and therefore every
//! emitted token, is unchanged. So `beam-designb-compact`'s token stream ==
//! `beam-designb`'s token stream is the correctness oracle for `copy_into`.
//!
//! **Why no drain barrier is needed (K6 dissolved).** `copy_into` enqueues a
//! `PendingOp::Move` on the SAME scheduler FIFO / CUDA stream as the forward
//! fires. By the B3 ordering invariant the move happens-after the prior fires'
//! KV writes and happens-before the later fires' descriptor reads, so the
//! COMPACT_AT fire still reads BOS at flat 0, the move relocates it, and the
//! COMPACT_AT+1 fire reads it at `DST_FLAT` — all without a QUIESCE/drain.
//!
//! The mask stays fully IN-GRAPH (only small u32/bool scalars are host-fed to
//! drive an in-graph `select`); the host never `put`s a mask tensor, so the
//! deferred host-writer-mask staging gap does not arise.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const B: u32 = 2; // beams
const PAGE_T: u32 = 16; // tokens per pool page
const POOL_PAGES: u32 = 8; // shared pool pages
const POOL: u32 = POOL_PAGES * PAGE_T; // flat pool token positions
const BOS: i32 = 1;
const MAX_STEPS: usize = 8;
const COMPACT_AT: usize = 3; // step at which BOS is relocated
const SRC_FLAT: u32 = 0; // BOS lives at flat pool position 0
// A MATERIALISED, DEAD, IN-CAUSAL-RANGE destination cell: the decode is started
// at flat 2 (fill=2), so flat 1 (page 0, off 1) is never written yet physically
// exists (page 0 is materialised by the BOS write at flat 0) AND sits inside the
// query's causal/attended span (unlike a high pool cell, which FlashInfer's
// append attention excludes by index regardless of the custom-mask bit). This is
// the shape a real Design-B prune produces: a dead cell BELOW the write frontier.
const DST_FLAT: u32 = 1; // page 0, off 1 — materialised, dead, within causal range
const INIT_FILL: u32 = 2; // decode starts at flat 2, leaving flat 1 permanently dead

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    // Two modes on the SAME code/geometry so the ONLY difference is the physical
    // KV move: `move` in the input runs the mid-run copy_into (+ in-graph mask
    // remap); anything else runs the identical decode with NO move. The harness
    // asserts the two token streams are identical — isolating copy_into's effect.
    let do_move = input.contains("move");
    let vocab = wit_model::output_vocab_size();
    let v = vocab;

    // Fixed physical page pool (bulk growth), used for BOTH the fire descriptors
    // and the copy_into move. Flat pool position `wpos` maps to physical page
    // `pool_ids[wpos / PAGE_T]` at offset `wpos % PAGE_T`.
    let ws = WorkingSet::new();
    let pool = ws
        .reserve(POOL_PAGES)
        .map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids = pool.ids().to_vec(); // [POOL_PAGES]
    let tiled: Vec<u32> = (0..B).flat_map(|_| pool_ids.iter().copied()).collect();
    let phys0 = pool_ids[0];

    let init_mask: Vec<bool> = (0..B)
        .flat_map(|_| (0..POOL).map(|p| p == SRC_FLAT))
        .collect();

    // Loop-carried state (guest-seeded). klen stays at PAGE_T so the attention
    // scans ONLY the materialised page 0 (flat 0..15); the moved BOS cell lands at
    // flat 15 inside that same page, and the unmaterialised pool tail is never read.
    let mask = Channel::from_shaped([B, POOL], init_mask).named("mask");
    let scores = Channel::from(vec![0.0f32; B as usize]).named("scores");
    let toks = Channel::from(vec![BOS; B as usize]).named("toks");
    let pos = Channel::from(vec![0u32; B as usize]).named("pos");
    let fill = Channel::from(vec![INIT_FILL; 1]).named("fill");
    let klen = Channel::from(vec![PAGE_T; B as usize]).named("klen");
    let w_slot = Channel::from(vec![phys0; B as usize]).named("w_slot");
    let w_off = Channel::from(vec![0u32; B as usize]).named("w_off");
    let pages = Channel::from(tiled.clone()).named("pages");
    let pool_ids_ch = Channel::new([POOL_PAGES], dtype::u32).named("pool_ids");
    // In-graph mask-remap descriptor (host-fed scalars, NOT a host-fed mask):
    // remap_on gates an in-graph `select`; remap_src/remap_dst name the columns.
    let remap_on = Channel::new([1], dtype::bool).named("remap_on");
    let remap_src = Channel::new([1], dtype::u32).named("remap_src");
    let remap_dst = Channel::new([1], dtype::u32).named("remap_dst");
    let out = Channel::new([B], dtype::i32).named("out");
    let out_par = Channel::new([B], dtype::u32).named("out_par");
    let out_scr = Channel::new([B], dtype::f32).named("out_scr");

    let pidx_const: Vec<u32> = (0..=B).map(|b| b * POOL_PAGES).collect();
    let page_indptr = Channel::from_shaped([B + 1], pidx_const.clone()).named("page_indptr");
    let lanes_b = Channel::from((0u32..=B).collect::<Vec<_>>()).named("embed_indptr");

    let fwd = ForwardPass::new();
    fwd.embed(&toks, &lanes_b)?;
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
        Some(&mask),
    )?;
    fwd.epilogue(move || {
        // 1. top-B over the flattened [B,V] cand block (identical to beam-designb).
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, v]),
            log_softmax(intrinsics::logits()),
        );
        let (s, i) = top_k(reshape(cand, [B * v]), B);
        let parent = div(&i, v);
        let tok_i = cast(rem(&i, v), DType::I32);

        // 2. flat tail-append positions: wpos = fill + lane.
        let base = fill.take();
        let lane = iota(B);
        let base_b = broadcast(reshape(&base, [1]), [B]);
        let wpos = add(&base_b, &lane);

        // 3. mask evolution: inherit parent's ancestry, OR the new position.
        let inherited = gather(mask.take(), &parent);
        let col = broadcast(reshape(iota(POOL), [1, POOL]), [B, POOL]);
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, POOL]);
        let newpos = eq(&col, wpos_b);
        let new_mask = or(inherited, &newpos);

        // 3b. compaction mask-remap (in-graph, gated by the host `remap_on`
        // scalar): clear column `src`, set column `dst`. On non-compaction steps
        // remap_on is false, so `select` returns `new_mask` unchanged — the token
        // stream is then bit-identical to beam-designb up to and past the move.
        let don = remap_on.take();
        let rsrc = remap_src.take();
        let rdst = remap_dst.take();
        let is_src = eq(&col, broadcast(reshape(&rsrc, [1, 1]), [B, POOL]));
        let is_dst = eq(&col, broadcast(reshape(&rdst, [1, 1]), [B, POOL]));
        // Clear column `src`, set column `dst`: the moved BOS cell now lives at
        // its new home, so every beam attends `dst` instead of `src`.
        let cleared = and(&new_mask, not(is_src));
        let remapped = or(&cleared, is_dst);
        let cond = broadcast(reshape(&don, [1, 1]), [B, POOL]);
        let final_mask = select(cond, remapped, &new_mask);
        mask.put(&final_mask);

        // 4. explicit write descriptor (B2 write_kv_explicit).
        let pids = pool_ids_ch.take();
        let logical_slot = div(&wpos, PAGE_T);
        let w_slot_v = gather(&pids, &logical_slot);
        let w_off_v = rem(&wpos, PAGE_T);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        let filled = add(&base, B);
        klen.take();
        klen.put(broadcast(
            reshape(&Tensor::constant(vec![PAGE_T]), [1]),
            [B],
        ));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);
        let pages_ig = reshape(
            broadcast(reshape(&pids, [1, POOL_PAGES]), [B, POOL_PAGES]),
            [B * POOL_PAGES],
        );
        pages.take();
        pages.put(&pages_ig);
        page_indptr.take();
        page_indptr.put(&Tensor::constant(
            (0..=B).map(|b| b * POOL_PAGES).collect::<Vec<_>>(),
        ));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });

    let pipeline = Pipeline::new();
    let mut hyp_tokens: Vec<u32> = Vec::new();
    for step in 0..MAX_STEPS {
        pool_ids_ch.put(pool_ids.clone());
        // Drive the in-graph remap: only the COMPACT_AT fire's epilogue remaps
        // the mask (BOS column SRC_FLAT -> DST_FLAT), and only when this run is the
        // `move` mode. Every other step / the no-move run is a mask no-op.
        // Drive the in-graph remap: only the COMPACT_AT fire's epilogue remaps
        // the mask (BOS column SRC_FLAT -> DST_FLAT), and only when this run is the
        // `move` mode. Every other step / the no-move run is a mask no-op.
        if do_move && step == COMPACT_AT {
            remap_on.put(vec![true]);
            remap_src.put(vec![SRC_FLAT]);
            remap_dst.put(vec![DST_FLAT]);
        } else {
            remap_on.put(vec![false]);
            remap_src.put(vec![SRC_FLAT]);
            remap_dst.put(vec![SRC_FLAT]);
        }

        fwd.submit(&pipeline)
            .map_err(|e| format!("submit @{step}: {e}"))?;

        // The physical KV move rides the SAME FIFO right behind this fire: it
        // happens-after this step's KV writes (BOS is safely written), and
        // happens-before the next fire's descriptor reads (which attend
        // DST_FLAT). No drain barrier — B3 ordering carries it.
        if do_move && step == COMPACT_AT {
            let src_page = pool_ids[(SRC_FLAT / PAGE_T) as usize];
            let src_off = SRC_FLAT % PAGE_T;
            let dst_page = pool_ids[(DST_FLAT / PAGE_T) as usize];
            let dst_off = DST_FLAT % PAGE_T;
            ws.copy_into(&pipeline, &[dst_page], &[dst_off], &[src_page], &[src_off])
                .map_err(|e| format!("copy_into @{step}: {e}"))?;
        }

        let picked = out
            .take()
            .get::<i32>()
            .await
            .map_err(|e| format!("out.take @{step}: {e}"))?;
        let _parents = out_par
            .take()
            .get::<u32>()
            .await
            .map_err(|e| format!("out_par.take @{step}: {e}"))?;
        let _scr = out_scr
            .take()
            .get::<f32>()
            .await
            .map_err(|e| format!("out_scr.take @{step}: {e}"))?;
        if let Some(&t0) = picked.first() {
            hyp_tokens.push(t0 as u32);
        }
    }
    pipeline.close();

    let result = format!(
        "BEAM_DESIGNB_COMPACT B={B} steps={} do_move={do_move} compact_at={COMPACT_AT} \
         moved_flat={SRC_FLAT}->{DST_FLAT} tokens={hyp_tokens:?} (vocab={vocab})",
        hyp_tokens.len()
    );
    println!("BEAM_DESIGNB_COMPACT_E2E {result}");
    Ok(result)
}
