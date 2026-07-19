//! **Overview §6.2 beam search — DESIGN B (logical mask-out + lazy compaction).**
//!
//! Supersedes the Design A eager freeze / designated-heir / fresh-page-per-fork
//! scheme (the Design-A `beam` inferlet). The KV cache is a prefix tree
//! over a shared physical page POOL: each surviving beam appends its new token
//! at the next free FLAT pool position (`wpos = fill + lane`), and its ancestry
//! is encoded in the per-beam `AttnMask` — inherit the parent's mask
//! (`gather(mask, parent)`) then set the one new cell (`eq(col, wpos)`). That is
//! the entire fork/prune mechanism: no heir election, no freeze arithmetic, no
//! per-fork fresh-page selection, no per-step `gather(pages, parent)` reorder.
//!
//! `Pages`/`PageIndptr` are CONSTANT in VALUE (the pool is fixed between
//! compactions — a trace-known shape) but bound via CHANNELS: any channel-bound
//! descriptor port makes this a device-geometry fire, and the driver's resolver
//! skips const ports, so all descriptor ports are channel-fed (device-geometry
//! fire wire-form). Every beam references all pool pages and the mask does all
//! per-beam selection. `WSlot`/`WOff` (= `wpos / PAGE_T`, `wpos % PAGE_T`)
//! drive the explicit KV write (B2 `write_kv_explicit`). All seeds are
//! guest-known constants (the shared BOS prompt at pool position 0), so there is
//! no runtime physical-page seeding handshake — the Design A fire-0 seeding gap
//! does not arise here.
//!
//! The steady-state contract is host-verified by
//! `sdk/rust/ptir-dsl/tests/beam_designb_goldens.rs` (mask evolution across a
//! fork step + a pool page turn) on the CPU reference interpreter. Compaction
//! (a generic per-token, all-layers KV cell-move — `copy_into`) is OUT OF SCOPE
//! here; this inferlet exercises the steady-state append + mask-out path only.

use inferlet::ptir::prelude::*;
use inferlet::{Result, model as wit_model};

const B: u32 = 2; // beams
const POOL_PAGES: u32 = 8; // shared pool pages (over-allocated; compaction bounds this)
const BOS: i32 = 1;
const MAX_STEPS: usize = 8;

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    let vocab = wit_model::output_vocab_size();
    let v = vocab;
    let page_t = wit_model::kv_page_size();
    let pool_len = POOL_PAGES * page_t;

    // Design B owns a FIXED physical page pool: allocate POOL_PAGES real page
    // slots ONCE (bulk pool growth) and use their PHYSICAL ids for the Pages
    // port and the WSlot write descriptor. The flat pool position `wpos` maps to
    // physical page `pool_ids[wpos / PAGE_T]` at offset `wpos % PAGE_T`.
    let ws = WorkingSet::new();
    let pool = ws
        .reserve(POOL_PAGES)
        .map_err(|e| format!("ws.reserve pool: {e}"))?;
    let pool_ids = pool.ids().to_vec(); // [POOL_PAGES] physical
    let tiled: Vec<u32> = (0..B).flat_map(|_| pool_ids.iter().copied()).collect(); // [B*POOL_PAGES]
    let phys0 = pool_ids[0]; // physical page holding the shared prefix (pos 0)

    // Shared BOS prompt at pool position 0: both beams attend it (mask), and the
    // fire-0 write descriptor lands both BOS at (page pool_ids[0], off 0) — the
    // shared prefix cell. fill = 1 (position 0 filled).
    let init_mask: Vec<bool> = (0..B).flat_map(|_| (0..pool_len).map(|p| p == 0)).collect();

    // Loop-carried state (guest-seeded). w_slot/pages carry PHYSICAL page ids.
    let mask = Channel::from_shaped([B, pool_len], init_mask).named("mask");
    let scores = Channel::from(vec![0.0f32; B as usize]).named("scores");
    let toks = Channel::from(vec![BOS; B as usize]).named("toks");
    let pos = Channel::from(vec![0u32; B as usize]).named("pos");
    let fill = Channel::from(vec![1u32; 1]).named("fill"); // next free flat position
    let klen = Channel::from(vec![1u32; B as usize]).named("klen");
    let w_slot = Channel::from(vec![phys0; B as usize]).named("w_slot"); // physical page id
    let w_off = Channel::from(vec![0u32; B as usize]).named("w_off");
    let pages = Channel::from(tiled.clone()).named("pages"); // [B*POOL_PAGES] physical, const
    // Physical pool ids [POOL_PAGES], host-fed each fire, gathered in-graph to
    // map a flat pool-page index → physical page id for the write descriptor.
    let pool_ids_ch = Channel::new([POOL_PAGES], dtype::u32).named("pool_ids");
    let out = Channel::new([B], dtype::i32).named("out");
    let out_par = Channel::new([B], dtype::u32).named("out_par");
    let out_scr = Channel::new([B], dtype::f32).named("out_scr");

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
        let inherited = gather(mask.take(), &parent);
        let col = broadcast(reshape(iota(pool_len), [1, pool_len]), [B, pool_len]);
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, pool_len]);
        let newpos = eq(col, wpos_b);
        let new_mask = or(inherited, &newpos);
        mask.put(&new_mask);

        // 4. explicit write descriptor (B2 write_kv_explicit). WSlot is a
        // PHYSICAL page id: map the flat pool-page index (wpos / PAGE_T) through
        // the physical pool ids. pids [POOL_PAGES] is host-fed each fire.
        let pids = pool_ids_ch.take(); // [POOL_PAGES] physical page ids
        let logical_slot = div(&wpos, page_t);
        let w_slot_v = gather(&pids, &logical_slot); // [B] physical page id
        let w_off_v = rem(&wpos, page_t);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        // physical KV span after this step's appends (mask restricts attention).
        let filled = add(&base, B); // [1]
        klen.take();
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);
        // Re-emit the fixed Pages port each fire: the physical pool ids tiled B
        // times (every beam references all POOL_PAGES pool pages; the mask does
        // the per-beam selection). Built in-graph from the host-fed pids.
        let pages_ig = reshape(
            broadcast(reshape(&pids, [1, POOL_PAGES]), [B, POOL_PAGES]),
            [B * POOL_PAGES],
        );
        pages.take();
        pages.put(&pages_ig);
        // Re-emit the constant page_indptr each fire (channel-bound; peeked ports
        // still want a fresh value each pass). [0, POOL_PAGES, 2*POOL_PAGES].
        page_indptr.take();
        page_indptr.put(&Tensor::constant(
            (0..=B).map(|b| b * POOL_PAGES).collect::<Vec<_>>(),
        ));

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });

    // Beam decode loop: feed the fixed physical pool ids, submit run-ahead,
    // harvest tokens. No fresh-page put — Design B's pool is fixed (no per-fork
    // page handshake); compaction (out of scope here) would reclaim it.
    let pipeline = Pipeline::new();
    let mut hyp_tokens: Vec<u32> = Vec::new();
    for step in 0..MAX_STEPS {
        pool_ids_ch.put(pool_ids.clone());
        fwd.submit(&pipeline)
            .map_err(|e| format!("submit @{step}: {e}"))?;
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
        "BEAM_DESIGNB B={B} steps={} tokens={hyp_tokens:?} (mask-out §6.2 beam, vocab={vocab})",
        hyp_tokens.len()
    );
    println!("BEAM_DESIGNB_E2E {result}");
    Ok(result)
}
