//! Design B beam goldens (host-side, ptir-dsl + CPU reference interp).
//!
//! **Design B: logical mask-out + lazy compaction** (supersedes Design A's eager
//! freeze / designated-heir / fresh-page-per-fork scheme). The KV cache is a
//! prefix tree over a shared physical page pool: shared ancestors are written
//! once and referenced by mask; each surviving beam appends its new token at the
//! next free flat position in the pool; a beam's ancestry is encoded in its
//! per-beam attention mask, NOT in a per-fork physical page layout. Pruning is a
//! mask edit; garbage is reclaimed by a rare batched compaction (out of scope
//! here — the CompactPlan port is a separate contract item).
//!
//! The steady-state epilogue is DRAMATICALLY simpler than Design A: no heir
//! election, no freeze arithmetic, no per-fork fresh-page selection, no
//! per-step `gather(pages, parent)` page reorder. It computes:
//!   1. top-B (parent + token) over the flattened cand block,
//!   2. each survivor's flat tail-append position `wpos = fill + lane`,
//!   3. the new per-beam mask = inherit parent's mask (`gather(mask, parent)`)
//!      OR the new position (`eq(col, wpos)`), and
//!   4. the explicit write descriptor `w_slot = wpos / PAGE_T`, `w_off = wpos %
//!      PAGE_T` (consumed by B2's `write_kv_explicit`).
//! `Pages`/`PageIndptr` are CONSTANT (the shared pool is fixed between
//! compactions) — the mask does all the per-beam selection.
//!
//! These goldens assert the mask evolution across a fork step and the write
//! descriptor, on echo's reference interpreter. They are the Design B contract;
//! they replaced the retired Design A `ptir_beam.rs`/`beam_goldens` vectors (B5).

use pie_ptir::interp::{Instance, NoKernels, PassInputs, Value};
use pie_ptir::registry::{ModelProfile, Port};
use pie_ptir::validate::{bind, BoundTrace};

use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::{Channel, DType, Traced};

const B: u32 = 2; // beams
const V: u32 = 8; // vocab
const PAGE_T: u32 = 4; // tokens per pool page
const POOL_PAGES: u32 = 3; // pool pages (over-allocated; compaction bounds this)
const POOL: u32 = POOL_PAGES * PAGE_T; // 12 flat token positions in the shared pool

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Build the Design B steady-state beam epilogue via the neutral Builder.
///
/// Channel declaration order (host seeds by index):
///   0 mask[B,POOL] bool (seeded) — per-beam attention mask over the shared pool
///   1 scores[B] f32 (seeded)
///   2 toks[B] i32 (seeded)     — current per-beam token (embed)
///   3 pos[B] u32 (seeded)      — logical depth (RoPE position)
///   4 fill[1] u32 (seeded)     — next free flat position in the pool
///   5 klen[B] u32 (seeded)     — physical KV span (bound to KvLen)
///   6 w_slot[B] u32 (seeded)   — bound to WSlot
///   7 w_off[B] u32 (seeded)    — bound to WOff
///   8 out[B] i32 (host-reader) — harvested token
///   9 out_par[B] u32 (host-reader) — parent permutation (hypothesis backtrack)
///  10 out_scr[B] f32 (host-reader) — running scores
///  11 out_mask[B,POOL] bool (host-reader) — the NEW mask, for assertion
///  12 out_wslot[B] u32 (host-reader) — the write-descriptor page id
///  13 out_woff[B] u32 (host-reader) — the write-descriptor in-page offset
fn build_designb() -> Traced {
    ptir_dsl::model::configure(V, PAGE_T, 2);

    let mask = leak(Channel::seeded([B, POOL], dtype::bool).named("mask"));
    let scores = leak(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let toks = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let pos = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let fill = leak(Channel::from(vec![0u32; 1]).named("fill"));
    let klen = leak(Channel::from(vec![0u32; B as usize]).named("klen"));
    let w_slot = leak(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let out = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = leak(Channel::new([B], dtype::f32).named("out_scr"));
    let out_mask = leak(Channel::new([B, POOL], dtype::bool).named("out_mask"));
    let out_wslot = leak(Channel::new([B], dtype::u32).named("out_wslot"));
    let out_woff = leak(Channel::new([B], dtype::u32).named("out_woff"));

    // Constant pool geometry: every beam references all POOL_PAGES pool pages
    // (the mask restricts which positions it actually attends). Fixed between
    // compactions — no per-fire Pages computation.
    let pool_pages: Vec<u32> = (0..B).flat_map(|_| 0..POOL_PAGES).collect(); // [B*POOL_PAGES]
    let page_indptr: Vec<u32> = (0..=B).map(|b| b * POOL_PAGES).collect(); // [B+1]
    let pages_c = Tensor::constant(pool_pages);
    let page_indptr_c = Tensor::constant(page_indptr);
    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>());

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::Pages, pages_c);
    b.bind_port(Port::PageIndptr, page_indptr_c);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::AttnMask, mask);
    b.stage(Stage::Epilogue, move || {
        // 1. top-B over the flattened [B,V] cand block.
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, V]),
            log_softmax(intrinsics::logits()),
        );
        let (s, i) = top_k(reshape(cand, [B * V]), B);
        let parent = div(&i, V); // [B] which beam each survivor came from
        let tok_i = cast(rem(&i, V), DType::I32); // [B] new token

        // 2. flat tail-append positions in the shared pool: fill + lane.
        let base = fill.take(); // [1]
        let lane = iota(B); // [B]
        let base_b = broadcast(reshape(&base, [1]), [B]); // [1] -> [B]
        let wpos = add(&base_b, &lane); // [B] flat positions

        // 3. mask evolution: inherit parent's ancestry mask, OR the new position.
        let inherited = gather(mask.take(), &parent); // bool [B,POOL] row-gather
        let col = broadcast(reshape(iota(POOL), [1, POOL]), [B, POOL]); // [B,POOL] = 0..POOL-1
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, POOL]); // [B,POOL] = wpos[b]
        let newpos = eq(col, wpos_b); // bool [B,POOL]: the one new cell per beam
        let new_mask = or(inherited, &newpos); // bool [B,POOL]
        mask.put(&new_mask);

        // 4. explicit write descriptor for B2's write_kv_explicit.
        let w_slot_v = div(&wpos, PAGE_T);
        let w_off_v = rem(&wpos, PAGE_T);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        // physical KV span after this step's appends (all beams see the filled
        // prefix of the shared pool; the mask restricts attention).
        let filled = add(&base, B); // [1]
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        out_mask.put(&new_mask);
        out_wslot.put(&w_slot_v);
        out_woff.put(&w_off_v);
    });
    b.build().expect("design B beam epilogue binds")
}

fn beam_profile() -> ModelProfile {
    ModelProfile { vocab: V, page_size: PAGE_T, num_layers: 2, ..ModelProfile::dummy() }
}

fn u32s(v: &[u32]) -> Value {
    Value::U32(v.to_vec())
}

/// A `[B,POOL]` bool mask value with `positions` set true in each beam's row.
fn mask_of(rows: &[&[u32]]) -> Value {
    let mut m = vec![false; (B * POOL) as usize];
    for (b, positions) in rows.iter().enumerate() {
        for &p in *positions {
            m[b * POOL as usize + p as usize] = true;
        }
    }
    Value::Bool(m)
}

/// Craft a `[B,V]` logit block whose top-B over the flattened cand picks both
/// survivors from beam `parent_beam` with tokens `t0`,`t1`.
fn logits_forcing_parent(parent_beam: u32, t0: u32, t1: u32) -> PassInputs {
    let mut l = vec![0.0f32; (B * V) as usize];
    let row = (parent_beam * V) as usize;
    l[row + t0 as usize] = 20.0;
    l[row + t1 as usize] = 19.0;
    PassInputs { logits: Some(Value::F32(l)), ..Default::default() }
}

/// Force `parent = [0, 1]` (each beam continues itself): one dominant token per
/// row, so the top-B are beam0's peak + beam1's peak.
fn logits_diagonal(t0: u32, t1: u32) -> PassInputs {
    let mut l = vec![0.0f32; (B * V) as usize];
    l[t0 as usize] = 20.0; // beam 0 peak
    l[(V + t1) as usize] = 20.0; // beam 1 peak
    PassInputs { logits: Some(Value::F32(l)), ..Default::default() }
}

/// Harvested host-reader outputs of one committed Design B step (drains all of
/// channels 8..=13 so the next step's capacity-1 host-reader puts can land).
struct Harvest {
    tok: Value,
    par: Value,
    mask: Value,
    wslot: Value,
    woff: Value,
}

fn harvest(inst: &mut Instance, bound: &BoundTrace) -> Harvest {
    let tok = inst.host_take(bound, 8).unwrap();
    let par = inst.host_take(bound, 9).unwrap();
    let _scr = inst.host_take(bound, 10).unwrap();
    let mask = inst.host_take(bound, 11).unwrap();
    let wslot = inst.host_take(bound, 12).unwrap();
    let woff = inst.host_take(bound, 13).unwrap();
    Harvest { tok, par, mask, wslot, woff }
}

// ───────────────────────────────────────────────────────────────────────────
// GOLDEN — Design B fork from a shared prefix. Prompt occupies positions {0,1}
// (both beams share it). Both survivors fork from beam 0 (parent=[0,0]); their
// new tokens append at flat positions wpos=[2,3]. Expect the ancestry masks
// mask[0]={0,1,2}, mask[1]={0,1,3} (shared prefix {0,1} + own new cell), and the
// write descriptor w_slot=[0,0], w_off=[2,3] (positions 2,3 in pool page 0).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn golden_designb_fork_from_shared_prefix() {
    let traced = build_designb();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();

    // Seed: prompt of length 2 shared by both beams (positions {0,1}); fill=2.
    let seeds: Vec<(u32, Value)> = vec![
        (0, mask_of(&[&[0, 1], &[0, 1]])), // mask: both beams see the prompt
        (1, Value::F32(vec![0.0, 0.0])),   // scores
        (2, Value::I32(vec![1, 1])),       // toks
        (3, u32s(&[2, 2])),                // pos (logical depth after prompt)
        (4, u32s(&[2])),                   // fill (2 prompt positions filled)
        (5, u32s(&[2, 2])),                // klen
        (6, u32s(&[0, 0])),                // w_slot
        (7, u32s(&[0, 0])),                // w_off
    ];
    let mut inst = Instance::new(&bound, &seeds).unwrap();

    // Force both survivors from beam 0, tokens 2 and 3.
    let inputs = logits_forcing_parent(0, 2, 3);
    let r0 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(r0.committed, "no host-writer late edge in Design B ⇒ first fire commits: {:?}", r0.missed);

    let h = harvest(&mut inst, &bound);
    assert_eq!(h.par, u32s(&[0, 0]), "out_par: both fork from beam 0");
    assert_eq!(h.tok, Value::I32(vec![2, 3]), "out: tokens [2,3]");
    // The core Design B mechanism: mask evolution = shared prefix + own new cell.
    assert_eq!(
        h.mask,
        mask_of(&[&[0, 1, 2], &[0, 1, 3]]),
        "mask: beam0={{0,1,2}}, beam1={{0,1,3}} (shared {{0,1}} + own append)"
    );
    // Explicit write descriptor: positions 2,3 land in pool page 0 at offs 2,3.
    assert_eq!(h.wslot, u32s(&[0, 0]), "w_slot: positions 2,3 in pool page 0");
    assert_eq!(h.woff, u32s(&[2, 3]), "w_off: offsets 2,3 within page 0");
}

/// Design B across TWO steps: fork from a shared prefix, then each survivor
/// continues itself — the masks accumulate as independent ancestry paths, and
/// the flat append crosses a pool page boundary (page turn). Prompt {0,1}.
///   step 0: parent=[0,0] append wpos=[2,3] → masks {0,1,2},{0,1,3}, fill 4.
///   step 1: parent=[0,1] append wpos=[4,5] → masks {0,1,2,4},{0,1,3,5};
///           w_slot=[1,1], w_off=[0,1] (positions 4,5 are page 1 offs 0,1).
#[test]
fn golden_designb_multistep_pageturn() {
    let traced = build_designb();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    let seeds: Vec<(u32, Value)> = vec![
        (0, mask_of(&[&[0, 1], &[0, 1]])),
        (1, Value::F32(vec![0.0, 0.0])),
        (2, Value::I32(vec![1, 1])),
        (3, u32s(&[2, 2])),
        (4, u32s(&[2])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[0, 0])),
        (7, u32s(&[0, 0])),
    ];
    let mut inst = Instance::new(&bound, &seeds).unwrap();

    // step 0: fork both from beam 0.
    let r0 = inst.step(&bound, &logits_forcing_parent(0, 2, 3), &mut NoKernels).unwrap();
    assert!(r0.committed, "{:?}", r0.missed);
    let h0 = harvest(&mut inst, &bound);
    assert_eq!(h0.par, u32s(&[0, 0]));
    assert_eq!(h0.mask, mask_of(&[&[0, 1, 2], &[0, 1, 3]]));
    assert_eq!(h0.wslot, u32s(&[0, 0]));
    assert_eq!(h0.woff, u32s(&[2, 3]));

    // step 1: each survivor continues itself (parent=[0,1]); appends 4,5.
    let r1 = inst.step(&bound, &logits_diagonal(4, 5), &mut NoKernels).unwrap();
    assert!(r1.committed, "{:?}", r1.missed);
    let h1 = harvest(&mut inst, &bound);
    assert_eq!(h1.par, u32s(&[0, 1]), "each beam continues itself");
    assert_eq!(
        h1.mask,
        mask_of(&[&[0, 1, 2, 4], &[0, 1, 3, 5]]),
        "masks accumulate independent ancestry: beam0={{0,1,2,4}}, beam1={{0,1,3,5}}"
    );
    assert_eq!(h1.wslot, u32s(&[1, 1]), "page turn: positions 4,5 are in pool page 1");
    assert_eq!(h1.woff, u32s(&[0, 1]), "page-1 offsets 0,1");
}
