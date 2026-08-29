//! **`copy_kv` MOVES REAL BYTES**: a lane's KV is grafted onto fresh pages and
//! a decode against the COPIED layout says, bit for bit, what a decode against
//! the original says.
//!
//! # What was broken, and what a weaker gate would have missed
//!
//! `Engine::copy_kv` took the trait's default body on the CUDA shell, so the
//! `prefix-tree-kv-cache` inferlet died at its first fork with
//! `pre-launch KV copy rejected: the cuda engine does not serve` copy_kv``.
//! The shell has a mover now ([`engine_cuda::store::Pools::copy_kv`]), and the
//! thing worth gating about a mover is not that it returns `Ok` — a body that
//! did nothing at all would do that — but that the pages it wrote are
//! ATTENDABLE. So this gate never inspects a page. It forks a sequence and
//! asks the model.
//!
//! # The shape, and why every piece of it is load-bearing
//!
//! ```text
//!   frame 1   lane slot 0, pages [0, 1], 20 tokens        the parent prefill
//!   copy_state  slot 0 -> slot 1                          the fork's history
//!   copy_kv     page 0 -> page 4        (whole page)      the fork's pages
//!               page 1 cells 0..4 -> page 5 cells 0..4    (the partial tail)
//!   frame 2   lane 0: slot 0, pages [0, 1], held 20       the control decode
//!             lane 1: slot 1, pages [4, 5], held 20       the forked decode
//! ```
//!
//! * **The pages are the CALLER'S** (`KvDelta::pages` non-empty), because a
//!   fork is the runtime keeping its own page table and the whole verb is
//!   about page ids it minted. A lane with the shell's own block-per-slot
//!   paging could not name page 4.
//! * **Twenty tokens at `page_size = 16`** so the parent spans TWO pages and
//!   the second is PARTIAL. That is the case the token-granular half of the
//!   contract exists for: a whole-page move for page 0, and four `KvMove`
//!   cells for the live tokens of page 1. A gate whose prompt fit in one page
//!   would only ever exercise the page half.
//! * **`copy_state` beside `copy_kv`**, because this checkpoint is a HYBRID:
//!   `qwen35-d0.8b` carries eighteen attention layers over the kv pages and
//!   eighteen gated-delta layers over the recurrent slabs. A fork that copied
//!   only the pages would leave the forked lane folding somebody else's
//!   history, and the logits would differ for a reason that has nothing to do
//!   with this gate's claim.
//! * **ONE FRAME, TWO LANES**, and this is the part that makes the assertion
//!   an identity rather than a plausibility. Both decodes ride one walk, one
//!   bucket, one set of kernels; the ONLY thing that differs between them is
//!   which page ids their attention gathers through. So a difference in the
//!   logits is a difference in the bytes under those ids, and nothing else —
//!   which is why the comparison is bit-for-bit and not approximate.
//!
//! # Gating
//!
//! Skips at RUN time rather than being `#[ignore]`d, saying which of the two
//! things it was missing (a device, or the checkpoint on disk).
//!
//! ```text
//! cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_kv_page_graft -- --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

mod common;

use engine_api::model_ir::Platform;
use engine_api::{
    Budgets, FrameSubmission, KvCopy, KvDelta, KvMove, Lane, MemoryDomain, Readout,
    RsReset, RsVerb, StateCopy, StateMove, Step,
};
use runtime::engine::backend::open;

/// The catalog row this gate serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Tokens per KV page. Small on purpose: the point of this gate is a prompt
/// that spans more than one page and leaves the last one partial.
const PAGE_SIZE: u32 = 16;

/// How many tokens the parent holds. `20 = 16 + 4`, so page 0 is whole and
/// page 1 carries four live cells.
const HELD: u32 = 20;

/// The parent's page ids, and the fork's. Both are the CALLER's numbering —
/// this gate is the runtime's side of article 8 — and they are disjoint so a
/// mover that wrote to the wrong end would be caught rather than aliased.
const PARENT: [u32; 2] = [0, 1];
const FORK: [u32; 2] = [4, 5];

/// The lane word the model's own `Classify` computes, reached the way the fire
/// path reaches it.
fn word(query_len: u32) -> u64 {
    let classify = runtime::engine::load::classify(SKU).expect("this build ships the gate's SKU");
    classify(&model::Request::new(query_len, false))
}

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// One lane, spelled once, because this gate builds five of them and the four
/// axes it never uses would otherwise be written out five times.
fn lane(slot: u32, pages: &[u32], held: u32, tokens: Vec<u32>, readout: Readout) -> Lane {
    Lane {
        slot,
        word: word(tokens.len() as u32),
        tokens,
        positions: Vec::new(),
        kv: KvDelta {
            held,
            pages: pages.to_vec(),
        },
        mask: None,
        adapter: None,
        drafts: false,
        captures_scores: false,
        rs: RsVerb::Fold,
        rs_reset: RsReset::Inferred,
        channels: Vec::new(),
        readout,
    }
}

#[test]
fn a_grafted_page_run_decodes_exactly_as_the_run_it_was_copied_from() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the kv graft gate: no CUDA device on this machine");
        return;
    }
    let Ok(checkpoint) = common::resolve_qwen35_snapshot() else {
        eprintln!("skipping the kv graft gate: no Qwen3.5-0.8B snapshot in the HF cache");
        return;
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    // 1. THE LOAD, through the same door the runtime uses.
    let mut engine = open::cuda(b"[model]\ndevice = \"cuda:0\"\n").expect("the cuda seam opens");
    let budgets = Budgets {
        max_lanes: 4,
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // vocabulary-wide logit column.
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: PAGE_SIZE,
        max_context: 512,
        slots: 4,
    };
    let request = runtime::engine::load::request(
        &checkpoint,
        Platform::Cuda,
        budgets,
        engine_api::Residency::uncapped(),
        0,
        1,
    )
    .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(request.trace.name, SKU);
    let loaded = engine.load(request).expect("the checkpoint lands");

    // 2. **THE CAPABILITY IS STATED, NOT DISCOVERED.** A caller reads which
    //    directions are served before it plans a fork; the three this shell
    //    has no storage for say so here rather than by refusing a plan.
    let served = loaded.caps.kv_copy;
    assert!(
        served.device_to_device,
        "the cuda shell moves cells between pages of its own pools"
    );
    assert!(
        !served.device_to_host && !served.host_to_device && !served.host_to_host,
        "and it reserves no pinned swap pool, so it says so: {served:?}"
    );

    // And the refusal is BY NAME rather than by a silent nothing: a plan whose
    // ends are host-pinned names storage this load does not hold.
    let host_to_host = engine.copy_kv(&KvCopy::default());
    assert!(
        matches!(
            host_to_host,
            Err(engine_api::Error::Unsupported { engine: "cuda", .. })
        ),
        "a host-pinned copy plan is refused by name, not served: {host_to_host:?}"
    );

    // 3. THE PARENT. Twenty tokens over the caller's own two pages.
    let prompt = tokenizer.encode("The capital of France is");
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");
    let tokens: Vec<u32> = prompt
        .iter()
        .copied()
        .cycle()
        .take(HELD as usize)
        .collect();
    let prefill = FrameSubmission::of(Step {
        lanes: vec![lane(0, &PARENT, 0, tokens.clone(), Readout::Last)],
        attachments: Vec::new(),
    });
    prefill.validate().expect("the parent prefill is well formed");
    let mut ticket = engine.submit(&prefill).expect("the parent prefill fires");
    engine
        .settle_frame(&mut ticket)
        .expect("and its numbers come back");
    let next = argmax(&ticket.steps[0].readouts[0].values);
    eprintln!(
        "parent's next token: {:?}",
        tokenizer.decode(&[next], false)
    );

    // 4. THE FORK, both halves of it.
    //
    //    The recurrent half first, because a hybrid's fork is two stores and
    //    a gate that copied one would be measuring the other's drift.
    engine
        .copy_state(&StateCopy {
            moves: vec![StateMove {
                src_slot_id: 0,
                dst_slot_id: 1,
                src_token_offset: 0,
                dst_token_offset: 0,
                token_count: 0,
            }],
        })
        .expect("the recurrent banks fork");

    //    Then the pages: one WHOLE page and one PARTIAL one, which is exactly
    //    the two halves `KvCopy` states apart. The four cells are named one
    //    `KvMove` each, the way a prefix tree names them; the shell coalesces
    //    the run behind this seam and the gate does not care that it does.
    let live_in_tail = HELD - PAGE_SIZE;
    let device = MemoryDomain::CudaDevice(0);
    let graft = KvCopy {
        src: device,
        dst: device,
        src_page_ids: vec![PARENT[0]],
        dst_page_ids: vec![FORK[0]],
        moves: (0..live_in_tail)
            .map(|at| KvMove {
                src_page_id: PARENT[1],
                src_token_offset: at,
                dst_page_id: FORK[1],
                dst_token_offset: at,
            })
            .collect(),
    };
    graft.validate().expect("the graft is a plan the contract describes");
    engine.copy_kv(&graft).expect("the pages graft");

    // 5. **ONE FRAME, TWO LANES, ONE WALK.** The control decode and the forked
    //    decode differ in their slot and their page ids and in nothing else —
    //    same token, same word, same bucket, same kernels — so the comparison
    //    below is an identity.
    let decode = FrameSubmission::of(Step {
        lanes: vec![
            lane(0, &PARENT, HELD, vec![next], Readout::Last),
            lane(1, &FORK, HELD, vec![next], Readout::Last),
        ],
        attachments: Vec::new(),
    });
    decode.validate().expect("the paired decode is well formed");
    let mut ticket = engine.submit(&decode).expect("the paired decode fires");
    engine
        .settle_frame(&mut ticket)
        .expect("and its numbers come back");

    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 2, "two lanes in, two readouts out");
    let control = &readouts[0];
    let forked = &readouts[1];
    assert_eq!(control.rows, 1, "`Readout::Last` is one row");
    assert_eq!(forked.rows, 1);
    assert_eq!(
        control.width, loaded.caps.profile.vocab,
        "a logits row is the vocabulary wide"
    );
    assert!(
        control.values.iter().all(|v| v.is_finite()),
        "the control decode's logits are finite"
    );
    assert!(
        forked.values.iter().all(|v| v.is_finite()),
        "and so are the forked decode's — a page read at the wrong stride \
         shows up here first"
    );

    // **BIT FOR BIT.** Not `abs() < eps`: the two lanes ran one walk with one
    // set of kernels, so any difference at all is a difference in the bytes
    // the graft wrote.
    let first = control
        .values
        .iter()
        .zip(&forked.values)
        .position(|(a, b)| a != b);
    assert!(
        first.is_none(),
        "a decode against the grafted pages {FORK:?} disagrees with one against \
         the pages they were copied from {PARENT:?}, first at vocabulary index {} \
         ({} against {}) — the graft moved the wrong bytes, or none",
        first.unwrap_or(0),
        control.values[first.unwrap_or(0)],
        forked.values[first.unwrap_or(0)],
    );
    eprintln!(
        "grafted continuation: {:?}",
        tokenizer.decode(&[argmax(&forked.values)], false)
    );
}
