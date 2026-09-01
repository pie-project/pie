//! **A LANE'S LOGITS MUST NOT DEPEND ON HOW MANY LANES RODE WITH IT.**
//!
//! `test_curated.py`'s `greedy-decoding-is-the-same-alone-and-in-a-crowd`
//! states the property at the serving door: the same prompt at temperature 0
//! answers the same alone and 8 ways at once. That gate is autoregressive, so
//! when it fails it says only "the text diverged" — twenty-four steps after
//! whatever moved.
//!
//! This one is the same claim with the feedback loop cut. ONE fire, one
//! teacher-forced prompt, lane 0 identical in every submission — same slot,
//! same tokens, same positions, same pages, same `held` — and the only thing
//! that changes between submissions is HOW MANY OTHER LANES are in the fire.
//! Every active pass steps in lockstep waves by design, so lane 0's row of the
//! rectangle is a function of lane 0's inputs and of nothing else. Bit for
//! bit, or the wave is not a wave.
//!
//! # WHAT THIS GATE CAUGHT THAT IT WAS NOT WRITTEN FOR, AND WHAT FIXED IT
//!
//! This gate ran RED on the default configuration for a long time — since at
//! least `8c2aaa7c1`, verified by checking that commit out and reproducing it
//! byte for byte — and the failure was not the cuBLASLt reassociation the
//! section below describes:
//!
//! ```text
//!   [decode  width  2] max |delta logit| 11.046875 at 99129
//!                      argmax solo 248068 "<think>" -> crowd 3710 "What"
//!   [decode  width  3] 0.000000      [decode  width  4] 0.000000
//!   [decode  width  8] 0.000000      [decode  width 16] 0.000000
//!   every prefill width               0.000000
//! ```
//!
//! Eleven logits is not an accumulation order, one width is not a kernel
//! boundary, and `[engine] graphs = "off"` made it all zero — so it was a
//! REPLAYED BODY computing something other than what the eager walk computes,
//! and nothing in any kernel's arithmetic.
//!
//! **AND THE ONE THING THAT MOVED BETWEEN THE CAPTURE AND THE REPLAY WAS THE
//! LANE SPLIT.** Load-time arming captures the mixed key `b8[c0:8 c1:8]` on a
//! synthetic fire of seven one-row prefill lanes beside one decode lane, so
//! the decode class's window begins at lane 7. Width 2 replays that body with
//! ONE five-row prefill neighbour beside the probe, so the decode class's
//! window begins at lane 1. Everything the bodies design retargets — the row
//! offsets, the staged seat, the schedules, the FA2 lane ids and their
//! fire-wide page tables — moved with it. What did not was the shell's own
//! pointer arithmetic on the LANE axis: on this SSM hybrid the decode class
//! runs the per-STEP gated-delta scans, whose slot map, fold predicate and
//! commit length are handed over sliced at `lane_offset`
//! (`engine_cuda::run::Run::recurrent`) — an address a body bakes and a
//! `record::BodyKey` does not fix. The replay read lane 7's recurrent bank
//! for a decode sitting on lane 1: a wrong-but-real state, which is exactly
//! why the answer was coherent and wrong rather than garbage.
//!
//! `engine_cuda::SHIFTED` had said all along that its promise is about ROWS
//! and that the lane axis is a separate question with two earned exceptions;
//! what nothing did was ASK that question at the admissibility door.
//! `engine_cuda::LANE_SHIFTED` is it, `Windows::admits` spends it on one
//! clause — a region beginning above the fire's lane zero must find its own
//! lane — and the gdn decode region is an island on this SKU now. Widths 1
//! and 3+ were always clean for their own reasons: width 1 is single-class,
//! so its window begins at lane zero, and width 3+ moves the bucket and walks
//! eagerly.
//!
//! # AND A SECOND FAULT, UNDERNEATH THE FIRST: THE TILED PATH
//!
//! With `graphs = "off"` removing the confound above, three SKUs at widths
//! 2, 3, 4, 8, 16 against a solo baseline:
//!
//! ```text
//!   qwen35-d0.8b-bf16   (dense bf16)          all 0.000000, prefill + decode
//!   gptoss-20b-mxfp4    (routed, mxfp4)       all 0.000000, prefill + decode
//!   qwen35-d0.8b-mlxu4  (TILED, U4g64tiled)   prefill 0.156-0.164 at 2..16
//!                                             decode  0.000000 at 2,
//!                                                     0.187500 at 3,4,8,16
//!   qwen36-35b-a3b-mini (TILED **and ROUTED**) prefill 0.063 at 2,
//!                                                     0.102 at 3..16
//!                                             decode  0.000000 at 2,
//!                                                     0.078 at 3..16
//! ```
//!
//! Deterministic across runs. **The tiled affine path is not
//! batch-invariant**; the dense path and the routed mxfp4 path are.
//!
//! That is the shape the mac-engine session predicted from their own vector
//! point and named by file:line — `kernels_cuda::linear::tiled::carve_for`
//! picks `kSplit` 32 at `rows <= 8` and 16 above, and `split` PARTITIONS K, so
//! two fires that differ only in row count walk the contraction in different
//! orders. A tenth of a logit is two or three bf16 ulp, which is what an
//! accumulation-order difference looks like and is nothing like the eleven
//! logits the graph fault produces.
//!
//! # THE DANGEROUS COMPOSITION, RUN
//!
//! The worry was TILED **and** ROUTED: an mlxu4 MoE row, where top-k over
//! many experts could turn a ulp into a DIFFERENT EXPERT and that into whole
//! logits — the a4b-class failure mac-engine root-caused on their plane.
//! Neither full row fits a 46 GiB card, so they carved a miniature that does:
//! `qwen36-35b-a3b-mini-mlxu4-kv-bf16`, 5 layers, 8-of-16 experts, **K
//! untouched** (hidden 2048, expert intermediate 512), vocabulary full.
//!
//! ```text
//!   width      2      3      4      5      6      7      8      9     12     16
//!   prefill  .063   .102   .102   .102   .102   .102   .102   .102   .102   .102
//!   decode   .000   .078   .078   .078   .078   .078   .078   .078   .078   .078
//! ```
//!
//! **The same structure, exactly**: three prefill regimes, two decode
//! regimes, the one boundary between 2 and 3, everything from 3 to 16
//! identical. So the fault composes with routing and the miniature reproduces
//! it — which makes it the bisection vehicle, small enough to run per change.
//!
//! **AND THE ARGMAX SURVIVED, AT EVERY WIDTH.** Routing did not amplify it
//! here: the routed delta (0.078) is SMALLER than the same SKU family's
//! non-routed one (0.188), and no expert flipped. So the amplification that
//! makes this class catastrophic is demonstrated as POSSIBLE on this plane
//! and is not demonstrated as REALIZED. Two honest reasons it might not
//! have fired: 8-of-16 is a less contested tail than the 8-of-256 the
//! source ships, and five layers compound less than thirty-six.
//!
//! **THE FIRST WAS TESTED AND IS NOT THE REASON.** A second carve at
//! `--experts 64` — same K, same layers, a tail four times as crowded:
//!
//! ```text
//!   width      2      3      4      5      6      7      8      9     12     16
//!   prefill  .047   .065   .065   .065   .065   .065   .065   .065   .065   .065
//!   decode   .000   .078   .078   .078   .078   .078   .078   .078   .078   .078
//! ```
//!
//! Same structure, same boundary, argmax intact — and the decode delta is
//! **identical to the 16-expert carve's, 0.078125 at both**, while
//! prefill's is SMALLER at 64 than at 16. Crowding the tail fourfold
//! amplified nothing.
//!
//! So on this plane the fault does not compound through expert selection:
//! its size is insensitive to how many experts there are. That is a real
//! negative result about the a4b-class CONSEQUENCE and it is not a
//! clearance of the fault — a tenth of a logit is still a tenth of a logit,
//! and thirty-six layers were not tested.
//!
//! # A THIRD FAULT, FOUND VERIFYING THE FIX FOR THE FIRST
//!
//! `e8455cb25` fixed the graph-replay divergence above — a recurrent scan's
//! slot table baking lane-offset ADDRESSES into the capture — and the dense
//! SKU is now `0.000000` at every width on the DEFAULT configuration, which
//! is what that fix promised and what it delivers.
//!
//! Re-running the tiled+routed miniature at the default to check my own
//! findings were independent of it turned up something else:
//!
//! ```text
//!   graphs on (default)   the same ONE-LANE fire twice is not bit-identical
//!   graphs off            solo repeats exactly; only the width fault remains
//! ```
//!
//! Deterministic across runs, and it is not batch-invariance at all — it is
//! the same input, the same single lane, twice in one process, answering
//! differently. That is a stronger property broken than anything else in this
//! file, and this gate only catches it because it checks its own baseline
//! before reading anything below it.
//!
//! It is model-dependent: the dense 0.8B is clean at the default, the
//! tiled+routed mini is not. What the mini has that it does not is expert
//! routing (and this family's hybrid recurrent layers). Handed to the
//! cuda-graph session as a possible remainder of the same class their fix
//! addressed, on the model shape their fix did not have to hand.
//!
//! # WHAT THE NUMBERS EXCLUDE, WHICH IS EVERY CANDIDATE NAMED SO FAR
//!
//! Ten widths on the tiled SKU, graphs off, deterministic:
//!
//! ```text
//!   width      2      3      4      5      6      7      8      9     12     16
//!   prefill  .164   .156   .156   .156   .156   .156   .156   .156   .156   .156
//!   decode   .000   .188   .188   .188   .188   .188   .188   .188   .188   .188
//! ```
//!
//! **Two regimes at decode and three at prefill, and the only boundary in
//! range is between 2 and 3 rows.** Every width from 3 to 16 is identical to
//! the byte, at the same vocabulary entry. That excludes all three row-count
//! switches this kernel has:
//!
//! * **`carve_for` — excluded by MEASUREMENT.** `THIN_ROWS` is 8 and widths
//!   8 and 9 agree, straddling the switch from `THIN_SPLIT` (32) to
//!   `WIDE_SPLIT` (16) without moving.
//!
//!   An earlier version of this list also argued it out by arithmetic — "the
//!   delta is at a vocabulary entry, so `lm_head` carries it, and its split
//!   clamps to `MIN_SPLIT` at every row count". **The second clause is sound
//!   and the first is a tautology**: this gate measures logits, so its
//!   maximum is at a vocabulary entry by construction, whatever produced it.
//!   The clamp says something true about `lm_head` and nothing about where
//!   these numbers come from. The measurement is the exclusion; the
//!   arithmetic was scaffolding on a circle.
//! * **`bucket` — NOT excluded, and the first version of this list said it
//!   was.** It instantiates at 1, 2, 4, 8, 16, and buckets 4, 8 and 16 do all
//!   agree — but that shows the arithmetic is bucket-independent ABOVE two
//!   rows, not that bucket is irrelevant. **The observed boundary coincides
//!   exactly with its 2-to-4 step**, and the template is instantiated on
//!   `rows = bucket(y.rows)` with `Carve::smem(rows)` sized from it. Buckets
//!   1 and 2 agreeing while 4 and up agree separately is the signature of a
//!   specialization that changes shape at small row counts, which is what a
//!   template parameter on the row count can do. It is the surviving
//!   candidate.
//! * **`PREFILL_ROWS`** — 12 takes `matmul_gemv` and 16 takes `matmul`, two
//!   different kernels, and they agree to the byte.
//!
//! So two of the three are out — one by measurement and arithmetic together —
//! and the third is not the alibi the first version of this list gave it.
//! What survives is a template specialization on the bucketed row count at
//! the 2-to-4 step, somewhere in the tiled gemv path. WHICH projection
//! carries it is not established — a logit maximum cannot say — and the
//! five-SKU pattern only says the tiled path is involved. That is where a fix
//! should start, and it is not where the file:line handover pointed.
//!
//! # And a correction: this is attributed, not proven
//!
//! What is measured is that **the mlxu4 SKU's LOGITS depend on lane count**,
//! at a 2-vs-3 threshold. The logits are the whole forward pass. The tiled
//! projection is the leading candidate because the two SKUs that do not use
//! it — dense bf16 and routed mxfp4 — are clean at every width on the same
//! attention and the same scheduler, which is what makes it a statement about
//! the quantized projection path rather than about batching in general.
//!
//! But with all three of that path's row-count switches excluded, the
//! MECHANISM is not identified, and "the tiled path is not batch-invariant"
//! is one inference further than the numbers reach on their own.
//!
//! # What was broken BEFORE, and what this file was written for

//!
//! Every kernel this tree owns keeps its LIVE extent, and every one of them is
//! row-parallel: a row's arithmetic reads that row. One entry does not.
//! `linear::gemm::act_x_wt` hands its row count to cuBLASLt, whose
//! shape→algorithm heuristic reads M — and on a narrow M it buys parallelism
//! by CUTTING THE CONTRACTION, summing the pieces afterwards. Two fires that
//! differed only in who else rode along walked K in different orders, and a
//! bf16 accumulation in a different order is different numbers.
//!
//! Measured here at HEAD 506560b84, this gate's own prefill probe against the
//! same lane fired alone:
//!
//! ```text
//!   width  2   max |delta logit| 0.230469
//!   width  4   max |delta logit| 0.156250
//!   width  8   max |delta logit| 0.179688
//!   width 16   max |delta logit| 0.164062
//! ```
//!
//! Two to four ulp of bf16 per step — small enough that the argmax survived it
//! HERE, and not small enough to survive twenty-four autoregressive steps of
//! it, which is what the python gate saw. `Ctx::opaque_rows` (D4's padding)
//! only QUANTIZED the drift: two fires agreed iff they shared a lattice
//! bucket, and a lane alone and the same lane in a crowd of eight do not.
//! `linear::dense`'s family algorithm removes it instead — one cuBLASLt
//! algorithm per WEIGHT, pinned to a split-free K walk, chosen without ever
//! being told this fire's width.
//!
//! # What this gate covers that its name does not say
//!
//! A weight may now carry TWO algorithms: the family's, chosen at
//! `dense::FAMILY_ROWS`, and a second one that serves the narrow fires and is
//! faster there. The second is admitted only after `dense::settle_small`
//! shows it lands the first's bits on this device, and this gate is where
//! that claim is worth something — the widths below step the row count across
//! the boundary, so lane 0's row is computed by the second algorithm in some
//! submissions and by the family's in others. The padded row counts this
//! file's five widths produce are 8, 16, 32, 64, 128 and 256; where the
//! handover falls inside that is per weight and measured, so this gate does
//! not name it — it asserts the only thing that matters, which is that
//! nobody can tell from the numbers which side of it a fire landed on.
//!
//! ```text
//! PIE_COMPILER_LAUNCHER=env cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_width_invariance -- --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

mod common;

use engine::{
    Budgets, FrameSubmission, KvDelta, Lane, LaneReadout, Readout, RsReset, RsVerb, Step,
};
use model_ir::Platform;
use runtime::engine::backend::Graphs;
use runtime::engine::backend::open;

/// The catalog row this gate serves.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// Lane 0's prompt — the probe whose row must not move.
const PROMPT: &str = "Explain why the sky appears blue.";

/// The neighbours' prompts, deliberately RAGGED: a fire whose lanes all carry
/// the same row count is the one composition a width-dependent reduction is
/// most likely to survive by accident.
const NEIGHBOURS: [&str; 15] = [
    "The capital of France is",
    "Name the largest planet in the solar system, and say why it is the largest.",
    "banana",
    "Write a haiku about a stone that has been sitting in a river for a very long time.",
    "One two three four five",
    "Kilimanjaro",
    "What colour is the sky on a clear day?",
    "A list: alpha, beta, gamma, delta, epsilon, zeta, eta, theta, iota, kappa.",
    "Hello",
    "Describe, in a single careful paragraph, how a compiler lowers a loop.",
    "Seven",
    "The quick brown fox jumps over the lazy dog and keeps on going for a while.",
    "Why?",
    "Recite the first several digits of pi as far as you can remember them.",
    "ok",
];

/// Pages per lane. Two 16-token pages is more context than any prompt here
/// needs; what matters is that each lane states its OWN pages, so no two lanes
/// write the same cells.
const PAGES_PER_LANE: u32 = 2;

/// The lane word the model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    let classify = runtime::engine::load::classify(SKU).expect("this build ships the gate's SKU");
    classify(&models::Request::new(query_len, false))
}

/// One teacher-forced lane, seated on its own slot over its own pages.
///
/// `held == 0` is a prefill — `RsReset::Inferred` reads it as "this sequence
/// begins", so the same submission twice is the same computation over the same
/// cells rather than a continuation. `held > 0` is the decode arm: one token
/// appended to the rows a previous fire wrote.
fn seated(slot: u32, held: u32, tokens: &[u32]) -> Lane {
    let pages = (slot * PAGES_PER_LANE..(slot + 1) * PAGES_PER_LANE).collect::<Vec<u32>>();
    Lane {
        slot,
        word: word(tokens.len() as u32),
        tokens: tokens.to_vec(),
        positions: Vec::new(),
        kv: KvDelta {
            held,
            pages,
            ..KvDelta::default()
        },
        mask: None,
        adapter: None,
        drafts: false,
        captures_scores: false,
        rs: RsVerb::Fold,
        rs_reset: RsReset::Inferred,
        channels: Vec::new(),
        readout: Readout::Last,
    }
}

/// A fire `width` lanes wide, carrying `probe` on lane 0 and prefilling
/// neighbours on the slots above it.
fn fire(probe: Lane, neighbours: &[Vec<u32>], width: usize) -> FrameSubmission {
    let mut lanes = vec![probe];
    for (at, tokens) in neighbours.iter().take(width - 1).enumerate() {
        lanes.push(seated(at as u32 + 1, 0, tokens));
    }
    FrameSubmission::of(Step {
        lanes,
        attachments: Vec::new(),
        media: Vec::new(),
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// Largest absolute difference between two rows, and where.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = (0.0f32, 0usize);
    for (at, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 {
            worst = (d, at);
        }
    }
    worst
}

#[test]
fn lane_zeros_logits_do_not_move_when_the_fire_gets_wider() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the width-invariance gate: no CUDA device on this machine");
        return;
    }
    // **AND THE SKU IS OVERRIDABLE, BECAUSE THE DEFAULT ONE IS THE PATH
    // ALREADY FIXED.** This gate has always run a DENSE bf16 Qwen3.5-0.8B —
    // which is `linear::gemm::act_x_wt`, the entry the header above says was
    // broken and is now batch-invariant. Nothing here has ever named a
    // QUANTIZED, TILED or ROUTED row, and those have their own row-count
    // boundaries: `kernels_cuda::linear::tiled::carve_for` picks `kSplit` 32
    // at `rows <= 8` and 16 above, and `split` partitions K — the same
    // reassociation one axis over — and `dispatch::linear` has two more
    // `act.rows >= PREFILL_ROWS` boundaries documented as reassociating.
    //
    // Handed over by the mac-engine session, which found exactly this shape on
    // their vector point: two kernels chosen by ROW COUNT, summing the same
    // products in different partial-sum orders, one bf16 ulp per ~8k elements
    // — invisible on a dense row and worth a different EXPERT once top-k
    // routing over 128 experts multiplies it.
    //
    // `PIE_WIDTH_INVARIANCE_SNAPSHOT` names another checkpoint to run it
    // against; a routed one is the interesting case.
    let checkpoint = match std::env::var("PIE_WIDTH_INVARIANCE_SNAPSHOT") {
        Ok(named) => named,
        Err(_) => match common::resolve_qwen35_snapshot() {
            Ok(found) => found,
            Err(_) => {
                eprintln!(
                    "skipping the width-invariance gate: no Qwen3.5-0.8B snapshot in the \
                     HF cache and no PIE_WIDTH_INVARIANCE_SNAPSHOT"
                );
                return;
            }
        },
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    // `PIE_WI_DEVICE` picks another card (`"1"` or `"cuda:1"`); the default is
    // ordinal 0 — the successor of the boot-document override this gate
    // carried while the seam took TOML.
    let ordinal = std::env::var("PIE_WI_DEVICE")
        .map(|d| runtime::engine::backend::ordinal_of(&d))
        .unwrap_or(0);
    // **AND `PIE_WI_GRAPHS` COMES BACK, TYPED.** The boot-document override
    // this gate carried took a whole `[engine]` table, and the seam's move to
    // a struct kept the card knob and dropped this one — which is the knob
    // that did the work. Turning capture off is how the two faults this file
    // records were SEPARATED: with it off the dense SKU's eleven-logit swing
    // vanished and the tiled path's tenth-of-a-logit stayed, which is the
    // whole reason they are known to be two faults and not one.
    //
    // `Graphs` parses the same three spellings the boot key always took, so
    // this is the old override's one useful degree of freedom in the type the
    // seam now speaks. Default is the shell's own, which is `On` — a gate that
    // silently ran eager would be measuring a configuration nobody deploys.
    let graphs = match std::env::var("PIE_WI_GRAPHS") {
        Ok(named) => named.parse().expect("PIE_WI_GRAPHS is on, shaped, or off"),
        Err(_) => Graphs::default(),
    };
    let mut engine = open::cuda(runtime::engine::backend::DeviceBoot {
        ordinal,
        graphs,
        ..Default::default()
    })
    .expect("the cuda seam opens");
    let budgets = Budgets {
        max_lanes: 16,
        max_tokens: 1024,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: 512,
        slots: 16,
        max_patches: None,
        max_images: None,
    };
    let request = runtime::engine::load::request(
        &checkpoint,
        Platform::Cuda,
        budgets,
        engine::Residency::uncapped(),
        0,
        1,
    )
    .expect("the checkpoint identifies and its SKU traces");
    // The default snapshot pins its SKU, because a gate that silently ran a
    // different model would be asserting invariance about something nobody
    // chose. An OVERRIDDEN snapshot names whatever it identifies as, and says
    // so: picking the row is the point of the override.
    match std::env::var("PIE_WIDTH_INVARIANCE_SNAPSHOT") {
        Err(_) => assert_eq!(request.trace.name, SKU),
        Ok(_) => eprintln!("[width-invariance] running {}", request.trace.name),
    }
    let loaded = engine.load(request).expect("the checkpoint lands");
    let vocab = loaded.caps.profile.vocab as usize;

    let probe = tokenizer.encode(PROMPT);
    let neighbours: Vec<Vec<u32>> = NEIGHBOURS.iter().map(|p| tokenizer.encode(p)).collect();

    fn read(
        engine: &mut dyn engine::Engine,
        probe: Lane,
        neighbours: &[Vec<u32>],
        width: usize,
    ) -> LaneReadout {
        let submission = fire(probe, neighbours, width);
        let mut ticket = engine
            .submit(&submission)
            .unwrap_or_else(|why| panic!("the width-{width} fire: {why}"));
        engine
            .settle_frame(&mut ticket)
            .unwrap_or_else(|why| panic!("the width-{width} fire settles: {why}"));
        ticket.steps[0].readouts[0].clone()
    }

    // ── CLAIM 1: A PREFILLING PROBE.
    //
    // The reference is taken twice, because a claim about width is only
    // available once the same fire twice is established as identical.
    let prefill = || seated(0, 0, &probe);
    let solo = read(engine.as_mut(), prefill(), &neighbours, 1);
    let solo_again = read(engine.as_mut(), prefill(), &neighbours, 1);
    assert_eq!(solo.values.len(), vocab);
    assert_eq!(
        solo.values, solo_again.values,
        "the same one-lane fire twice is not bit-identical, so nothing below can be read"
    );
    let first = argmax(&solo.values);
    eprintln!(
        "[prefill width  1] argmax {first} -> {:?}",
        tokenizer.decode(&[first], false)
    );

    let mut failures = Vec::new();
    // **1 AND 3 BESIDE 2 AND 4, WHICH IS THE SIGNATURE AND NOT A WIDER
    // SWEEP.** A kernel chosen by row count divides the batch to pick its
    // arm, so an ODD width and an EVEN one take different arms and a
    // DIVISIBILITY signature — 1,3 clean while 2,4 dirty — says "two kernels"
    // where a geometry fault would dirty all four. That is how the mac-engine
    // session separated their two `affine_qmv` arms from a window bug.
    //
    // Width 1 is NOT in the list: it is the solo baseline itself, so comparing
    // it would be comparing a read to itself. 3 is the odd probe.
    // The ten the header's table was measured at. 5,6,7,9,12 are not padding:
    // they are what excluded `bucket` and `carve_for`, because a group that
    // straddles a switch and does not move is what acquits it.
    for width in [2usize, 3, 4, 5, 6, 7, 8, 9, 12, 16] {
        let crowd = read(engine.as_mut(), prefill(), &neighbours, width);
        report("prefill", &solo, &crowd, width, &tokenizer, &mut failures);
    }

    // ── CLAIM 2: A DECODING PROBE BESIDE PREFILLING NEIGHBOURS.
    //
    // The composition the serving door actually produces, and the one
    // `test_curated.py`'s greedy gate rides: lane 0 carries ONE row and its
    // neighbours carry a prompt each, so the fire's total rows — and every
    // shape the linear layers are handed — move with the crowd while the
    // probe's own row does not.
    //
    // The seating fire below is always one lane wide and always the same
    // prompt at `held = 0`, so slot 0's cache is the same bytes before every
    // measurement; the probe then appends its first greedy token at `held =
    // n`.
    let held = probe.len() as u32;
    let step = vec![first];
    let decode = |engine: &mut dyn engine::Engine, width: usize| -> LaneReadout {
        let _ = read(engine, prefill(), &neighbours, 1);
        read(engine, seated(0, held, &step), &neighbours, width)
    };
    let solo = decode(engine.as_mut(), 1);
    let solo_again = decode(engine.as_mut(), 1);
    assert_eq!(
        solo.values, solo_again.values,
        "the same one-lane decode twice is not bit-identical, so nothing below can be read"
    );
    eprintln!(
        "[decode  width  1] argmax {} -> {:?}",
        argmax(&solo.values),
        tokenizer.decode(&[argmax(&solo.values)], false)
    );
    // **1 AND 3 BESIDE 2 AND 4, WHICH IS THE SIGNATURE AND NOT A WIDER
    // SWEEP.** A kernel chosen by row count divides the batch to pick its
    // arm, so an ODD width and an EVEN one take different arms and a
    // DIVISIBILITY signature — 1,3 clean while 2,4 dirty — says "two kernels"
    // where a geometry fault would dirty all four. That is how the mac-engine
    // session separated their two `affine_qmv` arms from a window bug.
    //
    // Width 1 is NOT in the list: it is the solo baseline itself, so comparing
    // it would be comparing a read to itself. 3 is the odd probe.
    // The ten the header's table was measured at. 5,6,7,9,12 are not padding:
    // they are what excluded `bucket` and `carve_for`, because a group that
    // straddles a switch and does not move is what acquits it.
    for width in [2usize, 3, 4, 5, 6, 7, 8, 9, 12, 16] {
        let crowd = decode(engine.as_mut(), width);
        report("decode ", &solo, &crowd, width, &tokenizer, &mut failures);
    }

    assert!(
        failures.is_empty(),
        "lane 0 was the same submission in every fire — same slot, tokens, positions, pages \
         and `held` — so its row of the rectangle is a function of its own inputs alone. \
         Every active pass steps in lockstep waves, and a wave whose arithmetic reads the \
         wave's WIDTH is not one:\n  {}",
        failures.join("\n  ")
    );
}

/// One width's verdict, printed and — if it moved — collected.
fn report(
    class: &str,
    solo: &LaneReadout,
    crowd: &LaneReadout,
    width: usize,
    tokenizer: &tokenizer::Tokenizer,
    failures: &mut Vec<String>,
) {
    assert_eq!(crowd.values.len(), solo.values.len());
    let (delta, at) = worst(&solo.values, &crowd.values);
    let token = argmax(&crowd.values);
    eprintln!(
        "[{class} width {width:>2}] max |delta logit| {delta:.6} at {at}  argmax {token} -> {:?}",
        tokenizer.decode(&[token], false)
    );
    if crowd.values != solo.values {
        failures.push(format!(
            "{class} at width {width}: lane 0's logits moved — max |delta| {delta:.6} at \
             vocab entry {at} (solo {:.6}, crowd {:.6}); argmax solo {} crowd {token}",
            solo.values[at],
            crowd.values[at],
            argmax(&solo.values),
        ));
    }
}
