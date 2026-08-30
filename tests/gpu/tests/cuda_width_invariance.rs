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
//! # What was broken
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
    classify(&model::Request::new(query_len, false))
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
    let Ok(checkpoint) = common::resolve_qwen35_snapshot() else {
        eprintln!("skipping the width-invariance gate: no Qwen3.5-0.8B snapshot in the HF cache");
        return;
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let mut engine = open::cuda(b"[model]\ndevice = \"cuda:0\"\n").expect("the cuda seam opens");
    let budgets = Budgets {
        max_lanes: 16,
        max_tokens: 1024,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: 512,
        slots: 16,
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
    assert_eq!(request.trace.name, SKU);
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
    for width in [2usize, 4, 8, 16] {
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
    for width in [2usize, 4, 8, 16] {
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
