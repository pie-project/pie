//! **`Readout::Rows` NAMES INTERIOR ROWS AND GETS THEM**, in the order it
//! named them.
//!
//! # What was broken
//!
//! The fire path refused every lane that asked for anything but its last row:
//! `direct launch rejected: the cuda engine does not serve` row-selected
//! readout``, which took `cacheback-speculative-decoding` and every
//! spec-decode verify inferlet with it — a verifier's whole job is to read the
//! logits of the rows it teacher-forced, not the one row a sampler wants.
//!
//! The rectangle was never the problem. The carve holds the out seam open past
//! the last node (that is what makes ANY readback possible), and the shell
//! already indexes a lane's row run by its first row and its length, because
//! that is how the capture columns are read. What was missing was the row
//! LIST: `Readout` is a submission word, and the read-back loop had never been
//! told one.
//!
//! # Three fires of ONE composition, which is what makes this exact
//!
//! Every fire below is the same prefill — same tokens, same slot, the same
//! stated pages, `held = 0`, the recurrent banks cleared by the same
//! `RsReset::Inferred` rule — so all three compute the same logits rectangle,
//! in the same bucket, through the same kernels. Only the `Readout` differs.
//!
//! **THE PAGES ARE STATED AND `held` IS STATED WITH THEM**, which is the whole
//! reason this is repeatable. A lane whose `KvDelta::pages` is EMPTY hands the
//! page table to the shell, and the shell then owns the token count too: it
//! advances `held` per slot at enqueue and reads its own counter on the next
//! fire, so a second "identical" submission is not a second prefill — it is a
//! continuation writing at positions `n..2n`. Stating the pages puts the count
//! back in the submission (`KvDelta::held` is only read for a lane that brought
//! its own table), and three submissions that state `held = 0` are three
//! prefills of the same tokens over the same cells.
//!
//! ```text
//!   fire 1   Readout::Rows([0, n-2, n-1])   three rows
//!   fire 2   Readout::Last                  one row: n-1, the shell's own
//!   fire 3   Readout::Rows([n-1, n-2, 0])   the same three, reversed
//! ```
//!
//! and the three claims they make together:
//!
//! 1. **The shape.** Three rows come back, each the vocabulary wide, and the
//!    values are `3 * vocab` of them — `LaneReadout`'s own contract.
//! 2. **The anchor.** Fire 1's THIRD row is fire 2's only row, bit for bit.
//!    That pins the arithmetic — `first_row + index` — against the one row
//!    this shell has always known how to find, so a gather that landed a row
//!    early or a row late fails here.
//! 3. **The order, and that the rows are distinct.** Fire 3 is fire 1's rows
//!    reversed, and the values come back reversed to match: the contract says
//!    "row-major, in the order of the requested list", so a shell that sorted
//!    or deduplicated the list would return the right numbers under the wrong
//!    names. Rows 0 and n-2 differing from row n-1 is what rules out the other
//!    failure — three copies of the last row, which claims 1 and 2 would both
//!    accept.
//!
//! And one refusal: a row index past the rows the lane carries is `Invalid`
//! before anything launches, because a readout that ran off the end of its
//! lane would hand back a neighbour's logits and nothing about them looks
//! wrong.
//!
//! ```text
//! cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_row_selected_readout -- --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

mod common;

use engine_api::model_ir::Platform;
use engine_api::tensor_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use engine_api::tensor_ir::container::HostRole;
use engine_api::tensor_ir::op::{IntrinsicId, Op};
use engine_api::tensor_ir::registry::{GeometryClass, ModelProfile, Stage};
use engine_api::tensor_ir::types::{DType, Shape};
use engine_api::{
    Attachment, BindExtents, Boundary, Budgets, FrameSubmission, InstanceBinding, KvDelta, Lane,
    LaneReadout, Readout, RsReset, RsVerb, Step,
};
use runtime::engine::backend::open;

/// The guest's one channel: the token it chose per readout row.
const OUT: u32 = 0;

/// **THE DEVICE HALF'S GUEST**: read the readout rectangle, argmax each row,
/// publish the row of tokens.
///
/// ```text
///   logits(k, vocab) ──▶ argmax(per row) ──▶ chan_put(out)  [k] i32
/// ```
///
/// No channel in, no descriptor port: this pass states no geometry of its own
/// (`GeometryClass::Host`) and reads nothing but the rectangle the model fire
/// produced, which is the whole point — what it publishes is a function of
/// WHICH ROWS the shell pointed `IntrinsicId::Logits` at, and of nothing else.
///
/// `ReduceArgmax` over a `[k, vocab]` value drops the last axis and answers
/// `[k]` i32, one token per row, so the guest's list is directly comparable
/// with the host mirror's rows one for one.
///
/// Authored here rather than taken from the corpus for the reason
/// `runtime/tests/cuda_program_epilogue.rs` gives about its own: a trace's
/// `logits` intrinsic carries a CONCRETE shape, so a golden authored at the
/// dummy profile's vocabulary cannot bind at this checkpoint's.
fn program(rows: u32, vocab: u32) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![ChannelDecl {
            shape: Shape::vector(rows),
            dtype: ChanDType::Concrete(DType::I32),
            capacity: 1,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(rows, vocab),
                    dtype: DType::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut {
                    chan: OUT,
                    value: 1,
                },
            ],
        }],
    }
}

/// [`program`], all the way to what `Engine::register_program` takes.
///
/// The emitted kernels are attached HERE rather than by the scheduler lane's
/// splice, because this gate holds the engine directly and so plays that part
/// too.
fn registration(rows: u32, profile: &ModelProfile) -> engine_api::ProgramRegistration {
    let bound = engine_api::tensor_ir::validate::bind(program(rows, profile.vocab), profile.clone())
        .unwrap_or_else(|why| panic!("the gate's program does not bind: {why:?}"));
    let stages = tensor_compiler::plan::compile_bound(&bound);
    let launch = tensor_compiler::codegen::launch::build(&bound, &stages);
    let backend = tensor_compiler::codegen::program::Backend::Cuda;
    let emitted = tensor_compiler::codegen::program::emit_program(backend, &stages, &bound);
    for kernel in &emitted {
        assert!(
            kernel.error.is_empty(),
            "the gate's program did not emit: {}",
            kernel.error
        );
    }
    engine_api::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted
            .into_iter()
            .map(|kernel| engine_api::EmittedKernel {
                kind: kernel.kind,
                stage_index: kernel.stage_index,
                region_index: kernel.region_index,
                entry_name: kernel.entry_name,
                source: kernel.source,
                error: kernel.error,
            })
            .collect(),
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

/// The catalog row this gate serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt, and the reason it is this one: its greedy continuation is a
/// single well-known token, so the last row can be checked against a value
/// three other gates in this tree already pin.
const PROMPT: &str = "The capital of France is";

/// What the last row's argmax decodes to.
const EXPECTED: &str = " Paris";

/// The lane's stated page table. Two pages of sixteen tokens is more than this
/// prompt needs; what matters is that the table is the CALLER'S, so `held` is
/// the submission's word rather than a counter the shell carries between
/// fires.
const PAGES: [u32; 2] = [0, 1];

/// The lane word the model's own `Classify` computes.
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

/// The one prefill this gate fires three times, differing only in `readout`.
///
/// `held = 0` over a STATED page table, so the count is the submission's and
/// not the shell's running total, and `RsReset::Inferred` reads `held == 0` as
/// "this sequence begins" — each fire clears the recurrent banks and rewrites
/// the same cells of the same pages. Three identical computations, by
/// construction rather than by hope.
fn prefill(tokens: &[u32], readout: Readout) -> FrameSubmission {
    attached_prefill(tokens, readout, Vec::new())
}

/// [`prefill`], with a guest pass hung off the fire's epilogue.
fn attached_prefill(
    tokens: &[u32],
    readout: Readout,
    attachments: Vec<Attachment>,
) -> FrameSubmission {
    FrameSubmission::of(Step {
        lanes: vec![Lane {
            slot: 0,
            word: word(tokens.len() as u32),
            tokens: tokens.to_vec(),
            positions: Vec::new(),
            kv: KvDelta {
                held: 0,
                pages: PAGES.to_vec(),
            },
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            rs: RsVerb::Fold,
            rs_reset: RsReset::Inferred,
            channels: Vec::new(),
            readout,
        }],
        attachments,
    })
}

#[test]
fn a_lane_reads_back_the_interior_rows_it_names_in_the_order_it_names_them() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the row-readout gate: no CUDA device on this machine");
        return;
    }
    let Ok(checkpoint) = common::resolve_qwen35_snapshot() else {
        eprintln!("skipping the row-readout gate: no Qwen3.5-0.8B snapshot in the HF cache");
        return;
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let mut engine = open::cuda(b"[model]\ndevice = \"cuda:0\"\n").expect("the cuda seam opens");
    let budgets = Budgets {
        max_lanes: 4,
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
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
    let vocab = loaded.caps.profile.vocab;

    let tokens = tokenizer.encode(PROMPT);
    let n = u32::try_from(tokens.len()).expect("a prompt of a countable length");
    assert!(
        n >= 3,
        "this gate names three distinct rows, so the prompt has to carry three"
    );
    let named = vec![0, n - 2, n - 1];

    // **THE REFUSAL FIRST, BECAUSE IT COSTS NOTHING.** A row past the lane's
    // own rows is `Invalid` at validation — before a kernel runs, which is
    // where a bounds error on a shared rectangle has to be caught.
    let overrun = prefill(&tokens, Readout::Rows(vec![n]));
    let refusal = overrun.validate();
    assert!(
        matches!(refusal, Err(engine_api::Error::Invalid(_))),
        "row {n} of a lane that carries {n} rows is refused, not gathered: {refusal:?}"
    );

    // ── FIRE 1. The three rows, in the caller's order.
    let mut ticket = engine
        .submit(&prefill(&tokens, Readout::Rows(named.clone())))
        .expect("a row-selected prefill fires");
    engine
        .settle_frame(&mut ticket)
        .expect("and its numbers come back");
    let rows: LaneReadout = ticket.steps[0].readouts[0].clone();

    // Claim 1: the shape `LaneReadout` promises.
    assert_eq!(rows.rows, 3, "three rows were named, so three came back");
    assert_eq!(rows.width, vocab, "each of them is the vocabulary wide");
    assert_eq!(
        rows.values.len(),
        3 * vocab as usize,
        "the values are row-major, `rows * width` of them"
    );
    assert!(
        rows.values.iter().all(|value| value.is_finite()),
        "an interior row read at the wrong stride shows up as a non-finite logit first"
    );

    // ── FIRE 2. The same computation, asked for its last row the way every
    //    caller before this wave asked for it.
    let mut ticket = engine
        .submit(&prefill(&tokens, Readout::Last))
        .expect("the same prefill fires again");
    engine
        .settle_frame(&mut ticket)
        .expect("and its numbers come back");
    let last: LaneReadout = ticket.steps[0].readouts[0].clone();
    assert_eq!(last.rows, 1, "`Readout::Last` is one row");
    assert_eq!(last.values.len(), vocab as usize);

    // Claim 2: THE ANCHOR, bit for bit. Row `n-1` reached through the row list
    // is the row `Readout::Last` has always answered with.
    let row = |at: usize| &rows.values[at * vocab as usize..(at + 1) * vocab as usize];
    assert_eq!(
        row(2),
        last.values.as_slice(),
        "the last of the named rows is not the row `Readout::Last` answers with — \
         the gather landed on the wrong arena row"
    );

    // And it is the RIGHT row, not merely a consistent one: a load that ran is
    // not a load that works.
    let text = tokenizer.decode(&[argmax(row(2))], false);
    eprintln!("row {} continues: {text:?}", n - 1);
    assert_eq!(
        text, EXPECTED,
        "the greedy continuation of {PROMPT:?} off row {} was {text:?}",
        n - 1
    );

    // Claim 3a: the three rows are three DIFFERENT rows. Without this, a shell
    // that answered the last row three times would pass everything above.
    assert_ne!(
        row(0),
        row(2),
        "row 0 and row {} are the same numbers, so the gather read one row three times",
        n - 1
    );
    assert_ne!(row(1), row(2), "row {} and row {} likewise", n - 2, n - 1);

    // ── FIRE 3. The same three rows, named backwards.
    let mut ticket = engine
        .submit(&prefill(&tokens, Readout::Rows(vec![n - 1, n - 2, 0])))
        .expect("the reversed row list fires");
    engine
        .settle_frame(&mut ticket)
        .expect("and its numbers come back");
    let reversed: LaneReadout = ticket.steps[0].readouts[0].clone();
    assert_eq!(reversed.rows, 3);
    let back = |at: usize| &reversed.values[at * vocab as usize..(at + 1) * vocab as usize];

    // Claim 3b: THE ORDER IS THE CALLER'S. Bit for bit again — the two fires
    // are the same computation, so the same row read twice is the same bytes.
    for (there, back_at) in [(0usize, 2usize), (1, 1), (2, 0)] {
        assert_eq!(
            row(there),
            back(back_at),
            "row list [0, {}, {}] and its reverse do not name the same rows in \
             mirrored positions — the shell sorted, deduplicated or re-indexed a \
             list the contract says it answers in order",
            n - 2,
            n - 1
        );
    }

    // ── CLAIM 4: THE DEVICE HALF READS THE SAME ROWS.
    //
    // Everything above reads the mirror, which `settle_frame` fills from the
    // host. It is not the reader that matters most: a guest's epilogue reads
    // `IntrinsicId::Logits` ON THE DEVICE and argmaxes there (design §9 — the
    // numbers never reach the host at all), and that is how every speculative
    // verifier in the corpus gets its tokens. The two readers are pointed at
    // the rectangle by different code — the mirror by `Shell::read_out_rows`,
    // the guest by `Plane::bind_intrinsic` — so a gate that checked only the
    // mirror would pass while a `k`-row verifier read its own last row
    // followed by `k - 1` rows of zeros past the end of the fire.
    //
    // So: attach a pass that argmaxes every readout row, and compare what it
    // publishes with the argmax of the mirror's own rows, in the same fire.
    let program = engine
        .register_program(&registration(3, &loaded.caps.profile))
        .expect("the gate's program compiles on this device");
    let instance = engine
        .bind_instance(&InstanceBinding {
            program,
            channels: vec![0],
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            // THE MODEL FIRE'S READOUT SHAPE. Three rows, which is what every
            // stage buffer this instance carries is carved for; a wrong guess
            // zero-fills rather than faulting.
            extents: BindExtents {
                sampled_rows: 3,
                ..BindExtents::default()
            },
        })
        .expect("the instance binds");

    // **BOTH BINDING SHAPES, BECAUSE THE SHELL HAS TWO.** A CONSECUTIVE run is
    // a base and an offset into the rectangle — which is `Readout::Last` and
    // every verifier in the corpus, `start .. start + k` — and a list that
    // skips is a table of row pointers. They are different lines in
    // `enqueue`, so they are different rows here.
    for named in [vec![n - 3, n - 2, n - 1], vec![0, n - 2, n - 1]] {
        let consecutive = named.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let mut ticket = engine
            .submit(&attached_prefill(
                &tokens,
                Readout::Rows(named.clone()),
                vec![Attachment {
                    lane: 0,
                    instance: instance.id,
                    at: Boundary::Epilogue,
                }],
            ))
            .unwrap_or_else(|why| panic!("the attached fire for {named:?}: {why}"));
        engine
            .settle_frame(&mut ticket)
            .expect("and its numbers come back");
        let mirror = &ticket.steps[0].readouts[0];
        assert_eq!(
            mirror.rows, 3,
            "an attachment must not change what the mirror answers"
        );
        let published = engine
            .take_channel(instance.id, OUT)
            .expect("taking the guest's channel")
            .unwrap_or_else(|| {
                panic!("the guest published nothing for {named:?}; its epilogue never ran")
            });
        let device: Vec<u32> = published
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        assert_eq!(
            device.len(),
            3,
            "three rows in, three tokens out for {named:?}"
        );
        for (at, &want_row) in named.iter().enumerate() {
            let mirrored = argmax(
                &mirror.values[at * vocab as usize..(at + 1) * vocab as usize],
            );
            assert_eq!(
                device[at], mirrored,
                "readout {named:?} (consecutive={consecutive}): at position {at} — row \
                 {want_row} of the lane — the guest's device argmax answered {} and the \
                 host mirror of the SAME fire answered {mirrored}. The two readers are \
                 pointed at the rectangle by different code, and this is where they \
                 disagree: a device rectangle that does not carry exactly the requested \
                 rows in the requested order reads zeros past the fire's rows, and an \
                 argmax over zeros is token 0",
                device[at]
            );
        }
        eprintln!("readout {named:?} (consecutive={consecutive}) -> device {device:?}");
    }
}
