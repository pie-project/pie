//! `palo B2`: one lane's guest program at the fire's boundary, end to end.
//!
//! # What this is for
//!
//! `driver-cuda/tests/program_parity.rs` proves the guest-program plane
//! computes the right thing: both halves of `driver::program` — the host
//! interpreter and the CUDA one — run the same golden traces over the same
//! inputs and must agree byte for byte. What it drives is a [`Plane`] with a
//! hand-staged f32 buffer standing in for a readout, because at P7 there was
//! no model fire to take one from.
//!
//! This test is that parity **through the serving stack**:
//!
//! ```text
//!   backend::cuda::open           -> Cuda
//!   Driver::load(request)         -> a real checkpoint on the device
//!   Driver::register_program      -> the corpus golden, compiled by NVRTC
//!   Driver::bind_instance         -> rings carved, seeds planted, extents stated
//!   Driver::publish_channel       -> the host half of the channel join
//!   Driver::fire(.. attachments)  -> forward, then the guest pass on its logits
//!   Driver::take_channel          -> what the guest published
//! ```
//!
//! and the readout the guest reads is not a fixture: it is the arena's own
//! `out` seam, at the row this lane's tokens landed on, read where it lies as
//! raw bf16. So what is being asserted is that the SAME logits reached the
//! guest program and the caller — which is the one thing no test under this
//! one can see, and the thing a wrong row offset or a wrong stride would
//! break silently.
//!
//! # The four claims
//!
//! 1. **Parity.** Every fire, the device's rings and the host interpreter's
//!    hold the same bytes, and what a reader takes out is cell for cell the
//!    same. Given the same logits, `driver::program::step` is the golden.
//! 2. **A blocked round retries.** A reader channel nobody drains fills, its
//!    `NEEDS_EMPTY` requirement fails, and the next fire answers
//!    [`DriverError::Exhausted`] — a *scheduling* answer the run-ahead lane
//!    retries, not a failure — with nothing launched and nothing written.
//!    Drain it and the same submission commits.
//! 3. **No attachment is byte-identical.** The first fire carries none and
//!    pins the same continuation the boot smoke does.
//! 4. **A replayed fire still fires the program.** Attachments are outside
//!    the captured body by design (§5's table: guest programs are "outside
//!    the graph"), so the graph key is a function of the composition alone.
//!    The last rounds run with capture on, replay from a warm key, and
//!    produce the guest's cells exactly as the eager ones did.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine --features driver-cuda-13 --test cuda_program_epilogue -- --nocapture
//! ```

#![cfg(feature = "_driver-cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use driver::driver_api::program::{LaunchChannel, LaunchPackage};
use driver::tensor_ir::container::HostRole;
use driver::tensor_ir::DType;
use driver::{
    Boundaries, ExecPlan, HostOp, InterpInstance, PassInputs, StepOutcome, Value,
    adopt_launch_package_with, concrete_dtype, encode_wire, host_put, host_take,
    make_host_instance, step, wire_cell_bytes,
};
use driver_api::model_ir::Plane;
use driver_api::{
    Attachment, BindExtents, Boundary, Budgets, DriverError, FireSubmission, InstanceBinding, Lane,
    Readout,
};
use driver_cuda::{Cuda, DeviceBoot, Graphs};
use tensor_ir::container::{ChanDType, ChannelDecl, StageProgram, TraceContainer};
use tensor_ir::op::{IntrinsicId, Op};
use tensor_ir::registry::Stage;
use tensor_ir::types::Shape;

/// The catalog row this gate serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt, and the token a correct load continues it with — the same pair
/// `cuda_boot_smoke` and `driver-cuda/tests/serve_smoke.rs` pin. Three paths
/// to one device must agree about it.
const PROMPT: &str = "The capital of France is";
const EXPECTED: &str = " Paris";

/// How many decode fires the gate runs after the prefill.
const DECODES: usize = 6;

/// The round whose output is deliberately left in the ring, so the round
/// after it finds the guest's `NEEDS_EMPTY` channel full and blocks.
const HOLD_THE_DRAIN: usize = 2;

/// The round the graph mode is turned on at. A key captures on its SECOND
/// sighting (Build log 11), so the rounds after that replay.
const CAPTURE_FROM: usize = 3;

/// The round the host biases the guest's decision on, and the token it biases
/// towards.
///
/// **THE ONE ASSERTION THE HOST→DEVICE PUMP CANNOT PASS BY ACCIDENT.** Every
/// other round publishes an all-zero bias, so the guest's constrained argmax
/// is the free argmax of the row — which proves it read the right LOGITS. A
/// round that pushes one token far above the rest proves the other direction:
/// the guest read the cell the host put into its ring this fire, rather than
/// a stale one, an empty one, or none.
const BIAS_ROUND: usize = 4;
const BIAS_TOKEN: u32 = 1234;

// ─────────────────────────────────────────────────────────────────────────
// The guest program
// ─────────────────────────────────────────────────────────────────────────

/// The gate's guest program: a biased greedy decode.
///
/// ```text
///   logits(1, vocab) ─reshape─▶ ┐
///                               ├─ add ─▶ argmax ─▶ chan_put(out)
///   chan_take(bias) ────────────┘
/// ```
///
/// **AUTHORED HERE RATHER THAN TAKEN FROM THE CORPUS**, and the reason is a
/// fact about PTIR rather than a convenience: a trace's `logits` intrinsic
/// carries its own concrete shape, so a golden authored at
/// `ModelProfile::dummy()`'s eight-token vocabulary refuses to bind at this
/// checkpoint's — `IntrinsicTypeRule { intr: Logits, stage: Epilogue }` — and
/// no amount of re-binding makes it read a 151k-wide row. What the corpus
/// gives (`program_parity`'s `dfa_ingraph`) is the SHAPE of a subject: reads
/// a bound logits row, takes from one channel, publishes to another, every op
/// integral where it matters. This is that shape at the load's own
/// vocabulary.
///
/// **NO DESCRIPTOR PORT, DELIBERATELY.** A port bound to a channel is how a
/// standalone pass states its own geometry — which tokens to embed, which
/// pages to read — and an attached program has none to state: the model fire
/// supplies all of it, which is exactly what `GeometryClass::Host` means. The
/// load serves `PortMask::DECODE_ENVELOPE` (`palo B3`) and this instance binds
/// `Host` anyway, which is the gate that matters: `Plane::envelope` resolves
/// nothing for a Host-class instance, so this pass's fires read the submission
/// exactly as they did before the plane existed. Binding a port here would
/// make the pass consume a channel cell per fire that nothing writes.
fn program(vocab: u32) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        externs: Vec::new(),
        channels: vec![
            // 0: the host's per-fire bias. `Writer` + a take is what makes the
            // program's readiness `NeedsFull` on it, which is the gate the
            // driver asks before it launches anything.
            ChannelDecl {
                shape: Shape::vector(vocab),
                dtype: ChanDType::Concrete(DType::F32),
                capacity: 1,
                host_role: HostRole::Writer,
                seeded: false,
            },
            // 1: the token the guest chose. `Reader` + a put is `NeedsEmpty`,
            // and a capacity-one ring nobody drains is what makes a round
            // block.
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(1, vocab),
                    dtype: DType::F32,
                },
                Op::Reshape {
                    value: 0,
                    shape: Shape::vector(vocab),
                },
                Op::ChanTake(0),
                Op::Add(1, 2),
                Op::ReduceArgmax(3),
                Op::Reshape {
                    value: 4,
                    shape: Shape::vector(1),
                },
                Op::ChanPut { chan: 1, value: 5 },
            ],
        }],
    }
}

/// Which channel is which, in the package's declaration order — the numbering
/// `publish_channel` and `take_channel` address.
const BIAS: u32 = 0;
const OUT: u32 = 1;

/// The gate's program, all the way to what `Driver::register_program` takes.
///
/// The emitted kernels are attached HERE rather than by
/// `pipeline::program::with_host_codegen`, which is the scheduler lane's
/// splice: this test holds the driver directly, so it plays that part too.
fn registration(
    profile: &driver::tensor_ir::registry::ModelProfile,
) -> driver_api::ProgramRegistration {
    let bound = tensor_ir::validate::bind(program(profile.vocab), profile.clone())
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
    driver_api::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted
            .into_iter()
            .map(|kernel| driver_api::EmittedKernel {
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

// ─────────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────────

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

/// One channel cell, as wire bytes.
fn wire(value: &Value, dtype: DType, lanes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; wire_cell_bytes(dtype, lanes)];
    encode_wire(value, &mut bytes);
    bytes
}

/// The bias this round publishes: zero everywhere, except on
/// [`BIAS_ROUND`], where one token is pushed far above whatever the model
/// thinks.
fn bias(round: usize, vocab: u32) -> Value {
    let mut values = vec![0.0f32; vocab as usize];
    if round == BIAS_ROUND {
        values[BIAS_TOKEN as usize] = 1.0e4;
    }
    Value::F32(values)
}

/// Greedy: the highest logit, first index on a tie — what the guest program's
/// `ReduceArgmax` computes, said here so the two can be compared.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// What the guest published, taken from BOTH halves and asserted equal.
///
/// The wire encoding is the comparison because it is the one both halves
/// speak: the interpreter's cells are wire bytes and the device's are native.
/// `None` from both is a round the program did not publish on.
fn take_both(
    driver: &mut Cuda,
    instance: u64,
    interp: &InterpInstance,
    plan: &ExecPlan,
    package: &LaunchPackage,
    at: &str,
) -> Option<u32> {
    use driver_api::Driver;
    let declared = &package.channels[OUT as usize];
    let dtype = concrete_dtype(declared.dtype);
    let lanes = numel(declared);
    let (op, value) = host_take(interp, plan, OUT);
    let device = driver
        .take_channel(instance, OUT)
        .unwrap_or_else(|error| panic!("{at}: taking the out channel: {error}"));
    match (op, value, device) {
        (HostOp::Ok, Some(value), Some(bytes)) => {
            let expected = wire(&value, dtype, lanes);
            assert_eq!(
                expected, bytes,
                "{at}: the device published {bytes:02x?} and the interpreter, given \
                 the same logits and the same bias, published {expected:02x?}"
            );
            Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        (HostOp::WouldBlock, None, None) => None,
        (op, value, device) => panic!(
            "{at}: the out channel — host {op:?}/{}, device {}",
            value.is_some(),
            device.is_some()
        ),
    }
}

fn numel(declared: &LaunchChannel) -> usize {
    declared
        .shape
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1)
}

// ─────────────────────────────────────────────────────────────────────────
// The gate
// ─────────────────────────────────────────────────────────────────────────

/// The fixture's own contract, checked without a device: the program binds at
/// a real checkpoint's vocabulary, plans, and emits CUDA.
///
/// Separate from the gate because it is the claim the corpus goldens cannot
/// make — every one of them is authored at eight tokens — and because a
/// failure here is a failure of the TEST rather than of the seam.
#[test]
fn the_gates_program_binds_at_a_real_vocabulary() {
    let profile = driver::tensor_ir::registry::ModelProfile {
        vocab: 151_936,
        ..driver::tensor_ir::registry::ModelProfile::dummy()
    };
    let registration = registration(&profile);
    let plan: ExecPlan =
        adopt_launch_package_with(registration.launch.clone(), Boundaries::CUDA)
            .expect("the gate's program adopts");
    assert!(
        plan.needs_logits,
        "the gate's program must read the readout, or it proves nothing about it"
    );
    assert_eq!(
        plan.package.channels.len(),
        2,
        "one bias in, one token out"
    );
    assert!(
        plan.takes_channel(BIAS) && plan.puts_channel(OUT),
        "the readiness gate this test leans on is a take and a put"
    );
    assert!(
        !registration.emitted_kernels.is_empty(),
        "nothing was emitted, so `register_program` would compile nothing"
    );
}

#[test]
#[allow(clippy::too_many_lines, reason = "one end-to-end run, told in order")]
fn a_guest_program_fires_at_the_boundary_of_a_real_model_fire() {
    use driver_api::Driver;

    if !driver_cuda::device::present() {
        eprintln!("skipping the epilogue gate: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping the epilogue gate: no Qwen3.5-0.8B snapshot in the hugging \
             face cache (set PIE_SMOKE_SNAPSHOT)"
        );
        return;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    // 1. THE LOAD, through the door the engine opens. `Cuda` and not
    //    `Box<dyn Driver>` for one reason: the graph-replay claim needs
    //    `Shell::set_mode`, and the A/B is ONE load (serve_smoke's argument —
    //    two loads are two residencies and a difference could be either).
    let mut driver = Cuda::new(
        DeviceBoot {
            ordinal: 0,
            graphs: Graphs::Off,
        },
        engine::driver::load::contract_for,
    );
    let budgets = Budgets {
        max_lanes: 4,
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: 512,
        slots: 4,
    };
    let request = engine::driver::load::request(&checkpoint, Plane::Cuda, budgets.clone(), 0)
        .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(request.plan.name, SKU);
    let loaded = driver.load(request).expect("the checkpoint lands");
    let vocab = loaded.caps.profile.vocab;
    let classify = engine::driver::load::classify(SKU).expect("this build ships the gate's SKU");

    // 2. THE FIRE WITH NO ATTACHMENT, which must be what it always was. Same
    //    prompt, same pinned continuation as the boot smoke — asserted here
    //    so that "attachments changed the forward" is a claim this file can
    //    refute on its own.
    let prompt = tokenizer.encode(PROMPT);
    let prefill = FireSubmission {
        lanes: vec![Lane {
            slot: 0,
            word: classify(&model::Request::new(prompt.len() as u32, false)),
            tokens: prompt.clone(),
            readout: Readout::Last,
            ..Lane::default()
        }],
        attachments: Vec::new(),
    };
    let ticket = driver.fire(&prefill).expect("the prefill fires");
    let first = argmax(&ticket.readouts[0].values);
    assert_eq!(
        tokenizer.decode(&[first], false),
        EXPECTED,
        "a fire with no attachment must be the fire it always was"
    );
    let mut held = prompt.len() as u32;
    let mut token = first;

    // 3. THE PROGRAM, registered and bound through the contract.
    let registration = registration(&loaded.caps.profile);
    let package = registration.launch.clone();
    let plan: ExecPlan =
        adopt_launch_package_with(package.clone(), Boundaries::CUDA).expect("it adopts");
    let program = driver
        .register_program(&registration)
        .expect("the program compiles on this device");
    let mut interp: InterpInstance = make_host_instance(&plan, &BTreeMap::new(), &BTreeMap::new());
    let bound = driver
        .bind_instance(&InstanceBinding {
            program,
            channels: (0..package.channels.len() as u64).collect(),
            seeds: Vec::new(),
            geometry: driver_api::tensor_ir::registry::GeometryClass::Host,
            // THE MODEL FIRE'S READOUT SHAPE. `Readout::Last` is one row, and
            // one row is what every stage buffer this instance carries is
            // carved for. Stated rather than defaulted because a wrong guess
            // zero-fills instead of faulting (Build log 15).
            extents: BindExtents {
                sampled_rows: 1,
                ..BindExtents::default()
            },
        })
        .expect("the instance binds");
    assert_eq!(bound.program, program, "the driver acknowledges the program");

    // 4. THE DECODE LOOP. Each round: publish the guest's input into both
    //    halves, fire the model with the epilogue attached, step the
    //    interpreter over the SAME readout, and compare what came out.
    let mut committed = 0usize;
    let mut blocked = 0usize;
    let mut guest_tokens: Vec<u32> = Vec::new();
    let bias_lanes = numel(&package.channels[BIAS as usize]);
    let bias_dtype = concrete_dtype(package.channels[BIAS as usize].dtype);
    for round in 0..DECODES {
        if round == CAPTURE_FROM {
            // §5's table puts guest programs OUTSIDE the graph, so the key is
            // a function of the composition alone and an attachment cannot
            // change it. Turning capture on mid-run is what makes that
            // testable on one load.
            driver.shell_mut().expect("loaded").set_mode(Graphs::On);
        }
        let submission = FireSubmission {
            lanes: vec![Lane::decode(
                0,
                classify(&model::Request::new(1, true)),
                token,
                held,
            )],
            attachments: vec![Attachment {
                lane: 0,
                instance: bound.id,
                at: Boundary::Epilogue,
            }],
        };

        // The same cell to both halves. Back-pressure has to be the same
        // answer on both sides, which is the first thing a join gets wrong.
        let cell = bias(round, vocab);
        let host = host_put(&interp, &plan, BIAS, &cell) == HostOp::Ok;
        let device = driver
            .publish_channel(bound.id, BIAS, &wire(&cell, bias_dtype, bias_lanes))
            .unwrap_or_else(|error| panic!("round {round}: publishing the bias: {error}"));
        assert_eq!(
            host, device,
            "round {round}: the host {host} the bias channel and the device {device}"
        );

        let ticket = match driver.fire(&submission) {
            Ok(ticket) => ticket,
            Err(error) => {
                // CLAIM 2. A guest program whose ring has no room is a
                // SCHEDULING answer the run-ahead lane retries in place — not
                // a failure, and not a fire that half-happened.
                assert!(
                    error.is_scheduling(),
                    "round {round}: a blocked guest program must be a scheduling \
                     answer the lane retries, and this was {error}"
                );
                assert!(
                    matches!(error, DriverError::Exhausted { .. }),
                    "round {round}: and `Exhausted` rather than `Impossible`, because \
                     draining the ring is exactly what makes it fit: {error}"
                );
                blocked += 1;
                // Drain both halves and retry the SAME submission: the
                // refused fire launched nothing, so `held` has not moved and
                // the lane's kv is untouched.
                let late = take_both(
                    &mut driver,
                    bound.id,
                    &interp,
                    &plan,
                    &package,
                    &format!("round {round} drain-after-block"),
                );
                assert!(
                    late.is_some(),
                    "round {round}: the fire blocked, so the ring it blocked on must \
                     have held the cell that filled it"
                );
                guest_tokens.push(late.expect("just checked"));
                driver
                    .fire(&submission)
                    .expect("the same fire commits once the ring has room")
            }
        };
        committed += 1;

        // The interpreter, over the readout the caller was handed. This is
        // the P7 parity's whole substance: the shell read the arena's out
        // seam at this lane's row and widened it, the emitted kernel read the
        // SAME bytes at the same offset as raw bf16, and `bits << 16` is both
        // widenings — so a disagreement here is a disagreement about the row,
        // the stride or the arithmetic, and nothing else.
        let readout = &ticket.readouts[0];
        assert_eq!(readout.width, vocab, "a logits row is the vocabulary wide");
        let outcome = step(
            &mut interp,
            &plan,
            &PassInputs {
                logits: Some(&readout.values),
                rows: 1,
                vocab,
                mtp_draft_row: None,
            },
        );
        assert_eq!(
            outcome,
            StepOutcome::Committed,
            "round {round}: the device committed, so the interpreter must too"
        );

        let free = argmax(&readout.values);
        if round == HOLD_THE_DRAIN {
            // Leave the guest's output in the ring so the NEXT round's gate
            // finds it full. Nothing this program does produces a blocked
            // round on its own.
            held += 1;
            token = free;
            continue;
        }
        let chose = take_both(
            &mut driver,
            bound.id,
            &interp,
            &plan,
            &package,
            &format!("round {round}"),
        )
        .unwrap_or_else(|| panic!("round {round}: the fire committed and published nothing"));
        assert!(chose < vocab, "round {round}: token {chose} of {vocab}");
        guest_tokens.push(chose);
        if round == BIAS_ROUND {
            // THE HOST'S CELL REACHED THE DEVICE THIS FIRE. Nothing about the
            // model says this token; only the bias the host published into
            // the ring a moment ago does.
            assert_eq!(
                chose, BIAS_TOKEN,
                "round {round}: the host pushed token {BIAS_TOKEN} to +1e4 and the \
                 guest chose {chose}, so the cell the host published did not reach \
                 the pass"
            );
        } else {
            // AND THE READOUT REACHED IT. An unbiased round's constrained
            // argmax is the free argmax of the row the CALLER was handed —
            // which is only true if both read the same row of the arena.
            assert_eq!(
                chose, free,
                "round {round}: the guest chose {chose} and the row the caller was \
                 handed argmaxes to {free}, so the two are not reading the same row"
            );
        }

        held += 1;
        token = free;
    }

    // CLAIM 4. The graph captured and replayed, and the guest's cells kept
    // coming across it.
    let stats = driver.shell().expect("loaded").graph_stats();
    eprintln!(
        "guest tokens: {:?}",
        guest_tokens
            .iter()
            .map(|&id| tokenizer.decode(&[id], false))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "{committed} committed, {blocked} blocked; graph {stats:?}"
    );
    assert!(
        blocked > 0,
        "the readiness gate was never exercised — holding the drain at round \
         {HOLD_THE_DRAIN} is supposed to fill the channel the program declares \
         `NEEDS_EMPTY` on"
    );
    assert_eq!(
        guest_tokens.len(),
        DECODES,
        "the guest published on every round, blocked or not"
    );

    driver.close_instance(bound.id).expect("the instance closes");
}
