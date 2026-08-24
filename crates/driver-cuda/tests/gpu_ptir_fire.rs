//! A PTIR program, running on the GPU, checked against the golden model.
//!
//! # What this claims, and why the comparison is the claim
//!
//! Every other test in this crate checks one seam: that NVRTC is reachable,
//! that a cubin loads, that the ring cursors move. None of them can tell you
//! the program computed the right thing, because none of them has a right
//! answer to compare against.
//!
//! This one does. `driver::step` is the reference interpreter — the
//! same code the Metal shell diffs against, itself proven equal to
//! `tensor_compiler::eval::interp` by `driver-pipeline`'s own oracle — and it
//! runs the *same adopted plan* over the *same seeds* on the CPU. So the GPU
//! and the CPU start from one program and one input, and a disagreement is a
//! disagreement about semantics.
//!
//! `.wiki/driver/progress-metal.md`'s tolerance contract is what "agree" means: magnitudes
//! may differ by one ulp on `exp`, and **an argmax index may not differ at
//! all**. The cases below are chosen so the observable is an index or an
//! exact integer, which is the half of the contract that admits no slack.
//!
//! # The chain under test
//!
//! Author a `TraceContainer` → `bind` → `compile_bound` → `codegen::launch::build`
//! → `emit_program(Backend::Cuda)` → `driver::adopt_launch_package_with`
//! → `ptir::Runtime::compile` (NVRTC) → `ptir::Prepared::build` → launch.
//! That is every step the engine takes, with nothing stubbed.
//!
//! Including the BOUNDARY VOCABULARY, which is a step it is easy to take
//! differently here than in production and which this file used to. The bare
//! `adopt_launch_package` adopts under `Boundaries::METAL` — the step
//! interpreter's `metal.identity`/`metal.discard` — and `serve/load.rs` says
//! at length what calling it on this backend costs: every `lora`,
//! `attn_page_mask` and `envelope_dot` program marked non-executable, never
//! compiled, never found by the fire, so a guest waits on a token that will
//! not come. These tests pass either way today, because their programs name
//! neither vocabulary's calls. That is precisely why the wrong one could sit
//! here: a chain-under-test that diverges from the engine at a step nothing
//! in it exercises is a chain that will agree with the engine until the day
//! it matters.

use driver::tensor_ir::DType;
use driver::{Boundaries, Extents, Versions, adopt_launch_package_with};
use driver_cuda::device::{Allocator, OwnedStream};
use driver_cuda::program::run::Lane;
use driver_cuda::program::{
    ChannelShape, Control, Disk, Prepared, Rings, Runtime, Target, compile, launch_control,
};
use tensor_compiler::codegen::program::{Backend, emit_program};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType as IrDType, Shape};
use tensor_ir::validate::bind;

mod common;
use common::{device_or_skip, gpu_guard};

/// Lanes in the vector the cases reduce over.
const LANES: u32 = 8;

fn profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = LANES;
    profile
}

fn chan(shape: Shape, dtype: IrDType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

/// A one-stage epilogue: read the seeded channel, apply `ops`, publish.
fn epilogue(out: IrDType, ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            chan(Shape::vector(LANES), IrDType::F32, HostRole::None, true),
            chan(Shape::SCALAR, out, HostRole::Reader, false),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

/// What one case answered, on each side.
struct Answers {
    device: Vec<u8>,
    golden: driver::Value,
    committed: bool,
}

/// Run one program on the GPU and, over the same plan and seed, on the CPU.
///
/// Returns `None` when there is no device, so a case skips rather than fails
/// on a GPU-less box — the same rule every `gpu_*` binary here follows.
fn run_both(container: TraceContainer, seed: &[f32]) -> Option<Answers> {
    let device = device_or_skip("PTIR fire")?;
    let (major, minor) = device.compute_capability().expect("compute capability");
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    // ── The host's chain, complete. ──
    let bound = bind(container, profile()).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels: Vec<driver_api::plan::EmittedKernel> = emitted
        .iter()
        .map(|k| driver_api::plan::EmittedKernel {
            kind: k.kind,
            stage_index: k.stage_index,
            region_index: k.region_index,
            entry_name: k.entry_name.clone(),
            source: k.source.clone(),
            error: k.error.clone(),
        })
        .collect();
    let plan = adopt_launch_package_with(package, Boundaries::CUDA)
        .expect("the driver adopts the package");
    assert!(
        plan.executable,
        "the plan must be executable: {}",
        plan.reject_reason.as_deref().unwrap_or("no reason given")
    );

    let directory = std::env::temp_dir().join(format!("pie-ptir-fire-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);
    let architecture = compile::arch_flag(major, minor);

    let mut runtime = Runtime::new(disk.clone());
    let compiled = runtime
        .compile(
            0xF1E,
            &plan,
            &kernels,
            Versions::from_compiler(Backend::Cuda.emitter_version()),
            Target {
                major,
                minor,
                device: u64::try_from(device.ordinal()).unwrap_or(0),
                nvrtc: compile::version().expect("nvrtc"),
            },
        )
        .unwrap_or_else(|failure| panic!("the program must compile: {}", failure.reason()));
    let control = Control::compile(&disk, &architecture, "fire-test").expect("control kernels");

    // ── The device side. ──
    let stage_plan = plan.package.plans.first().expect("one stage");
    let shapes = [
        ChannelShape {
            numel: LANES as usize,
            dtype: DType::F32,
            capacity: 1,
        },
        ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: 1,
        },
    ];
    let mut rings = Rings::new(&alloc, &shapes, stream.as_ref()).expect("rings");
    let seed_bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
    rings
        .seed(0, 0, &seed_bytes, stream.as_ref())
        .expect("seed");
    stream.as_ref().synchronize().expect("sync");

    // Readiness first: the input must hold a value and the output must have
    // room. A fire launched without asking would read the zeroed cell.
    //
    // The four index sets are DERIVED from the plan rather than written
    // here, and this test is where that derivation is checked against a
    // real compiled program: it used to hardcode `&[0]` and `&[1]`, which
    // is right for this two-channel epilogue and says nothing about a
    // program whose local slots and global indices differ.
    let sets = driver_cuda::program::channel::stage_channels(stage_plan)
        .expect("the plan binds its slots");
    assert_eq!(sets.need_full, vec![0], "the seeded input");
    assert_eq!(sets.need_empty, vec![1], "the reader");
    assert_eq!(sets.put, vec![1]);
    // AND `taken` IS EMPTY, which is where the derivation disagrees with
    // what stood here. These programs use `Op::ChanRead` — "peek: full →
    // copy, stays full" — and the hardcoded `&[0]` passed the input as
    // TAKEN, which advances its head and consumes a value the program
    // only looked at. Unobservable here because each case fires once; on a
    // decode loop it is one seeded value dropped per fire.
    assert!(
        sets.taken.is_empty(),
        "a chan_read consumes nothing, so no head advances: {:?}",
        sets.taken
    );
    let ready = launch_control::readiness(
        &control,
        &rings,
        &sets.need_full,
        &sets.need_empty,
        &alloc,
        stream.as_ref(),
    )
    .expect("readiness");
    assert!(ready, "a seeded input and an empty output is a ready pass");

    let extents = Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        ..Extents::default()
    };
    let prepared = Prepared::build(
        &alloc,
        stage_plan,
        &[Lane {
            rings: &rings,
            // The identity map: this fixture registers its channels at
            // slots 0..n, so the instance's dense index IS its slot.
            slots: &[0, 1],
            extents,
        }],
        stream.as_ref(),
    )
    .expect("prepare");
    for stage in compiled.stages.iter() {
        for region in stage.regions.iter() {
            prepared
                .launch_region(region, stream.as_ref())
                .expect("launch");
        }
    }
    stream.as_ref().synchronize().expect("the fire completes");

    let committed = prepared.committed(stream.as_ref()).expect("commit slot");
    // The status word says WHY when the commit slot is clear; without it a
    // refusal is indistinguishable from every other refusal.
    if let Ok((outcome, diagnosis)) = prepared.outcome(stream.as_ref()) {
        eprintln!("[fire] committed={committed} outcome={outcome:?} diagnosis={diagnosis:?}");
    }
    launch_control::commit(
        &control,
        &rings,
        &sets.taken,
        &sets.put,
        committed,
        &alloc,
        stream.as_ref(),
    )
    .expect("commit");
    stream.as_ref().synchronize().expect("sync");

    // The published cell is the one the commit just advanced past.
    let cursors = rings.cursors(stream.as_ref()).expect("cursors");
    let published = cursors[1].tail.wrapping_sub(1) % 2;
    let device_bytes = rings
        .read_cell(1, published, stream.as_ref())
        .expect("read the output");

    // ── The golden side: the SAME plan, on the CPU. ──
    let seeds: std::collections::BTreeMap<u32, driver::Value> =
        [(0u32, driver::Value::F32(seed.to_vec()))]
            .into_iter()
            .collect();
    let mut instance =
        driver::make_host_instance(&plan, &std::collections::BTreeMap::new(), &seeds);
    let outcome = driver::step(&mut instance, &plan, &driver::PassInputs::none());
    assert_eq!(
        outcome,
        driver::StepOutcome::Committed,
        "the reference interpreter must commit"
    );
    let golden = match driver::host_take(&instance, &plan, 1) {
        (driver::HostOp::Ok, Some(value)) => value,
        (op, _) => panic!("the reference must publish a value: {op:?}"),
    };

    let _ = std::fs::remove_dir_all(&directory);
    Some(Answers {
        device: device_bytes,
        golden,
        committed,
    })
}

/// The headline: an argmax over a tie, on the GPU, agreeing with the golden
/// model to the index.
///
/// The tie is the whole content. The maximum appears three times, and "first
/// wins" and "last wins" are both defensible rules — only one is the contract,
/// and it is the one the tolerance contract admits no slack on. A magnitude
/// may drift an ulp; a sampled token may not.
#[test]
fn an_argmax_over_a_tie_picks_the_same_lane_on_the_device_as_in_the_golden_model() {
    let _gpu = gpu_guard();
    let Some(answers) = run_both(
        epilogue(
            IrDType::I32,
            vec![
                Op::ChanRead(0),
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0],
    ) else {
        return;
    };
    assert!(answers.committed, "the fire must commit");

    let device = i32::from_le_bytes(answers.device[..4].try_into().expect("four bytes"));
    let golden = match &answers.golden {
        driver::Value::I32(lanes) => lanes[0],
        other => panic!("the golden model answered {other:?}"),
    };
    assert_eq!(
        device, golden,
        "the device and the reference interpreter must pick the SAME lane; a \
         tie broken differently is a different token, and the tolerance \
         contract admits no slack on an index"
    );
}

/// A reduction, where the fold ORDER is the observable. The canonical
/// reduction is a width-32 pairwise tree rather than a left fold, and at these
/// magnitudes the two orders give different bits — so this case fails if
/// either side ever "simplifies".
#[test]
fn a_reduction_agrees_with_the_golden_model_because_both_fold_in_one_order() {
    let _gpu = gpu_guard();
    let Some(answers) = run_both(
        epilogue(
            IrDType::F32,
            vec![
                Op::ChanRead(0),
                Op::ReduceSum(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[1e8, 1.0, -1e8, 1.0, 1e-8, 1.0, -1e-8, 1.0],
    ) else {
        return;
    };
    assert!(answers.committed, "the fire must commit");

    let device = f32::from_le_bytes(answers.device[..4].try_into().expect("four bytes"));
    let golden = match &answers.golden {
        driver::Value::F32(lanes) => lanes[0],
        other => panic!("the golden model answered {other:?}"),
    };
    assert_eq!(
        device.to_bits(),
        golden.to_bits(),
        "device {device} and reference {golden} disagree; at these magnitudes \
         a left fold and a pairwise tree give different bits, so this is an \
         order disagreement rather than rounding"
    );
}

/// An elementwise chain, so the case covers ops that write scratch and read it
/// back rather than only ops that reduce.
#[test]
fn an_elementwise_chain_agrees_with_the_golden_model() {
    let _gpu = gpu_guard();
    let Some(answers) = run_both(
        epilogue(
            IrDType::F32,
            vec![
                Op::ChanRead(0),
                Op::Neg(0),
                Op::Abs(1),
                Op::ReduceMax(2),
                Op::ChanPut { chan: 1, value: 3 },
            ],
        ),
        &[2.0, -7.5, 1.0, 3.25, -0.5, 7.0, -3.0, 6.0],
    ) else {
        return;
    };
    assert!(answers.committed, "the fire must commit");

    let device = f32::from_le_bytes(answers.device[..4].try_into().expect("four bytes"));
    let golden = match &answers.golden {
        driver::Value::F32(lanes) => lanes[0],
        other => panic!("the golden model answered {other:?}"),
    };
    assert_eq!(
        device.to_bits(),
        golden.to_bits(),
        "device {device}, reference {golden}"
    );
}

/// The regression that cost the most to find: a `chan_put` whose `sink_bytes`
/// is zero.
///
/// `driver-pipeline` leaves that field for the driver and says so — "filled
/// when the sink is bound, which is not this module's job" — and a driver that
/// takes it at its default does not write a short cell. The emitted kernel's
/// first act on a put is `if (logical_bytes > p.sink_bytes) fault`, so EVERY
/// put faults, the kernel clears the commit slot, and the fire comes back
/// refused with nothing to explain it: CUDA's `M1Status` is `__shared__` and
/// never reaches memory the host can read.
///
/// This test pins the fix from the outside, by asserting the thing that was
/// false: a fire whose only output is a put must commit.
#[test]
fn a_put_commits_which_means_its_sink_size_reached_the_kernel() {
    let _gpu = gpu_guard();
    let Some(answers) = run_both(
        epilogue(
            IrDType::F32,
            vec![
                Op::ChanRead(0),
                Op::ReduceMax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ) else {
        return;
    };
    assert!(
        answers.committed,
        "a put with no sink size faults with class 146 and clears the commit \
         slot; a committed fire is the only evidence the size arrived"
    );
    let device = f32::from_le_bytes(answers.device[..4].try_into().expect("four bytes"));
    assert_eq!(device, 8.0, "and the cell must hold the value, not zeros");
}

/// The five intrinsic side tables reach the device with what the emitted
/// kernel reads out of them, and each LANE reads its own row.
///
/// `Prepared::build` allocates them zeroed and, until `bind_intrinsic`,
/// nothing filled them — so a program that read `logits` read address ZERO.
/// Sampling is what a PTIR program is FOR (top-p, top-k, temperature and
/// argmax are stages, not driver flags), which makes an unbound intrinsic
/// the difference between a program that samples and one that cannot.
///
/// The claim is a ROUND TRIP through device memory, not a host-side
/// arithmetic check: the tables are strided `lane * INTRINSIC_SLOTS + intr`,
/// so the failure this catches is a write that lands on the fifteen slots
/// beside the one it meant — which no host assertion about the arguments
/// would see.
///
/// The row offsets are the half that matters most. One buffer, one layout,
/// and each lane reading a different row of it; a binding that gave every
/// lane row 0 would sample the first request N times and look exactly like
/// a correct fire. That is not hypothetical — it is the shape of a defect
/// this driver already shipped once, as `bind_intrinsic`'s doc records.
#[test]
fn every_lane_binds_its_own_row_of_the_logits() {
    use driver::tensor_ir::op::IntrinsicId;

    let _gpu = gpu_guard();
    let Some(_device) = device_or_skip("PTIR intrinsic binding") else {
        return;
    };
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let container = epilogue(
        IrDType::F32,
        vec![
            Op::ChanRead(0),
            Op::ReduceSum(0),
            Op::ChanPut { chan: 1, value: 1 },
        ],
    );
    let bound = bind(container, profile()).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("adopt");
    let stage_plan = plan.package.plans.first().expect("one stage");

    let shapes = [
        ChannelShape {
            numel: LANES as usize,
            dtype: DType::F32,
            capacity: 1,
        },
        ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: 1,
        },
    ];
    let rings = Rings::new(&alloc, &shapes, stream.as_ref()).expect("rings");

    // TWO LANES, with DIFFERENT extents. The second is what makes this a
    // multi-lane test rather than a single-lane one run twice: the lanes
    // of a real group submit different token counts, and a table laid out
    // for one lane has every lane after the first reading the previous
    // lane's tail — with every field a plausible number.
    let lane_extents = [
        Extents {
            row_count: 4,
            token_count: 4,
            sampled_rows: 4,
            ..Extents::default()
        },
        Extents {
            row_count: 2,
            token_count: 2,
            sampled_rows: 2,
            ..Extents::default()
        },
    ];
    // TWO MEMBERS, and both name THIS session's rings — which is the
    // single-instance case. A grouped fire across instances would give
    // each `Lane` its own, and the fire takes the pairing precisely so
    // that a caller cannot group them and silently share channels.
    let mut prepared = Prepared::build(
        &alloc,
        stage_plan,
        &lane_extents.map(|extents| Lane {
            rings: &rings,
            slots: &[0, 1],
            extents,
        }),
        stream.as_ref(),
    )
    .expect("prepare");
    assert_eq!(
        prepared.lanes(),
        2,
        "a fire has as many lanes as it was given extents"
    );

    // Unbound is address zero, which is the state this test exists to end.
    let (base, ..) = prepared
        .intrinsic_binding(IntrinsicId::Logits, 0, stream.as_ref())
        .expect("read back");
    assert_eq!(base, 0, "an unbound intrinsic points at nothing");

    let vocab: u32 = 128;
    let logits = alloc
        .alloc(vocab as usize * 4 * 4)
        .expect("a logits buffer");
    let address = logits.as_ptr() as u64;
    prepared
        .bind_intrinsic(
            IntrinsicId::Logits,
            address,
            driver_cuda::program::params::INTRINSIC_STORAGE_F32,
            vocab,
            vocab,
            // `lane + 3`, NOT the identity, and that is the point at one
            // lane: `row_of(0) == 0` is indistinguishable from a binding
            // that hardcodes zero, so the identity would let exactly the
            // defect this guards — every lane reading row 0 — pass. An
            // offset the lane index cannot produce by accident is what
            // makes the claim falsifiable before grouping lands.
            |lane| lane + 3,
            stream.as_ref(),
        )
        .expect("bind");
    stream.as_ref().synchronize().expect("sync");

    // Over the lanes the fire ACTUALLY has. Written against `lanes()`
    // rather than a constant so that it covers whatever the fire was
    // given — which is now two, and was one when the fire could only be
    // one.
    for lane in 0..prepared.lanes() {
        let (base, mode, width, stride, offset) = prepared
            .intrinsic_binding(IntrinsicId::Logits, lane, stream.as_ref())
            .expect("read back");
        assert_eq!(base, address, "lane {lane} points at the logits");
        assert_eq!(mode, DType::F32 as u8 as u32, "lane {lane} dtype");
        assert_eq!(width, vocab, "lane {lane} width is the vocabulary");
        assert_eq!(stride, vocab, "lane {lane} row stride");
        assert_eq!(offset, lane + 3, "lane {lane} reads ITS row, not row 0");
    }

    // A lane past the fire's is refused rather than read out of the next
    // allocation, which is the whole reason the pitch is a constant.
    assert!(
        prepared
            .intrinsic_binding(IntrinsicId::Logits, prepared.lanes(), stream.as_ref())
            .is_err(),
        "a lane past the fire's has no binding"
    );
}

/// A compiled program fires through `ptir::Session`, taking its input from
/// a HOST mirror and publishing back into one.
///
/// The other cases in this file drive the pieces directly — seed the
/// device ring, launch, read the device cell. This one goes through the
/// plane the engine actually uses: a pinned-style host mirror with four
/// control words, exactly what `pie_cuda_register_channel` hands over. So
/// it is the first test in which a value crosses BOTH planes, which is
/// what `ptir_programs` having no reader has meant all along.
///
/// The program is an argmax over eight lanes with a tie, chosen because
/// the observable is an INDEX: `.wiki/driver/progress-metal.md`'s tolerance contract
/// admits one ulp on magnitudes and no slack at all on an argmax, so a
/// disagreement here cannot be rounding.
#[test]
fn a_program_fires_from_a_host_mirror_and_publishes_back_into_one() {
    use driver_cuda::program::channel::{HostChannel, stage_channels};
    use driver_cuda::program::session::{Fired, Session};

    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR session") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let seed: [f32; LANES as usize] = [2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0];
    let container = epilogue(
        IrDType::I32,
        vec![
            Op::ChanRead(0),
            Op::ReduceArgmax(0),
            Op::ChanPut { chan: 1, value: 1 },
        ],
    );

    let bound = bind(container, profile()).expect("binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels: Vec<driver_api::plan::EmittedKernel> = emitted
        .iter()
        .map(|k| driver_api::plan::EmittedKernel {
            kind: k.kind,
            stage_index: k.stage_index,
            region_index: k.region_index,
            entry_name: k.entry_name.clone(),
            source: k.source.clone(),
            error: k.error.clone(),
        })
        .collect();
    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("adopts");
    assert!(plan.executable, "the plan must be executable");

    let directory = std::env::temp_dir().join(format!("pie-ptir-session-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);
    let architecture = compile::arch_flag(major, minor);
    let mut runtime = Runtime::new(disk.clone());
    let compiled = runtime
        .compile(
            0x5E5,
            &plan,
            &kernels,
            Versions::from_compiler(Backend::Cuda.emitter_version()),
            Target {
                major,
                minor,
                device: u64::try_from(device.ordinal()).unwrap_or(0),
                nvrtc: compile::version().expect("nvrtc"),
            },
        )
        .unwrap_or_else(|f| panic!("compiles: {}", f.reason()));
    let control = Control::compile(&disk, &architecture, "session-test").expect("control");
    let stage_plan = plan.package.plans.first().expect("one stage");

    // ── The HOST plane, as `register_channel` lays it out. ──
    let shapes = [
        ChannelShape {
            numel: LANES as usize,
            dtype: DType::F32,
            capacity: 1,
        },
        ChannelShape {
            numel: 1,
            dtype: DType::I32,
            capacity: 1,
        },
    ];
    let mut mirrors: Vec<Vec<u8>> = shapes
        .iter()
        .map(|s| vec![0u8; s.cell_bytes() * s.ring().expect("ring") as usize])
        .collect();
    let mut words: Vec<Vec<u64>> = vec![vec![0u64; 4], vec![0u64; 4]];
    // WHICH SIDE THE ENGINE IS ON, per channel. The bridge is directional
    // now: the driver takes only from a plane the engine writes and publishes
    // only into one it reads, because the mirror has ONE head/tail pair for
    // both and a driver that did both would read back its own writes.
    const ROLES: [u8; 2] = [
        driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER,
        driver_api::local::PIE_CHANNEL_HOST_ROLE_READER,
    ];

    // The registry, and the instance's map into it: registered in order, so
    // the dense index is the slot. `Session` no longer owns rings — a channel
    // has ONE ring wherever it is named from.
    let mut rings = Rings::new(&alloc, &shapes, stream.as_ref()).expect("rings");
    let mut session = Session::new(vec![0, 1], shapes.to_vec()).expect("session");
    let _sets = stage_channels(stage_plan).expect("sets");

    // The engine's side: publish the seed into the input's host mirror.
    {
        let mut host: Vec<HostChannel> = mirrors
            .iter_mut()
            .zip(words.iter_mut())
            .zip(shapes.iter())
            .enumerate()
            .map(|(index, ((m, w), s))| unsafe {
                HostChannel::new(
                    m.as_mut_ptr().cast(),
                    w.as_mut_ptr().cast(),
                    s.cell_bytes(),
                    s.ring().expect("ring"),
                    ROLES[index],
                )
            })
            .collect();
        let bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(host[0].publish(&bytes), "the engine publishes the seed");
        assert_eq!(host[0].depth(), 1);
    }

    let outcome = {
        let mut host: Vec<HostChannel> = mirrors
            .iter_mut()
            .zip(words.iter_mut())
            .zip(shapes.iter())
            .enumerate()
            .map(|(index, ((m, w), s))| unsafe {
                HostChannel::new(
                    m.as_mut_ptr().cast(),
                    w.as_mut_ptr().cast(),
                    s.cell_bytes(),
                    s.ring().expect("ring"),
                    ROLES[index],
                )
            })
            .collect();
        session
            .fire(
                &mut rings,
                &compiled,
                // `stage_plan` STOOD HERE. `Session::fire` derives the
                // per-stage plans from the compiled program itself now, so a
                // caller that hands one in is stating what the session
                // already knows.
                &control,
                &mut host,
                // This program reads no intrinsic, so a null base is
                // honest rather than a placeholder — `bind_intrinsic` is
                // skipped and nothing reads address zero.
                (0, 0, 0),
                |lane| lane,
                &[Extents {
                    row_count: 1,
                    token_count: 1,
                    sampled_rows: 1,
                    ..Extents::default()
                }],
                &alloc,
                stream.as_ref(),
            )
            .expect("the fire completes")
    };
    assert_eq!(
        outcome,
        Fired::Committed { published: 1 },
        "one cell published"
    );

    // ── And the engine reads its answer out of the host mirror. ──
    let mut host: Vec<HostChannel> = mirrors
        .iter_mut()
        .zip(words.iter_mut())
        .zip(shapes.iter())
        .enumerate()
        .map(|(index, ((m, w), s))| unsafe {
            HostChannel::new(
                m.as_mut_ptr().cast(),
                w.as_mut_ptr().cast(),
                s.cell_bytes(),
                s.ring().expect("ring"),
                ROLES[index],
            )
        })
        .collect();
    let cell = host[1].take().expect("the reader has a value");
    let got = i32::from_le_bytes(cell[..4].try_into().expect("four bytes"));
    assert_eq!(
        got, 1,
        "argmax over a tie takes the FIRST maximum, on both planes"
    );
}
