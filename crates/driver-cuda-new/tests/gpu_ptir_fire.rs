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
//! `PARITY-INTERP.md`'s tolerance contract is what "agree" means: magnitudes
//! may differ by one ulp on `exp`, and **an argmax index may not differ at
//! all**. The cases below are chosen so the observable is an index or an
//! exact integer, which is the half of the contract that admits no slack.
//!
//! # The chain under test
//!
//! Author a `TraceContainer` → `bind` → `compile_bound` → `codegen::launch::build`
//! → `emit_program(Backend::Cuda)` → `driver::adopt_launch_package`
//! → `ptir::Runtime::compile` (NVRTC) → `ptir::Prepared::build` → launch.
//! That is every step the engine takes, with nothing stubbed.

use driver_cuda_new::cuda::{Allocator, OwnedStream};
use driver_cuda_new::ptir::{
    ChannelShape, Control, Disk, Prepared, Rings, Runtime, Target, launch_control, nvrtc,
};
use driver::tensor_ir::DType;
use driver::{Extents, Versions, adopt_launch_package};
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
    let plan = adopt_launch_package(package).expect("the driver adopts the package");
    assert!(
        plan.executable,
        "the plan must be executable: {}",
        plan.reject_reason.as_deref().unwrap_or("no reason given")
    );

    let directory = std::env::temp_dir().join(format!("pie-ptir-fire-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);
    let architecture = nvrtc::arch_flag(major, minor);

    let mut runtime = Runtime::new(disk.clone());
    let compiled = runtime
        .compile(
            0xF1E,
            &plan,
            &kernels,
            Versions::mirrored(Backend::Cuda.emitter_version()),
            Target {
                major,
                minor,
                device: u64::try_from(device.ordinal()).unwrap_or(0),
                nvrtc: nvrtc::version().expect("nvrtc"),
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
    let ready = launch_control::readiness(&control, &rings, &[0], &[1], &alloc, stream.as_ref())
        .expect("readiness");
    assert!(ready, "a seeded input and an empty output is a ready pass");

    let extents = Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        ..Extents::default()
    };
    let prepared =
        Prepared::build(&alloc, stage_plan, &rings, extents, stream.as_ref()).expect("prepare");
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
        &[0],
        &[1],
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
