//! The oracle, device half: the GPU against the interpreter, on emitted code.
//!
//! # What this pins
//!
//! `oracle_interp.rs` pins this crate's channel-plane interpreter against
//! `tensor_compiler::eval::interp`, the original golden model, bit for bit.
//! This file closes the other half of `.wiki/driver/progress-metal.md`'s gate item 4: the same
//! program run **on the device** must agree with that interpreter.
//!
//! Because the CPU half already holds, diffing the device against
//! `pipeline::step` here is not the weaker claim it would otherwise be — the
//! two comparisons compose, and the device is transitively held to the golden
//! model without this file having to carry a `BoundTrace` through the GPU path.
//!
//! # Emitted kernels, not stand-ins
//!
//! `device_fire.rs` hand-writes MSL that honours the binding ABI, because what
//! it tests is the driver's half of the protocol — readiness, then regions,
//! then commit. That is the right fixture for that question and the wrong one
//! for this one: a hand-written kernel proves nothing about whether the
//! *emitter's* arithmetic matches the interpreter's.
//!
//! So these cases run [`tensor_compiler::codegen::program::emit_program`] with
//! [`Backend::Metal`] — the real generated MSL for the real compiled stages —
//! against the real `LaunchPackage` from the same compile. One trace produces
//! both the thing the GPU runs and the thing the interpreter runs, so a
//! disagreement is a disagreement about semantics.
//!
//! # What this found: one ulp, and only on transcendentals
//!
//! The device and the interpreter agree **exactly** on plain arithmetic, on
//! reductions, and on the argmax tie-break. They disagree by **one ulp on
//! `exp`**: for `exp(0.5)` Rust's `f32::exp` answers `1.6487212` and Metal's
//! answers `1.6487213`. Both are within half an ulp of the true value; neither
//! is wrong. Two libms rounded a transcendental differently, which is what
//! libms do, and no amount of care in this crate closes it.
//!
//! That is why `.wiki/driver/progress-metal.md`'s gate item 4 says "within its stated tolerance"
//! where item 3 says bit-identical: the interpreter oracle crosses a libm
//! boundary and the token-exactness gate does not. This file is where the
//! tolerance is stated — one ulp, claimed by the `exp` case alone, with a
//! companion case proving plain arithmetic still costs zero so the bound means
//! something.
//!
//! **The tolerance never applies to a decision.** `same_within` compares
//! integer, index and boolean lanes exactly whatever the tolerance says. A
//! magnitude may be a hair off and still be the same answer; an argmax index or
//! a sort permutation is either the same decision or a different token. The
//! reduction and tie-break cases below therefore run at zero tolerance, and
//! widening the constant cannot reach them.
//!
//! # The versions are read, not written
//!
//! [`Versions`] comes from `tensor_compiler`'s own constants rather than a
//! literal. `.wiki/driver/progress-metal.md`'s `identity.rs` entry is about exactly this: the
//! C++ carried `kMetalM1EmitterVersion = 23` while the host emitter had moved
//! to 36, so the compile-cache key had silently drifted. A test that hardcodes
//! the numbers cannot notice that; one that reads them fails to compile when
//! they move.

#![allow(clippy::print_stdout)]

use std::collections::BTreeMap;
use std::rc::Rc;

use driver_metal::channel::{
    ExecPlan, HostOp, PassInputs, StatusOutcome, StepOutcome, Ticket, Value, adopt_launch_package,
    encode_wire, host_take, make_host_instance, step,
};
use driver_metal::{Error, Region};
use driver_metal::gpu::{Archives, Context, DeviceInputs, Externals, Mode, Pool, Prepare, Ring, Runtime, Stepper, Tables};
use tensor_compiler::codegen::program::{Backend, emit_program};
use tensor_compiler::plan::compile_bound;
use tensor_compiler::plan::{COMPILER_VERSION, LANE_TABLE_ABI_VERSION, REGION_PLAN_VERSION};
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType, Shape};
use tensor_ir::validate::bind;

use driver_metal::channel::Versions;

/// The versions the driver checks a package against, taken from the compiler
/// that built it. See the module doc: a literal here is the `= 23` bug.
fn versions() -> Versions {
    Versions {
        compiler: COMPILER_VERSION,
        region_plan: REGION_PLAN_VERSION,
        lane_table: LANE_TABLE_ABI_VERSION,
        emitter: Backend::Metal.emitter_version(),
    }
}

struct Fixture {
    context: Context,
    runtime: Runtime,
    pool: Pool,
    tables: Tables,
    externals: Externals,
    _dir: tempfile::TempDir,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => return None,
        Err(e) => panic!("context: {e}"),
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let kernels = dir.path().join("kernels");
    std::fs::create_dir_all(kernels.join("ptir")).expect("kernels dir");
    // The REAL generated RNG preamble, not the `// rng` stub `device_fire.rs`
    // uses. Emitted PTIR kernels splice this include and call into it, so a
    // stub is a link error the moment a case reaches an `rng` op — and, more
    // to the point, a comparison run against a stubbed RNG would not be
    // running the RNG contract the interpreter implements.
    std::fs::write(
        kernels.join("ptir/ptir_rng.generated.metal"),
        tensor_compiler::codegen::rng::generate_msl_preamble(),
    )
    .expect("rng");
    let runtime =
        Runtime::new(kernels, Archives::new(Some(dir.path().join("archives")))).expect("runtime");
    Some(Fixture {
        context,
        runtime,
        pool: Pool::new(64 << 20),
        tables: Tables::new(),
        externals: Externals::new(),
        _dir: dir,
    })
}

fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

fn epilogue(channels: Vec<ChannelDecl>, ops: Vec<Op>) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

fn profile() -> ModelProfile {
    let mut p = ModelProfile::dummy();
    p.vocab = 8;
    p
}

/// Seed a device ring with one committed cell, by writing the wire bytes the
/// interpreter's codec would write and publishing the tail.
///
/// [`encode_wire`] is the same function `ChannelState::push` uses, which is the
/// point: if the device and the interpreter disagreed about the wire encoding
/// this would seed two different values and the comparison would be meaningless.
/// `Ring::new`'s own doc makes the same commitment from the other side ("a ring
/// that the interpreter would size one way and the device another is two rings
/// wearing one channel id").
fn seed_ring(ring: &Ring, value: &Value) {
    let mut bytes = vec![0u8; ring.cell_bytes()];
    encode_wire(value, &mut bytes);
    let cell = ring.pending_cell(0).expect("slot 0 exists");
    // SAFETY: the ring was just created and no GPU work names it yet; the
    // handle is `cell_bytes` wide and the source is exactly that long.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            cell.contents().cast::<u8>().as_ptr(),
            bytes.len(),
        );
    }
    // Publish by writing the tail word. `Ring` exposes no producer-side put
    // because on the serving path the *device* is the producer — a host that
    // could advance the tail could publish a cell the GPU had not written.
    // A test seeding a ring is the one legitimate exception, and it writes the
    // word rather than gaining an API that would exist only for it.
    //
    // SAFETY: the words buffer is 32 shared, 8-aligned bytes owned by this
    // ring, index 1 is the tail, and no GPU work names it yet.
    unsafe {
        ring.words()
            .contents()
            .cast::<u64>()
            .as_ptr()
            .add(1)
            .write(1);
    }
}

/// Read every committed cell a ring holds back into interpreter `Value`s.
fn drain_ring(ring: &Ring) -> Option<Value> {
    let (head, tail) = (ring.head(), ring.tail());
    if head >= tail {
        return None;
    }
    let cell = ring.committed_cell(head).expect("committed slot");
    // SAFETY: `execute`'s completion fence has been waited on by the time a
    // test reads, and the handle is one wire cell wide.
    let bytes = unsafe {
        std::slice::from_raw_parts(cell.contents().cast::<u8>().as_ptr(), ring.cell_bytes())
    };
    driver_metal::channel::decode_wire(bytes, ring.dtype(), ring.numel())
}

/// What one side of the comparison read back: a value per host-readable
/// channel, in channel order.
type Readback = Vec<(u32, Value)>;

/// Distance between two `f32`s in representable steps.
///
/// Monotone bit patterns make this a subtraction: for same-signed finites the
/// ordered bit patterns are adjacent integers, so their difference counts the
/// floats between them. Opposite signs are mapped through zero. `NaN` is at
/// infinite distance from everything including itself, because a `NaN` where a
/// number was expected is never "close".
fn ulps(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let key = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

/// Lane comparison within `tolerance` representable steps.
///
/// **Integer, index and boolean lanes are always exact**, whatever the
/// tolerance says, and that is the important half of this function. A tolerance
/// is defensible for a magnitude that a human reads; it is never defensible for
/// an argmax index or a sort permutation, because those are *decisions*, and a
/// decision is either the same decision or a different one. Widening the
/// tolerance can never quietly start accepting a different token.
fn same_within(a: &Value, b: &Value, tolerance: u64) -> bool {
    match (a, b) {
        (Value::F32(x), Value::F32(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| ulps(*p, *q) <= tolerance)
        }
        (Value::I32(x), Value::I32(y)) => x == y,
        (Value::U32(x), Value::U32(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        _ => false,
    }
}

/// The largest lane-wise gap between two readbacks, in ulps.
fn worst_gap(a: &[(u32, Value)], b: &[(u32, Value)]) -> u64 {
    a.iter()
        .zip(b)
        .filter_map(|((_, x), (_, y))| match (x, y) {
            (Value::F32(p), Value::F32(q)) => p.iter().zip(q).map(|(s, t)| ulps(*s, *t)).max(),
            _ => Some(0),
        })
        .max()
        .unwrap_or(0)
}

/// Run one container on the device and through the interpreter, and return
/// each side's readable-channel values.
fn run_both(
    f: &mut Fixture,
    container: TraceContainer,
    seeds: &[(u32, Value)],
) -> (Readback, Readback) {
    let bound = bind(container, profile()).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted: Vec<driver_api::plan::EmittedKernel> =
        emit_program(Backend::Metal, &stages, &bound)
            .into_iter()
            .map(|k| driver_api::plan::EmittedKernel {
                kind: k.kind,
                stage_index: k.stage_index,
                region_index: k.region_index,
                entry_name: k.entry_name,
                source: k.source,
                error: k.error,
            })
            .collect();

    let readable: Vec<u32> = bound
        .container
        .channels
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d.host_role, HostRole::Reader))
        .map(|(i, _)| i as u32)
        .collect();

    // ── the interpreter ─────────────────────────────────────────────────────
    let plan: ExecPlan = adopt_launch_package(package.clone()).expect("the package adopts");
    let seed_map: BTreeMap<u32, Value> = seeds.iter().cloned().collect();
    let mut inst = make_host_instance(&plan, &BTreeMap::new(), &seed_map);
    let cpu_outcome = step(&mut inst, &plan, &PassInputs::none());
    assert_eq!(
        cpu_outcome,
        StepOutcome::Committed,
        "the interpreter must commit for the comparison to mean anything"
    );
    let cpu: Vec<(u32, Value)> = readable
        .iter()
        .filter_map(|&c| match host_take(&inst, &plan, c) {
            (HostOp::Ok, Some(v)) => Some((c, v)),
            _ => None,
        })
        .collect();

    // ── the device ──────────────────────────────────────────────────────────
    let device_plan: ExecPlan = adopt_launch_package(package).expect("the package adopts");
    let rings: Vec<Rc<Ring>> = device_plan
        .package
        .channels
        .iter()
        .enumerate()
        .map(|(i, decl)| {
            let ring = Rc::new(
                Ring::new(
                    &f.context,
                    driver_metal::channel::concrete_dtype(decl.dtype),
                    decl.shape
                        .iter()
                        .map(|&d| d as usize)
                        .product::<usize>()
                        .max(1),
                    decl.capacity as usize,
                )
                .expect("ring"),
            );
            if let Some(v) = seed_map.get(&(i as u32)) {
                seed_ring(&ring, v);
            }
            ring
        })
        .collect();

    let signature = device_plan.package.plans[0].signature_hash;
    let program = f
        .runtime
        .compile(&f.context, signature, &device_plan, versions(), &emitted)
        .expect("the emitted kernels compile");

    // One ticket per channel, pinning each ring exactly where it stands.
    //
    // `check_words` requires channels, effects and tickets to be the same
    // length, because each is a per-channel fact and a short array would
    // silently check fewer channels than the fire has. And a ticket is not
    // optional for a putting fire: `Ticket::default()` is unpinned at both
    // ends, and an unpinned put is refused with `Reason::Unpinned` rather than
    // raced, because it cannot be ordered against another put to the same ring.
    //
    // A single fire against rings only this test touches is the degenerate
    // composition, so "where the ring is now" is the right pin.
    let tickets: Vec<Ticket> = rings
        .iter()
        .map(|r| Ticket {
            expected_head: r.head(),
            expected_tail: r.tail(),
        })
        .collect();
    let prepared = f
        .runtime
        .prepare(&f.context, &f.pool, &program, &rings, &tickets)
        .expect("prepare");
    let Prepare::Ready(fire) = prepared else {
        panic!(
            "the seeded rings should satisfy readiness, got {prepared:?}; \
             ring words: {:?}",
            rings
                .iter()
                .map(|r| (r.head(), r.tail(), r.poison(), r.closed()))
                .collect::<Vec<_>>()
        );
    };

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let execution = f
        .runtime
        .execute(
            &f.context,
            &mut stepper,
            &mut f.tables,
            &f.pool,
            &f.externals,
            &fire,
            &DeviceInputs::default(),
            Mode::Singleton,
        )
        .expect("execute");
    assert_eq!(
        execution.outcome,
        StatusOutcome::Committed,
        "the fire did not commit: {:?}",
        execution.report
    );

    let gpu: Vec<(u32, Value)> = readable
        .iter()
        .filter_map(|&c| drain_ring(&rings[c as usize]).map(|v| (c, v)))
        .collect();

    (cpu, gpu)
}

/// Assert the device and the interpreter agree within `tolerance` ulps.
///
/// `tolerance: 0` is the default and what most cases use. A case that passes a
/// nonzero tolerance is making a claim about the platform, not relaxing a
/// standard, and must say which operation earns it.
fn agree_within(cpu: &[(u32, Value)], gpu: &[(u32, Value)], tolerance: u64, case: &str) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{case}: different channels produced a value — interpreter {:?}, device {:?}",
        cpu.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
        gpu.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
    );
    assert!(!cpu.is_empty(), "{case}: neither side produced anything");
    for ((cc, cv), (gc, gv)) in cpu.iter().zip(gpu) {
        assert_eq!(cc, gc, "{case}: channel order differs");
        assert!(
            same_within(cv, gv, tolerance),
            "{case}: channel {cc} differs by more than {tolerance} ulp — \
             interpreter {cv:?}, device {gv:?}"
        );
    }
}

/// The common case: exact agreement.
fn agree(cpu: &[(u32, Value)], gpu: &[(u32, Value)], case: &str) {
    agree_within(cpu, gpu, 0, case);
}

// ─────────────────────────────── the cases ──────────────────────────────────

/// The tolerance `exp` earns, and nothing else does. See the module doc.
const TRANSCENDENTAL_ULPS: u64 = 1;

#[test]
fn a_transcendental_costs_one_ulp_between_metal_and_rust_and_no_more() {
    // This is the case that found the gap, and it is deliberately written to
    // fail in BOTH directions: more than one ulp fails the tolerance, and a
    // silent convergence to zero would make `worst_gap`'s report wrong without
    // failing — so the report is printed rather than asserted, and the bound is.
    //
    // `exp(0.5)` is the lane that disagrees on this machine: Rust's `f32::exp`
    // answers `1.6487212`, Metal's answers `1.6487213`. Both are within a half
    // ulp of the true value; neither is wrong. Two libms rounded a
    // transcendental differently, which is what libms do.
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let (cpu, gpu) = run_both(
        &mut f,
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::Exp(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[(
            0,
            Value::F32(vec![0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25]),
        )],
    );
    println!("exp: worst lane gap {} ulp", worst_gap(&cpu, &gpu));
    agree_within(&cpu, &gpu, TRANSCENDENTAL_ULPS, "exp");
}

#[test]
fn plain_arithmetic_costs_nothing_which_is_what_makes_the_transcendental_bound_meaningful() {
    // The companion to the case above. If everything drifted by an ulp, a
    // one-ulp tolerance would be a shrug. It does not: multiply, add and
    // subtract are IEEE-exact operations, both sides do them in f32, and both
    // sides land on the same bits. So the tolerance the `exp` case takes is a
    // statement about transcendentals specifically, not about the device.
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let (cpu, gpu) = run_both(
        &mut f,
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::Mul(0, 1),
                Op::Sub(2, 0),
                Op::ChanPut { chan: 2, value: 3 },
            ],
        ),
        &[
            (
                0,
                Value::F32(vec![1e7, 3.0, -1e7, 0.1, -0.5, 2.0, -2.0, 0.25]),
            ),
            (
                1,
                Value::F32(vec![1e-7, 7.0, 1e7, 0.3, 1.5, 0.125, -8.0, 1e20]),
            ),
        ],
    );
    agree(&cpu, &gpu, "mul + sub");
}

#[test]
fn the_device_and_the_interpreter_agree_on_a_reduction() {
    // The reduction is the case worth running on real silicon: the interpreter
    // folds a width-32 pairwise tree on purpose, and a GPU reduction that
    // reassociates lands on different bits. If the emitter ever emits a
    // threadgroup reduction whose order differs, this is what says so.
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let (cpu, gpu) = run_both(
        &mut f,
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::SCALAR, DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ReduceSum(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[(
            0,
            Value::F32(vec![1e8, 1.0, -1e8, 1.0, 1e-8, 1.0, -1e-8, 1.0]),
        )],
    );
    agree(&cpu, &gpu, "reduce_sum");
}

#[test]
fn the_device_and_the_interpreter_break_an_argmax_tie_the_same_way() {
    // The tie-break is a contract, not an artefact of iteration order, and a
    // GPU argmax is the place most likely to have picked the other one.
    let Some(mut f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let (cpu, gpu) = run_both(
        &mut f,
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::SCALAR, DType::I32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        &[(0, Value::F32(vec![2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0]))],
    );
    agree(&cpu, &gpu, "reduce_argmax");
}
