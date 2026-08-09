//! The oracle: this crate's channel-plane interpreter against the original.
//!
//! # What this pins
//!
//! This crate is a port of `driver-metal/csrc/src/pipeline/interp.hpp`, and
//! `interp.hpp` says four times in its own comments that it is itself a copy —
//! *"Mirrors interp.rs eval_op case for case"*, *"interp.rs step, minus the
//! per-layer taps"*. `interp.rs` is [`tensor_compiler::eval::interp`], whose
//! module doc calls it *"the golden model every backend diffs against"*.
//!
//! So the same semantics were maintained in three hand-written copies (the
//! Rust original, CUDA's `tier0_runner.hpp`, and the Metal `interp.hpp` this
//! crate ports), and nothing had ever checked that they agree. This file is
//! that check: one trace, compiled once, run through **both** interpreters,
//! and every observable compared.
//!
//! It is also why this layer is a crate rather than a module of one shell.
//! Two of those three copies existed because a device driver could not import
//! another device driver's directory; a crate both can import is the structural
//! form of the fix, and this file is what proves the surviving copy is right.
//!
//! It is the same argument as `pipeline::status`'s fault-table check — a
//! hand-copied table that nothing verifies drifts — which is why
//! `tensor-compiler` is already a dev-dependency and why this is a test rather
//! than a runtime dependency. The driver must not depend on the compiler to
//! *run*; it should have to prove it agrees with it to *ship*.
//!
//! # The shape of a case
//!
//! One [`TraceContainer`] is the single source. It is bound once, compiled
//! once, and then:
//!
//! * the **golden** side runs the `BoundTrace` through
//!   [`tensor_compiler::eval::interp::Instance`];
//! * the **driver** side lowers that same bound trace to a `LaunchPackage`
//!   through [`tensor_compiler::codegen::launch::build`] — the artefact the
//!   driver actually receives — adopts it, and runs
//!   [`driver::step`].
//!
//! Both sides therefore start from one program, and a disagreement is a
//! disagreement about semantics rather than about test setup. The lowering is
//! in the driver's path on purpose: a copy error in `adopt_launch_package` is
//! exactly as fatal as one in `eval_op`, and only running through the real
//! artefact catches it.
//!
//! # What this deliberately does not cover
//!
//! Per-layer tap stages. Both the C++ and this crate's `step` reject them at
//! classification (`prologue`/`descriptor`/`on_attn*` taps are the increment
//! the port does not claim), so a trace containing one is not a case where the
//! two are *expected* to agree. Every case here is an epilogue program, which
//! is what the channel-plane interpreter exists to run.
//!
//! This is also not yet the device comparison `.wiki/driver/progress-metal.md`'s gate item 4
//! finishes with. It is its first half, and the half that runs without a GPU:
//! proving the driver's interpreter *is* the golden model means a later device
//! test can diff against the cheap local copy and still be making the strong
//! claim.

use std::collections::BTreeMap;

use driver::{
    ExecPlan, HostOp, PassInputs, StepOutcome, Value as DriverValue, adopt_launch_package,
    host_take, make_host_instance, step,
};
use tensor_compiler::eval::interp::{
    Instance, NoKernels, PassInputs as GoldenInputs, Value as GoldenValue,
};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType, Shape};
use tensor_ir::validate::bind;

// ─────────────────────────────── building a case ────────────────────────────

/// A channel declaration. `seeded` channels receive their cell at bind time on
/// both sides; `HostRole::Reader` channels are the ones a case reads back.
fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

/// A one-stage epilogue program — the shape the channel-plane interpreter runs.
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

/// The two interpreters' verdicts on one pass, plus what each left on every
/// host-readable channel.
struct Verdicts {
    golden_committed: bool,
    driver_outcome: StepOutcome,
    golden_reads: Vec<(u32, GoldenValue)>,
    driver_reads: Vec<(u32, DriverValue)>,
}

/// Bind, compile, and run one container through both interpreters.
///
/// `seeds` are `(channel, value)` in the golden's `Value`; the driver's seeds
/// are converted from them rather than written twice, so a case cannot seed the
/// two sides differently — which would turn a real disagreement into a passing
/// test.
fn run_both(
    container: TraceContainer,
    profile: ModelProfile,
    seeds: &[(u32, GoldenValue)],
) -> Verdicts {
    let bound = bind(container, profile).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);

    // Which channels a host may read back, in channel order.
    let readable: Vec<u32> = bound
        .container
        .channels
        .iter()
        .enumerate()
        .filter(|(_, d)| matches!(d.host_role, HostRole::Reader))
        .map(|(i, _)| i as u32)
        .collect();

    // ── the golden ──────────────────────────────────────────────────────────
    let mut golden = Instance::new(&bound, seeds).expect("the golden binds");
    let report = golden
        .step(&bound, &GoldenInputs::default(), &mut NoKernels)
        .expect("the golden steps");
    let golden_reads: Vec<(u32, GoldenValue)> = readable
        .iter()
        .filter_map(|&c| golden.host_take(&bound, c).ok().map(|v| (c, v)))
        .collect();

    // ── the driver ──────────────────────────────────────────────────────────
    let plan: ExecPlan = adopt_launch_package(package).expect("the package adopts");
    let driver_seeds: BTreeMap<u32, DriverValue> =
        seeds.iter().map(|(c, v)| (*c, as_driver(v))).collect();
    let mut inst = make_host_instance(&plan, &BTreeMap::new(), &driver_seeds);
    let driver_outcome = step(&mut inst, &plan, &PassInputs::none());
    let driver_reads: Vec<(u32, DriverValue)> = readable
        .iter()
        .filter_map(|&c| match host_take(&inst, &plan, c) {
            (HostOp::Ok, Some(v)) => Some((c, v)),
            _ => None,
        })
        .collect();

    Verdicts {
        golden_committed: report.committed,
        driver_outcome,
        golden_reads,
        driver_reads,
    }
}

/// Assert the two interpreters agree on a pass: same commit verdict, same set
/// of readable channels, and bit-identical lanes on each.
///
/// Bit-identical, not approximate. Both sides are CPU f32 evaluating the same
/// canonical reduction order, so a difference of one ulp is a copy error in the
/// arithmetic, not rounding — and the tokens these values choose are decided by
/// argmax, where one ulp is a different answer.
fn agree(v: &Verdicts, case: &str) {
    assert_eq!(
        v.golden_committed,
        v.driver_outcome == StepOutcome::Committed,
        "{case}: commit verdicts differ — golden committed={}, driver={:?}",
        v.golden_committed,
        v.driver_outcome
    );
    assert_eq!(
        v.golden_reads.len(),
        v.driver_reads.len(),
        "{case}: different number of readable channels produced a value \
         (golden {:?}, driver {:?})",
        v.golden_reads.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
        v.driver_reads.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
    );
    for ((gc, gv), (dc, dv)) in v.golden_reads.iter().zip(&v.driver_reads) {
        assert_eq!(gc, dc, "{case}: channel order differs");
        assert!(
            same_value(gv, dv),
            "{case}: channel {gc} differs — golden {gv:?}, driver {dv:?}"
        );
    }
}

/// The golden's `Value` as the driver's. The two enums have the same four
/// variants; only `Bool` differs in representation (`bool` against the
/// one-byte-per-lane `u8` the wire codec stores), which is itself a fact worth
/// converting in exactly one place.
fn as_driver(v: &GoldenValue) -> DriverValue {
    match v {
        GoldenValue::F32(x) => DriverValue::F32(x.clone()),
        GoldenValue::I32(x) => DriverValue::I32(x.clone()),
        GoldenValue::U32(x) => DriverValue::U32(x.clone()),
        GoldenValue::Bool(x) => DriverValue::Bool(x.iter().map(|&b| u8::from(b)).collect()),
    }
}

/// Bit-identical comparison across the two `Value` types.
///
/// `f32` lanes compare by `to_bits`, so a `NaN` must match a `NaN` of the same
/// payload and `-0.0` does not pass for `0.0`. Both are pinned rules in
/// `pipeline::op`'s canonical helpers (`max(-0, +0) = +0`, `NaN` never wins an
/// argmax), so a comparison that let them slide would not test the rules.
fn same_value(g: &GoldenValue, d: &DriverValue) -> bool {
    match (g, d) {
        (GoldenValue::F32(a), DriverValue::F32(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        (GoldenValue::I32(a), DriverValue::I32(b)) => a == b,
        (GoldenValue::U32(a), DriverValue::U32(b)) => a == b,
        (GoldenValue::Bool(a), DriverValue::Bool(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| u8::from(x) == y)
        }
        _ => false,
    }
}

/// The profile every case uses: the dummy model at a small vocabulary, so a
/// logits row is short enough to write out and check by eye.
fn profile() -> ModelProfile {
    let mut p = ModelProfile::dummy();
    p.vocab = 8;
    p
}

fn f32s(x: &[f32]) -> GoldenValue {
    GoldenValue::F32(x.to_vec())
}

// ─────────────────────────────── the cases ──────────────────────────────────

#[test]
fn a_sort_descending_orders_and_indexes_identically_in_both_interpreters() {
    // `sort_desc` is the case with a tie in it on purpose: two lanes hold 3.0,
    // and which index each is assigned is a tie-break rule, not arithmetic.
    // `pipeline::op::sort_desc_order` pins it and the golden pins it, and this
    // is the only thing that checks they pinned the same one.
    let v = run_both(
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
                chan(Shape::vector(8), DType::U32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::SortDesc(0),
                Op::ChanPut { chan: 1, value: 1 },
                Op::ChanPut { chan: 2, value: 2 },
            ],
        ),
        profile(),
        &[(0, f32s(&[1.0, 3.0, -2.0, 0.0, 3.0, 9.5, -0.0, 4.25]))],
    );
    agree(&v, "sort_desc");
}

#[test]
fn a_reduction_agrees_bit_for_bit_because_both_fold_in_the_same_order() {
    // The canonical reduction is a width-32 pairwise tree, not a left fold. At
    // eight lanes the two orders give different bits for these values, so this
    // case fails if either side ever "simplifies" to a fold.
    let v = run_both(
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
        profile(),
        &[(0, f32s(&[1e8, 1.0, -1e8, 1.0, 1e-8, 1.0, -1e-8, 1.0]))],
    );
    agree(&v, "reduce_sum");
}

#[test]
fn an_argmax_over_a_tie_picks_the_same_lane_on_both_sides() {
    // The tie-break is the whole content of this case: the maximum appears
    // three times. A rule of "first wins" and a rule of "last wins" are both
    // defensible and only one is the contract.
    let v = run_both(
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
        profile(),
        &[(0, f32s(&[2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0]))],
    );
    agree(&v, "reduce_argmax");
}

#[test]
fn a_maximum_over_negative_and_positive_zero_answers_the_same_signed_zero() {
    // `max(-0.0, +0.0) = +0.0` is a pinned rule that `==` cannot see: both
    // answers compare equal to zero and only their bits differ. `same_value`
    // compares by `to_bits` for exactly this case.
    let v = run_both(
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::SCALAR, DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ReduceMax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        ),
        profile(),
        &[(0, f32s(&[-0.0, -1.0, 0.0, -2.0, -0.0, -3.0, -0.0, -4.0]))],
    );
    agree(&v, "reduce_max signed zero");
}

#[test]
fn a_matmul_skips_a_zero_operand_on_both_sides_so_an_infinity_never_becomes_nan() {
    // Matmul does *not* go through `canonical_reduce` on either side: both
    // contract k-outer into an accumulator, and both guard the inner loop with
    // `if xv == 0.0 { continue }`.
    //
    // That skip is a semantic choice, not an optimisation. `0.0 * inf` is
    // `NaN`, so an implementation that drops the guard turns a cell that should
    // be finite into `NaN` — and `-0.0 == 0.0`, so a negative zero is skipped
    // too. This case is built to catch that: `a[0][1]` is zero and row 1 of `b`
    // is infinite, so the guard is the only reason row 0 of the result is a
    // number at all. Without it the two sides disagree; with it they agree
    // bit for bit, which is what pins the shared quirk rather than assuming it.
    let v = run_both(
        epilogue(
            vec![
                chan(Shape::matrix(2, 3), DType::F32, HostRole::None, true),
                chan(Shape::matrix(3, 4), DType::F32, HostRole::None, true),
                chan(Shape::matrix(2, 4), DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::MatMul(0, 1),
                Op::ChanPut { chan: 2, value: 2 },
            ],
        ),
        profile(),
        &[
            // a[0] = [1e7, 0.0, -1e7]  → the zero must skip b's infinite row
            // a[1] = [3.5, -0.0, 2.0]  → and so must the NEGATIVE zero
            (0, f32s(&[1e7, 0.0, -1e7, 3.5, -0.0, 2.0])),
            (
                1,
                f32s(&[
                    1.0,
                    2.0,
                    0.5,
                    -1.0,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::INFINITY,
                    1.0,
                    -2.0,
                    0.25,
                    4.0,
                ]),
            ),
        ],
    );
    agree(&v, "matmul");
}

#[test]
fn a_readiness_miss_blocks_both_interpreters_and_neither_consumes() {
    // The input channel is NOT seeded, so the take cannot be satisfied. The
    // claim is that both sides refuse the pass rather than running it on a
    // dummy and committing — the pass-atomic rule, checked from the outside.
    let v = run_both(
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, false),
                chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
            ],
            vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
        ),
        profile(),
        &[],
    );
    assert!(!v.golden_committed, "the golden must not commit on a miss");
    assert!(
        matches!(v.driver_outcome, StepOutcome::Blocked(_)),
        "the driver must report the block, got {:?}",
        v.driver_outcome
    );
    agree(&v, "readiness miss");
}

#[test]
fn an_elementwise_chain_over_every_dtype_agrees() {
    // Not a delicate case — a breadth case. It walks a comparison into a
    // select, so `Bool` crosses the boundary where the two crates disagree
    // about representation (`bool` against one byte per lane) and the
    // conversion in `as_driver`/`same_value` is exercised rather than assumed.
    let v = run_both(
        epilogue(
            vec![
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::None, true),
                chan(Shape::vector(8), DType::F32, HostRole::Reader, false),
            ],
            vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::Gt(0, 1),
                Op::Select {
                    cond: 2,
                    a: 0,
                    b: 1,
                },
                Op::ChanPut { chan: 2, value: 3 },
            ],
        ),
        profile(),
        &[
            (0, f32s(&[1.0, -1.0, 0.0, 5.0, -0.0, 2.5, 9.0, -9.0])),
            (1, f32s(&[0.0, 1.0, 0.0, -5.0, 0.0, 2.5, -9.0, 9.0])),
        ],
    );
    agree(&v, "gt + select");
}
