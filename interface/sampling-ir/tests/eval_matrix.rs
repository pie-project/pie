//! Matrix (rank ≥ 2) per-row eval tests for the shape-typed, **binding-free (①)**
//! IR — the L7 (hotel) shape-aware follow-up to alpha's scalar/vector `eval`.
//!
//! Each program declares typed `inputs`, references them with `Op::Input(index)`,
//! and is evaluated over an `[m, n]` block with a positional `InputBindings`; the
//! per-row results are checked against hand-computed values. Covers: per-row
//! argmax / reduce / scan, shape-directed `Broadcast` (`[m] → [m, n]`),
//! `GatherRow` (`p[i, draft[i]]`), matrix `Rng` (flattened `row*n + col`), and
//! per-row `PivotThreshold`.

#![cfg(feature = "eval")]

use pie_sampling_ir::eval::{InputBindings, Value, eval};
use pie_sampling_ir::types::*;

fn mat(m: u32, n: u32) -> InputDecl {
    InputDecl::new(Shape::matrix(m, n), DType::F32)
}
fn vec_in(n: u32, dt: DType) -> InputDecl {
    InputDecl::new(Shape::vector(n), dt)
}
fn f32s(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(x) => x.clone(),
        _ => panic!("expected F32, got {v:?}"),
    }
}

// ── per-row argmax over a [m, n] block ──────────────────────────────────────
#[test]
fn matrix_argmax_per_row() {
    let (m, n) = (3u32, 4u32);
    let p = SamplingProgram {
        inputs: vec![mat(m, n)],
        ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
        outputs: vec![OutputDecl::new(1, OutputKind::Token)],
    };
    p.validate().expect("valid");

    let logits = [
        0.1, 0.2, 0.9, 0.3, // row0 → 2
        1.5, 0.4, 0.1, 0.2, // row1 → 0
        0.0, 0.1, 0.2, 0.7, // row2 → 3
    ];
    let out = eval(&p, &InputBindings::new(&[Value::F32(logits.to_vec())], 7)).expect("eval");
    assert_eq!(out, vec![Value::I32(vec![2, 0, 3])]);
}

// ── per-row softmax (broadcast denom) then argmax ───────────────────────────
#[test]
fn matrix_softmax_rows() {
    let (m, n) = (2u32, 3u32);
    let p = SamplingProgram {
        inputs: vec![mat(m, n)],
        ops: vec![
            Op::Input(0),                                           // 0
            Op::Exp(0),                                             // 1
            Op::ReduceSum(1),                                       // 2 [m]
            Op::Broadcast { value: 2, shape: Shape::matrix(m, n) }, // 3 [m,n]
            Op::Div(1, 3),                                          // 4 probs
            Op::ReduceArgmax(4),                                    // 5 [m]
        ],
        outputs: vec![
            OutputDecl::new(4, OutputKind::Distribution),
            OutputDecl::new(5, OutputKind::Token),
        ],
    };
    p.validate().expect("valid");

    let logits = [2.0, 1.0, 0.0, /* r0→0 */ 0.0, 0.5, 3.0 /* r1→2 */];
    let out = eval(&p, &InputBindings::new(&[Value::F32(logits.to_vec())], 1)).expect("eval");
    let probs = f32s(&out[0]);
    assert!((probs[0..3].iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert!((probs[3..6].iter().sum::<f32>() - 1.0).abs() < 1e-5);
    assert_eq!(out[1], Value::I32(vec![0, 2]));
}

// ── per-row broadcast of a per-row temperature ([m] → [m, n]) ───────────────
#[test]
fn matrix_broadcast_per_row_scalar() {
    let (m, n) = (2u32, 3u32);
    let p = SamplingProgram {
        inputs: vec![mat(m, n), vec_in(m, DType::F32)],
        ops: vec![
            Op::Input(0),                                          // 0 logits
            Op::Input(1),                                          // 1 temp [m]
            Op::Broadcast { value: 1, shape: Shape::matrix(m, n) }, // 2 [m,n]
            Op::Div(0, 2),                                         // 3 scaled
        ],
        outputs: vec![OutputDecl::new(3, OutputKind::Distribution)],
    };
    p.validate().expect("valid");

    let logits = Value::F32(vec![2.0, 4.0, 6.0, 10.0, 20.0, 30.0]);
    let temp = Value::F32(vec![2.0, 10.0]); // row0 / 2, row1 / 10
    let out = eval(&p, &InputBindings::new(&[logits, temp], 1)).expect("eval");
    assert_eq!(f32s(&out[0]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

// ── GatherRow: p[i, draft[i]] (the lossless accept-ratio lookup) ────────────
#[test]
fn matrix_gather_row() {
    let (m, n) = (3u32, 4u32);
    let p = SamplingProgram {
        inputs: vec![mat(m, n), vec_in(m, DType::I32)],
        ops: vec![
            Op::Input(0),                     // 0 = "p"
            Op::Input(1),                     // 1 draft [m]
            Op::GatherRow { src: 0, idx: 1 }, // 2 [m]
        ],
        outputs: vec![OutputDecl::new(2, OutputKind::Distribution)],
    };
    p.validate().expect("valid");

    let pmat = Value::F32(vec![
        0.1, 0.2, 0.3, 0.4, // row 0, draft 3 → 0.4
        0.9, 0.05, 0.03, 0.02, // row 1, draft 0 → 0.9
        0.25, 0.25, 0.25, 0.25, // row 2, draft 9 (OOB) → 0.0 (fill-0)
    ]);
    let draft = Value::I32(vec![3, 0, 9]);
    let out = eval(&p, &InputBindings::new(&[pmat, draft], 1)).expect("eval");
    assert_eq!(f32s(&out[0]), vec![0.4, 0.9, 0.0]);
}

// ── matrix Rng: flattened row*n+col ⇒ rows are decorrelated ─────────────────
#[test]
fn matrix_rng_rows_decorrelate() {
    let (m, n) = (4u32, 8u32);
    let p = SamplingProgram {
        inputs: vec![],
        ops: vec![Op::Rng { stream: 0, shape: Shape::matrix(m, n), kind: RngKind::Uniform }],
        outputs: vec![OutputDecl::new(0, OutputKind::Distribution)],
    };
    p.validate().expect("valid");

    let out = eval(&p, &InputBindings::new(&[], 12345)).expect("eval");
    let u = f32s(&out[0]);
    assert_eq!(u.len() as u32, m * n);
    assert!(u.iter().all(|&x| x > 0.0 && x < 1.0));
    assert_ne!(&u[0..8], &u[8..16]); // row 0 vs row 1 — no aliasing
    let out2 = eval(&p, &InputBindings::new(&[], 12345)).expect("eval");
    assert_eq!(u, f32s(&out2[0])); // determinism
}

// ── per-row pivot threshold (top-1 per row → masked argmax) ─────────────────
#[test]
fn matrix_pivot_threshold_per_row() {
    let (m, n) = (2u32, 4u32);
    let p = SamplingProgram {
        inputs: vec![mat(m, n)],
        ops: vec![
            Op::Input(0),                                                     // 0 logits
            Op::Const(Literal::U32(1)),                                       // 1 k = 1
            Op::Const(Literal::F32(-1.0e30)),                                 // 2 mask fill
            Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(1) }, // 3 mask [m,n] (k=id 1)
            Op::Select { cond: 3, a: 0, b: 2 },                               // 4 masked logits
            Op::ReduceArgmax(4),                                              // 5 per-row token
        ],
        outputs: vec![OutputDecl::new(5, OutputKind::Token)],
    };
    p.validate().expect("valid");

    let logits = [
        0.1, 0.9, 0.2, 0.3, // row 0 top-1 @ col 1
        0.5, 0.4, 0.8, 0.1, // row 1 top-1 @ col 2
    ];
    let out = eval(&p, &InputBindings::new(&[Value::F32(logits.to_vec())], 1)).expect("eval");
    assert_eq!(out, vec![Value::I32(vec![1, 2])]);
}

// ── Broadcast preserves the source dtype (foxtrot's mirostat fix) ───────────
#[test]
fn broadcast_preserves_int_dtype() {
    // Const i32 scalar → Broadcast [3] → use as a Gather index into a vector.
    let p = SamplingProgram {
        inputs: vec![vec_in(5, DType::F32)],
        ops: vec![
            Op::Input(0),                                       // 0 src vec
            Op::Const(Literal::I32(2)),                         // 1 i32 scalar
            Op::Broadcast { value: 1, shape: Shape::vector(3) }, // 2 i32 [3]
            Op::Gather { src: 0, idx: 2 },                      // 3 src[2] ×3
        ],
        outputs: vec![OutputDecl::new(3, OutputKind::Distribution)],
    };
    p.validate().expect("valid");

    let src = Value::F32(vec![10.0, 11.0, 12.0, 13.0, 14.0]);
    let out = eval(&p, &InputBindings::new(&[src], 1)).expect("eval");
    assert_eq!(f32s(&out[0]), vec![12.0, 12.0, 12.0]);
}
