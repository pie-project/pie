//! Smoke test for the `eval` feature against the **shape-typed, binding-free
//! (v4)** IR. The full §9/matrix/MC parity suite is owned by L7 (hotel), rebased
//! onto the shape-typed `eval`; this is a compile-keeper.

#![cfg(feature = "eval")]

use pie_sampling_ir::bytecode::{decode, encode};
use pie_sampling_ir::eval::{InputBindings, Value, eval};
use pie_sampling_ir::types::*;

#[test]
fn argmax_roundtrip_and_eval() {
    let p = SamplingProgram {
        inputs: vec![InputDecl::new(Shape::vector(8), DType::F32)],
        ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
        outputs: vec![OutputDecl::new(1, OutputKind::Token)],
    };
    let back = decode(&encode(&p)).expect("decode");
    assert_eq!(p, back);

    let logits = Value::F32(vec![0.1, 0.2, 0.9, 0.0, 0.5, 0.3, 0.4, 0.15]);
    let out = eval(&back, &InputBindings::new(&[logits], 7)).expect("eval");
    assert_eq!(out, vec![Value::I32(vec![2])]);
}

#[test]
fn top_p_temperature_masked_argmax() {
    let v = 6u32;
    let p = SamplingProgram {
        inputs: vec![InputDecl::new(Shape::vector(v), DType::F32)],
        ops: vec![
            Op::Input(0),
            Op::Const(Literal::F32(1.0)),
            Op::Const(Literal::F32(0.9)),
            Op::Const(Literal::F32(-1.0e30)),
            Op::Div(0, 1),
            Op::Exp(4),
            Op::ReduceSum(5),
            Op::Div(5, 6),
            Op::PivotThreshold { input: 7, predicate: Predicate::CummassLe(2) },
            Op::Select { cond: 8, a: 7, b: 3 },
            Op::ReduceArgmax(9),
        ],
        outputs: vec![OutputDecl::new(10, OutputKind::Token)],
    };
    p.validate().expect("valid");
    let logits = Value::F32(vec![3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
    let out = eval(&p, &InputBindings::new(&[logits], 1)).expect("eval");
    assert_eq!(out, vec![Value::I32(vec![0])]);
}
