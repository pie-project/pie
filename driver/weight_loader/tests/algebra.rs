use std::collections::HashMap;

use pie_weight_loader::ir::{LayoutExpr, LayoutPlan};
use pie_weight_loader::optimizer::optimize;
use pie_weight_loader::reference::{TensorValue, evaluate};
use pie_weight_loader::typecheck::typecheck;
use pie_weight_loader::types::{
    Axis, DType, Encoding, ExprId, Layout, QuantScheme, QuantSpec, Sharding, TensorDecl, TensorId,
};

#[test]
fn typecheck_rejects_bad_select_range() {
    let mut plan = LayoutPlan::new();
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: decl(0, "x", &[2, 3], Encoding::Raw(DType::BF16)),
    });
    plan.push(LayoutExpr::Select {
        input: source,
        axis: Axis(0),
        start: 1,
        length: 2,
        decl: decl(1, "bad", &[2, 3], Encoding::Raw(DType::BF16)),
    });

    let err = typecheck(&plan).unwrap_err().to_string();
    assert!(err.contains("exceeds dim"));
}

#[test]
fn reference_evaluates_join_select_stack_reorder() {
    let mut plan = LayoutPlan::new();
    let a_decl = decl(0, "a", &[2, 2], Encoding::Raw(DType::I32));
    let b_decl = decl(1, "b", &[2, 2], Encoding::Raw(DType::I32));
    let a = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: a_decl.clone(),
    });
    let b = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: b_decl.clone(),
    });
    let joined_decl = decl(2, "joined", &[4, 2], Encoding::Raw(DType::I32));
    let joined = plan.push(LayoutExpr::Join {
        inputs: vec![a, b],
        axis: Axis(0),
        decl: joined_decl,
    });
    let selected_decl = decl(3, "selected", &[2, 2], Encoding::Raw(DType::I32));
    let selected = plan.push(LayoutExpr::Select {
        input: joined,
        axis: Axis(0),
        start: 1,
        length: 2,
        decl: selected_decl,
    });
    let stacked_decl = decl(4, "stacked", &[2, 2, 2], Encoding::Raw(DType::I32));
    let stacked = plan.push(LayoutExpr::Stack {
        inputs: vec![selected, selected],
        axis: Axis(0),
        decl: stacked_decl,
    });
    let reordered_decl = decl(5, "runtime", &[2, 2, 2], Encoding::Raw(DType::I32));
    let reordered = plan.push(LayoutExpr::Reorder {
        input: stacked,
        perm: vec![1, 0, 2],
        decl: reordered_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: reordered,
        runtime_name: "runtime".to_string(),
        decl: reordered_decl,
    });
    plan.outputs.push(realized);

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(a_decl, vec![1, 2, 3, 4]).unwrap(),
    );
    sources.insert(
        TensorId(1),
        TensorValue::new(b_decl, vec![5, 6, 7, 8]).unwrap(),
    );

    let outputs = evaluate(&plan, &sources).unwrap();
    assert_eq!(outputs["runtime"].data, vec![3, 4, 3, 4, 5, 6, 5, 6]);
}

#[test]
fn optimizer_collapses_select_chain_without_changing_values() {
    let mut plan = LayoutPlan::new();
    let source_decl = decl(0, "x", &[6], Encoding::Raw(DType::I32));
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: source_decl.clone(),
    });
    let first = plan.push(LayoutExpr::Select {
        input: source,
        axis: Axis(0),
        start: 1,
        length: 4,
        decl: decl(1, "s1", &[4], Encoding::Raw(DType::I32)),
    });
    let second_decl = decl(2, "s2", &[2], Encoding::Raw(DType::I32));
    let second = plan.push(LayoutExpr::Select {
        input: first,
        axis: Axis(0),
        start: 2,
        length: 2,
        decl: second_decl.clone(),
    });
    let realized = plan.push(LayoutExpr::Realize {
        input: second,
        runtime_name: "out".to_string(),
        decl: second_decl,
    });
    plan.outputs.push(realized);

    let optimized = optimize(plan.clone()).unwrap();
    assert!(optimized.exprs.iter().any(|expr| matches!(
        expr,
        LayoutExpr::Select { input, start: 3, .. } if *input == ExprId(0)
    )));

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(source_decl, vec![0, 1, 2, 3, 4, 5]).unwrap(),
    );
    assert_eq!(
        evaluate(&plan, &sources).unwrap()["out"].data,
        evaluate(&optimized, &sources).unwrap()["out"].data
    );
}

#[test]
fn optimizer_pushes_select_through_join_and_drops_dead_full_join() {
    let mut plan = LayoutPlan::new();
    let a_decl = decl(0, "a", &[2], Encoding::Raw(DType::I32));
    let b_decl = decl(1, "b", &[2], Encoding::Raw(DType::I32));
    let a = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: a_decl.clone(),
    });
    let b = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: b_decl.clone(),
    });
    let joined = plan.push(LayoutExpr::Join {
        inputs: vec![a, b],
        axis: Axis(0),
        decl: decl(2, "joined", &[4], Encoding::Raw(DType::I32)),
    });
    let selected_decl = decl(3, "selected", &[2], Encoding::Raw(DType::I32));
    let selected = plan.push(LayoutExpr::Select {
        input: joined,
        axis: Axis(0),
        start: 1,
        length: 2,
        decl: selected_decl.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: selected,
        runtime_name: "out".to_string(),
        decl: selected_decl,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    assert!(optimized.exprs.len() < plan.exprs.len() + 2);
    assert!(
        optimized
            .exprs
            .iter()
            .any(|expr| matches!(expr, LayoutExpr::Join { .. }))
    );

    let mut sources = HashMap::new();
    sources.insert(TensorId(0), TensorValue::new(a_decl, vec![1, 2]).unwrap());
    sources.insert(TensorId(1), TensorValue::new(b_decl, vec![3, 4]).unwrap());
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["out"].data,
        vec![2, 3]
    );
}

#[test]
fn encode_decode_transcode_are_explicit_reference_ops() {
    let mut plan = LayoutPlan::new();
    let raw = decl(0, "raw", &[4], Encoding::Raw(DType::BF16));
    let encoded = decl(
        1,
        "awq",
        &[4],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let transcoded = decl(
        2,
        "gptq",
        &[4],
        Encoding::Quant(quant(QuantScheme::GptqInt4, DType::BF16)),
    );
    let decoded = decl(3, "decoded", &[4], Encoding::Raw(DType::BF16));
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: raw.clone(),
    });
    let enc = plan.push(LayoutExpr::Encode {
        scheme: QuantScheme::AwqInt4,
        input: source,
        metadata_outputs: Vec::new(),
        decl: encoded,
    });
    let trans = plan.push(LayoutExpr::Transcode {
        from: QuantScheme::AwqInt4,
        to: QuantScheme::GptqInt4,
        data: enc,
        metadata: Vec::new(),
        metadata_outputs: Vec::new(),
        decl: transcoded,
    });
    let dec = plan.push(LayoutExpr::Decode {
        scheme: QuantScheme::GptqInt4,
        data: trans,
        metadata: Vec::new(),
        decl: decoded.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: dec,
        runtime_name: "decoded".to_string(),
        decl: decoded,
    });
    plan.outputs.push(out);

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(raw, vec![11, 12, 13, 14]).unwrap(),
    );
    assert_eq!(
        evaluate(&plan, &sources).unwrap()["decoded"].data,
        vec![11, 12, 13, 14]
    );
}

fn decl(id: u32, name: &str, shape: &[i64], encoding: Encoding) -> TensorDecl {
    TensorDecl {
        id: TensorId(id),
        name: name.to_string(),
        shape: shape.to_vec(),
        encoding,
        layout: Layout::dense(1),
        sharding: Sharding::replicated(),
        alignment: 1,
    }
}

fn quant(scheme: QuantScheme, dtype: DType) -> QuantSpec {
    QuantSpec {
        scheme,
        logical_dtype: dtype,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: None,
        scale_dtype: Some(DType::F32),
        zero_point_dtype: None,
        block_shape: Vec::new(),
    }
}
