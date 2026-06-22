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
fn optimizer_distributes_cast_over_join() {
    let mut plan = LayoutPlan::new();
    let a_decl = decl(0, "a", &[2], Encoding::Raw(DType::F32));
    let b_decl = decl(1, "b", &[2], Encoding::Raw(DType::F32));
    let a = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: a_decl.clone(),
    });
    let b = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: b_decl.clone(),
    });
    let joined_decl = decl(2, "joined", &[4], Encoding::Raw(DType::F32));
    let joined = plan.push(LayoutExpr::Join {
        inputs: vec![a, b],
        axis: Axis(0),
        decl: joined_decl,
    });
    let cast_decl = decl(3, "out", &[4], Encoding::Raw(DType::BF16));
    let cast = plan.push(LayoutExpr::Cast {
        input: joined,
        dtype: DType::BF16,
        decl: cast_decl.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: cast,
        runtime_name: "out".to_string(),
        decl: cast_decl,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    let join_has_cast_inputs = optimized.exprs.iter().any(|expr| {
        let LayoutExpr::Join { inputs, .. } = expr else {
            return false;
        };
        inputs.iter().all(|input| {
            matches!(
                optimized.exprs[input.0 as usize],
                LayoutExpr::Cast {
                    dtype: DType::BF16,
                    ..
                }
            )
        })
    });
    assert!(join_has_cast_inputs);

    let mut sources = HashMap::new();
    sources.insert(TensorId(0), TensorValue::new(a_decl, vec![1, 2]).unwrap());
    sources.insert(TensorId(1), TensorValue::new(b_decl, vec![3, 4]).unwrap());
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["out"].data,
        evaluate(&plan, &sources).unwrap()["out"].data
    );
}

#[test]
fn optimizer_pushes_select_through_decode() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "q",
        &[8, 2],
        Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
    );
    let scale_decl = decl(1, "q.scale", &[8], Encoding::Raw(DType::F32));
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: q_decl.clone(),
    });
    let scale = plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: scale_decl.clone(),
    });
    let decoded_decl = decl(2, "decoded", &[8, 2], Encoding::Raw(DType::BF16));
    let decoded = plan.push(LayoutExpr::Decode {
        scheme: QuantScheme::Fp8E4M3,
        data: q,
        metadata: vec![scale],
        decl: decoded_decl,
    });
    let out_decl = decl(3, "out", &[4, 2], Encoding::Raw(DType::BF16));
    let selected = plan.push(LayoutExpr::Select {
        input: decoded,
        axis: Axis(0),
        start: 2,
        length: 4,
        decl: out_decl.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: selected,
        runtime_name: "out".to_string(),
        decl: out_decl,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    assert!(optimized.exprs.iter().any(|expr| matches!(
        expr,
        LayoutExpr::Decode { decl, metadata, .. }
            if decl.shape == vec![4, 2] && metadata.iter().any(|id| matches!(
                optimized.exprs[id.0 as usize],
                LayoutExpr::Select { length: 4, .. }
            ))
    )));

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(q_decl, (0..16).collect()).unwrap(),
    );
    sources.insert(
        TensorId(1),
        TensorValue::new(scale_decl, (0..8).collect()).unwrap(),
    );
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["out"].data,
        evaluate(&plan, &sources).unwrap()["out"].data
    );
}

#[test]
fn optimizer_fuses_cast_into_decode() {
    let mut plan = LayoutPlan::new();
    let q_decl = decl(
        0,
        "q",
        &[4],
        Encoding::Quant(quant(QuantScheme::Fp8E4M3, DType::BF16)),
    );
    let q = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: q_decl.clone(),
    });
    let decoded_decl = decl(1, "decoded", &[4], Encoding::Raw(DType::BF16));
    let decoded = plan.push(LayoutExpr::Decode {
        scheme: QuantScheme::Fp8E4M3,
        data: q,
        metadata: Vec::new(),
        decl: decoded_decl,
    });
    let cast_decl = decl(2, "out", &[4], Encoding::Raw(DType::F32));
    let cast = plan.push(LayoutExpr::Cast {
        input: decoded,
        dtype: DType::F32,
        decl: cast_decl.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: cast,
        runtime_name: "out".to_string(),
        decl: cast_decl,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    assert!(optimized.exprs.iter().any(|expr| matches!(
        expr,
        LayoutExpr::Decode { decl, .. } if decl.encoding == Encoding::Raw(DType::F32)
    )));
    assert!(
        !optimized
            .exprs
            .iter()
            .any(|expr| matches!(expr, LayoutExpr::Cast { .. }))
    );

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(q_decl, vec![1, 2, 3, 4]).unwrap(),
    );
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["out"].data,
        evaluate(&plan, &sources).unwrap()["out"].data
    );
}

#[test]
fn optimizer_hoists_aligned_encode_through_select() {
    let mut plan = LayoutPlan::new();
    let raw = decl(0, "raw", &[128], Encoding::Raw(DType::BF16));
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: raw.clone(),
    });
    let selected_decl = decl(1, "selected", &[32], Encoding::Raw(DType::BF16));
    let selected = plan.push(LayoutExpr::Select {
        input: source,
        axis: Axis(0),
        start: 32,
        length: 32,
        decl: selected_decl,
    });
    let encoded_decl = decl(
        2,
        "encoded",
        &[32],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let encoded = plan.push(LayoutExpr::Encode {
        scheme: QuantScheme::AwqInt4,
        input: selected,
        metadata_outputs: Vec::new(),
        decl: encoded_decl.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: encoded,
        runtime_name: "encoded".to_string(),
        decl: encoded_decl,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    assert!(optimized.exprs.iter().any(|expr| matches!(
        expr,
        LayoutExpr::Encode { decl, .. } if decl.shape == vec![128]
    )));
    assert!(optimized.exprs.iter().any(|expr| matches!(
        expr,
        LayoutExpr::Select { decl, .. }
            if decl.encoding == Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16))
    )));

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(raw, (0..128).collect()).unwrap(),
    );
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["encoded"].data,
        evaluate(&plan, &sources).unwrap()["encoded"].data
    );
}

#[test]
fn optimizer_fuses_encode_decode_to_transcode() {
    let mut plan = LayoutPlan::new();
    let awq = decl(
        0,
        "awq",
        &[4],
        Encoding::Quant(quant(QuantScheme::AwqInt4, DType::BF16)),
    );
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: awq.clone(),
    });
    let decoded = decl(1, "decoded", &[4], Encoding::Raw(DType::BF16));
    let dec = plan.push(LayoutExpr::Decode {
        scheme: QuantScheme::AwqInt4,
        data: source,
        metadata: Vec::new(),
        decl: decoded,
    });
    let gptq = decl(
        2,
        "gptq",
        &[4],
        Encoding::Quant(quant(QuantScheme::GptqInt4, DType::BF16)),
    );
    let enc = plan.push(LayoutExpr::Encode {
        scheme: QuantScheme::GptqInt4,
        input: dec,
        metadata_outputs: Vec::new(),
        decl: gptq.clone(),
    });
    let out = plan.push(LayoutExpr::Realize {
        input: enc,
        runtime_name: "gptq".to_string(),
        decl: gptq,
    });
    plan.outputs.push(out);

    let optimized = optimize(plan.clone()).unwrap();
    assert_eq!(
        optimized
            .exprs
            .iter()
            .filter(|expr| matches!(expr, LayoutExpr::Transcode { .. }))
            .count(),
        1
    );

    let mut sources = HashMap::new();
    sources.insert(
        TensorId(0),
        TensorValue::new(awq, vec![11, 12, 13, 14]).unwrap(),
    );
    assert_eq!(
        evaluate(&optimized, &sources).unwrap()["gptq"].data,
        evaluate(&plan, &sources).unwrap()["gptq"].data
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
