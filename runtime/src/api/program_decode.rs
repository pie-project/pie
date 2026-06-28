//! `tensor::Program → pie_sampling_ir::SamplingProgram` decode — the runtime
//! half of the host program decoder (alpha owns the IR half:
//! [`program_from_parts`](pie_sampling_ir::program_from_parts) +
//! [`OutputKind::from_dtype`](pie_sampling_ir::OutputKind::from_dtype)).
//!
//! Each WIT `op-kind` maps field-by-field to [`witmap::OpKind`](pie_sampling_ir::OpKind)
//! → [`Op`](pie_sampling_ir::Op); each `input` to an [`InputDecl`]; the bare
//! output value-ids are handed to `program_from_parts`, which types the SSA body
//! and infers each output's [`OutputKind`] from its dtype (the Submit-for-MVP
//! convention (a): int ⇒ `Token`, `f32` ⇒ `Scalar`).
//!
//! Decoding **untrusted inferlet input** is fallible by construction — a shape
//! with rank > `MAX_RANK`, a malformed SSA reference, or a bad output dtype
//! returns `Err`, never panics.
//!
//! NOTE (bravo): this is the working scaffold against the live wasmtime
//! bindings; alpha's oracle-verified field-copy is a clean swap at this exact
//! signature.

use crate::api::pie::core::tensor as wit;
use pie_sampling_ir::{
    DType, InputDecl, Literal, Op, OpKind, OutputDecl, OutputKind, Predicate, Readiness, RngKind,
    SamplingProgram, Shape,
};

/// Decode the structured `(inputs, ops, outputs)` handed to the WIT `program`
/// resource constructor into a fully-validated [`SamplingProgram`].
///
/// The IR is **positional SSA**: the op at list position `i` defines value ids
/// `[next_id .. next_id + result_count())`, so operand ids in each `op-kind`
/// reference values by position. The guest emit numbers each `op.outputs[k].id`
/// positionally by construction (golf/foxtrot-confirmed), so the explicit ids
/// carried in the WIT are redundant with the position. We still **verify** them
/// (id == position, `outputs` arity == the kind's `result_count`): on this trust
/// boundary a non-positional id or wrong arity is emit drift or adversarial
/// input, and is rejected with a clean `Err` rather than silently miscompiled.
pub fn decode_program(
    inputs: Vec<wit::Input>,
    ops: Vec<wit::Op>,
    outputs: Vec<wit::Output>,
) -> Result<SamplingProgram, String> {
    let inputs: Vec<InputDecl> = inputs
        .into_iter()
        .map(|i| {
            // lane-3 (#21): readiness now survives the WIT wire (golf lane-1
            // `record input.ready` + foxtrot lane-2 emit). Carry it through so a
            // `Late` input routes to the device-alias late channel (the gather in
            // `attach_program`); v4 programs without it decode `Submit` (alpha's
            // additive default — `tensor::Readiness::Submit`).
            Ok(InputDecl::with_ready(
                shape_from_wit(i.shape)?,
                dtype_from_wit(i.dtype),
                readiness_from_wit(i.ready),
            ))
        })
        .collect::<Result<_, String>>()?;

    let mut ir_ops: Vec<Op> = Vec::with_capacity(ops.len());
    let mut next_id: u32 = 0;
    for (op_index, op) in ops.into_iter().enumerate() {
        let ir_op = op_kind_from_wit(op.kind)?.to_op();
        let arity = ir_op.result_count();
        if op.outputs.len() as u32 != arity {
            return Err(format!(
                "op {op_index}: declares {} output value(s) but op-kind defines {arity}",
                op.outputs.len()
            ));
        }
        for (k, v) in op.outputs.iter().enumerate() {
            let expected = next_id + k as u32;
            if v.id != expected {
                return Err(format!(
                    "op {op_index}: output value id {} is not positional SSA (expected {expected})",
                    v.id
                ));
            }
        }
        next_id += arity;
        ir_ops.push(ir_op);
    }

    // Each declared output carries its marshaling kind EXPLICITLY (#18): the WIT
    // front door no longer drops it, so typed float kinds
    // (`Logits`/`Logprobs`/`Distribution`, all `F32`) survive instead of
    // re-inferring to `Scalar` via `from_dtype`.
    let output_decls: Vec<OutputDecl> = outputs
        .into_iter()
        .map(|o| OutputDecl::new(o.id, output_kind_from_wit(o.kind)))
        .collect();

    pie_sampling_ir::program_from_parts_typed(inputs, ir_ops, output_decls)
        .map_err(|e| format!("invalid sampling program: {e:?}"))
}

fn output_kind_from_wit(k: wit::OutputKind) -> OutputKind {
    match k {
        wit::OutputKind::Token => OutputKind::Token,
        wit::OutputKind::Distribution => OutputKind::Distribution,
        wit::OutputKind::Logits => OutputKind::Logits,
        wit::OutputKind::Logprobs => OutputKind::Logprobs,
        wit::OutputKind::Entropy => OutputKind::Entropy,
        wit::OutputKind::Scalar => OutputKind::Scalar,
        wit::OutputKind::Embedding => OutputKind::Embedding,
    }
}

fn shape_from_wit(dims: Vec<u32>) -> Result<Shape, String> {
    Shape::new(&dims).ok_or_else(|| {
        format!(
            "tensor shape rank {} exceeds MAX_RANK ({})",
            dims.len(),
            pie_sampling_ir::MAX_RANK
        )
    })
}

fn dtype_from_wit(d: wit::Dtype) -> DType {
    match d {
        wit::Dtype::F32 => DType::F32,
        wit::Dtype::I32 => DType::I32,
        wit::Dtype::U32 => DType::U32,
        wit::Dtype::Bool => DType::Bool,
    }
}

fn readiness_from_wit(r: wit::Readiness) -> Readiness {
    match r {
        wit::Readiness::Submit => Readiness::Submit,
        wit::Readiness::Late => Readiness::Late,
    }
}

fn literal_from_wit(l: wit::Literal) -> Literal {
    match l {
        wit::Literal::F32(v) => Literal::F32(v),
        wit::Literal::I32(v) => Literal::I32(v),
        wit::Literal::U32(v) => Literal::U32(v),
        wit::Literal::Bool(v) => Literal::Bool(v),
    }
}

fn predicate_from_wit(p: wit::Predicate) -> Predicate {
    match p {
        wit::Predicate::RankLe(k) => Predicate::RankLe(k),
        wit::Predicate::CummassLe(v) => Predicate::CummassLe(v),
        wit::Predicate::ProbGe(v) => Predicate::ProbGe(v),
    }
}

fn rng_kind_from_wit(k: wit::RngKind) -> RngKind {
    match k {
        wit::RngKind::Uniform => RngKind::Uniform,
        wit::RngKind::Gumbel => RngKind::Gumbel,
    }
}

fn op_kind_from_wit(k: wit::OpKind) -> Result<OpKind, String> {
    Ok(match k {
        wit::OpKind::Input(i) => OpKind::Input(i),
        wit::OpKind::Const(l) => OpKind::Const(literal_from_wit(l)),
        wit::OpKind::Exp(a) => OpKind::Exp(a),
        wit::OpKind::Log(a) => OpKind::Log(a),
        wit::OpKind::Neg(a) => OpKind::Neg(a),
        wit::OpKind::Recip(a) => OpKind::Recip(a),
        wit::OpKind::Abs(a) => OpKind::Abs(a),
        wit::OpKind::Sign(a) => OpKind::Sign(a),
        wit::OpKind::Add((a, b)) => OpKind::Add((a, b)),
        wit::OpKind::Sub((a, b)) => OpKind::Sub((a, b)),
        wit::OpKind::Mul((a, b)) => OpKind::Mul((a, b)),
        wit::OpKind::Div((a, b)) => OpKind::Div((a, b)),
        wit::OpKind::MaxElem((a, b)) => OpKind::MaxElem((a, b)),
        wit::OpKind::MinElem((a, b)) => OpKind::MinElem((a, b)),
        wit::OpKind::Gt((a, b)) => OpKind::Gt((a, b)),
        wit::OpKind::Ge((a, b)) => OpKind::Ge((a, b)),
        wit::OpKind::Eq((a, b)) => OpKind::Eq((a, b)),
        wit::OpKind::Select((c, a, b)) => OpKind::Select((c, a, b)),
        wit::OpKind::ReduceSum(a) => OpKind::ReduceSum(a),
        wit::OpKind::ReduceMax(a) => OpKind::ReduceMax(a),
        wit::OpKind::ReduceMin(a) => OpKind::ReduceMin(a),
        wit::OpKind::ReduceArgmax(a) => OpKind::ReduceArgmax(a),
        wit::OpKind::Broadcast((a, shape)) => OpKind::Broadcast((a, shape_from_wit(shape)?)),
        wit::OpKind::Cumsum(a) => OpKind::CumSum(a),
        wit::OpKind::Cumprod(a) => OpKind::CumProd(a),
        wit::OpKind::SortDesc(a) => OpKind::SortDesc(a),
        wit::OpKind::PivotThreshold((a, p)) => {
            OpKind::PivotThreshold((a, predicate_from_wit(p)))
        }
        wit::OpKind::Gather((a, b)) => OpKind::Gather((a, b)),
        wit::OpKind::GatherRow((a, b)) => OpKind::GatherRow((a, b)),
        wit::OpKind::ScatterAdd((a, b, c)) => OpKind::ScatterAdd((a, b, c)),
        wit::OpKind::ScatterSet((a, b, c)) => OpKind::ScatterSet((a, b, c)),
        wit::OpKind::MaskApply((a, b)) => OpKind::MaskApply((a, b)),
        wit::OpKind::Rng((stream, shape, kind)) => {
            OpKind::Rng((stream, shape_from_wit(shape)?, rng_kind_from_wit(kind)))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every WIT `op-kind` variant (incl. each literal / predicate / rng-kind
    /// sub-variant) maps to the canonical `OpKind` — the host-side analog of
    /// alpha's `witmap` drift oracle. The match's exhaustiveness guarantees all
    /// 32 are *handled*; this guards against a *mis*-map the compiler can't see
    /// (e.g. an `Add`/`Sub` swap, or a `Cumsum → CumProd` slip).
    #[test]
    fn op_kind_from_wit_maps_every_variant() {
        let cases: Vec<(wit::OpKind, OpKind)> = vec![
            (wit::OpKind::Input(3), OpKind::Input(3)),
            (wit::OpKind::Const(wit::Literal::F32(0.5)), OpKind::Const(Literal::F32(0.5))),
            (wit::OpKind::Const(wit::Literal::I32(-7)), OpKind::Const(Literal::I32(-7))),
            (wit::OpKind::Const(wit::Literal::U32(9)), OpKind::Const(Literal::U32(9))),
            (wit::OpKind::Const(wit::Literal::Bool(true)), OpKind::Const(Literal::Bool(true))),
            (wit::OpKind::Exp(0), OpKind::Exp(0)),
            (wit::OpKind::Log(0), OpKind::Log(0)),
            (wit::OpKind::Neg(0), OpKind::Neg(0)),
            (wit::OpKind::Recip(0), OpKind::Recip(0)),
            (wit::OpKind::Abs(0), OpKind::Abs(0)),
            (wit::OpKind::Sign(0), OpKind::Sign(0)),
            (wit::OpKind::Add((0, 1)), OpKind::Add((0, 1))),
            (wit::OpKind::Sub((0, 1)), OpKind::Sub((0, 1))),
            (wit::OpKind::Mul((0, 1)), OpKind::Mul((0, 1))),
            (wit::OpKind::Div((0, 1)), OpKind::Div((0, 1))),
            (wit::OpKind::MaxElem((0, 1)), OpKind::MaxElem((0, 1))),
            (wit::OpKind::MinElem((0, 1)), OpKind::MinElem((0, 1))),
            (wit::OpKind::Gt((0, 1)), OpKind::Gt((0, 1))),
            (wit::OpKind::Ge((0, 1)), OpKind::Ge((0, 1))),
            (wit::OpKind::Eq((0, 1)), OpKind::Eq((0, 1))),
            (wit::OpKind::Select((0, 1, 2)), OpKind::Select((0, 1, 2))),
            (wit::OpKind::ReduceSum(0), OpKind::ReduceSum(0)),
            (wit::OpKind::ReduceMax(0), OpKind::ReduceMax(0)),
            (wit::OpKind::ReduceMin(0), OpKind::ReduceMin(0)),
            (wit::OpKind::ReduceArgmax(0), OpKind::ReduceArgmax(0)),
            (wit::OpKind::Broadcast((0, vec![2, 4])), OpKind::Broadcast((0, Shape::matrix(2, 4)))),
            (wit::OpKind::Cumsum(0), OpKind::CumSum(0)),
            (wit::OpKind::Cumprod(0), OpKind::CumProd(0)),
            (wit::OpKind::SortDesc(0), OpKind::SortDesc(0)),
            (wit::OpKind::PivotThreshold((0, wit::Predicate::RankLe(40))), OpKind::PivotThreshold((0, Predicate::RankLe(40)))),
            (wit::OpKind::PivotThreshold((0, wit::Predicate::CummassLe(1))), OpKind::PivotThreshold((0, Predicate::CummassLe(1)))),
            (wit::OpKind::PivotThreshold((0, wit::Predicate::ProbGe(1))), OpKind::PivotThreshold((0, Predicate::ProbGe(1)))),
            (wit::OpKind::Gather((0, 1)), OpKind::Gather((0, 1))),
            (wit::OpKind::GatherRow((0, 1)), OpKind::GatherRow((0, 1))),
            (wit::OpKind::ScatterAdd((0, 1, 2)), OpKind::ScatterAdd((0, 1, 2))),
            (wit::OpKind::ScatterSet((0, 1, 2)), OpKind::ScatterSet((0, 1, 2))),
            (wit::OpKind::MaskApply((0, 1)), OpKind::MaskApply((0, 1))),
            (wit::OpKind::Rng((0, vec![4], wit::RngKind::Uniform)), OpKind::Rng((0, Shape::vector(4), RngKind::Uniform))),
            (wit::OpKind::Rng((7, vec![2, 4], wit::RngKind::Gumbel)), OpKind::Rng((7, Shape::matrix(2, 4), RngKind::Gumbel))),
        ];
        for (w, expected) in cases {
            assert_eq!(op_kind_from_wit(w).expect("maps"), expected);
        }
    }

    fn vinput(vocab: u32) -> wit::Input {
        wit::Input { shape: vec![vocab], dtype: wit::Dtype::F32, ready: wit::Readiness::Submit }
    }
    fn op_in0(vocab: u32) -> wit::Op {
        wit::Op {
            outputs: vec![wit::Value { id: 0, shape: vec![vocab], dtype: wit::Dtype::F32 }],
            kind: wit::OpKind::Input(0),
        }
    }
    fn tok_out(id: u32) -> wit::Output {
        wit::Output { id, kind: wit::OutputKind::Token }
    }

    #[test]
    fn decode_carries_input_readiness_lane3() {
        // lane-3: the WIT `record input.ready` survives decode → `InputDecl.ready`,
        // so a `Late` input routes to the device-alias late channel (the
        // `attach_program` gather) instead of decoding all-`Submit`. The
        // additive-v4 default (`Submit`) is preserved.
        let argmax = || {
            vec![
                op_in0(8),
                wit::Op {
                    outputs: vec![wit::Value { id: 1, shape: vec![], dtype: wit::Dtype::I32 }],
                    kind: wit::OpKind::ReduceArgmax(0),
                },
            ]
        };
        let late_in = wit::Input {
            shape: vec![8],
            dtype: wit::Dtype::F32,
            ready: wit::Readiness::Late,
        };
        let prog = decode_program(vec![late_in], argmax(), vec![tok_out(1)]).expect("decodes");
        assert_eq!(prog.inputs[0].ready, Readiness::Late);

        // The `Submit` default (`vinput`) round-trips to `Submit`.
        let prog2 = decode_program(vec![vinput(8)], argmax(), vec![tok_out(1)]).expect("decodes");
        assert_eq!(prog2.inputs[0].ready, Readiness::Submit);
    }

    #[test]
    fn decode_rejects_non_positional_output_id() {
        // op1 (ReduceArgmax) declares its output id as 5, not the positional 1 —
        // adversarial/drifted SSA must reject, not silently miscompile.
        let ops = vec![
            op_in0(8),
            wit::Op {
                outputs: vec![wit::Value { id: 5, shape: vec![], dtype: wit::Dtype::I32 }],
                kind: wit::OpKind::ReduceArgmax(0),
            },
        ];
        let err = decode_program(vec![vinput(8)], ops, vec![tok_out(1)]).unwrap_err();
        assert!(err.contains("not positional SSA"), "got: {err}");
    }

    #[test]
    fn decode_rejects_output_arity_mismatch() {
        // ReduceArgmax defines exactly 1 value, but the WIT op declares 2 — a
        // bad arity must reject before it corrupts the positional id stream.
        let ops = vec![
            op_in0(8),
            wit::Op {
                outputs: vec![
                    wit::Value { id: 1, shape: vec![], dtype: wit::Dtype::I32 },
                    wit::Value { id: 2, shape: vec![], dtype: wit::Dtype::I32 },
                ],
                kind: wit::OpKind::ReduceArgmax(0),
            },
        ];
        let err = decode_program(vec![vinput(8)], ops, vec![tok_out(1)]).unwrap_err();
        assert!(err.contains("op-kind defines"), "got: {err}");
    }

    #[test]
    fn decode_accepts_sortdesc_two_positional_ids() {
        // SortDesc defines 2 values (ids 1,2) — the arity/positional guards must
        // accept the contiguous multi-output, and `program_from_parts` types it.
        let ops = vec![
            op_in0(8),
            wit::Op {
                outputs: vec![
                    wit::Value { id: 1, shape: vec![8], dtype: wit::Dtype::F32 },
                    wit::Value { id: 2, shape: vec![8], dtype: wit::Dtype::I32 },
                ],
                kind: wit::OpKind::SortDesc(0),
            },
        ];
        // output the sorted-value vector (id 1) — f32 ⇒ Scalar is wrong shape, so
        // use the index vector (id 2, i32 ⇒ Token) as the program output.
        let program = decode_program(vec![vinput(8)], ops, vec![tok_out(2)]).expect("valid sortdesc");
        assert_eq!(program.ops.len(), 2);
    }

    #[test]
    fn decode_carries_logits_output_kind_18() {
        // #18: a typed F32 `Logits` output survives the WIT front door instead of
        // re-inferring to `Scalar` (the old `from_dtype(F32)=Scalar` bug). The
        // value (input 0, `[8]` f32) is declared `Logits`; decode must preserve it
        // so the driver marshals it as logits, not a scalar.
        let prog = decode_program(
            vec![vinput(8)],
            vec![op_in0(8)],
            vec![wit::Output { id: 0, kind: wit::OutputKind::Logits }],
        )
        .expect("decodes a logits output");
        assert_eq!(prog.outputs.len(), 1);
        assert_eq!(prog.outputs[0].kind, OutputKind::Logits, "Logits must survive, not Scalar");
        // And the kind rides the encoded bytecode (the byte the driver reader reads).
        let bytecode = pie_sampling_ir::encode(&prog);
        let back = pie_sampling_ir::decode(&bytecode).expect("re-decode");
        assert_eq!(back.outputs[0].kind, OutputKind::Logits);
    }
}
