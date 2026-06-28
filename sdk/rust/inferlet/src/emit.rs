//! Guest emit — lower a canonical `pie-sampling-ir` [`SamplingProgram`] to the
//! WIT [`tensor::Program`](crate::tensor::Program) front door.
//!
//! This is the **guest half** of the `Op ↔ op-kind` bridge. The op-kind mapping
//! rides alpha's zero-drift `OpKind` oracle ([`sampling_edsl::ir::OpKind`],
//! `witmap.rs`): `op.to_op_kind()` already produces a structure that is field-for
//! -field identical to the WIT `op-kind` variant, so [`op_kind_to_wit`] is a pure
//! rename onto the generated bindings. Only the leaf types
//! (literal / predicate / rng-kind / shape / dtype) need a structural copy.
//!
//! The program is binding-free: `Op::Input(i)` references a typed input slot, and
//! the attach-time bindings (`LoweredProgram::bindings`) are golf's SDK concern
//! (`Vec<InputBinding>` for [`Forward::sampler`](crate::forward::Forward::sampler)).
//! This module owns only `SamplingProgram → tensor::Program`.
//!
//! [`SamplingProgram`]: crate::sampling::ir::SamplingProgram

use crate::Result;
use crate::tensor;
use sampling_edsl::ir;

/// Lower a binding-free [`SamplingProgram`](ir::SamplingProgram) to a reusable
/// [`tensor::Program`](crate::tensor::Program). The program is type-inferred (and
/// thereby validated) so each op's explicit `outputs: list<value>` can be
/// materialized from the inferred value types — exactly what the host validator
/// re-checks on decode.
pub fn emit_program(program: &ir::SamplingProgram) -> Result<tensor::Program> {
    let parts = lower_parts(program)?;
    Ok(tensor::Program::new(&parts.inputs, &parts.ops, &parts.outputs))
}

/// The pure (host-testable) half of [`emit_program`]: the three `program`
/// constructor argument lists. Separated from the resource construction so the
/// op-kind/shape/dtype mapping is unit-testable off-wasm (the `program` resource
/// constructor is a host call, only live under `wasm32`).
pub(crate) struct ProgramParts {
    pub inputs: Vec<tensor::Input>,
    pub ops: Vec<tensor::Op>,
    pub outputs: Vec<tensor::Output>,
}

pub(crate) fn lower_parts(program: &ir::SamplingProgram) -> Result<ProgramParts> {
    // Per-value-id inferred types (also rejects undefined operands / type
    // mismatches) — used to fill each op's explicit `outputs`.
    let types =
        ir::value_types(program).map_err(|e| format!("invalid sampling program: {e:?}"))?;

    let inputs = program
        .inputs
        .iter()
        .map(|d| tensor::Input {
            shape: shape_to_wit(d.shape),
            dtype: dtype_to_wit(d.dtype),
            ready: readiness_to_wit(d.ready),
        })
        .collect();

    let mut ops = Vec::with_capacity(program.ops.len());
    let mut next_id: ir::ValueId = 0;
    for op in &program.ops {
        let n = op.result_count();
        let outputs = (0..n)
            .map(|k| {
                let id = next_id + k;
                let vt = types[id as usize];
                tensor::Value {
                    id,
                    shape: shape_to_wit(vt.shape),
                    dtype: dtype_to_wit(vt.dtype),
                }
            })
            .collect();
        ops.push(tensor::Op {
            outputs,
            kind: op_kind_to_wit(op.to_op_kind()),
        });
        next_id += n;
    }

    // Carry the DECLARED OutputKind through the front door (not bare ids) so
    // typed float kinds (logits/logprobs/distribution) survive the decode
    // instead of re-inferring to scalar (#18).
    let outputs = program
        .outputs
        .iter()
        .map(|o| tensor::Output { id: o.value, kind: output_kind_to_wit(o.kind) })
        .collect();
    Ok(ProgramParts {
        inputs,
        ops,
        outputs,
    })
}

// ---------------------------------------------------------------------------
// Leaf-type conversions (IR → generated WIT bindings).
// ---------------------------------------------------------------------------

pub(crate) fn dtype_to_wit(d: ir::DType) -> tensor::Dtype {
    match d {
        ir::DType::F32 => tensor::Dtype::F32,
        ir::DType::I32 => tensor::Dtype::I32,
        ir::DType::U32 => tensor::Dtype::U32,
        ir::DType::Bool => tensor::Dtype::Bool,
    }
}

/// WIT `readiness`: when an input-slot's binding value is ready at attach. Late
/// inputs (e.g. the grammar mask) ride the host late H2D channel; the host
/// reconstructs `InputDecl.ready` from this on decode (lane-3).
fn readiness_to_wit(r: ir::Readiness) -> tensor::Readiness {
    match r {
        ir::Readiness::Submit => tensor::Readiness::Submit,
        ir::Readiness::Late => tensor::Readiness::Late,
    }
}

/// WIT `output-kind`: the declared marshaling kind of a program output, carried
/// explicitly so the typed float kinds (logits/logprobs/distribution) survive
/// the front door instead of re-inferring to scalar (#18).
fn output_kind_to_wit(k: ir::OutputKind) -> tensor::OutputKind {
    match k {
        ir::OutputKind::Token => tensor::OutputKind::Token,
        ir::OutputKind::Distribution => tensor::OutputKind::Distribution,
        ir::OutputKind::Logits => tensor::OutputKind::Logits,
        ir::OutputKind::Logprobs => tensor::OutputKind::Logprobs,
        ir::OutputKind::Entropy => tensor::OutputKind::Entropy,
        ir::OutputKind::Scalar => tensor::OutputKind::Scalar,
        ir::OutputKind::Embedding => tensor::OutputKind::Embedding,
    }
}

/// WIT `shape = list<u32>`: `[]` scalar, `[n]` vector, `[m, n]` matrix.
pub(crate) fn shape_to_wit(s: ir::Shape) -> Vec<u32> {
    s.dims().to_vec()
}

fn literal_to_wit(l: ir::Literal) -> tensor::Literal {
    match l {
        ir::Literal::F32(x) => tensor::Literal::F32(x),
        ir::Literal::I32(x) => tensor::Literal::I32(x),
        ir::Literal::U32(x) => tensor::Literal::U32(x),
        ir::Literal::Bool(x) => tensor::Literal::Bool(x),
    }
}

fn predicate_to_wit(p: ir::Predicate) -> tensor::Predicate {
    match p {
        ir::Predicate::RankLe(n) => tensor::Predicate::RankLe(n),
        ir::Predicate::CummassLe(v) => tensor::Predicate::CummassLe(v),
        ir::Predicate::ProbGe(v) => tensor::Predicate::ProbGe(v),
    }
}

fn rng_kind_to_wit(k: ir::RngKind) -> tensor::RngKind {
    match k {
        ir::RngKind::Uniform => tensor::RngKind::Uniform,
        ir::RngKind::Gumbel => tensor::RngKind::Gumbel,
    }
}

/// Map a canonical [`OpKind`](ir::OpKind) (alpha's oracle) onto the generated WIT
/// `op-kind`. Structurally a field-by-field copy — the value-id operands pass
/// through unchanged; only nested leaf types are converted.
fn op_kind_to_wit(k: ir::OpKind) -> tensor::OpKind {
    use tensor::OpKind as W;
    match k {
        ir::OpKind::Input(i) => W::Input(i),
        ir::OpKind::Const(l) => W::Const(literal_to_wit(l)),
        ir::OpKind::Exp(a) => W::Exp(a),
        ir::OpKind::Log(a) => W::Log(a),
        ir::OpKind::Neg(a) => W::Neg(a),
        ir::OpKind::Recip(a) => W::Recip(a),
        ir::OpKind::Abs(a) => W::Abs(a),
        ir::OpKind::Sign(a) => W::Sign(a),
        ir::OpKind::Add(t) => W::Add(t),
        ir::OpKind::Sub(t) => W::Sub(t),
        ir::OpKind::Mul(t) => W::Mul(t),
        ir::OpKind::Div(t) => W::Div(t),
        ir::OpKind::MaxElem(t) => W::MaxElem(t),
        ir::OpKind::MinElem(t) => W::MinElem(t),
        ir::OpKind::Gt(t) => W::Gt(t),
        ir::OpKind::Ge(t) => W::Ge(t),
        ir::OpKind::Eq(t) => W::Eq(t),
        ir::OpKind::Select(t) => W::Select(t),
        ir::OpKind::ReduceSum(a) => W::ReduceSum(a),
        ir::OpKind::ReduceMax(a) => W::ReduceMax(a),
        ir::OpKind::ReduceMin(a) => W::ReduceMin(a),
        ir::OpKind::ReduceArgmax(a) => W::ReduceArgmax(a),
        ir::OpKind::Broadcast((v, s)) => W::Broadcast((v, shape_to_wit(s))),
        ir::OpKind::CumSum(a) => W::Cumsum(a),
        ir::OpKind::CumProd(a) => W::Cumprod(a),
        ir::OpKind::SortDesc(a) => W::SortDesc(a),
        ir::OpKind::PivotThreshold((v, p)) => W::PivotThreshold((v, predicate_to_wit(p))),
        ir::OpKind::Gather(t) => W::Gather(t),
        ir::OpKind::GatherRow(t) => W::GatherRow(t),
        ir::OpKind::MaskApply(t) => W::MaskApply(t),
        ir::OpKind::ScatterAdd(t) => W::ScatterAdd(t),
        ir::OpKind::ScatterSet(t) => W::ScatterSet(t),
        ir::OpKind::Rng((stream, s, kind)) => W::Rng((stream, shape_to_wit(s), rng_kind_to_wit(kind))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sampling_edsl::program::{grammar, mirostat, spec_verify_lossless};

    /// Reverse the WIT `op-kind` back to a canonical [`OpKind`](ir::OpKind) so the
    /// emit can be checked op-for-op against `op.to_op_kind()` — full op-kind
    /// parity coverage without depending on the generated types' `PartialEq`.
    fn wit_op_kind_to_ir(k: &tensor::OpKind) -> ir::OpKind {
        use tensor::OpKind as W;
        match *k {
            W::Input(i) => ir::OpKind::Input(i),
            W::Const(ref l) => ir::OpKind::Const(wit_literal_to_ir(l)),
            W::Exp(a) => ir::OpKind::Exp(a),
            W::Log(a) => ir::OpKind::Log(a),
            W::Neg(a) => ir::OpKind::Neg(a),
            W::Recip(a) => ir::OpKind::Recip(a),
            W::Abs(a) => ir::OpKind::Abs(a),
            W::Sign(a) => ir::OpKind::Sign(a),
            W::Add(t) => ir::OpKind::Add(t),
            W::Sub(t) => ir::OpKind::Sub(t),
            W::Mul(t) => ir::OpKind::Mul(t),
            W::Div(t) => ir::OpKind::Div(t),
            W::MaxElem(t) => ir::OpKind::MaxElem(t),
            W::MinElem(t) => ir::OpKind::MinElem(t),
            W::Gt(t) => ir::OpKind::Gt(t),
            W::Ge(t) => ir::OpKind::Ge(t),
            W::Eq(t) => ir::OpKind::Eq(t),
            W::Select(t) => ir::OpKind::Select(t),
            W::ReduceSum(a) => ir::OpKind::ReduceSum(a),
            W::ReduceMax(a) => ir::OpKind::ReduceMax(a),
            W::ReduceMin(a) => ir::OpKind::ReduceMin(a),
            W::ReduceArgmax(a) => ir::OpKind::ReduceArgmax(a),
            W::Broadcast((v, ref s)) => ir::OpKind::Broadcast((v, wit_shape_to_ir(s))),
            W::Cumsum(a) => ir::OpKind::CumSum(a),
            W::Cumprod(a) => ir::OpKind::CumProd(a),
            W::SortDesc(a) => ir::OpKind::SortDesc(a),
            W::PivotThreshold((v, ref p)) => {
                ir::OpKind::PivotThreshold((v, wit_predicate_to_ir(p)))
            }
            W::Gather(t) => ir::OpKind::Gather(t),
            W::GatherRow(t) => ir::OpKind::GatherRow(t),
            W::MaskApply(t) => ir::OpKind::MaskApply(t),
            W::ScatterAdd(t) => ir::OpKind::ScatterAdd(t),
            W::ScatterSet(t) => ir::OpKind::ScatterSet(t),
            W::Rng((stream, ref s, k)) => {
                ir::OpKind::Rng((stream, wit_shape_to_ir(s), wit_rng_kind_to_ir(k)))
            }
        }
    }

    fn wit_shape_to_ir(s: &[u32]) -> ir::Shape {
        ir::Shape::new(s).expect("valid shape rank")
    }
    fn wit_literal_to_ir(l: &tensor::Literal) -> ir::Literal {
        match *l {
            tensor::Literal::F32(x) => ir::Literal::F32(x),
            tensor::Literal::I32(x) => ir::Literal::I32(x),
            tensor::Literal::U32(x) => ir::Literal::U32(x),
            tensor::Literal::Bool(x) => ir::Literal::Bool(x),
        }
    }
    fn wit_predicate_to_ir(p: &tensor::Predicate) -> ir::Predicate {
        match *p {
            tensor::Predicate::RankLe(n) => ir::Predicate::RankLe(n),
            tensor::Predicate::CummassLe(v) => ir::Predicate::CummassLe(v),
            tensor::Predicate::ProbGe(v) => ir::Predicate::ProbGe(v),
        }
    }
    fn wit_rng_kind_to_ir(k: tensor::RngKind) -> ir::RngKind {
        match k {
            tensor::RngKind::Uniform => ir::RngKind::Uniform,
            tensor::RngKind::Gumbel => ir::RngKind::Gumbel,
        }
    }
    fn wit_dtype_to_ir(d: tensor::Dtype) -> ir::DType {
        match d {
            tensor::Dtype::F32 => ir::DType::F32,
            tensor::Dtype::I32 => ir::DType::I32,
            tensor::Dtype::U32 => ir::DType::U32,
            tensor::Dtype::Bool => ir::DType::Bool,
        }
    }

    /// The emitted parts reconstruct the source program exactly: input slots,
    /// op count, every op-kind (via the oracle), and each op's explicit output
    /// ids/shapes/dtypes (from `value_types`).
    fn assert_emit_matches(program: &ir::SamplingProgram) {
        let parts = lower_parts(program).expect("emit");
        let types = ir::value_types(program).expect("types");

        // Input slots: shape + dtype, in order.
        assert_eq!(parts.inputs.len(), program.inputs.len());
        for (got, decl) in parts.inputs.iter().zip(&program.inputs) {
            assert_eq!(got.shape, decl.shape.dims());
            assert_eq!(wit_dtype_to_ir(got.dtype), decl.dtype);
        }

        // Ops: same count, op-kind round-trips through the oracle, explicit
        // outputs match the inferred SSA value types with contiguous ids.
        assert_eq!(parts.ops.len(), program.ops.len());
        let mut next_id: u32 = 0;
        for (wop, op) in parts.ops.iter().zip(&program.ops) {
            assert_eq!(
                wit_op_kind_to_ir(&wop.kind),
                op.to_op_kind(),
                "op-kind drift for {op:?}"
            );
            let n = op.result_count();
            assert_eq!(wop.outputs.len() as u32, n);
            for (k, v) in wop.outputs.iter().enumerate() {
                let id = next_id + k as u32;
                assert_eq!(v.id, id);
                assert_eq!(v.shape, types[id as usize].shape.dims());
                assert_eq!(wit_dtype_to_ir(v.dtype), types[id as usize].dtype);
            }
            next_id += n;
        }

        // Program outputs carry (value id, declared marshaling kind), in order —
        // the OutputKind survives the front door (#18). The generated WIT variant
        // has no `PartialEq`, so compare the kind via its frozen discriminant.
        assert_eq!(parts.outputs.len(), program.outputs.len());
        for (got, decl) in parts.outputs.iter().zip(&program.outputs) {
            assert_eq!(got.id, decl.value);
            let got_kind = match got.kind {
                tensor::OutputKind::Token => 0u8,
                tensor::OutputKind::Distribution => 1,
                tensor::OutputKind::Logits => 2,
                tensor::OutputKind::Logprobs => 3,
                tensor::OutputKind::Entropy => 4,
                tensor::OutputKind::Scalar => 5,
                tensor::OutputKind::Embedding => 6,
            };
            assert_eq!(got_kind, decl.kind.to_u8(), "OutputKind drift");
        }
    }

    #[test]
    fn emit_grammar_program() {
        let (b, _) = grammar(128).unwrap();
        assert_emit_matches(&b.program);
    }

    #[test]
    fn emit_mirostat_program() {
        // Two outputs (Token + surprise scalar) — exercises multi-output decls.
        let (b, _) = mirostat(128).unwrap();
        assert_emit_matches(&b.program);
    }

    #[test]
    fn emit_lossless_program() {
        // Matrix shapes + GatherRow + dual Rng streams — the densest op coverage.
        let (b, _) = spec_verify_lossless(64, 4).unwrap();
        assert_emit_matches(&b.program);
    }

    #[test]
    fn emit_every_sugar_sampler() {
        use sampling_edsl::{SamplerSpec, build_sampler};
        for spec in [
            SamplerSpec::Argmax,
            SamplerSpec::TopP { temperature: 0.8, p: 0.9 },
            SamplerSpec::TopK { temperature: 0.8, k: 40 },
            SamplerSpec::MinP { temperature: 0.8, p: 0.05 },
            SamplerSpec::TopKTopP { temperature: 0.8, k: 40, p: 0.9 },
            SamplerSpec::Multinomial { temperature: 0.8 },
        ] {
            let b = build_sampler(spec, 128).expect("build");
            assert_emit_matches(&b.program);
        }
    }
}
