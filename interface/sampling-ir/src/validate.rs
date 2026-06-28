//! IR validation: SSA well-formedness + per-op shape/dtype checking.
//!
//! [`validate`] is the single gate the host runs on every inferlet-supplied
//! program (after lowering WIT-ops → [`SamplingProgram`]) before it reaches the
//! driver.

use alloc::vec::Vec;
use core::fmt;

use crate::types::*;

/// A validation failure. `op_index` is the 0-based position in the op list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationError {
    /// A program must have at least one op.
    NoOps,
    /// An `Input(index)` referenced a slot outside [`SamplingProgram::inputs`].
    InputIndexOutOfRange { op_index: u32, index: u32 },
    /// An operand value id was undefined at this point (out of range, or a
    /// forward reference — both forbidden in SSA order).
    ValueIdOutOfRange { op_index: u32, value: ValueId },
    /// An op's operand shapes are incompatible.
    ShapeMismatch { op_index: u32 },
    /// An op's operand dtypes are incompatible.
    DTypeMismatch { op_index: u32 },
    /// A program declared no outputs.
    NoOutputs,
    /// An output referenced an undefined value id.
    OutputIdOutOfRange { value: ValueId },
    /// An output's declared [`OutputKind`] is inconsistent with its value dtype.
    OutputKindTypeMismatch { index: u32 },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::NoOps => f.write_str("program has no ops"),
            ValidationError::InputIndexOutOfRange { op_index, index } => {
                write!(f, "op {op_index}: input slot {index} out of range")
            }
            ValidationError::ValueIdOutOfRange { op_index, value } => {
                write!(f, "op {op_index}: operand value id {value} is undefined here")
            }
            ValidationError::ShapeMismatch { op_index } => {
                write!(f, "op {op_index}: incompatible operand shapes")
            }
            ValidationError::DTypeMismatch { op_index } => {
                write!(f, "op {op_index}: incompatible operand dtypes")
            }
            ValidationError::NoOutputs => f.write_str("program declares no outputs"),
            ValidationError::OutputIdOutOfRange { value } => {
                write!(f, "output value id {value} is undefined")
            }
            ValidationError::OutputKindTypeMismatch { index } => {
                write!(f, "output #{index}: declared kind is inconsistent with the value dtype")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ValidationError {}

/// Validate a [`SamplingProgram`].
pub fn validate(p: &SamplingProgram) -> Result<(), ValidationError> {
    if p.ops.is_empty() {
        return Err(ValidationError::NoOps);
    }
    let types = compute_types(p)?;

    if p.outputs.is_empty() {
        return Err(ValidationError::NoOutputs);
    }
    for (i, out) in p.outputs.iter().enumerate() {
        let t = types
            .get(out.value as usize)
            .ok_or(ValidationError::OutputIdOutOfRange { value: out.value })?;
        if !out.kind.accepts_dtype(t.dtype) {
            return Err(ValidationError::OutputKindTypeMismatch { index: i as u32 });
        }
    }
    Ok(())
}

/// The per-value [`ValueType`] table (index = value id), inferring each op's
/// result(s) in SSA order. Errors on an undefined operand or type mismatch.
pub fn value_types(p: &SamplingProgram) -> Result<Vec<ValueType>, ValidationError> {
    compute_types(p)
}

fn compute_types(p: &SamplingProgram) -> Result<Vec<ValueType>, ValidationError> {
    let mut types: Vec<ValueType> = Vec::with_capacity(p.ops.len());
    for (op_index, op) in p.ops.iter().enumerate() {
        match infer(op_index as u32, op, &types, &p.inputs)? {
            Results::One(t) => types.push(t),
            Results::Two(a, b) => {
                types.push(a);
                types.push(b);
            }
        }
    }
    Ok(types)
}

/// The [`ValueType`] of every declared output, in `outputs` order. Validates as
/// a side effect.
pub fn output_types(p: &SamplingProgram) -> Result<Vec<ValueType>, ValidationError> {
    validate(p)?;
    let types = compute_types(p)?;
    Ok(p.outputs.iter().map(|o| types[o.value as usize]).collect())
}

/// The declared [`OutputKind`] of every output, in `outputs` order.
pub fn output_kinds(p: &SamplingProgram) -> Result<Vec<OutputKind>, ValidationError> {
    validate(p)?;
    Ok(p.outputs.iter().map(|o| o.kind).collect())
}

/// Assemble + validate a [`SamplingProgram`] from host-decoder parts, inferring
/// each output's [`OutputKind`] from its value dtype.
///
/// This is the IR half of the runtime's `tensor::Program → SamplingProgram`
/// decode: the caller maps each WIT `op-kind` → [`Op`] (via
/// [`OpKind::to_op`](crate::OpKind::to_op)) and `input` → [`InputDecl`], then
/// hands the program's **bare output value-ids** here. The WIT front door carries
/// no [`OutputKind`], so it is inferred per [`OutputKind::from_dtype`] (the
/// Submit-for-MVP convention: int ⇒ `Token`, `F32` ⇒ `Scalar`; a `Bool` output
/// is rejected). The returned program is fully validated — `encode` it for the
/// driver carrier, or read [`output_kinds`] for host marshaling routing.
pub fn program_from_parts(
    inputs: Vec<InputDecl>,
    ops: Vec<Op>,
    output_ids: &[ValueId],
) -> Result<SamplingProgram, ValidationError> {
    // Type the SSA body first (validates ops, operand ids, shapes, dtypes) so we
    // can read each output value's dtype. Outputs are not needed to type the body.
    let scratch = SamplingProgram { inputs, ops, outputs: Vec::new() };
    let types = compute_types(&scratch)?;
    let mut outputs = Vec::with_capacity(output_ids.len());
    for (i, &id) in output_ids.iter().enumerate() {
        let t = types
            .get(id as usize)
            .ok_or(ValidationError::OutputIdOutOfRange { value: id })?;
        let kind = OutputKind::from_dtype(t.dtype)
            .ok_or(ValidationError::OutputKindTypeMismatch { index: i as u32 })?;
        outputs.push(OutputDecl::new(id, kind));
    }
    let SamplingProgram { inputs, ops, .. } = scratch;
    let program = SamplingProgram { inputs, ops, outputs };
    validate(&program)?;
    Ok(program)
}

/// Like [`program_from_parts`] but takes the **declared** [`OutputDecl`]s
/// (value id + [`OutputKind`]) instead of bare value-ids — so the front door
/// carries the kind explicitly and the typed float kinds
/// (`Logits`/`Logprobs`/`Distribution`, all `F32`) survive instead of
/// re-inferring to `Scalar` via [`OutputKind::from_dtype`]. Each declared kind
/// is validated against its value's dtype ([`OutputKind::accepts_dtype`]); a
/// mismatch is rejected. This is the typed-output (#18) decode path.
pub fn program_from_parts_typed(
    inputs: Vec<InputDecl>,
    ops: Vec<Op>,
    outputs: Vec<OutputDecl>,
) -> Result<SamplingProgram, ValidationError> {
    // Type the SSA body first so each declared output's value dtype is known.
    let scratch = SamplingProgram { inputs, ops, outputs: Vec::new() };
    let types = compute_types(&scratch)?;
    for (i, od) in outputs.iter().enumerate() {
        let t = types
            .get(od.value as usize)
            .ok_or(ValidationError::OutputIdOutOfRange { value: od.value })?;
        if !od.kind.accepts_dtype(t.dtype) {
            return Err(ValidationError::OutputKindTypeMismatch { index: i as u32 });
        }
    }
    let SamplingProgram { inputs, ops, .. } = scratch;
    let program = SamplingProgram { inputs, ops, outputs };
    validate(&program)?;
    Ok(program)
}

// ===========================================================================
// Late-bind use-site analysis
// ===========================================================================

/// The first op index that directly reads the value materialized by
/// `Op::Input(input_index)` — the runtime's inject-before barrier for an input
/// bound to a *late* tensor at attach time. `None` if the slot is never
/// materialized or its value never consumed. (Binding/readiness are attach-time;
/// the caller supplies which input indices are late.)
pub fn input_first_use(p: &SamplingProgram, input_index: InputIndex) -> Option<u32> {
    // Find the value id produced by the first `Op::Input(input_index)`.
    let mut next_id = 0u32;
    let mut produced: Option<ValueId> = None;
    for op in &p.ops {
        if let Op::Input(idx) = *op
            && idx == input_index
        {
            produced = Some(next_id);
            break;
        }
        next_id += op.result_count();
    }
    let value = produced?;
    p.ops
        .iter()
        .position(|o| o.operands().contains(&value))
        .map(|i| i as u32)
}

/// First-use barriers for a set of late input slots (e.g. the indices an attach
/// bound to a `Tensor` with [`Readiness::Late`]). Each entry is
/// `(input_index, first_use_op_index)`. Validates the program first.
pub fn late_input_barriers(
    p: &SamplingProgram,
    late_indices: &[InputIndex],
) -> Result<Vec<(InputIndex, Option<u32>)>, ValidationError> {
    validate(p)?;
    Ok(late_indices.iter().map(|&i| (i, input_first_use(p, i))).collect())
}

/// The input indices the program declares [`Readiness::Late`] — derived directly
/// from each [`InputDecl::ready`] (readiness is now a program property, no longer
/// supplied externally). The runtime injects each before its [`input_first_use`]
/// barrier; feed the result to [`late_input_barriers`].
pub fn late_inputs(p: &SamplingProgram) -> Vec<InputIndex> {
    p.inputs
        .iter()
        .enumerate()
        .filter(|(_, inp)| matches!(inp.ready, Readiness::Late | Readiness::SelfSpecDraftInput))
        .map(|(i, _)| i as InputIndex)
        .collect()
}

// ===========================================================================
// Per-op type inference
// ===========================================================================

enum Results {
    One(ValueType),
    Two(ValueType, ValueType),
}

fn ty(types: &[ValueType], id: ValueId, op_index: u32) -> Result<ValueType, ValidationError> {
    types
        .get(id as usize)
        .copied()
        .ok_or(ValidationError::ValueIdOutOfRange { op_index, value: id })
}

/// Elementwise broadcast of two operand shapes (equal, or one scalar).
fn broadcast2(a: Shape, b: Shape) -> Option<Shape> {
    if a == b {
        Some(a)
    } else if a.is_scalar() {
        Some(b)
    } else if b.is_scalar() {
        Some(a)
    } else {
        None
    }
}

/// `Op::Broadcast` rule: `src` left-aligned against `target` (trailing axes
/// padded with `1`); each axis equals the target or is `1`.
fn can_broadcast_to(src: Shape, target: Shape) -> bool {
    if src.rank() > target.rank() {
        return false;
    }
    let sd = src.dims();
    let td = target.dims();
    for i in 0..target.rank() {
        let s = if i < sd.len() { sd[i] } else { 1 };
        if s != td[i] && s != 1 {
            return false;
        }
    }
    true
}

fn infer(
    op_index: u32,
    op: &Op,
    types: &[ValueType],
    inputs: &[InputDecl],
) -> Result<Results, ValidationError> {
    let shape_err = || ValidationError::ShapeMismatch { op_index };
    let dtype_err = || ValidationError::DTypeMismatch { op_index };
    let g = |id: ValueId| ty(types, id, op_index);
    let one = |t: ValueType| Ok(Results::One(t));

    match *op {
        // -- leaves --
        Op::Input(index) => {
            let slot = inputs
                .get(index as usize)
                .ok_or(ValidationError::InputIndexOutOfRange { op_index, index })?;
            one(slot.ty())
        }
        Op::Const(lit) => one(ValueType::scalar(lit.dtype())),

        // -- unary float-only map --
        Op::Exp(a) | Op::Log(a) | Op::Recip(a) => {
            let t = g(a)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            one(t)
        }
        // -- unary numeric map --
        Op::Neg(a) | Op::Abs(a) | Op::Sign(a) => {
            let t = g(a)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            one(t)
        }

        // -- binary arithmetic map --
        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::MaxElem(a, b) | Op::MinElem(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if !ta.dtype.is_numeric() || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            one(ValueType::new(broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?, ta.dtype))
        }
        Op::Div(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if ta.dtype != DType::F32 || tb.dtype != DType::F32 {
                return Err(dtype_err());
            }
            one(ValueType::new(broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?, DType::F32))
        }
        // -- comparisons → bool --
        Op::Gt(a, b) | Op::Ge(a, b) | Op::Eq(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if !ta.dtype.is_numeric() || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            one(ValueType::new(broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?, DType::Bool))
        }
        Op::Select { cond, a, b } => {
            let (tc, ta, tb) = (g(cond)?, g(a)?, g(b)?);
            if tc.dtype != DType::Bool || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            let ab = broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?;
            one(ValueType::new(broadcast2(ab, tc.shape).ok_or_else(shape_err)?, ta.dtype))
        }

        // -- reductions over the last axis --
        Op::ReduceSum(v) | Op::ReduceMax(v) | Op::ReduceMin(v) => {
            let t = g(v)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            one(ValueType::new(t.shape.drop_last().ok_or_else(shape_err)?, t.dtype))
        }
        Op::ReduceArgmax(v) => {
            let t = g(v)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            one(ValueType::new(t.shape.drop_last().ok_or_else(shape_err)?, DType::I32))
        }
        Op::Broadcast { value, shape } => {
            let t = g(value)?;
            if !can_broadcast_to(t.shape, shape) {
                return Err(shape_err());
            }
            one(ValueType::new(shape, t.dtype))
        }

        // -- scans (float) --
        Op::CumSum(v) | Op::CumProd(v) => {
            let t = g(v)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            if t.shape.rank() == 0 {
                return Err(shape_err());
            }
            one(t)
        }

        // -- sort: two results, value-first --
        Op::SortDesc(v) => {
            let t = g(v)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            let n = match *t.shape.dims() {
                [n] => n,
                _ => return Err(shape_err()),
            };
            Ok(Results::Two(ValueType::vector(n, DType::F32), ValueType::vector(n, DType::U32)))
        }
        Op::PivotThreshold { input, predicate } => {
            let t = g(input)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            if t.shape.rank() != 1 && t.shape.rank() != 2 {
                return Err(shape_err());
            }
            match predicate {
                Predicate::RankLe(k_id) => {
                    let kt = g(k_id)?;
                    // `k` is an integer count: a shared scalar, or — for a matrix
                    // input — a per-row `[rows]` vector (one `k` per row).
                    if !kt.dtype.is_int() {
                        return Err(dtype_err());
                    }
                    let per_row_ok =
                        t.shape.rank() == 2 && *kt.shape.dims() == [t.shape.rows()];
                    if !kt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
                Predicate::CummassLe(thr) | Predicate::ProbGe(thr) => {
                    let tt = g(thr)?;
                    if tt.dtype != DType::F32 {
                        return Err(dtype_err());
                    }
                    // The threshold is a shared scalar, or — for a matrix input —
                    // a per-row `[rows]` vector (one threshold per row).
                    let per_row_ok =
                        t.shape.rank() == 2 && *tt.shape.dims() == [t.shape.rows()];
                    if !tt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
            }
            one(ValueType::new(t.shape, DType::Bool))
        }

        // -- indexing --
        Op::Gather { src, idx } => {
            let (ts, ti) = (g(src)?, g(idx)?);
            if ts.shape.rank() != 1 {
                return Err(shape_err());
            }
            if !ti.dtype.is_int() || ti.shape.rank() != 1 {
                return Err(dtype_err());
            }
            one(ValueType::vector(ti.shape.last_len().ok_or_else(shape_err)?, ts.dtype))
        }
        Op::GatherRow { src, idx } => {
            let (ts, ti) = (g(src)?, g(idx)?);
            let m = match *ts.shape.dims() {
                [m, _n] => m,
                _ => return Err(shape_err()),
            };
            if !ti.dtype.is_int() {
                return Err(dtype_err());
            }
            match *ti.shape.dims() {
                [k] if k == m => {}
                _ => return Err(shape_err()),
            }
            one(ValueType::vector(m, ts.dtype))
        }
        Op::MaskApply { logits, mask } => {
            let (tl, tm) = (g(logits)?, g(mask)?);
            // `mask` is a packed `[ceil(n/32)]` U32 bitmask over the `n`-length
            // logits (word-indexed, not broadcast). Result ≡ logits.
            let n = tl.shape.last_len().ok_or_else(shape_err)?;
            let words = n.div_ceil(32);
            if tm.dtype != DType::U32 {
                return Err(dtype_err());
            }
            match *tm.shape.dims() {
                [w] if w == words => {}
                _ => return Err(shape_err()),
            }
            one(ValueType::new(tl.shape, tl.dtype))
        }
        Op::ScatterAdd { base, idx, vals } => one(scatter_ty(op_index, base, idx, vals, types, true)?),
        Op::ScatterSet { base, idx, vals } => one(scatter_ty(op_index, base, idx, vals, types, false)?),

        // -- rng --
        Op::Rng { shape, .. } => {
            if shape.rank() == 0 {
                return Err(shape_err());
            }
            one(ValueType::new(shape, DType::F32))
        }
    }
}

fn scatter_ty(
    op_index: u32,
    base: ValueId,
    idx: ValueId,
    vals: ValueId,
    types: &[ValueType],
    require_numeric: bool,
) -> Result<ValueType, ValidationError> {
    let shape_err = || ValidationError::ShapeMismatch { op_index };
    let dtype_err = || ValidationError::DTypeMismatch { op_index };
    let g = |id: ValueId| ty(types, id, op_index);

    let (tb, ti, tv) = (g(base)?, g(idx)?, g(vals)?);
    let base_len = match *tb.shape.dims() {
        [n] => n,
        _ => return Err(shape_err()),
    };
    if require_numeric && !tb.dtype.is_numeric() {
        return Err(dtype_err());
    }
    if !ti.dtype.is_int() || ti.shape.rank() != 1 {
        return Err(dtype_err());
    }
    if tv.shape.rank() != 1 || ti.shape.last_len() != tv.shape.last_len() {
        return Err(shape_err());
    }
    if tv.dtype != tb.dtype {
        return Err(dtype_err());
    }
    Ok(ValueType::vector(base_len, tb.dtype))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn logits(dims: &[u32]) -> InputDecl {
        InputDecl::new(Shape::new(dims).unwrap(), DType::F32)
    }
    fn islot(dims: &[u32], dtype: DType) -> InputDecl {
        InputDecl::new(Shape::new(dims).unwrap(), dtype)
    }

    #[test]
    fn argmax_ok() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
            outputs: vec![OutputDecl::new(1, OutputKind::Token)],
        };
        assert_eq!(validate(&p), Ok(()));
    }

    #[test]
    fn input_index_out_of_range() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(2)],
            outputs: vec![OutputDecl::new(0, OutputKind::Logits)],
        };
        assert_eq!(validate(&p), Err(ValidationError::InputIndexOutOfRange { op_index: 0, index: 2 }));
    }

    #[test]
    fn forward_ref_rejected() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(0), Op::Exp(5)],
            outputs: vec![OutputDecl::new(1, OutputKind::Logits)],
        };
        assert_eq!(
            validate(&p),
            Err(ValidationError::ValueIdOutOfRange { op_index: 1, value: 5 })
        );
    }

    #[test]
    fn sortdesc_reserves_two_ids() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(0), Op::SortDesc(0), Op::ReduceSum(1)],
            outputs: vec![OutputDecl::new(3, OutputKind::Scalar)],
        };
        assert_eq!(validate(&p), Ok(()));
        let t = value_types(&p).unwrap();
        assert_eq!(t[1], ValueType::vector(8, DType::F32));
        assert_eq!(t[2], ValueType::vector(8, DType::U32));
    }

    #[test]
    fn const_is_scalar() {
        let p = SamplingProgram {
            inputs: vec![logits(&[4])],
            ops: vec![Op::Input(0), Op::Const(Literal::F32(0.7)), Op::Div(0, 1)],
            outputs: vec![OutputDecl::new(2, OutputKind::Logits)],
        };
        assert_eq!(validate(&p), Ok(()));
        let t = value_types(&p).unwrap();
        assert_eq!(t[1], ValueType::scalar(DType::F32));
        assert_eq!(t[2], ValueType::vector(4, DType::F32));
    }

    #[test]
    fn select_requires_bool_cond() {
        let p = SamplingProgram {
            inputs: vec![logits(&[4])],
            ops: vec![Op::Input(0), Op::Select { cond: 0, a: 0, b: 0 }],
            outputs: vec![OutputDecl::new(1, OutputKind::Logits)],
        };
        assert_eq!(validate(&p), Err(ValidationError::DTypeMismatch { op_index: 1 }));
    }

    #[test]
    fn matrix_per_row_softmax_validates() {
        let k = 4;
        let v = 32;
        let p = SamplingProgram {
            inputs: vec![logits(&[k, v])],
            ops: vec![
                Op::Input(0),
                Op::ReduceMax(0),
                Op::Broadcast { value: 1, shape: Shape::matrix(k, v) },
                Op::Sub(0, 2),
                Op::Exp(3),
                Op::ReduceSum(4),
                Op::Broadcast { value: 5, shape: Shape::matrix(k, v) },
                Op::Div(4, 6),
                Op::ReduceArgmax(7),
            ],
            outputs: vec![OutputDecl::new(8, OutputKind::Token)],
        };
        assert_eq!(validate(&p), Ok(()));
        let t = value_types(&p).unwrap();
        assert_eq!(t[8], ValueType::vector(k, DType::I32));
    }

    #[test]
    fn broadcast_row_and_scalar() {
        assert!(can_broadcast_to(Shape::vector(4), Shape::matrix(4, 8)));
        assert!(can_broadcast_to(Shape::SCALAR, Shape::matrix(4, 8)));
        assert!(!can_broadcast_to(Shape::vector(8), Shape::matrix(4, 8)));
    }

    #[test]
    fn gather_row_per_row_pick_validates() {
        let k = 4;
        let v = 32;
        let p = SamplingProgram {
            inputs: vec![logits(&[k, v]), islot(&[k], DType::I32)],
            ops: vec![Op::Input(0), Op::Input(1), Op::GatherRow { src: 0, idx: 1 }],
            outputs: vec![OutputDecl::new(2, OutputKind::Scalar)],
        };
        assert_eq!(validate(&p), Ok(()));
        let t = value_types(&p).unwrap();
        assert_eq!(t[2], ValueType::vector(k, DType::F32));
    }

    #[test]
    fn matrix_pivot_per_row_threshold_validates() {
        // matrix top-p with a per-row [rows] threshold vector (batched top-p).
        let k = 4;
        let v = 16;
        let mk = |thr_len: u32| SamplingProgram {
            inputs: vec![logits(&[k, v]), islot(&[thr_len], DType::F32)],
            ops: vec![
                Op::Input(0), // [k,v]
                Op::Input(1), // [thr_len] per-row p
                Op::PivotThreshold { input: 0, predicate: Predicate::CummassLe(1) }, // [k,v] mask
            ],
            outputs: vec![OutputDecl::new(0, OutputKind::Logits)], // expose the F32 logits
        };
        // per-row [k] threshold validates
        assert_eq!(validate(&mk(k)), Ok(()));
        // wrong-length [k+1] threshold → pivot shape mismatch
        assert_eq!(validate(&mk(k + 1)), Err(ValidationError::ShapeMismatch { op_index: 2 }));
    }

    #[test]
    fn pivot_rank_le_k_is_a_value_id() {
        // #25: RankLe `k` is a value-id — an integer scalar (or per-row [rows]
        // vector), de-hardwired like top-p `p`.
        let mk = |k_dtype: DType| SamplingProgram {
            inputs: vec![logits(&[16]), islot(&[], k_dtype)],
            ops: vec![
                Op::Input(0),
                Op::Input(1),
                Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(1) },
            ],
            outputs: vec![OutputDecl::new(0, OutputKind::Logits)],
        };
        // a U32 scalar `k` validates
        assert_eq!(validate(&mk(DType::U32)), Ok(()));
        // a float `k` is rejected — `k` is an integer count
        assert_eq!(validate(&mk(DType::F32)), Err(ValidationError::DTypeMismatch { op_index: 2 }));
    }

    #[test]
    fn output_kind_type_mismatch_rejected() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
            outputs: vec![OutputDecl::new(1, OutputKind::Distribution)],
        };
        assert_eq!(validate(&p), Err(ValidationError::OutputKindTypeMismatch { index: 0 }));
    }

    #[test]
    fn output_types_and_kinds() {
        let p = SamplingProgram {
            inputs: vec![logits(&[8])],
            ops: vec![Op::Input(0), Op::SortDesc(0), Op::ReduceArgmax(1)],
            outputs: vec![
                OutputDecl::new(3, OutputKind::Token),
                OutputDecl::new(1, OutputKind::Logits),
            ],
        };
        assert_eq!(
            output_types(&p).unwrap(),
            vec![ValueType::scalar(DType::I32), ValueType::vector(8, DType::F32)]
        );
        assert_eq!(output_kinds(&p).unwrap(), vec![OutputKind::Token, OutputKind::Logits]);
    }

    #[test]
    fn input_first_use_barrier() {
        // slot 1 (μ) materialized at op1 → value 1; first read at op2 (Sub).
        let p = SamplingProgram {
            inputs: vec![logits(&[8]), islot(&[], DType::F32)],
            ops: vec![
                Op::Input(0),        // id0
                Op::Input(1),        // id1 (μ)
                Op::Sub(0, 1),       // op2 reads id1
                Op::ReduceArgmax(2), // op3
            ],
            outputs: vec![OutputDecl::new(3, OutputKind::Token)],
        };
        assert_eq!(input_first_use(&p, 1), Some(2));
        assert_eq!(input_first_use(&p, 0), Some(2)); // logits id0 first read at Sub too
        assert_eq!(
            late_input_barriers(&p, &[1]).unwrap(),
            vec![(1u32, Some(2u32))]
        );
    }

    #[test]
    fn output_kind_from_dtype() {
        assert_eq!(OutputKind::from_dtype(DType::I32), Some(OutputKind::Token));
        assert_eq!(OutputKind::from_dtype(DType::U32), Some(OutputKind::Token));
        assert_eq!(OutputKind::from_dtype(DType::F32), Some(OutputKind::Scalar));
        assert_eq!(OutputKind::from_dtype(DType::Bool), None);
    }

    #[test]
    fn program_from_parts_infers_kinds() {
        // mirostat-shaped decode: [Token i32, Scalar f32] from bare value-ids.
        let inputs = vec![logits(&[8]), islot(&[], DType::F32)];
        let ops = vec![
            Op::Input(0),        // id0 : logits [8] f32
            Op::Input(1),        // id1 : μ scalar f32
            Op::ReduceArgmax(0), // id2 : token i32 (argmax over f32 → int)
        ];
        let p = program_from_parts(inputs, ops, &[2, 1]).unwrap();
        assert_eq!(
            output_kinds(&p).unwrap(),
            vec![OutputKind::Token, OutputKind::Scalar]
        );
        // round-trips through the wire format like any other program.
        assert_eq!(crate::bytecode::decode(&crate::bytecode::encode(&p)).unwrap(), p);
    }

    #[test]
    fn program_from_parts_rejects_bool_output() {
        let inputs = vec![logits(&[8])];
        let ops = vec![Op::Input(0), Op::Gt(0, 0)]; // id1 : bool
        assert_eq!(
            program_from_parts(inputs, ops, &[1]),
            Err(ValidationError::OutputKindTypeMismatch { index: 0 })
        );
    }

    #[test]
    fn program_from_parts_rejects_unknown_output_id() {
        let inputs = vec![logits(&[8])];
        let ops = vec![Op::Input(0), Op::ReduceArgmax(0)]; // ids 0,1
        assert_eq!(
            program_from_parts(inputs, ops, &[9]),
            Err(ValidationError::OutputIdOutOfRange { value: 9 })
        );
    }
}
