//! Symbolic types: the shape vocabulary that keeps one plan valid across
//! batch shapes.
//!
//! A planned value's type is a dtype plus a list of [`Dimension`]s, each
//! either a concrete `u32` or a [`SymbolicExtent`] the runtime substitutes.
//! These also serialize, since [`crate::codegen::launch::LaunchPlanValue`]
//! ships them to an engine.

use alloc::vec;
use alloc::vec::Vec;

use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::Port;
use eta_ir::types::{Dtype, Shape, ValueType};
use eta_ir::validate::BoundTrace;

eta_ir::declare_tagged_enum! {
    #[derive(serde::Serialize, serde::Deserialize)]
    /// Runtime-varying dimensions represented symbolically in compiler
    /// types. Discriminants are the wire encoding, serialized because
    /// [`crate::codegen::launch::LaunchPlanValue`]'s axes are written in
    /// this tag space and ship to an engine.
    pub enum SymbolicExtent {
        /// Number of live KV-cache entries.
        KvLen = 0, "kv_len";
        /// Number of KV-cache pages.
        PageCount = 1, "page_count";
        /// Number of rows (requests) in the batch.
        RowCount = 2, "row_count";
        /// Number of input tokens in the pass.
        TokenCount = 3, "token_count";
        /// Number of rows read out for sampling.
        SampledRows = 4, "sampled_rows";
        /// Attention query length.
        QueryLen = 5, "query_len";
        /// Attention key length.
        KeyLen = 6, "key_len";
    }
}

/// One dimension of a [`SymbolicType`]: a fixed size or a runtime extent.
/// Also the launch package's axis type (an element of
/// [`crate::codegen::launch::LaunchPlanValue`]'s `axes`), hence it
/// serializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Dimension {
    /// A size known at plan time.
    Static(u32),
    /// A runtime-varying extent the launch substitutes.
    Symbolic(SymbolicExtent),
}

/// A planned value's type: a dtype and a per-dimension shape that keeps one
/// plan valid across batch shapes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolicType {
    /// The element type.
    pub dtype: Dtype,
    /// The shape, outermost dimension first; empty for a scalar.
    pub dims: Vec<Dimension>,
}

impl SymbolicType {
    fn static_type(value_type: ValueType) -> Self {
        Self {
            dtype: value_type.dtype,
            dims: value_type
                .shape
                .dims()
                .iter()
                .copied()
                .map(Dimension::Static)
                .collect(),
        }
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Whether the type has no dimensions, or every dimension is `1`.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
            || self
                .dims
                .iter()
                .all(|dimension| *dimension == Dimension::Static(1))
    }
}

pub(crate) fn symbolic_result_type(
    bound: &BoundTrace,
    original_op: &Op,
    value_type: ValueType,
    mapped_op: &Op,
    original_types: &[ValueType],
    normalized_types: &[SymbolicType],
) -> SymbolicType {
    match original_op {
        Op::ChanTake(channel) | Op::ChanRead(channel) => {
            symbolic_channel_type(bound, *channel, value_type)
        }
        Op::IntrinsicVal { intr, .. } => symbolic_intrinsic_type(bound, *intr, value_type),
        Op::ReduceSum(value)
        | Op::ReduceMax(value)
        | Op::ReduceMin(value)
        | Op::ReduceArgmax(value) => {
            let mapped = mapped_value(mapped_op, *value);
            let mut ty = normalized_types[mapped as usize].clone();
            ty.dims.pop();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::Transpose(value) => {
            let mapped = mapped_value(mapped_op, *value);
            let mut ty = normalized_types[mapped as usize].clone();
            if ty.dims.len() == 2 {
                ty.dims.swap(0, 1);
            }
            ty.dtype = value_type.dtype;
            ty
        }
        Op::Gather { .. } => {
            let operands = mapped_op.operands();
            let src = &normalized_types[operands[0] as usize];
            let index = &normalized_types[operands[1] as usize];
            let mut dims = index.dims.clone();
            dims.extend_from_slice(&src.dims[1..]);
            SymbolicType {
                dtype: value_type.dtype,
                dims,
            }
        }
        Op::GatherRow { .. } => {
            let index = mapped_op.operands()[1];
            SymbolicType {
                dtype: value_type.dtype,
                dims: normalized_types[index as usize].dims.clone(),
            }
        }
        Op::ScatterAdd { .. } | Op::ScatterSet { .. } => {
            let base = mapped_op.operands()[0];
            let mut ty = normalized_types[base as usize].clone();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::MaskApply { .. } => {
            let logits = mapped_op.operands()[0];
            normalized_types[logits as usize].clone()
        }
        Op::SortDesc(_) | Op::CumSum(_) | Op::CumProd(_) => {
            let input = mapped_op.operands()[0];
            let mut ty = normalized_types[input as usize].clone();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::TopK { k, .. } => {
            let input = mapped_op.operands()[0];
            let mut ty = normalized_types[input as usize].clone();
            if let Some(last) = ty.dims.last_mut() {
                *last = Dimension::Static(*k);
            }
            ty.dtype = value_type.dtype;
            ty
        }
        Op::MatMul(_, _) => {
            let operands = mapped_op.operands();
            let left = &normalized_types[operands[0] as usize];
            let right = &normalized_types[operands[1] as usize];
            // `MatMul` operands are exactly rank 2 (checked by `validate::bind`),
            // so `left.dims[0]` and the last of `right.dims` both exist.
            SymbolicType {
                dtype: value_type.dtype,
                dims: vec![left.dims[0], *right.dims.last().expect("matmul right rank")],
            }
        }
        Op::CausalMask { positions, .. }
        | Op::SlidingWindowMask { positions, .. }
        | Op::SinkWindowMask { positions, .. } => {
            let mapped = mapped_op.operands()[0];
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*positions as usize],
                value_type,
            );
            ty
        }
        Op::Broadcast { value, .. } => {
            let mapped = mapped_value(mapped_op, *value);
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*value as usize],
                value_type,
            );
            ty
        }
        Op::Reshape { value, .. } => {
            let mapped = mapped_value(mapped_op, *value);
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*value as usize],
                value_type,
            );
            ty
        }
        // Rank-preserving default: the op's declared shape, with any symbolic
        // dimension carried from the first operand of equal rank. Named
        // rather than `_` so a new reducer/gather can't silently fall
        // through to it.
        Op::Const(..)
        | Op::Exp(..)
        | Op::Log(..)
        | Op::Neg(..)
        | Op::Recip(..)
        | Op::Abs(..)
        | Op::Sign(..)
        | Op::Cast { .. }
        | Op::Add(..)
        | Op::Sub(..)
        | Op::Mul(..)
        | Op::Div(..)
        | Op::MaxElem(..)
        | Op::MinElem(..)
        | Op::Rem(..)
        | Op::Gt(..)
        | Op::Ge(..)
        | Op::Eq(..)
        | Op::Ne(..)
        | Op::Lt(..)
        | Op::Le(..)
        | Op::And(..)
        | Op::Or(..)
        | Op::Not(..)
        | Op::Select { .. }
        | Op::PivotThreshold { .. }
        | Op::Iota { .. }
        | Op::Rng { .. }
        | Op::RngKeyed { .. }
        | Op::ChanPut { .. }
        | Op::KernelCall { .. }
        | Op::SinkCall { .. } => {
            let mut ty = SymbolicType::static_type(value_type);
            if let Some((original, mapped)) = original_op
                .operands()
                .into_iter()
                .zip(mapped_op.operands())
                .find(|(_, mapped)| {
                    normalized_types
                        .get(*mapped as usize)
                        .is_some_and(|source| source.rank() == ty.rank())
                })
            {
                propagate_preserved_dimensions(
                    &mut ty,
                    &normalized_types[mapped as usize],
                    original_types[original as usize],
                    value_type,
                );
            }
            ty
        }
    }
}

pub(crate) fn propagate_preserved_dimensions(
    target: &mut SymbolicType,
    source: &SymbolicType,
    source_static: ValueType,
    target_static: ValueType,
) {
    for (index, source_dimension) in source.dims.iter().enumerate() {
        if index < target.dims.len()
            && matches!(source_dimension, Dimension::Symbolic(_))
            && source_static.shape.dims().get(index) == target_static.shape.dims().get(index)
        {
            target.dims[index] = *source_dimension;
        }
    }
}

/// The normalized SSA id standing in for `original_value`. Normalization
/// renumbers, so a shape-changing op's operand must be read off the
/// normalized op; `_` falls through to the original id since there is no
/// renumbering to follow for any other op kind.
pub(crate) fn mapped_value(mapped_op: &Op, original_value: u32) -> u32 {
    match mapped_op {
        Op::ReduceSum(value)
        | Op::ReduceMax(value)
        | Op::ReduceMin(value)
        | Op::ReduceArgmax(value)
        | Op::Transpose(value)
        | Op::Broadcast { value, .. }
        | Op::Reshape { value, .. } => *value,
        _ => original_value,
    }
}

pub(crate) fn symbolic_channel_type(
    _bound: &BoundTrace,
    _channel: u32,
    value_type: ValueType,
) -> SymbolicType {
    SymbolicType::static_type(value_type)
}

pub(crate) fn symbolic_port_type(port: Port, value_type: ValueType) -> SymbolicType {
    let mut ty = SymbolicType::static_type(value_type);
    match port {
        Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff => {
            set_first_symbolic(&mut ty, SymbolicExtent::TokenCount)
        }
        Port::Pages => set_first_symbolic(&mut ty, SymbolicExtent::PageCount),
        Port::PageIndptr => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
        Port::KvLen => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
        Port::Readout => set_first_symbolic(&mut ty, SymbolicExtent::SampledRows),
        Port::AttnMask => {
            if !ty.dims.is_empty() {
                ty.dims[0] = Dimension::Symbolic(SymbolicExtent::QueryLen);
            }
            if ty.dims.len() > 1 {
                let last = ty.dims.len() - 1;
                ty.dims[last] = Dimension::Symbolic(SymbolicExtent::KeyLen);
            }
        }
        Port::EmbedIndptr => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
        // RS buffered-slot family, mirroring KV: slab-id vector page-indexed,
        // CSR bounds and per-row live length row-indexed, write descriptor
        // token-indexed.
        Port::RsBufferPages => set_first_symbolic(&mut ty, SymbolicExtent::PageCount),
        Port::RsBufferIndptr | Port::RsBufferLen | Port::RsFoldLen => {
            set_first_symbolic(&mut ty, SymbolicExtent::RowCount)
        }
        Port::RsWSlot | Port::RsWOff => set_first_symbolic(&mut ty, SymbolicExtent::TokenCount),
    }
    ty
}

pub(crate) fn set_first_symbolic(ty: &mut SymbolicType, extent: SymbolicExtent) {
    if let Some(first) = ty.dims.first_mut() {
        *first = Dimension::Symbolic(extent);
    }
}

/// The symbolic type of an intrinsic's result. An exhaustive match, so a new
/// intrinsic fails to compile here rather than silently falling through to
/// the static type.
pub(crate) fn symbolic_intrinsic_type(
    bound: &BoundTrace,
    intrinsic: IntrinsicId,
    value_type: ValueType,
) -> SymbolicType {
    let mut ty = SymbolicType::static_type(value_type);
    match intrinsic {
        IntrinsicId::Logits => {
            if ty.rank() >= 2 && value_type.shape.last_len() == Some(bound.profile.vocab) {
                ty.dims[0] = Dimension::Symbolic(SymbolicExtent::SampledRows);
            }
        }
        // These stay static: e.g. `AttnScore`'s width is a declared plane
        // count bounded by `ATTN_SCORE_KV_MAX`, and lifting it to `KvLen`
        // would cut one launch per distinct kv length.
        IntrinsicId::MtpLogits
        | IntrinsicId::MtpDrafts
        | IntrinsicId::Hidden
        | IntrinsicId::Query
        | IntrinsicId::ValueHead
        | IntrinsicId::AttnScore
        | IntrinsicId::Layer => {}
    }
    ty
}

pub(crate) fn symbolic_shape_matches_static(value_type: &SymbolicType, shape: Shape) -> bool {
    symbolic_dims_match_static(&value_type.dims, shape.dims())
}

pub(crate) fn symbolic_dims_match_static(symbolic: &[Dimension], concrete: &[u32]) -> bool {
    symbolic.len() == concrete.len()
        && symbolic.iter().zip(concrete).all(|(symbolic, concrete)| {
            matches!(symbolic, Dimension::Symbolic(_)) || *symbolic == Dimension::Static(*concrete)
        })
}

pub(crate) fn symbolic_dims_match_expected(
    actual: &[Dimension],
    expected: &[Dimension],
    concrete: &[u32],
) -> bool {
    actual.len() == expected.len()
        && actual.len() == concrete.len()
        && actual
            .iter()
            .zip(expected)
            .zip(concrete)
            .all(|((actual, expected), concrete)| {
                actual == expected
                    || matches!(
                        (actual, expected),
                        (Dimension::Static(actual), Dimension::Symbolic(_))
                            if actual == concrete
                    )
            })
}

