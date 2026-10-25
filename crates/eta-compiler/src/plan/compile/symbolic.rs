use alloc::vec;
use alloc::vec::Vec;

use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::Port;
use eta_ir::types::{Dtype, Shape, ValueType};
use eta_ir::validate::BoundTrace;

eta_ir::declare_tagged_enum! {
    #[derive(serde::Serialize, serde::Deserialize)]
    pub enum SymbolicExtent {
        RowCount = 0, "row_count";
        TokenCount = 1, "token_count";
        SampledRows = 2, "sampled_rows";
        QueryLen = 3, "query_len";
        KeyLen = 4, "key_len";
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Dimension {
    Static(u32),
    Symbolic(SymbolicExtent),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolicType {
    pub dtype: Dtype,
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

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

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
        Op::Const(..)
        | Op::Exp(..)
        | Op::Log(..)
        | Op::Neg(..)
        | Op::Recip(..)
        | Op::Sin(..)
        | Op::Cos(..)
        | Op::Sqrt(..)
        | Op::Rsqrt(..)
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
        Port::EmbedTokens | Port::Positions | Port::Slots => {
            set_first_symbolic(&mut ty, SymbolicExtent::TokenCount)
        }
        Port::KvLen | Port::Fold => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
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
    }
    ty
}

pub(crate) fn set_first_symbolic(ty: &mut SymbolicType, extent: SymbolicExtent) {
    if let Some(first) = ty.dims.first_mut() {
        *first = Dimension::Symbolic(extent);
    }
}

pub(crate) fn symbolic_intrinsic_type(
    bound: &BoundTrace,
    intrinsic: IntrinsicId,
    value_type: ValueType,
) -> SymbolicType {
    let mut ty = SymbolicType::static_type(value_type);
    match intrinsic {
        IntrinsicId::Logits => {
            if ty.rank() >= 2 && value_type.shape.last_len() == Some(bound.vocab()) {
                ty.dims[0] = Dimension::Symbolic(SymbolicExtent::SampledRows);
            }
        }
        IntrinsicId::MtpLogits
        | IntrinsicId::MtpDrafts
        | IntrinsicId::Hidden
        | IntrinsicId::Velocity
        | IntrinsicId::Pixels
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
