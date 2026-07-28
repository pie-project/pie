//! Writing plan bytes.
//!
//! Everything that appends to a `Vec<u8>` destined for the wire lives here, so
//! the `PTRP` layout can be read in one place. [`super::decode`] is its mirror
//! and the two must be edited together.

use alloc::vec::Vec;

use pie_ir::container::{encode_op, put_u16, put_u32};
use pie_ir::op::Op;
use pie_ir::types::Shape;

use super::normalize::{NodeIndex, result_layout};
use super::region::{RegionKind, RegionPartition};
use super::symbolic::{Dimension, SymbolicType};
use super::{COMPILER_VERSION, CompiledStage, REGION_PLAN_VERSION};

pub(crate) const PLAN_MAGIC: [u8; 4] = *b"PTRP";

pub(crate) fn encode_static_shape(bytes: &mut Vec<u8>, shape: Shape) {
    bytes.push(shape.rank() as u8);
    for &dimension in shape.dims() {
        put_u32(bytes, dimension);
    }
}

pub(crate) fn encode_symbolic_type(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
    bytes.push(value_type.dtype as u8);
    bytes.push(value_type.dims.len() as u8);
    for dimension in &value_type.dims {
        match dimension {
            Dimension::Static(value) => {
                bytes.push(0);
                put_u32(bytes, *value);
            }
            Dimension::Symbolic(role) => {
                bytes.push(1);
                bytes.push(*role as u8);
            }
        }
    }
}

pub(crate) fn encode_symbolic_shape(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
    bytes.push(value_type.dims.len() as u8);
    for dimension in &value_type.dims {
        put_u32(
            bytes,
            match dimension {
                Dimension::Static(value) => *value,
                Dimension::Symbolic(_) => 0,
            },
        );
    }
}

/// Plan op encoding reuses container op records, with zero dimensions denoting
/// symbolic extents whose roles and runtime values live in the adjacent plan
/// type table and lane record.
pub(crate) fn encode_planned_op(bytes: &mut Vec<u8>, op: &Op, result_type: Option<&SymbolicType>) {
    let result_type = || result_type.expect("shape-bearing op defines a value");
    match op {
        Op::Broadcast { value, .. } | Op::Reshape { value, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *value);
            encode_symbolic_shape(bytes, result_type());
        }
        Op::Rng { stream, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *stream);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::RngKeyed { state, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *state);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::IntrinsicVal { intr, dtype, .. } => {
            bytes.push(op.tag());
            put_u16(bytes, *intr as u16);
            bytes.push(*dtype as u8);
            encode_symbolic_shape(bytes, result_type());
        }
        Op::KernelCall {
            name, args, dtype, ..
        } => {
            bytes.push(op.tag());
            put_u16(bytes, *name);
            bytes.push(*dtype as u8);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(args.len() as u8);
            for &argument in args {
                put_u32(bytes, argument);
            }
        }
        _ => encode_op(bytes, op),
    }
}

/// Serialize a complete stage plan. Every variable-sized record is
/// length-delimited so backend readers can reject unknown versions cleanly.
pub fn encode_stage_plan(stage: &CompiledStage) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PLAN_MAGIC);
    put_u16(&mut bytes, REGION_PLAN_VERSION);
    put_u16(&mut bytes, COMPILER_VERSION);
    bytes.push(stage.normalized.stage as u8);
    bytes.extend_from_slice(&stage.signature.hash.to_le_bytes());
    put_u32(&mut bytes, stage.signature.canonical_bytes.len() as u32);
    bytes.extend_from_slice(&stage.signature.canonical_bytes);

    put_u32(&mut bytes, stage.normalized.channel_bindings.len() as u32);
    for &channel in &stage.normalized.channel_bindings {
        put_u32(&mut bytes, channel);
    }
    put_u32(&mut bytes, stage.normalized.names.len() as u32);
    for name in &stage.normalized.names {
        put_u16(&mut bytes, name.len() as u16);
        bytes.extend_from_slice(name.as_bytes());
    }

    put_u32(&mut bytes, stage.normalized.ops.len() as u32);
    // `result_layout` once, not a prefix sum re-walked per op: the inline
    // `ops[..op_index].map(result_count).sum()` this replaces made encoding
    // quadratic in the op count, on the same curve as `build_region`.
    let (result_bases, _) = result_layout(&stage.normalized.ops);
    for (op_index, op) in stage.normalized.ops.iter().enumerate() {
        let mut encoded = Vec::new();
        let result_base = result_bases[op_index] as usize;
        encode_planned_op(
            &mut encoded,
            op,
            stage.normalized.value_types.get(result_base),
        );
        put_u32(&mut bytes, encoded.len() as u32);
        bytes.extend_from_slice(&encoded);
        put_u32(
            &mut bytes,
            stage.normalized.source_ops[op_index].len() as u32,
        );
        for &source in &stage.normalized.source_ops[op_index] {
            put_u32(&mut bytes, source);
        }
    }
    put_u32(&mut bytes, stage.normalized.value_types.len() as u32);
    for (value_type, domain) in stage
        .normalized
        .value_types
        .iter()
        .zip(&stage.normalized.value_domains)
    {
        encode_symbolic_type(&mut bytes, value_type);
        bytes.push(*domain as u8);
    }
    encode_partition(&mut bytes, &stage.singleton);
    encode_partition(&mut bytes, &stage.fused);
    bytes
}

pub(crate) fn encode_partition(bytes: &mut Vec<u8>, partition: &RegionPartition) {
    bytes.push(partition.kind as u8);
    bytes.push(u8::from(partition.whole_stage_fallback));
    put_u32(bytes, partition.regions.len() as u32);
    for region in &partition.regions {
        match region.kind {
            RegionKind::Generated => {
                bytes.push(0);
                bytes.push(0);
            }
            RegionKind::Library(library) => {
                bytes.push(1);
                bytes.push(library as u8);
            }
        }
        bytes.push(region.schedule as u8);
        encode_node_slice(bytes, &region.nodes);
        encode_u32_slice(bytes, &region.inputs);
        encode_u32_slice(bytes, &region.outputs);
        put_u32(bytes, region.sinks.len() as u32);
        for sink in &region.sinks {
            put_u32(bytes, sink.channel_slot);
            put_u32(bytes, sink.value);
        }
    }
}

/// The one place node indices become wire `u32`s. The wire cannot tell the
/// node space from the value space, so the cast is spelled out here rather
/// than wherever a region is built.
fn encode_node_slice(bytes: &mut Vec<u8>, nodes: &[NodeIndex]) {
    put_u32(bytes, nodes.len() as u32);
    for node in nodes {
        put_u32(bytes, node.get());
    }
}

pub(crate) fn encode_u32_slice(bytes: &mut Vec<u8>, values: &[u32]) {
    put_u32(bytes, values.len() as u32);
    for &value in values {
        put_u32(bytes, value);
    }
}
