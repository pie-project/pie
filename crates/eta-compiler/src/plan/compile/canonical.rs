//! Canonical byte forms fed to the signature hash. Not a wire format and
//! nothing decodes them; they exist so [`super::signature`] and
//! [`super::fold`] have one agreed spelling of a shape, a type, or an op.
//!
//! The layout is locked to the CUDA engine: `program_identity.hpp` walks
//! the same fields in the same order to key its graph cache. Reordering a
//! field here is an ABI break, not a refactor; bump
//! [`super::COMPILER_VERSION`] if the bytes have to move.

use alloc::vec::Vec;

use eta_ir::container::{encode_op, put_u16, put_u32};
use eta_ir::op::Op;
use eta_ir::types::{Shape, wire_dtype};

use super::symbolic::{Dimension, SymbolicType};

pub(crate) fn canonical_static_shape(bytes: &mut Vec<u8>, shape: Shape) {
    bytes.push(shape.rank() as u8);
    for &dimension in shape.dims() {
        put_u32(bytes, dimension);
    }
}

pub(crate) fn canonical_symbolic_type(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
    bytes.push(wire_dtype(value_type.dtype));
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

/// A shape with every runtime-varying extent flattened to zero. Batch shape
/// must not enter a signature, so a symbolic dimension contributes a fixed
/// zero rather than its current value.
fn canonical_symbolic_shape(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
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

/// An op in canonical form. Shape-bearing ops are respelled with their
/// symbolic result shape so batch size stays out of the hash; everything
/// else reuses the container's own op encoding ([`encode_op`]).
pub(crate) fn canonical_op(bytes: &mut Vec<u8>, op: &Op, result_type: Option<&SymbolicType>) {
    // every arm below is a shape-bearing op, which defines exactly one
    // value; falling back to a default shape would hash two
    // differently-shaped stages to the same signature.
    let result_type = || result_type.expect("shape-bearing op defines a value");
    match op {
        Op::Broadcast { value, .. } | Op::Reshape { value, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *value);
            canonical_symbolic_shape(bytes, result_type());
        }
        Op::Rng { stream, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *stream);
            canonical_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::RngKeyed { state, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *state);
            canonical_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::IntrinsicVal { intr, dtype, .. } => {
            bytes.push(op.tag());
            put_u16(bytes, *intr as u16);
            bytes.push(wire_dtype(*dtype));
            canonical_symbolic_shape(bytes, result_type());
        }
        Op::KernelCall {
            name, args, dtype, ..
        } => {
            bytes.push(op.tag());
            put_u16(bytes, *name);
            bytes.push(wire_dtype(*dtype));
            canonical_symbolic_shape(bytes, result_type());
            bytes.push(args.len() as u8);
            for &argument in args {
                put_u32(bytes, argument);
            }
        }
        _ => encode_op(bytes, op),
    }
}
