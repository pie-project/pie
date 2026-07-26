//! Canonical byte forms fed to the signature hash.
//!
//! These are not a wire format and nothing decodes them. They exist because
//! [`super::signature`] and [`super::fold`] need one agreed spelling of a shape,
//! a type, or an op so that two stages that should share a compiled executable
//! hash the same — and two that should not, do not.
//!
//! **The layout is locked to the CUDA driver.** `program_identity.hpp` walks the
//! same fields in the same order to key its graph cache, and
//! `stage_identity_matches_the_driver` in `compiler/tests` pins the two
//! together. Reordering a field here is an ABI break, not a refactor; bump
//! [`super::COMPILER_VERSION`] if the bytes have to move.

use alloc::vec::Vec;

use pie_ir::container::{encode_op, put_u16, put_u32};
use pie_ir::op::Op;
use pie_ir::types::Shape;

use super::symbolic::{Dimension, SymbolicType};

pub(crate) fn canonical_static_shape(bytes: &mut Vec<u8>, shape: Shape) {
    bytes.push(shape.rank() as u8);
    for &dimension in shape.dims() {
        put_u32(bytes, dimension);
    }
}

pub(crate) fn canonical_symbolic_type(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
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

/// A shape with every runtime-varying extent flattened to zero.
///
/// Batch shape must not enter a signature — one plan serves many batch sizes —
/// so a symbolic dimension contributes a fixed zero rather than its current
/// value. The role that says *which* extent it is travels in the type table.
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

/// An op in canonical form.
///
/// Shape-bearing ops are respelled with their *symbolic* result shape so that
/// batch size stays out of the hash; everything else reuses the container's own
/// op encoding ([`encode_op`]) rather than growing a second spelling of it.
pub(crate) fn canonical_op(bytes: &mut Vec<u8>, op: &Op, result_type: Option<&SymbolicType>) {
    // Invariant: every arm below that calls this is a shape-bearing op, and a
    // shape-bearing op defines exactly one value — `pie_ir::op`'s table says
    // so via `results`, and `validate::bind` checks each op's result count
    // against it. Falling back to a default shape instead would hash two
    // differently-shaped stages to the same signature and hand one stage the
    // other's compiled plan.
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
            bytes.push(*dtype as u8);
            canonical_symbolic_shape(bytes, result_type());
        }
        Op::KernelCall {
            name, args, dtype, ..
        } => {
            bytes.push(op.tag());
            put_u16(bytes, *name);
            bytes.push(*dtype as u8);
            canonical_symbolic_shape(bytes, result_type());
            bytes.push(args.len() as u8);
            for &argument in args {
                put_u32(bytes, argument);
            }
        }
        _ => encode_op(bytes, op),
    }
}
