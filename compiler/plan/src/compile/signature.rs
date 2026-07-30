//! The canonical stage signature.
//!
//! A signature is the bytes that decide plan identity: two stages with the same
//! signature may share a compiled executable. It therefore has to capture
//! everything that changes generated code and nothing that does not --
//! runtime extents stay symbolic, so batch shape is deliberately absent.

use alloc::vec::Vec;

use pie_ir::container::{PortSource, put_u16, put_u32};
use pie_ir::registry::Stage;
use pie_ir::validate::BoundTrace;

use super::COMPILER_VERSION;
use super::canonical::{canonical_op, canonical_static_shape, canonical_symbolic_type};
use super::normalize::NormalizedStage;
use super::symbolic::{symbolic_channel_type, symbolic_port_type};

pub(crate) const SIGNATURE_MAGIC: [u8; 4] = *b"PTSG";

/// The canonical bytes and hash that decide a stage's plan identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSignature {
    /// FNV-1a-64 of [`canonical_bytes`](Self::canonical_bytes); the plan cache
    /// key and the emitted kernel's entry-point name.
    pub hash: u64,
    /// The canonical encoding two stages must share byte-for-byte to share a
    /// plan. Runtime extents stay symbolic, so batch shape is absent.
    pub canonical_bytes: Vec<u8>,
}

pub(crate) fn signature_ports(
    bound: &BoundTrace,
    stage: Stage,
) -> impl Iterator<Item = &pie_ir::container::PortBinding> {
    bound
        .container
        .ports
        .iter()
        .filter(move |_| stage != Stage::Epilogue)
}

pub(crate) fn stage_signature(bound: &BoundTrace, stage: &NormalizedStage) -> StageSignature {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE_MAGIC);
    put_u16(&mut bytes, COMPILER_VERSION);
    bytes.push(stage.stage as u8);

    put_u32(&mut bytes, stage.channel_bindings.len() as u32);
    for &global in &stage.channel_bindings {
        let declaration = &bound.container.channels[global as usize];
        let value_type = symbolic_channel_type(bound, global, bound.channel_types[global as usize]);
        canonical_symbolic_type(&mut bytes, &value_type);
        put_u32(&mut bytes, declaration.capacity);
        bytes.push(declaration.host_role as u8);
        bytes.push(u8::from(declaration.seeded));
        let extern_direction = bound
            .container
            .externs
            .iter()
            .find(|external| external.chan == global)
            .map(|external| external.dir as u8 + 1)
            .unwrap_or(0);
        bytes.push(extern_direction);
    }

    let ports: Vec<_> = signature_ports(bound, stage.stage).collect();
    put_u32(&mut bytes, ports.len() as u32);
    for binding in ports {
        bytes.push(binding.port as u8);
        match &binding.source {
            PortSource::Channel(global) => {
                bytes.push(0);
                // Invariant: `localize_stage` ran before this and put every
                // channel the stage's ports name into `channel_bindings` —
                // that is what localization means. Skipping an unfound
                // channel would drop a port from the signature, so two stages
                // differing only in that port would share a plan.
                put_u32(
                    &mut bytes,
                    stage
                        .channel_bindings
                        .iter()
                        .position(|channel| channel == global)
                        .expect("port channel localized") as u32,
                );
                let port_type =
                    symbolic_port_type(binding.port, bound.channel_types[*global as usize]);
                canonical_symbolic_type(&mut bytes, &port_type);
            }
            PortSource::Const { dtype, shape, data } => {
                bytes.push(1);
                bytes.push(*dtype as u8);
                canonical_static_shape(&mut bytes, *shape);
                put_u32(&mut bytes, data.len() as u32);
                bytes.extend_from_slice(data);
            }
        }
    }

    put_u32(&mut bytes, stage.names.len() as u32);
    for name in &stage.names {
        put_u16(&mut bytes, name.len() as u16);
        bytes.extend_from_slice(name.as_bytes());
    }

    put_u32(&mut bytes, stage.ops.len() as u32);
    let mut next_value = 0usize;
    for op in &stage.ops {
        canonical_op(&mut bytes, op, stage.value_types.get(next_value));
        next_value += op.result_count() as usize;
    }
    put_u32(&mut bytes, stage.value_types.len() as u32);
    for (value_type, domain) in stage.value_types.iter().zip(&stage.value_domains) {
        canonical_symbolic_type(&mut bytes, value_type);
        bytes.push(*domain as u8);
    }
    StageSignature {
        hash: pie_ir::fnv1a64(&bytes),
        canonical_bytes: bytes,
    }
}
