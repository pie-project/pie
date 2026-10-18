use crate::codegen::error::{EmitError, EmitterKind};
use alloc::format;
use alloc::string::String;

use eta_ir::op::OP_TABLE;

use super::runtime::{Dialect, runtime_source};

const ENTRY: &str = include_str!("../../../runtime/cuda/ptir_m1_entry.cuh");

pub fn valid_identifier(name: &str) -> bool {
    let alpha = |byte: u8| byte.is_ascii_alphabetic() || byte == b'_';
    let mut bytes = name.bytes();
    match bytes.next() {
        None => false,
        Some(first) if !alpha(first) => false,
        Some(_) => bytes.all(|byte| alpha(byte) || byte.is_ascii_digit()),
    }
}

pub fn supported_tag(tag: u8) -> bool {
    OP_TABLE.iter().any(|entry| entry.tag == tag)
}

pub fn emit_singleton_region(
    dialect: Dialect,
    entry_name: &str,
    op_tag: u8,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(
            EmitterKind::CudaSingleton,
        ));
    }
    if !supported_tag(op_tag) {
        return Err(EmitError::UnsupportedSingletonOpcode { tag: op_tag });
    }
    let mut source = runtime_source(dialect);
    source.push_str("\nextern \"C\" __global__ void ");
    source.push_str(entry_name);
    source.push_str(ENTRY);
    source.push_str(&format!("{}", op_tag as u32));
    source.push_str(
        "u, status, descriptors, params, a0, a1, a2, o0, o1, temporary);\n\
         }\n",
    );
    Ok(source)
}
