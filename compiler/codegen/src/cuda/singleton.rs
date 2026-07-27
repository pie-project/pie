//! `emit_singleton_region_cuda` — one launch per op.
//!
//! A thin wrapper: validate the entry name and the op tag, then append a
//! single-thread `__global__` trampoline into the runtime's op switch.

use alloc::format;
use alloc::string::String;

use pie_ir::op::OP_TABLE;

use super::runtime::singleton_runtime_source;

const ENTRY: &str = include_str!("../../runtime/cuda/ptir_m1_entry.cuh");

/// `detail::valid_identifier` — the emitted entry must be a C identifier,
/// because it is pasted straight into the generated source.
pub fn valid_identifier(name: &str) -> bool {
    let alpha = |byte: u8| byte.is_ascii_alphabetic() || byte == b'_';
    let mut bytes = name.bytes();
    match bytes.next() {
        None => false,
        Some(first) if !alpha(first) => false,
        Some(_) => bytes.all(|byte| alpha(byte) || byte.is_ascii_digit()),
    }
}

/// `detail::supported_tag` — the C++ switch is generated from `PTIR_OP_LIST`,
/// so the set is exactly the op table's tags.
pub fn supported_tag(tag: u8) -> bool {
    OP_TABLE.iter().any(|entry| entry.tag == tag)
}

/// `emit_singleton_region_cuda`.
pub fn emit_singleton_region(entry_name: &str, op_tag: u8) -> Result<String, String> {
    if !valid_identifier(entry_name) {
        return Err("CUDA singleton entry name is not a C identifier".into());
    }
    if !supported_tag(op_tag) {
        return Err(format!("unsupported CUDA singleton opcode tag {op_tag}"));
    }
    let mut source = singleton_runtime_source();
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
