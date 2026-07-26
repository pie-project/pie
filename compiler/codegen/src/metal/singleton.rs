//! `emit_singleton_region_msl` — the tier-1 kernel: one dispatch runs exactly
//! one op through the runtime's `ptir_m1_execute` switch.

use alloc::string::String;
use core::fmt::Write as _;

use super::preamble::RUNTIME_TEMPLATE;

/// `emit_singleton_region_msl`. Total: the tag is passed straight through to
/// the runtime switch, which faults at run time on anything it does not
/// implement.
pub fn emit_singleton_region(function_name: &str, op_tag: u8) -> String {
    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    let _ = writeln!(source, "kernel void {function_name}(");
    source.push_str("    device M1Status* status [[buffer(0)]],\n");
    source.push_str("    const device M1ValueDesc* descriptors [[buffer(1)]],\n");
    source.push_str("    const device uchar* a0 [[buffer(2)]],\n");
    source.push_str("    const device uchar* a1 [[buffer(3)]],\n");
    source.push_str("    const device uchar* a2 [[buffer(4)]],\n");
    source.push_str("    device uchar* o0 [[buffer(5)]],\n");
    source.push_str("    device uchar* o1 [[buffer(6)]],\n");
    source.push_str("    device uchar* temporary [[buffer(7)]],\n");
    source.push_str("    const device M1OpParams* params [[buffer(8)]],\n");
    source.push_str("    uint gid [[thread_position_in_grid]]) {\n");
    let _ = writeln!(
        source,
        "  if (gid == 0) ptir_m1_execute({op_tag}u, status, descriptors, params, \
         a0, a1, a2, o0, o1, temporary);"
    );
    source.push_str("}\n");
    source
}
