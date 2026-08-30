//! The embedded runtime and the shared MSL struct preambles.
//!
//! `common_effect_preamble()` and `grouped_preamble()` are the C++ raw string
//! literals verbatim, including their leading newline — `R"MSL(` opens the
//! literal immediately before the line break.

use alloc::string::String;
use core::fmt::Write as _;
use std::sync::LazyLock;

use crate::codegen::layout;

/// MSL prefixes for the two dispatch shapes. Same layouts, different names —
/// the single-lane effect kernels call them `M1*`, the grouped kernels `M3*`.
const M1: &str = "M1";
const M3: &str = "M3";

/// The M1 runtime template, from
/// `crates/eta-compiler/runtime/metal/ptir_m1_runtime.metal` (the C++ oracle
/// this was ported from read the same text out of the engine's tree; the file
/// moved here when the emitter became the only implementation). Embedded so the
/// emitter is self-contained — the C++ took it as a `runtime_template` argument
/// read from disk at init.
///
/// It still carries `#include "ptir_rng.generated.metal"`; expanding that is
/// the engine's job at library-build time, not the emitter's.
pub const RUNTIME_TEMPLATE: &str = include_str!("../../../runtime/metal/ptir_m1_runtime.metal");

/// The grouped (M3) lane-table structs, in a file rather than a literal for the
/// same reason `RUNTIME_TEMPLATE` is: a C++ reader needs them too. The Metal
/// engine's `msl_compile_test` reconstructs the emitter's full sources from the
/// golden dump plus these two prefixes, and a raw string literal in this crate
/// is reachable from nothing but this crate.
///
/// This file is a *replica*, not the authority: [`grouped_preamble`] prints the
/// same structs from [`layout::HOST_SHARED`], and `file_matches_emitted_text`
/// below fails the build if the two ever disagree. Without that the file would
/// be a seventh hand-written copy of the lane table — the exact failure
/// [`layout`] exists to rule out, reintroduced by the C++ reader's need for a
/// path it can open.
pub const GROUPED_PREAMBLE: &str = include_str!("../../../runtime/metal/ptir_m1_grouped.metal");

/// `common_effect_preamble()` — the structs the single-lane readiness and
/// commit kernels read out of the lane table.
pub fn common_effect_preamble() -> &'static str {
    static TEXT: LazyLock<String> = LazyLock::new(|| {
        let mut out = String::from("\n#include <metal_stdlib>\nusing namespace metal;\n");
        out.push_str(&layout::STATUS.emit_msl(M1));
        for shared in layout::HOST_SHARED {
            out.push_str(&shared.emit_msl(M1));
        }
        out
    });
    &TEXT
}

/// `emit_word_arguments()` — one `words_N` ring buffer per channel, from
/// buffer 2 upward.
pub fn emit_word_arguments(source: &mut String, count: usize) {
    for channel in 0..count {
        let _ = write!(
            source,
            ", device ulong* words_{channel} [[buffer({})]]",
            channel + 2
        );
    }
}

/// `grouped_preamble()` — the M3 (grouped, multi-lane) lane-table structs.
pub fn grouped_preamble() -> &'static str {
    static TEXT: LazyLock<String> = LazyLock::new(|| {
        let mut out = String::from("\n");
        for shared in layout::HOST_SHARED {
            out.push_str(&shared.emit_msl(M3));
        }
        // `M3ChannelMeta`, `M3GroupLayout` and `M3RowMeta` are grouped-dispatch
        // arguments rather than lane-table members: the host does not build
        // them as `#[repr(C)]` structs, so there is nothing to pin them to and
        // they stay as text.
        out.push_str(
            r#"struct M3ChannelMeta {
  ulong words;
  uint capacity;
  uint flags;
};
struct M3GroupLayout {
  uint lane_count;
  uint value_count;
  uint scratch_stride;
  uint temporary_offset;
  uint vocab;
  uint reserved0;
  uint reserved1;
  uint reserved2;
};
struct M3RowMeta {
  uint offset;
  uint count;
  uint mtp_offset;
  uint reserved;
};
"#,
        );
        out
    });
    &TEXT
}

#[cfg(test)]
mod tests {
    use crate::codegen::runtime_scan::assert_execute_covers_the_table;

    #[test]
    fn metal_execute_covers_the_op_table() {
        assert_execute_covers_the_table(super::RUNTIME_TEMPLATE, "ptir_m1_runtime.metal");
    }

    use super::*;

    /// `ptir_m1_grouped.metal` exists because the Metal engine's
    /// `msl_compile_test` needs a path it can open; it is not a second place to
    /// declare the lane table. Adding a field to [`layout::HOST_SHARED`] and
    /// not to the file (or the reverse) fails here instead of shipping a
    /// kernel that reads different offsets than the host wrote.
    #[test]
    fn file_matches_emitted_text() {
        assert_eq!(
            GROUPED_PREAMBLE,
            grouped_preamble(),
            "crates/eta-compiler/runtime/metal/ptir_m1_grouped.metal has drifted from \
             what layout::HOST_SHARED prints; re-copy the emitted text into the file"
        );
    }
}
