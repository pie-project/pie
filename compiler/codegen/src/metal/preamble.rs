//! The embedded runtime and the shared MSL struct preambles.
//!
//! `common_effect_preamble()` and `grouped_preamble()` are the C++ raw string
//! literals verbatim, including their leading newline — `R"MSL(` opens the
//! literal immediately before the line break.

use alloc::string::String;
use core::fmt::Write as _;
use std::sync::LazyLock;

use crate::layout;

/// MSL prefixes for the two dispatch shapes. Same layouts, different names —
/// the single-lane effect kernels call them `M1*`, the grouped kernels `M3*`.
const M1: &str = "M1";
const M3: &str = "M3";

/// `driver/metal/src/kernels/ptir_m1_runtime.metal`, embedded so the emitter
/// is self-contained (the C++ takes it as a `runtime_template` argument the
/// driver reads from disk at init).
///
/// It still carries `#include "ptir_rng.generated.metal"`; expanding that is
/// the driver's job at library-build time, not the emitter's.
pub const RUNTIME_TEMPLATE: &str = include_str!("../../runtime/metal/ptir_m1_runtime.metal");

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
