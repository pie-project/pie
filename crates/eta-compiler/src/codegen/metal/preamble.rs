use alloc::string::String;
use core::fmt::Write as _;
use std::sync::LazyLock;

use crate::codegen::layout;

const M1: &str = "M1";
const M3: &str = "M3";

pub const RUNTIME_TEMPLATE: &str = include_str!("../../../runtime/metal/ptir_m1_runtime.metal");

pub const GROUPED_PREAMBLE: &str = include_str!("../../../runtime/metal/ptir_m1_grouped.metal");

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

pub fn emit_word_arguments(source: &mut String, count: usize) {
    for channel in 0..count {
        let _ = write!(
            source,
            ", device ulong* words_{channel} [[buffer({})]]",
            channel + 2
        );
    }
}

pub fn grouped_preamble() -> &'static str {
    static TEXT: LazyLock<String> = LazyLock::new(|| {
        let mut out = String::from("\n");
        for shared in layout::HOST_SHARED {
            out.push_str(&shared.emit_msl(M3));
        }
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

    use super::*;

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
