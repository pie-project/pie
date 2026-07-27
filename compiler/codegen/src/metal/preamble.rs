//! The embedded runtime and the shared MSL struct preambles.
//!
//! `common_effect_preamble()` and `grouped_preamble()` are the C++ raw string
//! literals verbatim, including their leading newline — `R"MSL(` opens the
//! literal immediately before the line break.

use alloc::string::String;
use core::fmt::Write as _;

/// `driver/metal/src/kernels/ptir_m1_runtime.metal`, embedded so the emitter
/// is self-contained (the C++ takes it as a `runtime_template` argument the
/// driver reads from disk at init).
///
/// It still carries `#include "ptir_rng.generated.metal"`; expanding that is
/// the driver's job at library-build time, not the emitter's.
pub const RUNTIME_TEMPLATE: &str = include_str!("../../runtime/metal/ptir_m1_runtime.metal");

/// The grouped (M3) lane-table structs, in a file rather than a literal for the
/// same reason `RUNTIME_TEMPLATE` is: a C++ reader needs them too. The Metal
/// driver's `msl_compile_test` reconstructs the emitter's full sources from the
/// golden dump plus these two prefixes, and a raw string literal in this crate
/// is reachable from nothing but this crate.
pub const GROUPED_PREAMBLE: &str =
    include_str!("../../runtime/metal/ptir_m1_grouped.metal");

/// `common_effect_preamble()` — the structs the single-lane readiness and
/// commit kernels read out of the lane table.
pub fn common_effect_preamble() -> &'static str {
    r#"
#include <metal_stdlib>
using namespace metal;
struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };
struct M1LaneHeader { uint abi_version; uint lane_count; uint channel_count; uint flags; };
struct M1LaneRecord {
  ulong logits_base;
  uint logits_row_offset;
  uint logits_row_count;
  uint kv_len;
  uint page_count;
  uint row_count;
  uint token_count;
  uint sampled_rows;
  uint query_len;
  uint key_len;
  uint channel_slot_offset;
  ulong rng_state;
  ulong commit_slot;
  ulong active_row_mask;
  ulong sample_output_channel_mask;
  ulong row_valid;
  uint row_valid_offset;
  uint reserved0;
};
struct M1LaneChannelSlot {
  ulong committed_cell;
  ulong pending_cell;
  ulong expected_head;
  ulong expected_tail;
};
"#
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
    GROUPED_PREAMBLE
}
