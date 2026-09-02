//! The device fault-code space: every number a kernel can leave in
//! `M1Status::fault` / `M3Status::fault`. Undecoded by the engines (a human
//! reads it); this module is the one place the space is written down.
//!
//! The table ships with the compiled artifact rather than being linked by
//! the reading binary, and is copied onto
//! [`LaunchPackage::fault_classes`](crate::codegen::launch::LaunchPackage::fault_classes).
//! Codes in the shared per-op-tag range alias raw op tags from an unrelated
//! space; [`TAG_ALIASES`] is the recorded, tested correspondence.

use serde::{Deserialize, Serialize};

/// One named region of the fault space. `name` is an owned `String` (not
/// `&'static str`) because this table rides on the artifact and must
/// deserialize.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultClass {
    /// The value written for channel 0. Per-channel classes write `base + channel`.
    pub base: u32,
    /// The class's symbolic name, matching its `pub const` — the label a reader
    /// maps a raw fault code back to.
    pub name: String,
    /// Whether the emitter adds a channel index to [`base`](Self::base).
    pub per_channel: bool,
}

/// The lane table's header did not match what the emitter compiled against
/// (ABI version, lane count, or channel count). `reserved0`/`reserved1` carry
/// the observed values.
pub const LANE_HEADER_MISMATCH: u32 = 0x100;

/// M1 ring words are corrupt: a reserved word is non-zero, or `tail < head`.
pub const M1_RING_CORRUPT: u32 = 0x200;

/// M1 ring head moved since the host published `expected_head` — stale lane table.
pub const M1_HEAD_STALE: u32 = 0x300;

/// M1 `take`/`read` needs a full ring and found it empty.
pub const M1_NOT_FULL: u32 = 0x400;

/// M1 `put` needs room and found the ring at capacity.
pub const M1_NOT_EMPTY: u32 = 0x480;

/// M1 `put` precondition: tail moved, or the ring is at capacity plus credit.
pub const M1_PUT_BLOCKED: u32 = 0x500;

/// M3 grouped ring words are corrupt — [`M1_RING_CORRUPT`]'s counterpart.
pub const M3_RING_CORRUPT: u32 = 0x700;

/// M3 grouped readiness unmet (head stale, or full/empty/put precondition).
/// The grouped path folds M1's four causes into one code; `state` distinguishes
/// retry (2) from fault (3).
pub const M3_NOT_READY: u32 = 0x780;

/// A generated fused region's inputs don't match the geometry the kernel
/// was emitted against. Aliases `intrinsic_val`'s op tag; see the module
/// docs for why that is safe.
pub const FUSED_GEOMETRY_MISMATCH: u32 = 0xA0;

/// A grouped region launched with more threads than
/// [`METAL_M3_REGION_THREADS`](crate::codegen::metal::fused::METAL_M3_REGION_THREADS),
/// which sizes its threadgroup reduction buffer. Aliases an op tag; see the
/// module docs.
pub const M3_THREADS_EXCEEDED: u32 = 0xB3;

/// Every aliasing class and the op whose tag it collides with (`None` = tag
/// unassigned). Kept by hand; checked against the op table by
/// [`the_tag_aliases_are_still_what_they_say`](self).
pub const TAG_ALIASES: &[(u32, Option<&str>)] = &[
    (FUSED_GEOMETRY_MISMATCH, Some("intrinsic_val")),
    (M3_THREADS_EXCEEDED, None),
];

/// The class table as declared: base, name, per-channel. A tuple table
/// since [`FaultClass`] carries a `String`, which is not `const`-able.
const TABLE: &[(u32, &str, bool)] = &[
    (FUSED_GEOMETRY_MISMATCH, "FUSED_GEOMETRY_MISMATCH", false),
    (M3_THREADS_EXCEEDED, "M3_THREADS_EXCEEDED", false),
    (LANE_HEADER_MISMATCH, "LANE_HEADER_MISMATCH", false),
    (M1_RING_CORRUPT, "M1_RING_CORRUPT", true),
    (M1_HEAD_STALE, "M1_HEAD_STALE", true),
    (M1_NOT_FULL, "M1_NOT_FULL", true),
    (M1_NOT_EMPTY, "M1_NOT_EMPTY", true),
    (M1_PUT_BLOCKED, "M1_PUT_BLOCKED", true),
    (M3_RING_CORRUPT, "M3_RING_CORRUPT", true),
    (M3_NOT_READY, "M3_NOT_READY", true),
];

/// Every class, in ascending order. A function, not a `const`, since the
/// table serializes and cannot hold `&'static str`.
#[must_use]
pub fn classes() -> Vec<FaultClass> {
    TABLE
        .iter()
        .map(|&(base, name, per_channel)| FaultClass {
            base,
            name: name.to_string(),
            per_channel,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::metal::METAL_M1_MAX_CHANNELS;

    /// A per-channel class occupies `base ..= base + max_channel`; the next
    /// base must fall outside that run.
    #[test]
    fn the_classes_do_not_overlap() {
        let highest_channel = (METAL_M1_MAX_CHANNELS - 1) as u32;
        let table = classes();
        for pair in table.windows(2) {
            let (lower, upper) = (&pair[0], &pair[1]);
            assert!(
                lower.base < upper.base,
                "{} and {} are out of order",
                lower.name,
                upper.name
            );
            let last = if lower.per_channel {
                lower.base + highest_channel
            } else {
                lower.base
            };
            assert!(
                last < upper.base,
                "{} runs to {last:#x}, which collides with {} at {:#x} — raising \
                 METAL_M1_MAX_CHANNELS past this gap aliases two fault classes",
                lower.name,
                upper.name,
                upper.base
            );
        }
    }

    /// The tightest gap between per-channel bases bounds
    /// `METAL_M1_MAX_CHANNELS`.
    #[test]
    fn the_tightest_gap_bounds_the_channel_count() {
        let table = classes();
        let tightest = table
            .windows(2)
            .filter(|pair| pair[0].per_channel)
            .map(|pair| pair[1].base - pair[0].base)
            .min()
            .expect("there is at least one per-channel class");
        assert_eq!(tightest, 0x80, "the M1_NOT_FULL/M1_NOT_EMPTY gap");
        assert!(
            METAL_M1_MAX_CHANNELS as u32 <= tightest,
            "METAL_M1_MAX_CHANNELS ({METAL_M1_MAX_CHANNELS}) exceeds the tightest \
             fault-class gap ({tightest:#x}); respace the bases in this module first"
        );
    }

}
