use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaultClass {
    pub base: u32,
    pub name: String,
    pub per_channel: bool,
}

pub const LANE_HEADER_MISMATCH: u32 = 0x100;

pub const M1_RING_CORRUPT: u32 = 0x200;

pub const M1_HEAD_STALE: u32 = 0x300;

pub const M1_NOT_FULL: u32 = 0x400;

pub const M1_NOT_EMPTY: u32 = 0x480;

pub const M1_PUT_BLOCKED: u32 = 0x500;

pub const M3_RING_CORRUPT: u32 = 0x700;

pub const M3_NOT_READY: u32 = 0x780;

pub const FUSED_GEOMETRY_MISMATCH: u32 = 0xA0;

pub const M3_THREADS_EXCEEDED: u32 = 0xB3;

pub const M4_REDUCE_TOO_WIDE: u32 = 0xB4;

pub const TAG_ALIASES: &[(u32, Option<&str>)] = &[
    (FUSED_GEOMETRY_MISMATCH, Some("intrinsic_val")),
    (M3_THREADS_EXCEEDED, None),
    (M4_REDUCE_TOO_WIDE, None),
];

const TABLE: &[(u32, &str, bool)] = &[
    (FUSED_GEOMETRY_MISMATCH, "FUSED_GEOMETRY_MISMATCH", false),
    (M3_THREADS_EXCEEDED, "M3_THREADS_EXCEEDED", false),
    (M4_REDUCE_TOO_WIDE, "M4_REDUCE_TOO_WIDE", false),
    (LANE_HEADER_MISMATCH, "LANE_HEADER_MISMATCH", false),
    (M1_RING_CORRUPT, "M1_RING_CORRUPT", true),
    (M1_HEAD_STALE, "M1_HEAD_STALE", true),
    (M1_NOT_FULL, "M1_NOT_FULL", true),
    (M1_NOT_EMPTY, "M1_NOT_EMPTY", true),
    (M1_PUT_BLOCKED, "M1_PUT_BLOCKED", true),
    (M3_RING_CORRUPT, "M3_RING_CORRUPT", true),
    (M3_NOT_READY, "M3_NOT_READY", true),
];

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

    #[test]
    fn fault_every_case() {
        the_classes_do_not_overlap();
        the_tightest_gap_bounds_the_channel_count();
    }

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
