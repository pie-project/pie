//! The device fault-code space — every number a kernel can leave in
//! `M1Status::fault` / `M3Status::fault`, declared once.
//!
//! Nothing decodes these: the drivers surface the number and a human reads it.
//! That is exactly why they were worth naming — an undecoded diagnostic has no
//! test that fails when two conditions start reporting the same value, so the
//! only protection is that the space is written down and checked.
//!
//! ## Shape of the space
//!
//! The readiness classes are *per channel*: the emitter writes `BASE + channel`,
//! so each base owns a run of consecutive codes and the bases must be spaced
//! wider than the largest channel index a kernel can see. Two of the gaps are
//! only `0x80` — [`the_classes_do_not_overlap`](self) is what keeps a future
//! channel-count raise from silently aliasing "ring empty" onto "ring full".
//!
//! ## The CUDA divergence, recorded not fixed
//!
//! `m1_fault(&status, tag)` in the CUDA fused emitter passes the **op tag**,
//! which is an unrelated `0x00..=0xFF` space, and the Metal fused emitter
//! passes the fixed [`FUSED_UNSUPPORTED_OP`]. So a `fault` value below `0x100`
//! means "which op" on one backend and "unsupported op" on the other. Both are
//! diagnostics with no consumer, so unifying them is an ABI decision rather
//! than a cleanup, and it is not made here.

/// One named region of the fault space.
pub struct FaultClass {
    /// The value written for channel 0. Per-channel classes write `base + channel`.
    pub base: u32,
    pub name: &'static str,
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

/// The Metal fused emitter's "this op has no lowering" code. Below `0x100`, and
/// so in the same numeric range the CUDA emitter fills with op tags — see the
/// module docs.
pub const FUSED_UNSUPPORTED_OP: u32 = 0xA0;

/// Every class, in ascending order.
pub const CLASSES: &[FaultClass] = &[
    FaultClass {
        base: FUSED_UNSUPPORTED_OP,
        name: "FUSED_UNSUPPORTED_OP",
        per_channel: false,
    },
    FaultClass {
        base: LANE_HEADER_MISMATCH,
        name: "LANE_HEADER_MISMATCH",
        per_channel: false,
    },
    FaultClass {
        base: M1_RING_CORRUPT,
        name: "M1_RING_CORRUPT",
        per_channel: true,
    },
    FaultClass {
        base: M1_HEAD_STALE,
        name: "M1_HEAD_STALE",
        per_channel: true,
    },
    FaultClass {
        base: M1_NOT_FULL,
        name: "M1_NOT_FULL",
        per_channel: true,
    },
    FaultClass {
        base: M1_NOT_EMPTY,
        name: "M1_NOT_EMPTY",
        per_channel: true,
    },
    FaultClass {
        base: M1_PUT_BLOCKED,
        name: "M1_PUT_BLOCKED",
        per_channel: true,
    },
    FaultClass {
        base: M3_RING_CORRUPT,
        name: "M3_RING_CORRUPT",
        per_channel: true,
    },
    FaultClass {
        base: M3_NOT_READY,
        name: "M3_NOT_READY",
        per_channel: true,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::METAL_M1_MAX_CHANNELS;

    /// A per-channel class occupies `base ..= base + max_channel`. If the next
    /// base falls inside that run, one number names two conditions and the
    /// reader of a crash report has no way to tell which.
    #[test]
    fn the_classes_do_not_overlap() {
        let highest_channel = (METAL_M1_MAX_CHANNELS - 1) as u32;
        for pair in CLASSES.windows(2) {
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

    /// The gaps are the budget for `METAL_M1_MAX_CHANNELS`. Naming the tightest
    /// one makes the ceiling a fact rather than something to rediscover.
    #[test]
    fn the_tightest_gap_bounds_the_channel_count() {
        let tightest = CLASSES
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
