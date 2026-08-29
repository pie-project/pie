//! The device fault-code space — every number a kernel can leave in
//! `M1Status::fault` / `M3Status::fault`, declared once.
//!
//! Nothing decodes these: the engines surface the number and a human reads it.
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
//! ## The op-tag half of the space
//!
//! Both runtimes' interpreters report an unhandled op as `m1_fault(status,
//! p.tag)` — the raw op tag, a `0x00..=0xFF` value from an unrelated space.
//! (An earlier version of this comment called that a CUDA-only divergence with
//! Metal passing a fixed code; `ptir_m1_runtime.metal:583,741,816` says
//! otherwise.) So every class below `0x100` shares numbers with the op table,
//! and [`FUSED_GEOMETRY_MISMATCH`] is in fact `intrinsic_val`'s tag exactly.
//!
//! And they meet: an emitted Metal region carries the runtime preamble, so one
//! kernel both writes [`FUSED_GEOMETRY_MISMATCH`] and reports unhandled ops as
//! `m1_fault(status, p.tag)`. A `fault` of `0xA0` from that kernel is either
//! "the region's geometry did not match" or "`intrinsic_val` had no arm", and
//! nothing in `M1Status` separates them — plain `m1_fault` does not touch the
//! `reserved` words that `m1_fault_op`'s three-way site code lives in.
//!
//! That is a live ambiguity, not a hypothetical, and it is recorded rather than
//! fixed: respacing the codes above the tag space rewrites emitted bytes in
//! every fused golden, which is an ABI change needing device re-verification,
//! not a cleanup. [`TAG_ALIASES`] is the record, and it is recomputed from the
//! op table on every run — so assigning a new op the tag `0xB3`, or adding a
//! sub-`0x100` class, fails a test instead of quietly widening the damage.

/// One named region of the fault space.
///
/// `Copy` and the comparison traits are derived because the engines re-export
/// this rather than restating it, and a decoder walks the table by value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultClass {
    /// The value written for channel 0. Per-channel classes write `base + channel`.
    pub base: u32,
    /// The class's symbolic name, matching its `pub const` — the label a reader
    /// maps a raw fault code back to.
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

/// A generated fused region's inputs do not have the geometry the kernel was
/// emitted against: a zero vocabulary, or a row window that runs past the row
/// metadata. Three guards in the Metal fused emitter write it — the draft
/// window, the argmax input, and the intrinsic row block.
///
/// Below `0x100`, and equal to `intrinsic_val`'s op tag. See the module docs
/// for why that is safe and what checks it stays safe.
pub const FUSED_GEOMETRY_MISMATCH: u32 = 0xA0;

/// A grouped region was launched with more threads than
/// [`METAL_M3_REGION_THREADS`](crate::codegen::metal::fused::METAL_M3_REGION_THREADS),
/// which is what its threadgroup reduction buffer is sized for. The kernel
/// reports this instead of reading past the buffer.
///
/// Below `0x100`, like [`FUSED_GEOMETRY_MISMATCH`], but `0xB3` is unassigned in
/// the op table today; the module docs say what keeps that from mattering.
pub const M3_THREADS_EXCEEDED: u32 = 0xB3;

/// Every sub-`0x100` class and the op whose tag it collides with, since those
/// classes share numbers with the op table and the two meet inside one emitted
/// kernel. `None` means the tag is unassigned today.
///
/// Kept by hand so that changing it is a decision; checked against the op table
/// by [`the_tag_aliases_are_still_what_they_say`](self) so that *not* changing
/// it cannot be an accident.
pub const TAG_ALIASES: &[(u32, Option<&str>)] = &[
    (FUSED_GEOMETRY_MISMATCH, Some("intrinsic_val")),
    (M3_THREADS_EXCEEDED, None),
];

/// Every class, in ascending order.
pub const CLASSES: &[FaultClass] = &[
    FaultClass {
        base: FUSED_GEOMETRY_MISMATCH,
        name: "FUSED_GEOMETRY_MISMATCH",
        per_channel: false,
    },
    FaultClass {
        base: M3_THREADS_EXCEEDED,
        name: "M3_THREADS_EXCEEDED",
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
    use crate::codegen::metal::METAL_M1_MAX_CHANNELS;

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

    /// The sub-`0x100` classes are the ones that share numbers with the op
    /// table, so the set of them and the ops they hit is a fact about two
    /// tables that move independently. Recomputing it here is what turns
    /// "someone wrote 0xB3 down once" into "0xB3 is still unassigned".
    #[test]
    fn the_tag_aliases_are_still_what_they_say() {
        let below: Vec<&FaultClass> = CLASSES.iter().filter(|c| c.base < 0x100).collect();
        assert_eq!(
            below.len(),
            TAG_ALIASES.len(),
            "{} fault classes sit in the op-tag space but TAG_ALIASES lists {} — \
             a class below 0x100 needs an entry saying which op it collides with",
            below.len(),
            TAG_ALIASES.len()
        );
        for (class, (code, alias)) in below.iter().zip(TAG_ALIASES) {
            assert_eq!(
                class.base, *code,
                "TAG_ALIASES is out of step with CLASSES at {}",
                class.name
            );
            let actual = u8::try_from(*code)
                .ok()
                .and_then(tensor_ir::op::spec)
                .map(|spec| spec.name);
            assert_eq!(
                actual, *alias,
                "{} ({code:#x}) now collides with {actual:?}, not {alias:?}. A crash \
                 report showing {code:#x} from a generated region cannot be told \
                 apart from one showing the op tag; decide whether to respace the \
                 class or accept the alias, then update TAG_ALIASES",
                class.name
            );
        }
    }
}
