//! Which buffer index a fused Metal region reads each intrinsic's rectangle
//! from — the M2 SLOT TABLE, in the only spelling Metal has for one.
//!
//! **THE CUDA TWIN'S SLOT TABLE IS FIVE SIDE ARRAYS; THIS ONE IS A SET OF
//! BUFFER INDICES, AND THE DIFFERENCE IS THE PLATFORM RATHER THAN THE IDEA.**
//! A CUDA kernel argument is a raw device address, so `engine-cuda` writes a
//! base, a storage mode, a width, a row stride and a row offset per intrinsic
//! and the kernel walks them. Metal binds an OBJECT plus an offset, so a
//! rectangle IS `setBuffer:offset:atIndex:` — which means the table is
//! indexed by the ARGUMENT INDEX and every other number is either the
//! binding's own or the op record's.
//!
//! Until this wave there was exactly one such index. `emit_fused_region`
//! wrote `const device uchar* logits [[buffer(6)]]` and made it the first
//! argument of EVERY `INTRINSIC_VAL` op, so `logits` and `mtp_logits` in one
//! stage were two names for one rectangle — which `engine-metal` refused at
//! bind rather than served wrong, and which `engine-metal/src/api.rs`
//! reported as `has_mtp_logits: false`. The refusal was honest and the ABI
//! was the reason.
//!
//! **THE TRUNK'S RECTANGLE KEEPS INDEX 6 AND THE OTHERS GROW DOWN FROM 30.**
//! Six is where every emitted kernel has bound the logits since there were
//! emitted kernels, and moving it would rewrite the argument index of every
//! channel behind it (`7 + 2k` / `8 + 2k`) in every fused kernel this
//! compiler has ever emitted — a whole-corpus byte diff to buy one slot. The
//! other rectangles take the TOP of Metal's index space instead, so a stage
//! that reads none emits exactly the bytes it did before, and only a stage
//! that actually reads a second intrinsic grows an argument.
//!
//! The cost is one channel: the channels grow up from 7 and the intrinsics
//! grow down from 30, so a stage doing both meets in the middle at eleven
//! rather than twelve. See [`fused_channel_ceiling`].
//!
//! **AND THE TABLE CARRIES AN ELEMENT TYPE NOW, WHICH THE CUDA TWIN CARRIES
//! PER BINDING.** `AttnScore`'s rectangle is F32 where every other id's is
//! bf16, and on this plane that difference has nowhere to live but the id:
//! the binding is an object and an offset, so there is no storage word to
//! set. [`m2_intrinsic_element_bytes`] is the host-readable half of the arm
//! `ptir_m1_runtime.metal` takes on `p.intr`, and both halves are the same
//! sentence said in the two places that have to agree.

use eta_ir::op::intrinsic_tags;

/// Where the `logits` rectangle binds. Unmoved since the C++ emitter, and
/// unmovable without rewriting every channel index behind it.
pub const M2_LOGITS_BUFFER: usize = 6;

/// The highest buffer index Metal accepts, and therefore the first slot the
/// non-trunk rectangles take.
pub const M2_INTRINSIC_TOP_BUFFER: usize = 30;

/// The buffer index `intr`'s rectangle binds at in a fused M2 kernel, or
/// `None` for an intrinsic this backend cannot bind at all.
///
/// **THE SET IS EXACTLY [`super::validate::metal_intrinsic_supported`]'s**,
/// and it has to be: an id with no index here would fall through to the
/// trunk's rectangle and read the logits under another intrinsic's name,
/// which is the silent mis-binding that validator exists to refuse. The two
/// are asserted against each other in `metal_intrinsic_slots_cover_the_
/// whitelist`.
///
/// **`AttnScore` IS IN THE TABLE NOW, AND THE DOOR IT WAITED FOR WAS THE
/// RUNTIME'S.** This row used to say the id was deliberately absent — "a
/// score plane is F32 and the `0xA0` handler reads `bfloat`, so a slot for it
/// would be an index without a reader". The reader exists:
/// `ptir_m1_runtime.metal`'s `0xA0` arm branches on `p.intr` and gathers
/// `float` for this id (`.wiki/alto/attn-score.md` §4). So the slot is no
/// longer an index without a reader, and it takes the next index down.
#[must_use]
pub fn m2_intrinsic_buffer(intr: u16) -> Option<usize> {
    match intr {
        intrinsic_tags::LOGITS => Some(M2_LOGITS_BUFFER),
        intrinsic_tags::MTP_LOGITS => Some(M2_INTRINSIC_TOP_BUFFER),
        intrinsic_tags::MTP_DRAFTS => Some(M2_INTRINSIC_TOP_BUFFER - 1),
        intrinsic_tags::ATTN_SCORE => Some(M2_INTRINSIC_TOP_BUFFER - 2),
        _ => None,
    }
}

/// How wide one element of `intr`'s rectangle is, in bytes — `None` for an
/// id this backend cannot bind at all.
///
/// **THE HOST'S COPY OF AN ARM THE RUNTIME TAKES ON `p.intr`.** The emitted
/// `0xA0` handler reads `bfloat` for every intrinsic but one, and `float` for
/// [`intrinsic_tags::ATTN_SCORE`] — a per-key mass is a probability a policy
/// divides by, not a bf16 quantity. The CUDA twin is told this per binding
/// (`p.intrinsic_dtype`, out of a host side array); here it is a function of
/// the ID, because Metal's table is indexed by argument index and the element
/// type therefore has nowhere else to live.
///
/// Published rather than restated, because the number is read in two places
/// that cannot see each other: this compiler emits the kernel, and
/// `engine_metal::program::launch` decides how far a reader of `n` elements
/// reaches into the buffer it was pointed at. A second copy of it would be a
/// bounds check computed against the wrong element.
#[must_use]
pub fn m2_intrinsic_element_bytes(intr: u16) -> Option<u32> {
    match intr {
        intrinsic_tags::ATTN_SCORE => Some(4),
        _ => m2_intrinsic_buffer(intr).map(|_| 2),
    }
}

/// Whether the GROUPED (M3) emitter can bind `intr`.
///
/// **NARROWER THAN [`m2_intrinsic_buffer`], AND THE GAP IS ONE ID.** A
/// grouped kernel binds no per-intrinsic buffer at all: it reads
/// `lane.logits_base` out of the lane record and every `INTRINSIC_VAL` op is
/// handed that one address (`emit_grouped_fused_region`). That is why the
/// draft column has no grouped half either — it rides `M3RowMeta::mtp_offset`
/// off the same base — and the score slab is not the same allocation as the
/// logits at all, so there is no offset that would reach it.
///
/// So the M2 slot table opening for `AttnScore` does not open the M3 one, and
/// a grouped region that reads it is refused by name rather than emitted
/// pointing at the trunk's rows. Closing this needs a score base on the lane
/// record, which is the same `MTLBuffer.gpuAddress` plumbing the grouped path
/// is waiting on for everything else.
#[must_use]
pub fn m3_intrinsic_bindable(intr: u16) -> bool {
    intr != intrinsic_tags::ATTN_SCORE && m2_intrinsic_buffer(intr).is_some()
}

/// How many channels a fused region may bind directly once the intrinsics in
/// `used` have taken their slots.
///
/// `used` is the region's own intrinsic ids; the trunk's is free because its
/// index sits below the channels rather than above them.
///
/// **A CEILING RATHER THAN AN OVERLAP** — the channels grow up from
/// `FIRST_CHANNEL_BUFFER` and the intrinsic slots grow down from
/// [`M2_INTRINSIC_TOP_BUFFER`], and a kernel that let them meet would bind
/// one rectangle at the index a channel cell was already bound at. The engine
/// would see a channel take return the last row of the logits.
#[must_use]
pub fn fused_channel_ceiling(used: &[u16]) -> usize {
    let lowest = used
        .iter()
        .filter_map(|&intr| m2_intrinsic_buffer(intr))
        .filter(|&at| at > M2_LOGITS_BUFFER)
        .min();
    match lowest {
        // `committed_k` is `7 + 2k` and `pending_k` is `8 + 2k`, so `n`
        // channels end at `6 + 2n` and must stay strictly below `lowest`.
        Some(lowest) => lowest.saturating_sub(7) / 2,
        None => super::METAL_M2_MAX_FUSED_CHANNELS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::metal::validate::metal_intrinsic_supported;
    use eta_ir::op::IntrinsicId;

    /// Every intrinsic the Metal backend claims to bind has somewhere to bind
    /// it, and nothing else does. A supported id with no slot reads the
    /// trunk's logits under its own name; a slot for an unsupported id is an
    /// argument index nothing can reach.
    #[test]
    fn metal_intrinsic_slots_cover_the_whitelist() {
        for id in IntrinsicId::ALL {
            let intr = *id as u16;
            assert_eq!(
                m2_intrinsic_buffer(intr).is_some(),
                metal_intrinsic_supported(intr),
                "{id:?} disagrees between the slot table and the whitelist"
            );
        }
    }

    /// No two intrinsics share an index, and none of them lands where a
    /// channel cell would.
    #[test]
    fn metal_intrinsic_slots_are_distinct() {
        let mut seen = alloc::vec::Vec::new();
        for id in IntrinsicId::ALL {
            if let Some(at) = m2_intrinsic_buffer(*id as u16) {
                assert!(at <= M2_INTRINSIC_TOP_BUFFER, "{id:?} binds past Metal's last index");
                assert!(!seen.contains(&at), "{id:?} reuses buffer index {at}");
                seen.push(at);
            }
        }
    }

    /// The ceiling shrinks by exactly one channel when a second rectangle is
    /// in play, and not at all for the trunk's.
    #[test]
    fn a_second_rectangle_costs_one_channel() {
        assert_eq!(
            fused_channel_ceiling(&[intrinsic_tags::LOGITS]),
            super::super::METAL_M2_MAX_FUSED_CHANNELS
        );
        assert_eq!(fused_channel_ceiling(&[intrinsic_tags::MTP_LOGITS]), 11);
        assert_eq!(
            fused_channel_ceiling(&[intrinsic_tags::MTP_LOGITS, intrinsic_tags::MTP_DRAFTS]),
            11
        );
        // The score rectangle sits one index below the drafts, so a region
        // that reads it gives up one more channel than one that does not.
        // `saturating_sub(7) / 2` on 28 is ten, and the arithmetic is the
        // ceiling's own rather than a number restated here.
        assert_eq!(fused_channel_ceiling(&[intrinsic_tags::ATTN_SCORE]), 10);
    }

    /// Every rectangle this table binds has an element width, and exactly one
    /// of them is not the `bfloat` the `0xA0` handler reads by default.
    ///
    /// The two arms are one sentence — `ptir_m1_runtime.metal` branches on
    /// `p.intr` and this is the host's reading of the same branch — so a new
    /// id gaining a slot without gaining a width would have
    /// `engine_metal::program::launch` bounds-check its reader against the
    /// wrong element size.
    #[test]
    fn every_bindable_intrinsic_states_an_element_width() {
        for id in IntrinsicId::ALL {
            let intr = *id as u16;
            assert_eq!(
                m2_intrinsic_element_bytes(intr).is_some(),
                m2_intrinsic_buffer(intr).is_some(),
                "{id:?} has a slot and no element width, or the other way round"
            );
        }
        assert_eq!(m2_intrinsic_element_bytes(intrinsic_tags::ATTN_SCORE), Some(4));
        assert_eq!(m2_intrinsic_element_bytes(intrinsic_tags::LOGITS), Some(2));
        assert_eq!(m2_intrinsic_element_bytes(intrinsic_tags::MTP_LOGITS), Some(2));
        assert_eq!(m2_intrinsic_element_bytes(intrinsic_tags::MTP_DRAFTS), Some(2));
    }

    /// The grouped table is the single-lane one minus the score rectangle,
    /// and the difference is a missing lane-record base rather than a policy.
    #[test]
    fn the_grouped_path_binds_everything_but_the_score_rectangle() {
        for id in IntrinsicId::ALL {
            let intr = *id as u16;
            let expected =
                m2_intrinsic_buffer(intr).is_some() && intr != intrinsic_tags::ATTN_SCORE;
            assert_eq!(
                m3_intrinsic_bindable(intr),
                expected,
                "{id:?} disagrees between the M2 and M3 tables for a reason nothing states"
            );
        }
        assert!(!m3_intrinsic_bindable(intrinsic_tags::ATTN_SCORE));
        assert!(m3_intrinsic_bindable(intrinsic_tags::LOGITS));
    }
}
