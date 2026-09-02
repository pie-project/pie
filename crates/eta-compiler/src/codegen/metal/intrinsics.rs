//! Buffer-index slot table for a fused Metal region's intrinsic rectangles.
//! Metal binds an object+offset (`setBuffer:offset:atIndex:`), so the table
//! is indexed by argument index; the trunk's `logits` rectangle stays fixed
//! at index 6, and other rectangles take the top of Metal's index space
//! downward so untouched stages keep emitting identical bytes. Channels grow
//! up from 7 and intrinsics grow down from 30 (see [`fused_channel_ceiling`]).

use eta_ir::op::intrinsic_tags;

/// Where the `logits` rectangle binds. Fixed: moving it would rewrite every
/// channel index behind it in every emitted kernel.
pub const M2_LOGITS_BUFFER: usize = 6;

/// The highest buffer index Metal accepts, and therefore the first slot the
/// non-trunk rectangles take.
pub const M2_INTRINSIC_TOP_BUFFER: usize = 30;

/// The buffer index `intr`'s rectangle binds at in a fused M2 kernel, or
/// `None` for an intrinsic this backend cannot bind at all. The set matches
/// [`super::validate::metal_intrinsic_supported`] exactly: an id with no
/// index here would fall through to the trunk's rectangle and silently read
/// the logits under another intrinsic's name.
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
/// id this backend cannot bind at all. `ATTN_SCORE` is F32 (4 bytes); every
/// other bindable intrinsic is bf16 (2 bytes). Must stay consistent with
/// `engine_metal::program::launch`, which uses this for bounds checks.
#[must_use]
pub fn m2_intrinsic_element_bytes(intr: u16) -> Option<u32> {
    match intr {
        intrinsic_tags::ATTN_SCORE => Some(4),
        _ => m2_intrinsic_buffer(intr).map(|_| 2),
    }
}

/// Whether the GROUPED (M3) emitter can bind `intr`. Same set as
/// [`m2_intrinsic_buffer`]: a grouped kernel binds no per-intrinsic buffer,
/// every rectangle arrives as a `ulong` on the lane record instead
/// (`LaneRecord::attn_score_base`/`attn_score_row_stride` for the score
/// plane).
#[must_use]
pub fn m3_intrinsic_bindable(intr: u16) -> bool {
    m2_intrinsic_buffer(intr).is_some()
}

/// How many channels a fused region may bind directly once the intrinsics in
/// `used` have taken their slots.
///
/// `used` is the region's own intrinsic ids; the trunk's is free because its
/// index sits below the channels rather than above them. Channels grow up
/// from `FIRST_CHANNEL_BUFFER`, intrinsic slots grow down from
/// [`M2_INTRINSIC_TOP_BUFFER`]; they must never meet or a channel and a
/// rectangle would share an index.
#[must_use]
pub fn fused_channel_ceiling(used: &[u16]) -> usize {
    let lowest = used
        .iter()
        .filter_map(|&intr| m2_intrinsic_buffer(intr))
        .filter(|&at| at > M2_LOGITS_BUFFER)
        .min();
    match lowest {
        // channels end at `6 + 2n` and must stay strictly below `lowest`
        Some(lowest) => lowest.saturating_sub(7) / 2,
        None => super::METAL_M2_MAX_FUSED_CHANNELS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use eta_ir::op::IntrinsicId;

    // Checks the M2 and M3 tables agree id for id.
    #[test]
    fn the_grouped_path_binds_every_rectangle_the_single_lane_form_does() {
        for id in IntrinsicId::ALL {
            let intr = *id as u16;
            assert_eq!(
                m3_intrinsic_bindable(intr),
                m2_intrinsic_buffer(intr).is_some(),
                "{id:?} disagrees between the M2 and M3 tables for a reason nothing states"
            );
        }
        assert!(m3_intrinsic_bindable(intrinsic_tags::ATTN_SCORE));
        assert!(m3_intrinsic_bindable(intrinsic_tags::LOGITS));
    }
}
