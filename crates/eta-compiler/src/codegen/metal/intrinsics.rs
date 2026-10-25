use eta_ir::op::{IntrinsicId, intrinsic_tags};

pub const M2_LOGITS_BUFFER: usize = 6;

pub const M2_INTRINSIC_TOP_BUFFER: usize = 30;

/// The single-lane form's argument index of the instance's address table:
/// one `ulong` a slot ([`eta_ir::op::intrinsic_slot`]), the stage's
/// [`eta_ir::op::intrinsic_pitch`] of them.
pub const M2_INTRINSIC_TABLE_BUFFER: usize = M2_INTRINSIC_TOP_BUFFER - 3;

/// The grouped and streamed forms' argument index of the dispatch's address
/// tables, one an instance, dispatch lane by dispatch lane.
pub const M3_INTRINSIC_TABLE_BUFFER: usize = 12;

/// Whether an intrinsic is read through the address table: a per-lane one
/// (`velocity`, `hidden`, `pixels`), each lane at its own slot.
#[must_use]
pub fn addressed(intr: u16) -> bool {
    addressed_slot(intr, 0).is_some()
}

/// The table slot a per-lane intrinsic of the pass's lane `lane` is read at;
/// `None` for the pass's intrinsics, each at an argument of its own.
#[must_use]
pub fn addressed_slot(intr: u16, lane: u32) -> Option<u32> {
    IntrinsicId::from_u16(intr)
        .filter(|id| id.per_lane())
        .map(|id| eta_ir::op::intrinsic_slot(id, lane))
}

#[must_use]
pub fn m2_intrinsic_buffer(intr: u16) -> Option<usize> {
    match intr {
        intrinsic_tags::LOGITS => Some(M2_LOGITS_BUFFER),
        intrinsic_tags::MTP_LOGITS => Some(M2_INTRINSIC_TOP_BUFFER),
        intrinsic_tags::MTP_DRAFTS => Some(M2_INTRINSIC_TOP_BUFFER - 1),
        intrinsic_tags::ATTN_SCORE => Some(M2_INTRINSIC_TOP_BUFFER - 2),
        _ if addressed(intr) => Some(M2_INTRINSIC_TABLE_BUFFER),
        _ => None,
    }
}

#[must_use]
pub fn m2_intrinsic_element_bytes(intr: u16) -> Option<u32> {
    match intr {
        intrinsic_tags::ATTN_SCORE | intrinsic_tags::MTP_DRAFTS => Some(4),
        _ => m2_intrinsic_buffer(intr).map(|_| 2),
    }
}

#[must_use]
pub fn m3_intrinsic_bindable(intr: u16) -> bool {
    m2_intrinsic_buffer(intr).is_some()
}

/// The channels the single-lane form seats below the lowest of `used`, the
/// intrinsic argument indices a region binds.
#[must_use]
pub fn fused_channel_ceiling(used: &[usize]) -> usize {
    let lowest = used
        .iter()
        .copied()
        .filter(|&at| at > M2_LOGITS_BUFFER)
        .min();
    match lowest {
        Some(lowest) => lowest.saturating_sub(7) / 2,
        None => super::METAL_M2_MAX_FUSED_CHANNELS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
