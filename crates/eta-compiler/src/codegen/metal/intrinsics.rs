use eta_ir::op::intrinsic_tags;

pub const M2_LOGITS_BUFFER: usize = 6;

pub const M2_INTRINSIC_TOP_BUFFER: usize = 30;

#[must_use]
pub fn m2_intrinsic_buffer(intr: u16) -> Option<usize> {
    match intr {
        intrinsic_tags::LOGITS | intrinsic_tags::VELOCITY | intrinsic_tags::HIDDEN => {
            Some(M2_LOGITS_BUFFER)
        }
        intrinsic_tags::MTP_LOGITS => Some(M2_INTRINSIC_TOP_BUFFER),
        intrinsic_tags::MTP_DRAFTS => Some(M2_INTRINSIC_TOP_BUFFER - 1),
        intrinsic_tags::ATTN_SCORE => Some(M2_INTRINSIC_TOP_BUFFER - 2),
        intrinsic_tags::PIXELS => Some(M2_INTRINSIC_TOP_BUFFER - 3),
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

#[must_use]
pub fn fused_channel_ceiling(used: &[u16]) -> usize {
    let lowest = used
        .iter()
        .filter_map(|&intr| m2_intrinsic_buffer(intr))
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

    use eta_ir::op::IntrinsicId;

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
