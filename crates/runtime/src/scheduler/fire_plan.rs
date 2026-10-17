#[derive(Clone, Copy, Debug)]
pub(crate) struct MemberFacts {
    pub(crate) hook_program: bool,
    pub(crate) custom_mask: bool,
    pub(crate) truncated: bool,
    pub(crate) max_layers: Option<u32>,
    pub(crate) multi_token: bool,
    pub(crate) geometry_class: eta_ir::registry::GeometryClass,
    pub(crate) arrival: usize,
}

pub(crate) fn member_order(members: &[MemberFacts]) -> Vec<usize> {
    let mut member_order: Vec<usize> = (0..members.len()).collect();
    member_order.sort_by_key(|&index| {
        let member = &members[index];
        (
            member.geometry_class,
            member.custom_mask,
            member.truncated,
            std::cmp::Reverse(member.max_layers.unwrap_or(u32::MAX)),
            member.hook_program,
            !member.multi_token,
            member.arrival,
        )
    });

    let contiguous = |bit: fn(&MemberFacts) -> bool| -> bool {
        let mut runs = 0;
        let mut inside = false;
        for &index in &member_order {
            let hit = bit(&members[index]);
            if hit && !inside {
                runs += 1;
            }
            inside = hit;
        }
        runs <= 1
    };
    let mask_ok = contiguous(|m| m.custom_mask);
    let hook_ok = contiguous(|m| m.hook_program);
    let trunc_ok = contiguous(|m| m.truncated);
    if !(mask_ok && hook_ok && trunc_ok) {
        static FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !FIRED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!(
                mask_ok,
                hook_ok,
                trunc_ok,
                order = ?member_order,
                "a step's members do not fire in contiguous runs; this combination wants the gather fallback"
            );
        }
    }
    member_order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(hook_program: bool, device_resolved_geometry: bool, arrival: usize) -> MemberFacts {
        MemberFacts {
            hook_program,
            custom_mask: false,
            truncated: false,
            max_layers: None,
            multi_token: false,
            geometry_class: if device_resolved_geometry {
                eta_ir::registry::GeometryClass::DecodeEnvelope
            } else {
                eta_ir::registry::GeometryClass::Host
            },
            arrival,
        }
    }

    fn masked_member(arrival: usize) -> MemberFacts {
        MemberFacts {
            custom_mask: true,
            ..member(false, false, arrival)
        }
    }

    fn band(k: u32, arrival: usize) -> MemberFacts {
        MemberFacts {
            truncated: true,
            max_layers: Some(k),
            ..member(false, false, arrival)
        }
    }

    #[test]
    fn fire_plan_every_case() {
        truncated_members_seriate_deepest_first();
        full_depth_hook_sorts_before_truncated_members();
        masked_members_seriate_last();
        mixed_hooks_order_hook_free_first();
        device_geometry_members_are_forced_last();
    }

    fn truncated_members_seriate_deepest_first() {
        let members = [
            band(8, 0),
            band(24, 1),
            band(16, 2),
            member(false, false, 3),
        ];
        assert_eq!(
            member_order(&members),
            vec![3, 1, 2, 0],
            "full depth first, then bands deepest-first"
        );
    }

    fn full_depth_hook_sorts_before_truncated_members() {
        let members = [
            band(8, 0),
            member(true, false, 1),
            band(12, 2),
            member(false, false, 3),
        ];
        assert_eq!(
            member_order(&members),
            vec![3, 1, 2, 0],
            "[plain-full, hook-full, k12, k8]"
        );
        let masked = [band(8, 0), member(true, false, 1), masked_member(2)];
        assert_eq!(
            member_order(&masked),
            vec![1, 0, 2],
            "[hook-full, k8, masked] — mask stays the outermost suffix"
        );
    }

    fn masked_members_seriate_last() {
        let members = [
            masked_member(0),
            member(false, false, 1),
            masked_member(2),
            member(false, false, 3),
        ];
        assert_eq!(member_order(&members), vec![1, 3, 0, 2]);
        let mixed = [masked_member(0), member(true, false, 1)];
        assert_eq!(member_order(&mixed), vec![1, 0]);
    }

    fn mixed_hooks_order_hook_free_first() {
        let members = [
            member(true, false, 0),
            member(false, false, 1),
            member(true, false, 2),
            member(false, false, 3),
            member(false, false, 4),
        ];
        assert_eq!(member_order(&members), vec![1, 3, 4, 0, 2]);
    }

    fn device_geometry_members_are_forced_last() {
        let members = [
            member(false, true, 0),
            member(true, false, 1),
            member(false, false, 2),
        ];
        assert_eq!(member_order(&members), vec![2, 1, 0]);
    }
}
