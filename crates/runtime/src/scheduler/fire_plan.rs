#[derive(Clone, Copy, Debug)]
pub(crate) struct MemberFacts {
    pub(crate) custom_mask: bool,
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
            !member.multi_token,
            member.arrival,
        )
    });
    member_order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(device_resolved_geometry: bool, arrival: usize) -> MemberFacts {
        MemberFacts {
            custom_mask: false,
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
            ..member(false, arrival)
        }
    }

    #[test]
    fn fire_plan_every_case() {
        masked_members_seriate_last();
        device_geometry_members_are_forced_last();
    }

    fn masked_members_seriate_last() {
        let members = [
            masked_member(0),
            member(false, 1),
            masked_member(2),
            member(false, 3),
        ];
        assert_eq!(member_order(&members), vec![1, 3, 0, 2]);
    }

    fn device_geometry_members_are_forced_last() {
        let members = [member(true, 0), member(false, 1), member(false, 2)];
        assert_eq!(member_order(&members), vec![1, 2, 0]);
    }
}
