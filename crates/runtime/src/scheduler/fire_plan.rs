#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DivClass {
    #[allow(dead_code)] // v0's two sites are never Shared.
    Shared,
    #[allow(dead_code)]
    Correction,
    Weight,
    Structural,
}

impl DivClass {
    fn as_str(self) -> &'static str {
        match self {
            DivClass::Shared => "shared",
            DivClass::Correction => "correction",
            DivClass::Weight => "weight",
            DivClass::Structural => "structural",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Granularity {
    Request,
    Token,
}

impl Granularity {
    fn as_str(self) -> &'static str {
        match self {
            Granularity::Request => "per-request",
            Granularity::Token => "per-token",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lowering {
    Uniform,
    Prefix {
        fast_rows: u32,
    },
    PerLane,
    #[allow(dead_code)]
    Conditional,
}

impl Lowering {
    fn describe(self) -> String {
        match self {
            Lowering::Uniform => "uniform".to_string(),
            Lowering::Prefix { fast_rows } => format!("prefix(fast_rows={fast_rows})"),
            Lowering::PerLane => "per-lane".to_string(),
            Lowering::Conditional => "conditional".to_string(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Site {
    pub(crate) name: &'static str,
    #[allow(dead_code)]
    pub(crate) class: DivClass,
    #[allow(dead_code)]
    pub(crate) granularity: Granularity,
    pub(crate) lowering: Lowering,
    pub(crate) note: String,
}

pub(crate) const SITE_QKV_POSTPROCESS: &str = "qkv_postprocess";

pub(crate) const SITE_PROJECTION_WEIGHTS: &str = "projection_weights";

pub(crate) const SITE_ATTENTION_MASK: &str = "attention_mask";

#[allow(
    dead_code,
    reason = "the vocabulary outlives its producer on purpose; the module doc measures why"
)]
pub(crate) const SITE_EXPERT_WEIGHTS: &str = "expert_weights";

#[allow(
    dead_code,
    reason = "same as `SITE_EXPERT_WEIGHTS`: nothing reports a site, so only tests build one"
)]
pub(crate) fn expert_weights_site(experts: u32, top_k: u32) -> Site {
    Site {
        name: SITE_EXPERT_WEIGHTS,
        class: DivClass::Weight,
        granularity: Granularity::Token,
        lowering: Lowering::PerLane,
        note: format!(
            "top-{top_k} of {experts} experts selected per token; \
             grouped GEMM over gathered tokens is the device-side lowering"
        ),
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MemberFacts {
    pub(crate) hook_program: bool,
    pub(crate) lora: bool,
    pub(crate) custom_mask: bool,
    pub(crate) truncated: bool,
    pub(crate) max_layers: Option<u32>,
    pub(crate) multi_token: bool,
    pub(crate) geometry_class: eta_ir::registry::GeometryClass,
    pub(crate) arrival: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FirePlan {
    pub(crate) member_order: Vec<usize>,
    pub(crate) sites: Vec<Site>,
}

impl FirePlan {
    #[allow(dead_code)]
    pub(crate) fn report(&self) -> String {
        let mut out = vec![format!("{} members", self.member_order.len())];
        for site in &self.sites {
            out.push(format!(
                "  {:<20} {:<11} {:<11} -> {}{}",
                site.name,
                site.class.as_str(),
                site.granularity.as_str(),
                site.lowering.describe(),
                if site.note.is_empty() {
                    String::new()
                } else {
                    format!("   ({})", site.note)
                }
            ));
        }
        out.join("\n")
    }
}

#[allow(dead_code)]
pub(crate) fn plan_fire(members: &[MemberFacts]) -> FirePlan {
    plan_fire_with_model(members, &[])
}

pub(crate) fn plan_fire_with_model(members: &[MemberFacts], model_sites: &[Site]) -> FirePlan {
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

    {
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
                eprintln!(
                    "[seriation] window axis fragmented under the Act-2 order (mask_ok={mask_ok} hook_ok={hook_ok} trunc_ok={trunc_ok}, order={member_order:?}) — this combination wants the gather fallback"
                );
            }
        }
    }
    let hook_members = members.iter().filter(|m| m.hook_program).count();
    let qkv_postprocess = if hook_members == 0 {
        Site {
            name: SITE_QKV_POSTPROCESS,
            class: DivClass::Structural,
            granularity: Granularity::Request,
            lowering: Lowering::Uniform,
            note: "no hook lanes; the fused QKV path covers the step".to_string(),
        }
    } else {
        let fast_rows = member_order
            .iter()
            .take_while(|&&index| !members[index].hook_program)
            .count() as u32;
        Site {
            name: SITE_QKV_POSTPROCESS,
            class: DivClass::Structural,
            granularity: Granularity::Request,
            lowering: Lowering::Prefix { fast_rows },
            note: format!("{hook_members} hook lane(s) peeled off the fused QKV path"),
        }
    };

    let lora_members = members.iter().filter(|m| m.lora).count();
    let projection_weights = if lora_members == 0 {
        Site {
            name: SITE_PROJECTION_WEIGHTS,
            class: DivClass::Weight,
            granularity: Granularity::Request,
            lowering: Lowering::Uniform,
            note: "no adapters; base weights only".to_string(),
        }
    } else {
        Site {
            name: SITE_PROJECTION_WEIGHTS,
            class: DivClass::Weight,
            granularity: Granularity::Request,
            lowering: Lowering::PerLane,
            note: format!("{lora_members} lora lane(s); corrections applied by span"),
        }
    };

    let mask_members = members.iter().filter(|m| m.custom_mask).count();
    let attention_mask = if mask_members == 0 {
        Site {
            name: SITE_ATTENTION_MASK,
            class: DivClass::Structural,
            granularity: Granularity::Request,
            lowering: Lowering::Uniform,
            note: "no masked lanes; the plain attention arm covers the step".to_string(),
        }
    } else {
        let unmasked_rows = member_order
            .iter()
            .take_while(|&&index| !members[index].custom_mask)
            .count() as u32;
        Site {
            name: SITE_ATTENTION_MASK,
            class: DivClass::Structural,
            granularity: Granularity::Request,
            lowering: Lowering::Prefix {
                fast_rows: unmasked_rows,
            },
            note: format!(
                "{mask_members} masked lane(s) seriated to the tail; the unmasked prefix keeps the plain attention arm (NS-2 consumes the split)"
            ),
        }
    };

    let mut sites = vec![qkv_postprocess, projection_weights, attention_mask];
    sites.extend_from_slice(model_sites);

    FirePlan {
        member_order,
        sites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn member(
        hook_program: bool,
        lora: bool,
        device_resolved_geometry: bool,
        arrival: usize,
    ) -> MemberFacts {
        MemberFacts {
            hook_program,
            lora,
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
            hook_program: false,
            lora: false,
            custom_mask: true,
            truncated: false,
            max_layers: None,
            multi_token: false,
            geometry_class: eta_ir::registry::GeometryClass::Host,
            arrival,
        }
    }

    #[test]
    fn fire_plan_every_case() {
        truncated_members_seriate_deepest_first();
        full_depth_hook_sorts_before_truncated_members();
        masked_members_seriate_last_and_the_site_counts_the_prefix();
        all_plain_members_plan_uniform_everywhere();
        all_hook_members_plan_an_empty_prefix();
        mixed_hooks_order_hook_free_first_and_count_the_prefix();
        lora_mixing_plans_per_lane_weights();
        device_geometry_members_are_forced_last();
    }

    fn truncated_members_seriate_deepest_first() {
        let band = |k: u32, arrival: usize| {
            let mut m = member(false, false, false, arrival);
            m.truncated = true;
            m.max_layers = Some(k);
            m
        };
        let members = [
            band(8, 0),
            band(24, 1),
            band(16, 2),
            member(false, false, false, 3),
        ];
        let plan = plan_fire_with_model(&members, &[]);
        assert_eq!(
            plan.member_order,
            vec![3, 1, 2, 0],
            "full depth first, then bands deepest-first"
        );
    }

    fn full_depth_hook_sorts_before_truncated_members() {
        let band = |k: u32, arrival: usize| {
            let mut m = member(false, false, false, arrival);
            m.truncated = true;
            m.max_layers = Some(k);
            m
        };
        let members = [
            band(8, 0),
            member(true, false, false, 1),
            band(12, 2),
            member(false, false, false, 3),
        ];
        let plan = plan_fire_with_model(&members, &[]);
        assert_eq!(
            plan.member_order,
            vec![3, 1, 2, 0],
            "[plain-full, hook-full, k12, k8]"
        );
        let masked = [band(8, 0), member(true, false, false, 1), masked_member(2)];
        let plan = plan_fire_with_model(&masked, &[]);
        assert_eq!(
            plan.member_order,
            vec![1, 0, 2],
            "[hook-full, k8, masked] — mask stays the outermost suffix"
        );
    }

    fn masked_members_seriate_last_and_the_site_counts_the_prefix() {
        let members = vec![
            masked_member(0),
            member(false, false, false, 1),
            masked_member(2),
            member(false, false, false, 3),
        ];
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![1, 3, 0, 2]);
        let mask_site = site(&plan, SITE_ATTENTION_MASK);
        assert_eq!(mask_site.class, DivClass::Structural);
        assert_eq!(mask_site.lowering, Lowering::Prefix { fast_rows: 2 });
        let mixed = vec![masked_member(0), member(true, false, false, 1)];
        let plan = plan_fire(&mixed);
        assert_eq!(plan.member_order, vec![1, 0]);
    }

    fn site<'a>(plan: &'a FirePlan, name: &str) -> &'a Site {
        plan.sites
            .iter()
            .find(|site| site.name == name)
            .expect("site is always planned")
    }

    fn all_plain_members_plan_uniform_everywhere() {
        let members: Vec<MemberFacts> = (0..4).map(|i| member(false, false, false, i)).collect();
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![0, 1, 2, 3]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Uniform
        );
        assert_eq!(
            site(&plan, SITE_PROJECTION_WEIGHTS).lowering,
            Lowering::Uniform
        );
    }

    fn all_hook_members_plan_an_empty_prefix() {
        let members: Vec<MemberFacts> = (0..3).map(|i| member(true, false, false, i)).collect();
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![0, 1, 2]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 0 }
        );
    }

    fn mixed_hooks_order_hook_free_first_and_count_the_prefix() {
        let members = vec![
            member(true, false, false, 0),
            member(false, false, false, 1),
            member(true, false, false, 2),
            member(false, false, false, 3),
            member(false, false, false, 4),
        ];
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![1, 3, 4, 0, 2]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 3 }
        );
    }

    fn lora_mixing_plans_per_lane_weights() {
        let members = vec![
            member(false, false, false, 0),
            member(false, true, false, 1),
            member(false, false, false, 2),
        ];
        let plan = plan_fire(&members);
        assert_eq!(
            site(&plan, SITE_PROJECTION_WEIGHTS).lowering,
            Lowering::PerLane
        );
        assert_eq!(site(&plan, SITE_PROJECTION_WEIGHTS).class, DivClass::Weight);
        assert_eq!(plan.member_order, vec![0, 1, 2]);
    }

    fn device_geometry_members_are_forced_last() {
        let members = vec![
            member(false, false, true, 0),
            member(true, false, false, 1),
            member(false, false, false, 2),
        ];
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![2, 1, 0]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 1 }
        );
    }
}
