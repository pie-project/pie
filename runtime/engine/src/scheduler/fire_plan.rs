//! Fire planning: the device-independent half of "how do these co-batch".
//!
//! [`LaunchGrouping::accepts`](super::worker::LaunchGrouping) answers *can*
//! these members share one step; this module answers *how* — the row order
//! and, per divergence site, a lowering. It is the port of the tart
//! prototype's planner vocabulary (`tart/plan.py` + `tart/ir.py`,
//! re-measured on L40S in `stage0-l40s.md`) into the home plan.md Part 2 and
//! pie-application-plan.md §4.3 assign it: beside the admission rule, in the
//! scheduler, because only the scheduler sees the model's divergence sites,
//! the attached programs, the device cost model, and what was admitted.
//!
//! v0 is deliberately narrow: [`plan_fire`] re-derives, as data, exactly the
//! decisions the scheduler already makes in scattered places — the
//! `(device_resolved_geometry, hook_program)` stable sort in
//! `batch::build_frame_submission` and the two per-site fast-path choices the
//! driver currently re-derives on its own (see [`SITE_QKV_POSTPROCESS`] and
//! [`SITE_PROJECTION_WEIGHTS`]). The member order is consumed and asserted
//! equivalent to the sort it replaces; the sites are consumed by nothing yet.
//! That is the point: when the next divergence axis lands it lands as one
//! more [`Site`], not as a fourth hard-coded mechanism.
//!
//! Per pie-application-plan.md §4.4 the planner emits candidates, not final
//! answers: class assignment is structural and device-independent, and a
//! later increment lets the runtime pick among candidate lowerings with
//! device cost. v0 has one candidate per site, so the distinction is latent.

/// How a divergence site's variation prices out, independent of device.
///
/// Port of `tart/ir.py Div`; the costs are the measured ones
/// (plan.md §"Why", pie-application-plan.md §3.3, re-measured on L40S in
/// `stage0-l40s.md` §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DivClass {
    /// Identical for every member; emit once, no branch.
    #[allow(dead_code)] // v0's two sites are never Shared; Stage 6's regions will be.
    Shared,
    /// Folds into an additive fix on an already-materialized output:
    /// ~1.1x the no-divergence floor (L40S 1.12x, 3090 1.01x) — far below
    /// any branch, so it never gets one.
    #[allow(dead_code)]
    // no Correction-class site yet; the class is part of the ported vocabulary.
    Correction,
    /// Same operator, per-member weights: one batched GEMM, no branch
    /// (Stage 4's lora; the padded-vs-batched call measured at 1.00x).
    Weight,
    /// Genuinely different operators: the fused region must split
    /// (~1.13-1.15x the floor on L40S vs 1.87x for not merging at all).
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

/// The lowering chosen for one site, named by its compiler analogue
/// (plan.md Part 2's table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lowering {
    /// Every lane agrees — devirtualization; the fast path covers the whole
    /// step and no per-lane machinery is needed.
    Uniform,
    /// The agreeing prefix takes the fast path, the tail does not — loop
    /// peeling. This is Stage 1's `fast_rows`, made possible by the member
    /// order putting the agreeing lanes first. v0 counts MEMBERS, not wire
    /// rows: the driver's `Dispatch::launch_hook_free_prefix_rows` still
    /// re-derives the row count independently from the wire layout, and a
    /// later increment hands the plan's answer across instead.
    Prefix { fast_rows: u32 },
    /// Per-lane weights/corrections applied by span — dictionary passing.
    /// Today this is Stage 4's per-adapter lora loop in the driver; the
    /// batched-GEMM upgrade (one kernel over a lane-indexed table) is a
    /// later candidate on the same site.
    PerLane,
    /// Genuinely different operators behind a guard. Reserved for Stage 6's
    /// conditional regions (coalesced, ~250us floor per region —
    /// `stage0-l40s.md` §3.4).
    #[allow(dead_code)] // reserved: no Structural site lowers to a real branch yet.
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

/// One place in the model where the step's members may diverge, with the
/// lowering this plan chose for it.
#[derive(Clone, Debug)]
pub(crate) struct Site {
    pub(crate) name: &'static str,
    #[allow(dead_code)] // read by report()/tests; the runtime consumer is a later increment.
    pub(crate) class: DivClass,
    pub(crate) lowering: Lowering,
    /// Why this lowering — for `report()`, mirroring `tart/plan.py`.
    pub(crate) note: String,
}

/// The QKV-postprocess site: attention-hook (`OnAttnProj`/`OnAttn`) programs
/// switch a lane off the fused QKV+norm+rope+KV-write kernel. Structural —
/// the fused region splits. Mirrors Stage 1's hard-coded pair: the
/// hooks-last member sort plus the driver's hook-free fast prefix
/// (`StageHooks::hook_free_prefix_rows`); the driver will eventually consume
/// this site's `fast_rows` instead of re-deriving it.
pub(crate) const SITE_QKV_POSTPROCESS: &str = "qkv_postprocess";

/// The projection-weights site: a program carrying the pass-wide `lora`
/// sink wants x(W+BA)^T where its neighbors want xW^T. Weight-class — same
/// operator, per-member weights. Mirrors Stage 4's hard-coded lowering: the
/// driver's per-adapter span loop in `llama_like` (applied per lane; the
/// single batched GEMM over a lane table is the upgrade candidate).
pub(crate) const SITE_PROJECTION_WEIGHTS: &str = "projection_weights";

/// The facts about one step member that planning reads — nothing else.
/// Device geometry is a wire-format fact, the other two are program facts
/// stamped at launch admission from the tracked instance
/// (`RegisterProgram` -> `LaneCommit` -> `TrackedInstance` ->
/// `PendingRequest`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemberFacts {
    /// The program declares an attention-hook stage (`OnAttnProj`/`OnAttn`).
    pub(crate) hook_program: bool,
    /// The program carries the pass-wide `lora` configuration sink.
    pub(crate) lora: bool,
    /// Device-resolved (chained-decode envelope) geometry: composes as the
    /// ordered suffix sub-batch, never interleaved with wire members.
    pub(crate) device_resolved_geometry: bool,
    /// Arrival position within the step group; the stable-order tiebreak.
    pub(crate) arrival: usize,
}

/// One step's plan: the member permutation plus a lowering per site.
#[derive(Clone, Debug)]
pub(crate) struct FirePlan {
    /// Indices into the planned members, in submission order. The existing
    /// sort key generalized: `(device_resolved_geometry, hook_program,
    /// arrival)`, stable. Device-geometry members last is PRIMARY (the
    /// driver's offset fixed-decode compose needs the envelope lanes as a
    /// contiguous program suffix); hooks-last within each class is what
    /// makes the qkv_postprocess prefix maximal.
    pub(crate) member_order: Vec<usize>,
    pub(crate) sites: Vec<Site>,
}

impl FirePlan {
    /// Debug rendering, mirroring `tart/plan.py BatchPlan.report`.
    #[allow(dead_code)] // debugging surface; tests exercise it.
    pub(crate) fn report(&self) -> String {
        let mut out = vec![format!("{} members", self.member_order.len())];
        for site in &self.sites {
            out.push(format!(
                "  {:<20} {:<11} -> {}{}",
                site.name,
                site.class.as_str(),
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

/// Plan one step group.
///
/// Everything here is device-independent (pie-application-plan.md §4.4):
/// the order and the class assignments are structural facts about the
/// members; picking among candidate lowerings with device cost is the
/// runtime's later job (v0 emits one candidate per site).
pub(crate) fn plan_fire(members: &[MemberFacts]) -> FirePlan {
    let mut member_order: Vec<usize> = (0..members.len()).collect();
    // sort_by_key is stable, so equal keys keep `arrival` order even
    // without the explicit third component; it is in the key anyway so the
    // contract survives callers passing members out of arrival order.
    member_order.sort_by_key(|&index| {
        let member = &members[index];
        (
            member.device_resolved_geometry,
            member.hook_program,
            member.arrival,
        )
    });

    let hook_members = members.iter().filter(|m| m.hook_program).count();
    let qkv_postprocess = if hook_members == 0 {
        Site {
            name: SITE_QKV_POSTPROCESS,
            class: DivClass::Structural,
            lowering: Lowering::Uniform,
            note: "no hook lanes; the fused QKV path covers the step".to_string(),
        }
    } else {
        // The agreeing prefix after ordering. All-hook degenerates to
        // fast_rows = 0 honestly: nothing takes the fused path, which is
        // exactly what the driver derives for such a step today.
        let fast_rows = member_order
            .iter()
            .take_while(|&&index| !members[index].hook_program)
            .count() as u32;
        Site {
            name: SITE_QKV_POSTPROCESS,
            class: DivClass::Structural,
            lowering: Lowering::Prefix { fast_rows },
            note: format!("{hook_members} hook lane(s) peeled off the fused QKV path"),
        }
    };

    let lora_members = members.iter().filter(|m| m.lora).count();
    let projection_weights = if lora_members == 0 {
        Site {
            name: SITE_PROJECTION_WEIGHTS,
            class: DivClass::Weight,
            lowering: Lowering::Uniform,
            note: "no adapters; base weights only".to_string(),
        }
    } else {
        Site {
            name: SITE_PROJECTION_WEIGHTS,
            class: DivClass::Weight,
            lowering: Lowering::PerLane,
            note: format!("{lora_members} lora lane(s); corrections applied by span"),
        }
    };

    FirePlan {
        member_order,
        sites: vec![qkv_postprocess, projection_weights],
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
            device_resolved_geometry,
            arrival,
        }
    }

    fn site<'a>(plan: &'a FirePlan, name: &str) -> &'a Site {
        plan.sites
            .iter()
            .find(|site| site.name == name)
            .expect("site is always planned")
    }

    /// The stable sort `plan_fire` replaces, applied to the same facts.
    fn legacy_order(members: &[MemberFacts]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..members.len()).collect();
        order.sort_by_key(|&i| (members[i].device_resolved_geometry, members[i].hook_program));
        order
    }

    #[test]
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

    #[test]
    fn all_hook_members_plan_an_empty_prefix() {
        let members: Vec<MemberFacts> = (0..3).map(|i| member(true, false, false, i)).collect();
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![0, 1, 2]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 0 }
        );
    }

    #[test]
    fn mixed_hooks_order_hook_free_first_and_count_the_prefix() {
        // Arrival order: hook, plain, hook, plain, plain.
        let members = vec![
            member(true, false, false, 0),
            member(false, false, false, 1),
            member(true, false, false, 2),
            member(false, false, false, 3),
            member(false, false, false, 4),
        ];
        let plan = plan_fire(&members);
        // Hook-free lanes first in arrival order, then hook lanes in
        // arrival order (the Stage 1 fix: the prefix no longer ends at
        // whichever hook lane arrived first).
        assert_eq!(plan.member_order, vec![1, 3, 4, 0, 2]);
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 3 }
        );
    }

    #[test]
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
        // lora does not perturb the member order: WEIGHT-class divergence
        // is a pointer, not a branch, so no seriation is needed for it.
        assert_eq!(plan.member_order, vec![0, 1, 2]);
    }

    #[test]
    fn device_geometry_members_are_forced_last() {
        // A device-resolved envelope lane arriving FIRST must still land
        // after every wire lane, hooks or not.
        let members = vec![
            member(false, false, true, 0),
            member(true, false, false, 1),
            member(false, false, false, 2),
        ];
        let plan = plan_fire(&members);
        assert_eq!(plan.member_order, vec![2, 1, 0]);
        // The prefix counts only the leading hook-free run; the hook-free
        // envelope lane behind the wire hook lane is not in it. This
        // mirrors today's driver derivation exactly (min row start over
        // attention-stage programs).
        assert_eq!(
            site(&plan, SITE_QKV_POSTPROCESS).lowering,
            Lowering::Prefix { fast_rows: 1 }
        );
    }

    #[test]
    fn member_order_matches_the_sort_it_replaced() {
        // Property-style over every (dev, hook) assignment of 4 members:
        // the planner's order must equal the old inline stable sort.
        for bits in 0..256u32 {
            let members: Vec<MemberFacts> = (0..4)
                .map(|i| member(bits & (1 << i) != 0, false, bits & (1 << (i + 4)) != 0, i))
                .collect();
            let plan = plan_fire(&members);
            assert_eq!(
                plan.member_order,
                legacy_order(&members),
                "divergence at bits={bits:#x}"
            );
        }
    }

    #[test]
    fn report_names_every_site() {
        let members = vec![member(false, true, false, 0), member(true, false, false, 1)];
        let report = plan_fire(&members).report();
        assert!(report.contains(SITE_QKV_POSTPROCESS));
        assert!(report.contains(SITE_PROJECTION_WEIGHTS));
        assert!(report.contains("prefix(fast_rows=1)"));
        assert!(report.contains("per-lane"));
    }
}
