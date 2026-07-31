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
//!
//! # Two site provenances
//!
//! Sites arrive from two distinct sources, and the split is semantic, not
//! historical:
//!
//! * **Member facts** — per-fire *attachment* divergence: what the admitted
//!   programs brought with them (attention hooks, a lora sink). These vary
//!   fire to fire and are derived here in [`plan_fire`] from
//!   [`MemberFacts`]; [`SITE_QKV_POSTPROCESS`] and
//!   [`SITE_PROJECTION_WEIGHTS`] are this provenance.
//! * **The traced form** — *model-structural* divergence: what the model
//!   itself declares, true for every member of every fire against it. An
//!   MoE trace's per-token expert selection ([`SITE_EXPERT_WEIGHTS`]) is
//!   this provenance; [`site_table::derive_sites`] walks a
//!   `pie_forward::ForwardPlan` and emits them once per model, not per
//!   fire.
//!
//! [`plan_fire_with_model`] merges both when the caller holds a traced
//! plan; `build_frame_submission` does not yet (see the [`site_table`]
//! module doc for why, honestly, it cannot).

pub(crate) mod site_table;

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

/// The extent a site's divergence varies over.
///
/// Port of the granularity axis `tart/ir.py` carries on WEIGHT-class nodes:
/// `matmul(x, W[i])` with `i` per-REQUEST is SGMV (adapters — Stage 4's
/// lora), with `i` per-TOKEN it is MoE grouped GEMM — the same expression,
/// two hand-written kernels, the syntactic identity plan.md Part 1 is built
/// on. Request-granularity sites work in member/row spans after the
/// planner's seriation; token-granularity sites cannot be seriated by
/// member order at all — their variation lives inside each member's rows,
/// so the lowering is always data-driven (gather → grouped GEMM →
/// scatter), never a prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Granularity {
    /// Varies per fire member (request/lane): adapters, hooks, depth.
    Request,
    /// Varies per token row within every member: the MoE expert axis.
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
    /// This is Stage 4's per-adapter lora correction in the driver. The
    /// driver additionally groups SAME-SHAPE lanes into one grouped-GEMM
    /// launch (llama_like's `LoraFireState`); that stays under this
    /// classification, not a new lowering here, because whether shapes
    /// share a kernel is device knowledge (§4.4) — the same reason
    /// `Prefix::fast_rows` is re-derived driver-side.
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
    /// The extent the divergence varies over; [`Granularity::Request`] for
    /// every site the member sort can seriate.
    #[allow(dead_code)] // read by report()/tests, like `class`.
    pub(crate) granularity: Granularity,
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
/// operator, per-member weights. The driver applies corrections by span
/// (`llama_like`'s `LoraFireState`), sharing one grouped-GEMM launch across
/// lanes whose shapes already agree — a device-side detail this
/// classification deliberately does not model (the §4.4 split; see the
/// grouping comment at that site).
pub(crate) const SITE_PROJECTION_WEIGHTS: &str = "projection_weights";

/// The expert-weights site: per-TOKEN weight divergence — an MoE trace's
/// expert-indexed matmuls (`pie_forward`'s `Matmul { selector }`, the
/// `layer.{l}.expert.{e}.*` templates whose `{e}` a `TopK` value resolves
/// per token). Weight-class like [`SITE_PROJECTION_WEIGHTS`], at the other
/// granularity: same operator, per-token weights, no branch.
///
/// NOT derived from [`MemberFacts`], and deliberately so: member facts have
/// no moe bit and fires do not carry one, because this site is a fact about
/// the TRACED FORM, not about the members — every member of a fire against
/// an MoE model diverges here, expert assignment being data. It is emitted
/// by the plan-derived site table ([`site_table::derive_sites`]: walk the
/// `ForwardPlan`, one Site per distinct selector parameterization) and
/// merged into a fire's plan by [`plan_fire_with_model`] when the caller
/// holds a traced plan — which `build_frame_submission` does not yet (the
/// [`site_table`] module doc records the wiring analysis).
pub(crate) const SITE_EXPERT_WEIGHTS: &str = "expert_weights";

/// The [`SITE_EXPERT_WEIGHTS`] vocabulary entry, as the plan-derived site
/// table will emit it.
///
/// The `PerLane` candidate here means "per selected weight, by span" —
/// dictionary passing, same as lora — and is deliberately NOT the final
/// word: the real lowering (grouped GEMM over gathered tokens) already
/// exists device-side, several families' worth (`qwen3_5_moe`'s
/// batched/aligned/CUTLASS pipelines, and the deepseek_v4 / kimi / gemma4 /
/// glm5 / mixtral / nemotron_h MoE paths), and per §4.4 choosing among
/// those strategies is the runtime/driver's device-knowledge call, exactly
/// as with the lora span-vs-grouped grouping above.
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

/// Plan one step group from member facts alone.
///
/// Equivalent to [`plan_fire_with_model`] with no model-structural sites —
/// the form `build_frame_submission` calls while the scheduler does not yet
/// hold a traced plan (the [`site_table`] module doc records why). Kept as
/// the named entry point so that wiring the traced plan through is one call
/// site's change, not a signature churn across every existing caller.
pub(crate) fn plan_fire(members: &[MemberFacts]) -> FirePlan {
    plan_fire_with_model(members, &[])
}

/// Plan one step group, merging both site provenances (module doc): the
/// member-fact sites this function derives, then `model_sites` — the
/// model-structural sites a caller derived once from the traced form
/// ([`site_table::derive_sites`]) — appended in the caller's order.
///
/// Everything here is device-independent (pie-application-plan.md §4.4):
/// the order and the class assignments are structural facts about the
/// members; picking among candidate lowerings with device cost is the
/// runtime's later job (v0 emits one candidate per site). Model-structural
/// sites never affect `member_order`: token-granularity divergence lives
/// inside each member's rows, so no member permutation can seriate it.
pub(crate) fn plan_fire_with_model(members: &[MemberFacts], model_sites: &[Site]) -> FirePlan {
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
            granularity: Granularity::Request,
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

    let mut sites = vec![qkv_postprocess, projection_weights];
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
        assert!(report.contains("per-request"));
    }

    /// Every site `plan_fire` emits is request-granularity: its divergence
    /// is a member fact, which is exactly why the member sort can seriate
    /// it. Token granularity never comes from member facts (that would be
    /// the fake derivation the expert site's doc rules out).
    #[test]
    fn planned_sites_are_request_granularity() {
        let members = vec![member(true, true, false, 0), member(false, false, false, 1)];
        let plan = plan_fire(&members);
        assert_eq!(plan.sites.len(), 2);
        for site in &plan.sites {
            assert_eq!(site.granularity, Granularity::Request, "{}", site.name);
        }
    }

    /// The expert-weights vocabulary entry: Weight-class at TOKEN
    /// granularity — `matmul(x, W[i])` with `i` per-token, the grouped-GEMM
    /// half of the SGMV/MoE identity — with the per-lane dictionary-passing
    /// candidate and a note pointing at the device-side lowering. Pinned
    /// here as the shape [`site_table::derive_sites`] emits (its own tests
    /// pin the derivation; this one pins the vocabulary).
    #[test]
    fn expert_weights_site_vocabulary() {
        let site = expert_weights_site(256, 8);
        assert_eq!(site.name, SITE_EXPERT_WEIGHTS);
        assert_eq!(site.class, DivClass::Weight);
        assert_eq!(site.granularity, Granularity::Token);
        assert_eq!(site.lowering, Lowering::PerLane);
        assert!(site.note.contains("top-8 of 256 experts"));
        assert!(site.note.contains("grouped GEMM"));

        // And it renders alongside the planned sites' vocabulary.
        let mut plan = plan_fire(&[member(false, false, false, 0)]);
        plan.sites.push(site);
        let report = plan.report();
        assert!(report.contains(SITE_EXPERT_WEIGHTS));
        assert!(report.contains("per-token"));
    }

    /// [`plan_fire_with_model`] merges the two provenances: the member-fact
    /// sites first, exactly as [`plan_fire`] derives them, then the
    /// model-structural sites in the caller's order — with the member order
    /// untouched (token-granularity divergence cannot be seriated by a
    /// member permutation).
    #[test]
    fn plan_fire_with_model_appends_model_sites() {
        let members = vec![
            member(false, true, false, 0),
            member(true, false, false, 1),
            member(false, false, false, 2),
        ];
        let base = plan_fire(&members);
        let plan = plan_fire_with_model(&members, &[expert_weights_site(256, 8)]);

        assert_eq!(plan.member_order, base.member_order);
        assert_eq!(plan.sites.len(), base.sites.len() + 1);
        for (merged, member_fact) in plan.sites.iter().zip(&base.sites) {
            assert_eq!(merged.name, member_fact.name);
            assert_eq!(merged.lowering, member_fact.lowering);
        }
        let appended = plan.sites.last().expect("model site appended");
        assert_eq!(appended.name, SITE_EXPERT_WEIGHTS);
        assert_eq!(appended.granularity, Granularity::Token);

        // And the merged plan reports all three.
        let report = plan.report();
        assert!(report.contains(SITE_QKV_POSTPROCESS));
        assert!(report.contains(SITE_PROJECTION_WEIGHTS));
        assert!(report.contains(SITE_EXPERT_WEIGHTS));
    }

    /// [`plan_fire`] is exactly the no-model-sites merge — the equivalence
    /// `build_frame_submission` relies on while it passes none.
    #[test]
    fn plan_fire_equals_the_empty_model_merge() {
        let members = vec![
            member(true, false, true, 0),
            member(false, true, false, 1),
            member(false, false, false, 2),
        ];
        let via_plain = plan_fire(&members);
        let via_merge = plan_fire_with_model(&members, &[]);
        assert_eq!(via_plain.member_order, via_merge.member_order);
        assert_eq!(via_plain.sites.len(), via_merge.sites.len());
        for (a, b) in via_plain.sites.iter().zip(&via_merge.sites) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.lowering, b.lowering);
            assert_eq!(a.note, b.note);
        }
    }

    /// End-to-end across the provenances: sites derived from a real MoE
    /// traced form merge into a fire plan alongside the member-fact sites.
    #[test]
    fn derived_moe_sites_merge_into_a_fire_plan() {
        let traced = pie_forward::family::qwen3_5_moe_mlp_block(
            &pie_forward::Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
        );
        let model_sites = site_table::derive_sites(&traced);
        let members = vec![member(false, false, false, 0), member(true, false, false, 1)];
        let plan = plan_fire_with_model(&members, &model_sites);
        assert_eq!(plan.sites.len(), 3);
        let expert = plan
            .sites
            .iter()
            .find(|site| site.name == SITE_EXPERT_WEIGHTS)
            .expect("expert site merged from the traced form");
        assert_eq!(expert.class, DivClass::Weight);
        assert!(expert.note.contains("top-8 of 256 experts"));
    }
}
