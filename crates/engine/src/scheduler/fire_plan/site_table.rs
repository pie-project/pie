//! The plan-derived site table: model-structural divergence sites, read
//! off the traced form.
//!
//! This is the Stage 5 "sites come from the traced form" step the
//! [`SITE_EXPERT_WEIGHTS`](super::SITE_EXPERT_WEIGHTS) doc promised:
//! [`derive_sites`] walks a `model_compiler::ForwardPlan` and emits the sites
//! the model's own structure declares — divergence that holds for every
//! member of every fire against this model, as opposed to the per-fire
//! attachment divergence [`plan_fire`](super::plan_fire) derives from
//! [`MemberFacts`](super::MemberFacts) (the two provenances; see the parent
//! module doc). [`plan_fire_with_model`](super::plan_fire_with_model)
//! merges both when a caller supplies a plan.
//!
//! # What the walk recognizes
//!
//! One pattern today: an op with a per-token weight selector —
//! `OpKind::Matmul { selector: Some(_) }`, the expert-indexed grouped GEMM
//! whose weight is a template (`layer.{l}.expert.{e}.gate_up`) resolved per
//! token by a `TopK` value. That is `Div::Weight` at token granularity
//! ([`expert_weights_site`](super::expert_weights_site)), and its
//! parameters are derived as follows:
//!
//! * `top_k` — from the `TopK { k }` op producing the selector. The
//!   `TraceBuilder::matmul_per_token` invariant (the selector must be a
//!   `dyn PerToken` value, which only `topk` creates) guarantees that
//!   producer exists in every in-vocabulary trace; a plan violating it is a
//!   tracer bug and panics here rather than mis-planning silently.
//! * `experts` — from the trailing `Dim::Const` of the `TopK` op's input
//!   (the router logits): the selector's indices index exactly that axis.
//!   Note what is NOT the source: the weight template itself. The string
//!   `layer.{l}.expert.{e}.gate_up` does not bound `{e}` — template
//!   cardinality is a *binding* fact the driver's weight resolver knows and
//!   the plan alone does not — so the router logits width is the only
//!   honest in-plan derivation. (For a hypothetical fragment taking the
//!   selector itself as a producer-less parameter, both `top_k` and
//!   `experts` would be underivable from the plan alone; no family traces
//!   such a fragment, and the builder invariant above rules it out.)
//!
//! Per-layer repetition dedups: an MoE model traces one `TopK` + two
//! selector-carrying matmuls per layer, all with the same `(experts, k)`,
//! and the site is a fact about the MODEL, so the table emits one site per
//! distinct parameterization, in first-appearance order — not one per
//! layer.
//!
//! # What is deliberately NOT a site: per-request recurrent state
//!
//! A GDN trace's `CausalConv1d`/`GatedDelta` ops address per-layer,
//! PER-REQUEST state slabs (`StateRef { store: RecurrentState }`), and it
//! is tempting to read that as another model-structural site. It is not,
//! and the distinction is the module's sharpest line: a divergence site is
//! a place where co-batched members want *different computation* — a
//! lowering must be chosen (prefix, per-lane spans, grouped GEMM, a
//! branch). RS-touching ops compute the SAME operation for every member;
//! each member merely addresses its own slab, exactly as every member
//! already addresses its own KV pages. What RS presence changes is
//! *admission*: the rs-buffer's capacity/aliasing hazard forces such fires
//! solo today (the scheduler's hand-maintained `touches_rs_buffer()`,
//! whose traced-form statement is `plan.ops.iter().any(|op|
//! op.kind.state_ref() ..)` — see `model_compiler::trace::OpKind::state_ref`).
//! That is a scheduling constraint, a fact for the admission rule
//! (`LaunchGrouping::accepts`), not a `Site` with a lowering; deriving it
//! from the plan belongs to the increment that wires plans to the
//! scheduler, on the admission side.
//!
//! # How the model sites reach `build_frame_submission` (landed via capabilities)
//!
//! The engine does NOT trace. It could seemingly do so at boot —
//! `bootstrap` holds the `ModelConfig` (an `arch_name`, driver configs)
//! and `model-compiler` is a direct rlib dep, so calling the family entry fn
//! is one line — but facts construction is where it breaks: the family fns
//! take facts the runtime does not have. `LlamaLikeFacts::fused_qkv`,
//! `Qwen35GdnFacts::fused_in_proj` etc. are BINDING facts — truths about
//! what the driver's deployment actually bound (contract joins/splits,
//! env-gated fusions like `PIE_QWEN35_FUSED_GDN_PROJ`) — and the rest come
//! from the checkpoint's config.json, which the runtime also does not
//! parse. The DRIVER is the party that traces, from exactly that evidence:
//! `declared_facts.cpp` builds facts from `HfConfig` plus the model's live
//! bindings, traces through the ABI, and structurally VALIDATES the traced
//! form against the real config and bound weight set. A runtime-side trace
//! built from guessed binding facts would be a second, unvalidated
//! derivation that can silently diverge from the plan the driver validated
//! — the exact class of drift this project exists to remove.
//!
//! So the wiring landed on the honest route this paragraph used to only
//! analyze: the driver reports its validated plan's SITE SUMMARY through
//! the capabilities handshake. The CUDA driver walks its declared plan with
//! the C++ mirror of [`derive_sites`] (`context.cpp`'s
//! `derive_expert_site_summary`; this module's tests pin the derivation)
//! and emits a `model_site_summary` capability row
//! (`::driver_api::ModelSiteSummary` — empty when `PIE_DECLARED_FORWARD`
//! is off, the validation refused, or the plan is dense). The summary rides
//! `DriverCapabilities` → worker `translate` → `bootstrap::DriverConfig` →
//! `DriverSpec`, where the driver's scheduler picks it up at spawn, maps it
//! through [`summary_sites`], and `build_frame_submission` merges the
//! result into every fire via
//! [`plan_fire_with_model`](super::plan_fire_with_model). An empty/absent
//! summary is exactly today's behavior. The sites remain INFORMATIONAL
//! this increment — nothing consumes a fire plan's site vec downstream yet
//! (same as v0), so a populated summary changes no submission bytes.

use model_compiler::{Dim, ForwardPlan, OpKind};

use super::{Site, expert_weights_site};

/// Map a driver-reported site summary (the capabilities handshake's
/// `model_site_summary` row) into the fire planner's vocabulary: one
/// [`expert_weights_site`](super::expert_weights_site) per reported entry,
/// in the driver's (first-appearance) order.
///
/// The summary states ONLY what [`derive_sites`] emits from a traced form
/// today — distinct `(experts, top_k)` parameterizations — so this map is
/// total; a summary entry the vocabulary cannot express does not exist.
pub(crate) fn summary_sites(summary: &::driver_api::ModelSiteSummary) -> Vec<Site> {
    summary
        .expert_sites
        .iter()
        .map(|site| expert_weights_site(site.experts, site.top_k))
        .collect()
}

/// Walk the traced form and emit the model-structural divergence sites its
/// structure declares (module doc: today, one
/// [`expert_weights_site`](super::expert_weights_site) per distinct
/// per-token selector parameterization; recurrent-state presence is
/// deliberately not one).
#[allow(dead_code)] // the production walk runs driver-side (context.cpp's C++ mirror — the driver holds the validated plan; module doc); this Rust original is the pinned reference the tests exercise.
pub(crate) fn derive_sites(plan: &ForwardPlan) -> Vec<Site> {
    // Distinct (experts, top_k) parameterizations, first-appearance order.
    // A Vec, not a set: plans have a handful of layers' worth of selector
    // ops and order stability matters more than asymptotics.
    let mut expert_params: Vec<(u32, u32)> = Vec::new();

    for op in &plan.ops {
        let OpKind::Matmul {
            selector: Some(selector),
            ..
        } = op.kind
        else {
            continue;
        };
        let producer = plan
            .ops
            .iter()
            .find(|candidate| candidate.outputs.contains(&selector))
            .unwrap_or_else(|| {
                panic!(
                    "{}: selector value {selector} has no producing op; \
                     matmul_per_token requires a dyn PerToken selector, which only topk creates",
                    plan.family
                )
            });
        let OpKind::TopK { k } = producer.kind else {
            panic!(
                "{}: selector value {selector} produced by {:?}, not TopK",
                plan.family, producer.kind
            );
        };
        let logits = *producer.inputs.first().unwrap_or_else(|| {
            panic!("{}: TopK op consumes the router logits", plan.family)
        });
        let experts = match plan.values[logits as usize].shape.0.last() {
            Some(&Dim::Const(experts)) => experts,
            other => panic!(
                "{}: router logits trailing dim must be a load-time constant \
                 (the expert count the selector indexes), got {other:?}",
                plan.family
            ),
        };
        if !expert_params.contains(&(experts, k)) {
            expert_params.push((experts, k));
        }
    }

    expert_params
        .into_iter()
        .map(|(experts, k)| expert_weights_site(experts, k))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::{DivClass, Granularity, Lowering, SITE_EXPERT_WEIGHTS};
    use super::*;
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts};
use model::qwen_3_5::forward::facts::{Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts};
    use model_compiler::{StateStore};

    /// The qwen3_5_moe MLP fragment (256 experts, top-8): the walk finds
    /// the selector-carrying matmuls, resolves k off the TopK op and the
    /// expert count off the router logits width, and dedups the gate_up /
    /// down pair into ONE model-level site of the pinned vocabulary shape.
    #[test]
    fn moe_fragment_derives_the_expert_site() {
        let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let sites = derive_sites(&plan);
        assert_eq!(sites.len(), 1, "gate_up + down share one selector group");
        let site = &sites[0];
        assert_eq!(site.name, SITE_EXPERT_WEIGHTS);
        assert_eq!(site.class, DivClass::Weight);
        assert_eq!(site.granularity, Granularity::Token);
        assert_eq!(site.lowering, Lowering::PerLane);
        assert!(site.note.contains("top-8 of 256 experts"), "{}", site.note);
    }

    /// Dense traces declare no model-structural divergence: nothing in a
    /// llama_like plan is `dyn`, whatever the configuration.
    #[test]
    fn llama_like_plans_derive_no_sites() {
        for facts in [
            LlamaLikeFacts::qwen3_0_6b(),
            LlamaLikeFacts::phi3_mini(),
            LlamaLikeFacts::mistral_7b_v03(),
            LlamaLikeFacts::olmo2_1b(),
        ] {
            let plan = model::shared::llama_like::forward::llama_like(&facts);
            assert!(
                derive_sites(&plan).is_empty(),
                "no model-structural sites in a dense trace ({})",
                plan.family
            );
        }
    }

    /// The dense 0.8b hybrid touches per-request recurrent state on every
    /// GDN layer — and derives NO site: RS presence is a scheduling
    /// constraint (the traced-form `touches_rs_buffer()`), not a divergence
    /// site (module doc). Pinned together so the distinction is a test, not
    /// only prose.
    #[test]
    fn dense_hybrid_touches_recurrent_state_but_derives_no_site() {
        let plan = model::qwen_3_5::forward::qwen3_5_hybrid(&Qwen35HybridFacts::qwen3_5_0_8b());
        assert!(
            plan.ops.iter().any(|op| op
                .kind
                .state_ref()
                .is_some_and(|s| s.store == StateStore::RecurrentState)),
            "the 0.8b hybrid has GDN layers"
        );
        assert!(derive_sites(&plan).is_empty());
    }

    /// The capabilities map: a driver-reported summary lands in the same
    /// vocabulary [`derive_sites`] emits — entry for entry, order kept,
    /// empty to empty (absent summary = today's behavior).
    #[test]
    fn summary_sites_maps_the_reported_entries() {
        assert!(summary_sites(&::driver_api::ModelSiteSummary::default()).is_empty());

        let summary = ::driver_api::ModelSiteSummary {
            expert_sites: vec![
                ::driver_api::ExpertSiteSummary {
                    experts: 256,
                    top_k: 8,
                },
                ::driver_api::ExpertSiteSummary {
                    experts: 64,
                    top_k: 4,
                },
            ],
        };
        let sites = summary_sites(&summary);
        assert_eq!(sites.len(), 2);
        for site in &sites {
            assert_eq!(site.name, SITE_EXPERT_WEIGHTS);
            assert_eq!(site.class, DivClass::Weight);
            assert_eq!(site.granularity, Granularity::Token);
            assert_eq!(site.lowering, Lowering::PerLane);
        }
        assert!(sites[0].note.contains("top-8 of 256 experts"));
        assert!(sites[1].note.contains("top-4 of 64 experts"));
    }

    /// Provenance agreement: the summary a driver would derive from the MoE
    /// traced form (the C++ mirror of [`derive_sites`]) maps through
    /// [`summary_sites`] to exactly what [`derive_sites`] emits from the
    /// same plan on this side — the handshake loses nothing.
    #[test]
    fn summary_of_a_moe_trace_round_trips_through_the_vocabulary() {
        let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
        let derived = derive_sites(&plan);
        let summary = ::driver_api::ModelSiteSummary {
            expert_sites: vec![::driver_api::ExpertSiteSummary {
                experts: 256,
                top_k: 8,
            }],
        };
        let mapped = summary_sites(&summary);
        assert_eq!(mapped.len(), derived.len());
        for (mapped, derived) in mapped.iter().zip(&derived) {
            assert_eq!(mapped.name, derived.name);
            assert_eq!(mapped.class, derived.class);
            assert_eq!(mapped.granularity, derived.granularity);
            assert_eq!(mapped.lowering, derived.lowering);
            assert_eq!(mapped.note, derived.note);
        }
    }

    /// A MoE-facts hybrid derives exactly one expert site: 24 layers of
    /// TopK + selector matmuls, all the same (experts, k), dedup to one
    /// model-level fact.
    #[test]
    fn moe_hybrid_derives_one_deduped_expert_site() {
        let mut facts = Qwen35HybridFacts::qwen3_5_0_8b();
        facts.mlp = Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
            hidden: facts.hidden(),
            ..Qwen35MoeMlpFacts::qwen3_5_35b_a3b()
        });
        let plan = model::qwen_3_5::forward::qwen3_5_hybrid(&facts);
        let sites = derive_sites(&plan);
        assert_eq!(sites.len(), 1, "per-layer repetition dedups to one site");
        assert_eq!(sites[0].name, SITE_EXPERT_WEIGHTS);
        assert!(sites[0].note.contains("top-8 of 256 experts"));
    }
}
