//! The plan-derived site table: model-structural divergence sites, read
//! off the traced form.
//!
//! This is the Stage 5 "sites come from the traced form" step the
//! [`SITE_EXPERT_WEIGHTS`](super::SITE_EXPERT_WEIGHTS) doc promised:
//! [`derive_sites`] walks a `model_ir::ForwardPlan` and emits the sites
//! the model's own structure declares — divergence that holds for every
//! member of every fire against this model, as opposed to the per-fire
//! attachment divergence [`plan_fire`](super::plan_fire) derives from
//! [`MemberFacts`](super::MemberFacts) (the two provenances; see the parent
//! module doc). [`plan_fire_with_model`](super::plan_fire_with_model)
//! merges both when a caller supplies a plan.
//!
//! # What the walk recognizes
//!
//! One pattern today: a mixture-of-experts router, recognized by the fire
//! that chooses the experts — `moe::topk_softmax`. The weights it routes to
//! are templates (`layer.{l}.expert.{e}.gate_up`) resolved per token by the
//! indices that fire produces, which is `Div::Weight` at token granularity
//! ([`expert_weights_site`](super::expert_weights_site)). Its parameters are
//! derived as follows:
//!
//! * `top_k` — the trailing `Dim::Const` of the chosen-expert rows the top-k
//!   PRODUCES, which are `[Tokens, Const(k)]`. Not a param on the wire: the
//!   routine reads `k` off the width of the buffer it is handed.
//! * `experts` — the trailing `Dim::Const` of the router logits the top-k
//!   CONSUMES: the indices index exactly that axis. Note what is NOT the
//!   source: the weight template itself. The string
//!   `layer.{l}.expert.{e}.gate_up` does not bound `{e}` — template
//!   cardinality is a *binding* fact the driver's weight resolver knows and
//!   the plan alone does not — so the router logits width is the only honest
//!   in-plan derivation.
//!
//! # Why the top-k and not the grouped GEMM
//!
//! This walk used to anchor on the consumer: `OpKind::Matmul { selector:
//! Some(_) }`, the expert-indexed grouped GEMM, reading `(experts, k)` back
//! through the selector operand to its producing `TopK`. Two things retired
//! that reading, and both of them retired it SILENTLY, which is the part
//! worth remembering.
//!
//! First, the no-ask contract collapsed every tier-1 statement into
//! `OpKind::Launch { kernel, .. }`. There is no `selector` field left to
//! match on, and no `TopK` variant either.
//!
//! Second — and this is the one that would have kept biting after a
//! mechanical repair — the CUDA reading of a MoE block does not state a
//! grouped GEMM at all. `moe::flashinfer_cutlass_moe_bf16` fuses the gather,
//! both projections and the activation into a single fire. A walk anchored
//! on `moe::moe_grouped_gemm` therefore finds nothing whatsoever in the
//! reading the driver actually validates and plans against, and derives an
//! EMPTY site table for a model that is unambiguously an MoE. The aligned
//! decode leg that does state a grouped GEMM hands it a per-BLOCK expert id
//! from `moe::moe_align_decode`, one hop further from the router still.
//!
//! The top-k is the invariant across both legs: aligned or fused, every MoE
//! block routes through exactly one, and it is the only fire in either
//! reading where both numbers are legible without following operands.
//!
//! Per-layer repetition dedups: an MoE model traces one router per layer,
//! all with the same `(experts, k)`, and the site is a fact about the MODEL,
//! so the table emits one site per distinct parameterization, in
//! first-appearance order — not one per layer.
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
//! op.kind.state_ref() ..)` — see `model_ir::trace::OpKind::state_ref`).
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
//! That names the honest route, and the route is NOT yet wired. What the
//! paragraph below described in the past tense is a design, and every one of
//! its three links is currently open — written out because a plan in the
//! past tense is worse than no plan at all.
//!
//! The design: the driver reports its validated plan's SITE SUMMARY through
//! the capabilities handshake. The CUDA driver would walk its declared plan
//! with a C++ mirror of [`derive_sites`] and emit a `model_site_summary`
//! capability row (`::driver_api::ModelSiteSummary` — empty when
//! `PIE_DECLARED_FORWARD` is off, the validation refused, or the plan is
//! dense). The summary would ride `DriverCapabilities` → worker `translate`
//! → `bootstrap::DriverConfig` → `DriverSpec`, where the scheduler picks it
//! up at spawn, maps it through [`summary_sites`], and
//! `build_frame_submission` merges the result into every fire via
//! [`plan_fire_with_model`](super::plan_fire_with_model).
//!
//! Where it stands, measured:
//!
//! 1. PRODUCER. `driver-cuda`'s `serve/load.rs` sets
//!    `model_site_summary: ModelSiteSummary::default()`, and no
//!    `derive_expert_site_summary` exists in this tree. The C++ mirror is
//!    unwritten, so the row is always empty at the source.
//! 2. CARRIER. `worker::translate` does copy `caps.model_site_summary`
//!    into its config, but [`DriverSpec`](crate::driver::DriverSpec) has
//!    three fields and none of them is this one. The summary stops there.
//! 3. CONSUMER. [`summary_sites`] has no caller outside its own tests.
//!
//! None of this is a live defect: the doc's last sentence is still true —
//! sites are INFORMATIONAL this increment, nothing downstream consumes a
//! fire plan's site vec, and an absent summary is exactly today's behavior.
//! The tests below pin the derivation so the mapping stays correct until
//! the three links close, and the empty path is the one that runs.

use model_ir::{Dim, ForwardPlan, OpKind};

use super::{Site, expert_weights_site};

/// Map a driver-reported site summary (the capabilities handshake's
/// `model_site_summary` row) into the fire planner's vocabulary: one
/// [`expert_weights_site`](super::expert_weights_site) per reported entry,
/// in the driver's (first-appearance) order.
///
/// The summary states ONLY what [`derive_sites`] emits from a traced form
/// today — distinct `(experts, top_k)` parameterizations — so this map is
/// total; a summary entry the vocabulary cannot express does not exist.
#[allow(
    dead_code,
    reason = "link 3 of the three open links the module doc enumerates: the \
              driver-reported summary is empty at the source and dropped at \
              `DriverSpec`, so nothing calls this yet. Kept because the tests \
              below pin the derivation against `derive_sites`, which is what \
              keeps the mapping correct until the spawn path consumes it"
)]
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
        // THE ROUTER TOP-K, by its symbol, and nothing downstream of it.
        //
        // This walk used to start at the expert-indexed grouped GEMM and read
        // its selector operand, because the semantic `OpKind::Matmul {
        // selector }` named one. Two things retired that: the no-ask contract
        // turned every tier-1 statement into `OpKind::Launch`, so there is no
        // `selector` field left to read; and the CUDA reading of a MoE block
        // no longer states a grouped GEMM at all -- `moe::flashinfer_cutlass_moe_bf16`
        // fuses the gather, both projections and the activation into one fire.
        // A walk anchored on the GEMM therefore found NOTHING in the reading
        // the driver actually validates, which is the only one that matters.
        //
        // The top-k is the invariant. Every MoE leg, aligned or fused, routes
        // through exactly one `moe::topk_softmax`, and that fire is where BOTH
        // numbers the site needs are legible: the expert count is the width of
        // the logits it consumes, and `k` is the width of the indices it
        // produces. Anchoring here also gives the dedup for free -- a gate_up
        // and a down that share a router share their one top-k.
        let OpKind::Launch { kernel, .. } = &op.kind else {
            continue;
        };
        if kernel != "moe::topk_softmax" {
            continue;
        }
        // BOTH NUMBERS OFF SHAPES, not off params. `moe::topk_softmax`
        // carries no `k` on the wire -- its rows are `[Tokens, Const(k)]` and
        // the routine reads the width -- and the expert count is the router
        // logits' trailing dim, which is the axis the indices index.
        let chosen = *op.outputs.first().unwrap_or_else(|| {
            panic!(
                "{}: the router top-k states its chosen experts",
                plan.family
            )
        });
        let k = match plan.values[chosen as usize].shape.0.last() {
            Some(&Dim::Const(k)) => k,
            other => panic!(
                "{}: the chosen-expert rows must be `[.., Const(top_k)]`, got {other:?}",
                plan.family
            ),
        };
        let logits = *op.inputs.first().unwrap_or_else(|| {
            panic!(
                "{}: the router top-k consumes the router logits",
                plan.family
            )
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

    /// The scalars the family texts used to read off their fact structs.
    /// A site derivation walks ops and weights; it never reads an epsilon or
    /// a rope base, so any well-formed value states the same text.
    const EPS: f32 = 1e-6;
    const THETA: f32 = 1_000_000.0;
    use super::super::{DivClass, Granularity, Lowering, SITE_EXPERT_WEIGHTS};
    use super::*;
    use model::qwen_3_5::forward::facts::{Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts};
    use model::shared::llama_like::forward::facts::LlamaLikeFacts;
    use model_ir::StateStore;

    /// The qwen3_5_moe MLP fragment (256 experts, top-8): the walk finds
    /// the router top-k, resolves k off the chosen-expert rows and the
    /// expert count off the router logits width, and dedups the gate_up /
    /// down pair into ONE model-level site of the pinned vocabulary shape.
    #[test]
    fn moe_fragment_derives_the_expert_site() {
        // THE CUDA READING, not the semantic one. `derive_sites` finds the
        // expert-indexed grouped GEMM by its SYMBOL now
        // (`moe::topk_softmax`), because `OpKind::Matmul { selector }`
        // retired with the no-ask contract -- and a semantic text states
        // `canon::matmul_select`, which is the role, not the statement. The
        // production walk runs driver-side over the validated plan, which is
        // always a backend reading, so this is the plan it sees.
        let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block_cuda(
            &Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
            &model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            EPS,
        );
        let sites = derive_sites(&plan);
        assert_eq!(sites.len(), 1, "gate_up + down share one router");
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
            let plan = model::shared::llama_like::forward::llama_like(&facts, EPS, THETA);
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
        let plan = model::qwen_3_5::forward::qwen3_5_hybrid(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            EPS,
            THETA,
        );
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
        // THE CUDA READING, not the semantic one. `derive_sites` finds the
        // expert-indexed grouped GEMM by its SYMBOL now
        // (`moe::topk_softmax`), because `OpKind::Matmul { selector }`
        // retired with the no-ask contract -- and a semantic text states
        // `canon::matmul_select`, which is the role, not the statement. The
        // production walk runs driver-side over the validated plan, which is
        // always a backend reading, so this is the plan it sees.
        let plan = model::qwen_3_5::forward::qwen3_5_moe_mlp_block_cuda(
            &Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
            &model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            EPS,
        );
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
        // The CUDA reading, for the reason `moe_fragment_derives_the_expert_site` states.
        let plan = model::qwen_3_5::forward::qwen3_5_hybrid_cuda::<
            model::qwen_3_5::forward::ShippedW1,
            model::qwen_3_5::forward::ShippedW2,
            model::qwen_3_5::forward::ShippedA,
            model::qwen_3_5::forward::ShippedKv,
        >(
            &facts,
            &model::qwen_3_5::forward::facts::Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            model_ir::FireClass::Decode,
            EPS,
            THETA,
        );
        let sites = derive_sites(&plan);
        assert_eq!(sites.len(), 1, "per-layer repetition dedups to one site");
        assert_eq!(sites[0].name, SITE_EXPERT_WEIGHTS);
        assert!(sites[0].note.contains("top-8 of 256 experts"));
    }
}
