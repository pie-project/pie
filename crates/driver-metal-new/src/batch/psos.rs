//! The decode PSO plan: which `(file, entrypoint)` pairs a configuration
//! compiles, and which kinds each serves.
//!
//! Many kinds share a pipeline — every dense projection is one matvec
//! entrypoint, five norms are one RMS kernel — so the C++ gathers each
//! distinct pair once and fans the compiled PSO out over its kinds. The
//! gathering half is pure: a function of the checkpoint's entrypoint names
//! and the family's feature set. That is this module; the metal half is
//! [`Compiler::compile_batch`](crate::Compiler), which already batches
//! file-based requests, deduplicates shared sources and archives the
//! result.
//!
//! ## Feature flags are load-bearing, and named
//!
//! Each flag's absence is as deliberate as its presence — `untied` exists
//! because claiming those kinds unconditionally handed the llama family a
//! valid gs_64/b_4 pipeline for a checkpoint that is neither ("not a load
//! failure, just wrong numbers, and it took the numerics test to say so");
//! `routed` exists because compiling mixture kernels for a dense checkpoint
//! lets an unrelated shader error fail a load that would have worked. The
//! C++ took them as named struct fields for the same reason its
//! `AffineFormat` has no default: two defaulted trailing ints once let call
//! sites pass the width and let the group fall back to 64, which "compiles,
//! binds, dispatches, and answers wrongly".
//!
//! ## The names are checked against the signature table
//!
//! The C++ builds format-dependent entrypoints through `pie::kernels::
//! entrypoint`, which refuses a name no shader instantiates — before that,
//! a mixture at an uninstantiated format reached the Metal compiler as a
//! string and failed there. The Rust plan takes the format-dependent names
//! as an [`EntryNames`] value (built by the loader from the checkpoint's
//! format), and the whole plan — fixed names included — is held against
//! `kernels-metal`'s signature table by a dev-dependency test: a name this
//! module can emit that no shader instantiates fails in CI, on any host.

use super::abi::Kernel;

/// The format-dependent entrypoints, spelled by the checkpoint's affine
/// format.
///
/// Built by the loader (`affine_qmv_fast_bfloat16_gs_64_b_4` and friends);
/// carried as a value so this plan stays a pure function of its inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryNames {
    /// `embed_gather_4bit_…` — the tied-embedding gather.
    pub embed_gather: String,
    /// `affine_qmv_fast_…` — the dense matvec.
    pub qmv_fast: String,
    /// `affine_qmv_fast_residual_…` — the residual-epilogue matvec.
    pub qmv_residual: String,
    /// `affine_qmv_routed_…` — the expert-indexed matvec.
    pub qmv_routed: String,
}

impl EntryNames {
    /// The names for a bf16 g64/b4 checkpoint — the shipped M=1 shape.
    #[must_use]
    pub fn bf16_g64_b4() -> Self {
        Self {
            embed_gather: "embed_gather_4bit_bfloat16_gs_64_b_4".to_owned(),
            qmv_fast: "affine_qmv_fast_bfloat16_gs_64_b_4".to_owned(),
            qmv_residual: "affine_qmv_fast_residual_bfloat16_gs_64_b_4".to_owned(),
            qmv_routed: "affine_qmv_routed_bfloat16_gs_64_b_4".to_owned(),
        }
    }
}

/// Which extensions a family's decode step asks for.
///
/// `DecodePsoFeatures`, field for field; see the module docs for why each
/// flag's absence is load-bearing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// The device-argmax substrate.
    pub argmax: bool,
    /// The residual-epilogue matvec variant.
    pub residual_qmv: bool,
    /// The GDN prologue/recurrent pair and the gated norm.
    pub gdn: bool,
    /// The gated-attention split and gate kernels.
    pub gated_attention: bool,
    /// The `d_256` decode attention.
    pub sdpa_d256: bool,
    /// The mixture: routing, sort/gather/combine, expert matvecs, the
    /// shared expert.
    pub routed: bool,
    /// The untied embedding/head pair — same entrypoints, different weight
    /// names, and only for families that do not compile their own.
    pub untied: bool,
    /// Build ONLY the two routing projections: for a checkpoint whose
    /// routing weights are in a second affine format, which nothing else in
    /// the model shares.
    pub routing_only: bool,
}

/// One pipeline to compile and the kinds it serves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PsoRequest {
    /// The `.metal` file, relative to the kernels directory.
    pub file: &'static str,
    /// The instantiated entrypoint.
    pub entry: String,
    /// The kinds this pipeline serves, fan-out order preserved.
    pub kinds: Vec<Kernel>,
}

/// The gathered plan: every distinct pipeline, plus where the residual
/// matvec sits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodePsoPlan {
    /// The compile requests, in gather order.
    pub requests: Vec<PsoRequest>,
    /// Index of the residual-epilogue matvec, which serves no kind: the
    /// encoder substitutes it per dispatch when the residual is fused.
    pub residual: Option<usize>,
}

impl DecodePsoPlan {
    /// The request that serves `kind`, if this configuration compiles one.
    ///
    /// Later requests win, which is how the GDN recurrent kernel
    /// deliberately overrides an earlier claim on the same kind.
    #[must_use]
    pub fn source_of(&self, kind: Kernel) -> Option<usize> {
        self.requests
            .iter()
            .rposition(|request| request.kinds.contains(&kind))
    }
}

/// Gather the M=1 decode plan for `names` and `features`.
#[must_use]
pub fn plan_decode_psos(names: &EntryNames, features: Features) -> DecodePsoPlan {
    let mut plan = DecodePsoPlan::default();
    let mut want = |file: &'static str, entry: &str, kinds: &[Kernel]| {
        plan.requests.push(PsoRequest {
            file,
            entry: entry.to_owned(),
            kinds: kinds.to_vec(),
        });
        plan.requests.len() - 1
    };

    want(
        "layout/embed_gather.metal",
        &names.embed_gather,
        &[Kernel::EmbedGather],
    );
    want(
        "norm/rms.metal",
        "rms_single_row_bfloat16",
        &[
            Kernel::Rms,
            Kernel::FfnRms,
            Kernel::QNorm,
            Kernel::KNorm,
            Kernel::FinalRms,
        ],
    );
    want(
        "quant/qmv.metal",
        &names.qmv_fast,
        &[
            Kernel::QmvIn,
            Kernel::QmvInZ,
            Kernel::QmvOut,
            Kernel::QmvQ,
            Kernel::QmvK,
            Kernel::QmvV,
            Kernel::QmvO,
            Kernel::QmvGate,
            Kernel::QmvUp,
            Kernel::QmvDown,
            Kernel::QmvLmHead,
            Kernel::GdnInA,
            Kernel::GdnInB,
        ],
    );
    want(
        "norm/residual_add.metal",
        "residual_add_bfloat16",
        &[Kernel::Residual, Kernel::LayerOut],
    );
    want(
        "rope/neox.metal",
        "neox_decode_bfloat16",
        &[Kernel::Rope, Kernel::RopeK],
    );
    want(
        "attn/kv_write.metal",
        "kv_append_bfloat16",
        &[Kernel::KvAppend],
    );
    want("mlp/gated.metal", "silu_mul_bfloat16", &[Kernel::SiluMul]);

    if features.residual_qmv {
        plan.residual = Some(want("quant/qmv.metal", &names.qmv_residual, &[]));
    }
    if features.gdn {
        want(
            "ssm/gdn_prep.metal",
            "gdn_prep_bfloat16",
            &[Kernel::GdnPrep],
        );
        // The slimmed recurrent kernel deliberately claims GdnCore after
        // anything earlier; `source_of` answers with the later claim.
        want(
            "ssm/gdn_prep.metal",
            "gdn_core_recurrent_bfloat16",
            &[Kernel::GdnCore],
        );
        want(
            "norm/gated_rms.metal",
            "gated_rms_bfloat16",
            &[Kernel::GatedRms],
        );
    }
    if features.gated_attention {
        want(
            "attn/gate.metal",
            "q_gate_split_bfloat16",
            &[Kernel::QSplit],
        );
        want("attn/gate.metal", "gate_bfloat16", &[Kernel::AttnGate]);
    }
    if features.sdpa_d256 {
        want(
            "attn/sdpa_vector.metal",
            "sdpa_vector_decode_bfloat16_d_256",
            &[Kernel::Sdpa],
        );
    }
    if features.untied {
        want(
            "layout/embed_gather.metal",
            &names.embed_gather,
            &[Kernel::EmbedUntied],
        );
        want("quant/qmv.metal", &names.qmv_fast, &[Kernel::LmHeadUntied]);
    }
    if features.routed {
        // SiluMul is deliberately not re-claimed: routed, the dense entry
        // already serves the shared expert's SwiGLU, and the sorted stack's
        // is the same kernel at another extent.
        want(
            "mlp/gated.metal",
            "silu_mul_bfloat16",
            &[Kernel::LlExpertSiluMul],
        );
        want("quant/qmv.metal", &names.qmv_fast, &[Kernel::LlRouter]);
        want(
            "moe/route.metal",
            "router_topk_bfloat16",
            &[Kernel::GoRouterTopK],
        );
        want("moe/route.metal", "route_sort", &[Kernel::LlMoeSort]);
        want("moe/route.metal", "route_gather", &[Kernel::LlMoeGather]);
        want("moe/route.metal", "combine_sorted", &[Kernel::LlMoeCombine]);
        want(
            "quant/qmv.metal",
            &names.qmv_routed,
            &[
                Kernel::LlExpertGate,
                Kernel::LlExpertUp,
                Kernel::LlExpertDown,
            ],
        );
        want(
            "quant/qmv.metal",
            &names.qmv_fast,
            &[
                Kernel::LlSharedGate,
                Kernel::LlSharedUp,
                Kernel::LlSharedDown,
                Kernel::LlSharedGateProj,
            ],
        );
        want(
            "moe/route.metal",
            "shared_expert_combine",
            &[Kernel::LlSharedCombine],
        );
    }
    if features.argmax {
        want(
            "sample/argmax.metal",
            "argmax_logits_bfloat16",
            &[Kernel::Argmax],
        );
    }
    if features.routing_only {
        // The second-format load: those two pipelines and nothing else.
        plan = DecodePsoPlan::default();
        plan.requests.push(PsoRequest {
            file: "quant/qmv.metal",
            entry: names.qmv_fast.clone(),
            kinds: vec![Kernel::LlRouter, Kernel::LlSharedGateProj],
        });
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds_served(plan: &DecodePsoPlan) -> Vec<Kernel> {
        Kernel::ALL
            .into_iter()
            .filter(|&kind| plan.source_of(kind).is_some())
            .collect()
    }

    #[test]
    fn the_base_plan_serves_the_m1_dense_surface_and_nothing_routed() {
        let plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), Features::default());
        let served = kinds_served(&plan);
        assert_eq!(served.len(), 25, "the always-on decode surface");
        assert!(served.contains(&Kernel::QmvLmHead));
        assert!(
            !served.contains(&Kernel::Sdpa),
            "d256 attention is a feature"
        );
        assert!(!served.contains(&Kernel::LlRouter));
        assert!(plan.residual.is_none());
        // Shared sources appear once per (file, entry): the batch compiler
        // dedups files, the plan dedups nothing else.
        assert_eq!(plan.requests.len(), 7);
    }

    #[test]
    fn features_add_their_kinds_and_the_full_set_is_disjoint_but_for_gdn() {
        let features = Features {
            argmax: true,
            residual_qmv: true,
            gdn: true,
            gated_attention: true,
            sdpa_d256: true,
            routed: true,
            untied: true,
            routing_only: false,
        };
        let plan = plan_decode_psos(&EntryNames::bf16_g64_b4(), features);
        let served = kinds_served(&plan);
        for kind in [
            Kernel::GdnPrep,
            Kernel::GdnCore,
            Kernel::GatedRms,
            Kernel::QSplit,
            Kernel::AttnGate,
            Kernel::Sdpa,
            Kernel::EmbedUntied,
            Kernel::LmHeadUntied,
            Kernel::LlExpertGate,
            Kernel::LlSharedCombine,
            Kernel::Argmax,
        ] {
            assert!(served.contains(&kind), "{kind:?} missing");
        }
        // Each kind is claimed by exactly one request.
        for kind in &served {
            let claims = plan
                .requests
                .iter()
                .filter(|request| request.kinds.contains(kind))
                .count();
            assert_eq!(claims, 1, "{kind:?} claimed {claims} times");
        }
        assert!(plan.residual.is_some());
        assert!(
            plan.requests[plan.residual.expect("set")].kinds.is_empty(),
            "the residual matvec serves no kind; the encoder substitutes it"
        );
    }

    #[test]
    fn later_claims_win_so_an_override_is_an_ordering_not_a_conflict() {
        // Synthesize the override the C++ comment describes: two requests
        // claiming GdnCore; source_of answers with the later.
        let mut plan = plan_decode_psos(
            &EntryNames::bf16_g64_b4(),
            Features {
                gdn: true,
                ..Features::default()
            },
        );
        let winner = plan.source_of(Kernel::GdnCore).expect("served");
        plan.requests[0].kinds.push(Kernel::GdnCore);
        assert_eq!(
            plan.source_of(Kernel::GdnCore),
            Some(winner),
            "the later claim still wins"
        );
    }

    #[test]
    fn routing_only_clears_everything_but_the_two_routing_projections() {
        let plan = plan_decode_psos(
            &EntryNames::bf16_g64_b4(),
            Features {
                gdn: true,
                routed: true,
                routing_only: true,
                ..Features::default()
            },
        );
        assert_eq!(plan.requests.len(), 1);
        assert_eq!(
            kinds_served(&plan),
            [Kernel::LlRouter, Kernel::LlSharedGateProj]
        );
        assert!(plan.residual.is_none());
    }

    /// The C++ `entrypoint()` refuses a name no shader instantiates, so an
    /// uninstantiated format fails at load with the formats that exist
    /// instead of inside the Metal compiler. This holds the same line for
    /// every name the plan can emit, against `kernels-metal`'s signature
    /// table, on any host.
    #[test]
    fn every_plannable_name_is_instantiated_by_a_shipped_shader() {
        let table: std::collections::HashSet<String> =
            kernels_metal::entrypoints().into_iter().collect();
        let full = Features {
            argmax: true,
            residual_qmv: true,
            gdn: true,
            gated_attention: true,
            sdpa_d256: true,
            routed: true,
            untied: true,
            routing_only: false,
        };
        let names = EntryNames::bf16_g64_b4();
        for plan in [
            plan_decode_psos(&names, full),
            plan_decode_psos(
                &names,
                Features {
                    routing_only: true,
                    ..full
                },
            ),
        ] {
            for request in &plan.requests {
                assert!(
                    table.contains(&request.entry),
                    "{} ({}) is not instantiated by any shipped shader",
                    request.entry,
                    request.file,
                );
                let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("crates/")
                    .join("kernels-metal/kernels")
                    .join(request.file);
                assert!(path.exists(), "{} does not exist", path.display());
            }
        }
    }
}
