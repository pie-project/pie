//! The Nemotron-H lineage: the MAMBA2 hybrid — selective-scan layers
//! interleaved with attention layers and with layers that are nothing
//! but an MLP.
//!
//! Three layer kinds, and a SCHEDULE rather than an interval. That is
//! the fact this generation contributes to the catalog: qwen3.5 and
//! kimi_k3 alternate on a period and can state their schedule as one
//! number, while NVIDIA's 47B carries `M---MM---M` in its tail — three
//! mixerless layers, then two scans back to back. [`spec`] holds the
//! transcription and the test that holds it to the published string.
//!
//! # What the row closed
//!
//! The deleted `nemotron_h_facts_from_hf` read the MLP width from
//! `moe_intermediate_size`, a key no published Nemotron-H states, so
//! every real checkpoint would have been traced with a dense MLP of
//! width ZERO and a router over no experts. It never fired that way
//! because the only stack this family was ever run on is the six-layer
//! synthetic fixture — which IS a mixture, and so took the one path that
//! worked. The rows below state the dense width the checkpoints ship,
//! and [`forward`] stops before the routed block when a stack has no
//! experts.
//!
//! Chat: ChatML that opens the assistant turn already inside a
//! `<think>`, which `instruct::create` reached through a registry row
//! pointing at qwen3's implementation. Stated per row here.

#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward — a mamba / attention / mlp hybrid.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
pub mod forward;

/// What a Nemotron-H checkpoint IS — ungated, because a row is written
/// in these words and a row must answer under every aspect.
pub mod spec;

/// What those numbers imply: a manifest, a deployment, a trace.
pub mod project;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::{Advertised, Deployment, Refusal};
use crate::manifest::Manifest;

use self::spec::NemotronHFacts;

/// The family label a GUEST PROGRAM matches on.
const ARCH: &str = "nemotron_h";

/// RMSNorm epsilon, shared by the whole generation.
///
/// `1e-5` in all three published configs — the qwen-2 answer, not the
/// llama/gemma `1e-6`. Stated once because all three agree, and stated
/// at all because no tensor extent carries it and the launch path used
/// to read it off a resident `HfConfig`.
const NORM_EPS: f32 = 1e-5;

/// The rotary base the attention layers rotate at.
///
/// **No Nemotron-H config states one.** `rope_theta` is absent from all
/// three, and the deleted normalizer's default — `10_000.0` — is what
/// the traced text has been rotating at since it was written. Stating it
/// here changes nothing and makes the number readable: a base that comes
/// from a default in a parser three crates away is a number nobody can
/// find.
const ROPE_THETA: f32 = 10_000.0;

/// One Nemotron-H checkpoint.
///
/// The shape is the stack; `max_model_len` is what the stack cannot say.
/// `rope_theta` and `norm_eps` are generation constants above rather
/// than fields, because all three checkpoints state the same eps and
/// none states a base at all — a per-row field would be three copies of
/// one measurement and an invitation for one of them to drift.
pub struct NemotronH {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers the stack is.
    pub shape: NemotronHFacts,
    /// `max_position_embeddings`, advertised and never fired on.
    pub max_model_len: u32,
}

/// The generation's rows.
///
/// Three, one per published base checkpoint. The instruction tunes
/// (`Nemotron-H-8B-Instruct-8K` and the reasoning variants) are the same
/// stacks — an instruction tune changes weights and a chat template, not
/// a schedule — so they match the base row, which is the arrangement
/// [`crate::gemma_4`]'s E4B pair documents at length.
pub const VARIANTS: &[NemotronH] = &[
    // nvidia/Nemotron-H-4B-Base-8K
    NemotronH {
        id: "nemotron-h-4b",
        shape: NemotronHFacts::nemotron_h_4b(),
        max_model_len: 8192,
    },
    // nvidia/Nemotron-H-8B-Base-8K
    NemotronH {
        id: "nemotron-h-8b",
        shape: NemotronHFacts::nemotron_h_8b(),
        max_model_len: 8192,
    },
    // nvidia/Nemotron-H-47B-Base-8K
    NemotronH {
        id: "nemotron-h-47b",
        shape: NemotronHFacts::nemotron_h_47b(),
        max_model_len: 8192,
    },
];

crate::rows_of!(NemotronH);

impl Variant for NemotronH {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// `mamba_groups` is why this method matters here.
    ///
    /// Every other field a load shape carries can be measured off a
    /// tensor. That one cannot: the checkpoint fuses B and C into one
    /// bank of `2 * n_groups * state_size` rows, so a loader holding the
    /// weights knows only the product and any factorization of it splits
    /// SOMETHING evenly. `contract::layer_mamba_tp` divides the bank by
    /// this number, and a wrong factor cuts a group in half without
    /// failing a shape check.
    fn load_shape(&self) -> LoadShape {
        LoadShape {
            layers: self.shape.layers(),
            head_dim: self.shape.attn.head_dim,
            n_experts: self.shape.moe.num_experts,
            mamba_groups: self.shape.mamba.n_groups,
            // No layer here attends through another's pages.
            kv_shared_layers: 0,
            tied_embeddings: self.shape.tied_embeddings,
        }
    }

    /// Servable: the paged KV store this build has, plus the recurrent
    /// slabs `RecurrentStateCache` allocates for the scan.
    ///
    /// # Errors
    ///
    /// Never, for this generation — both of its legs are traced.
    fn deployment(&self, load: Deployed<'_>) -> Result<Deployment, Refusal> {
        let _ = load;
        let mut deployment = project::deployment(
            &self.shape,
            ROPE_THETA,
            NORM_EPS,
            // ASKED of the table, not restated. This was a
            // `HEAD_DIM_KERNEL: u32 = 128` whose own doc called it "a
            // fact about the BINARY" -- which is exactly what
            // `ATTN_HEAD_DIMS` is, so a second copy of one of its
            // rows is a fourth place for the instantiation set to be
            // written down and go stale.
            crate::deployment::round_up_attn_head_dim(self.shape.attn.head_dim),
        );
        deployment.advertised = Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // Text only: no tower, so nothing for the encode entry point
            // to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// One author, and it reads the row's `mamba_groups` through
    /// [`Variant::load_shape`] rather than through a resident config.
    ///
    /// No `match` on naming: this generation has one spelling. The MLX
    /// table never held a `nemotron_h` row, and a row that dispatched on
    /// a naming it has no author for would be stating a capability that
    /// does not exist.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        self::contract::author_nemotron_h(builder)
    }

    /// # Errors
    ///
    /// Never: both legs of this hybrid are traced, and the dense MLP
    /// path is the one the guard in [`forward`] restored.
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, Refusal> {
        // METAL, refused by name. `llama_like_metal` is the only Metal
        // text in this build and it is not this model's — see
        // [`project::NO_METAL`] for what it states instead and why
        // reaching for it would trace a different model under this
        // row's id. The refusal is stated HERE, at the row, rather than
        // consulted from a list of architecture strings a driver keeps:
        // a list is a fourth place for the answer to live and a fourth
        // place for it to be wrong.
        if let crate::catalog::Backend::Metal(_) = load.backend {
            return Err(Refusal::Unsupported(project::NO_METAL));
        }
        Ok(project::trace(&self.shape, class))
    }

    /// ChatML that OPENS the assistant turn inside a `<think>`.
    ///
    /// The suffix is the whole difference from qwen's, and it is the
    /// reason this row cannot fall through to a default: a Nemotron-H
    /// served plain ChatML answers without ever entering the reasoning
    /// block it was tuned to write, which reads as a working model
    /// giving worse answers.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::NEMOTRON_CHATML,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::spec::NemotronLayerKind;
    use super::{ARCH, Deployed, NORM_EPS, NemotronH, ROPE_THETA, VARIANTS, Variant, rows};
    use crate::manifest::{Observed, Presence};

    fn row(id: &str) -> &'static NemotronH {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// What a checkpoint the row describes would publish.
    fn implied(v: &NemotronH) -> Observed {
        Observed::from_pairs(
            v.manifest()
                .tensors
                .iter()
                .filter(|t| t.presence != Presence::Absent)
                .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
        )
    }

    /// Ids are what a boundary carries, so they are held to the shape a
    /// boundary can carry: unique, non-empty, lowercase, hyphenated.
    #[test]
    fn every_row_has_an_id_a_boundary_can_carry() {
        let mut seen = std::collections::BTreeSet::new();
        for v in VARIANTS {
            assert!(!v.id.is_empty(), "a row with no name cannot be asked for");
            assert!(
                seen.insert(v.id),
                "'{}' names two rows, and a lookup would pick one",
                v.id
            );
            assert!(
                v.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "'{}' is not a name a URL or a CLI flag carries unescaped",
                v.id
            );
            assert!(
                v.id.starts_with("nemotron-h-"),
                "'{}' does not name its generation",
                v.id
            );
        }
    }

    /// `rows()` hands the catalog exactly the table, once.
    #[test]
    fn the_catalog_sees_one_entry_per_row() {
        assert_eq!(rows().len(), VARIANTS.len());
        let ids: Vec<&str> = rows().iter().map(|r| r.id()).collect();
        let table: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(
            ids, table,
            "the widening must preserve order, not just count"
        );
    }

    /// Every row deploys, and its deployment is the row's own numbers.
    #[test]
    fn every_row_deploys_at_the_depth_it_states() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("both legs are traced");
            assert_eq!(d.layers, v.shape.layers());
            assert_eq!(d.attention.len() as u32, v.shape.layers());
            assert_eq!(d.norm_eps, NORM_EPS);
            assert_eq!(
                d.shape.head_dim_kernel,
                crate::deployment::round_up_attn_head_dim(v.shape.attn.head_dim)
            );
            assert!(d.attention.iter().all(|a| a.rope_theta == ROPE_THETA));
            assert_eq!(d.advertised.arch, ARCH);
            assert_eq!(d.advertised.max_model_len, 8192);
            assert!(
                !d.advertised.media_encode,
                "no tower ships with this generation"
            );
            assert!(
                d.recurrent.is_some(),
                "a stack with scans carries recurrent state"
            );
        }
    }

    /// One family label over three rows of two different shapes.
    ///
    /// The label is a FAMILY and the id is a MODEL, and the difference
    /// is why nothing dispatches on the label: `nemotron_h` covers a
    /// 52-layer 4B, a 52-layer 8B and a 98-layer 47B whose Mamba state
    /// is twice as wide. A driver that picked a kernel from the string
    /// would pick one of the three.
    #[test]
    fn one_family_label_covers_three_distinct_rows() {
        let labels: std::collections::BTreeSet<&str> = VARIANTS
            .iter()
            .map(|v| {
                v.deployment(Deployed::single())
                    .expect("deploys")
                    .advertised
                    .arch
            })
            .collect();
        assert_eq!(labels.len(), 1, "one generation, one label");
        assert_eq!(labels.into_iter().next(), Some(ARCH));

        let ids: std::collections::BTreeSet<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), VARIANTS.len(), "three ids");
        assert!(
            ids.len() > 1,
            "the label cannot be the dispatch key it used to be"
        );

        // The ceiling IS shared, and the const says so. All three
        // published checkpoints are the 8K releases — the name carries
        // it — so this is a generation fact rather than a row field,
        // and this assertion is what would catch a 128K variant being
        // added as a fourth row without one.
        assert!(
            VARIANTS.iter().all(|v| v.max_model_len == 8192),
            "a Nemotron-H row that is not an 8K release needs its own ceiling stated"
        );
    }

    /// The slab list is the row's own schedule, per row.
    ///
    /// Three rows, two schedules — and the 47B's is the one no interval
    /// generates, so this is where a transcription slip in a 98-entry
    /// table would surface as a slab provisioned for a layer with no
    /// scan.
    #[test]
    fn each_row_provisions_slabs_for_its_own_scans() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("deploys");
            let r = d.recurrent.expect("a hybrid carries recurrent state");
            assert_eq!(r.linear_layers, v.shape.mamba_layers());
            assert!(
                r.linear_layers
                    .iter()
                    .all(|&l| v.shape.kind(l) == NemotronLayerKind::Mamba),
                "'{}' provisioned a slab for a layer that does not scan",
                v.id
            );
        }
        let biggest = row("nemotron-h-47b")
            .deployment(Deployed::single())
            .expect("deploys")
            .recurrent
            .expect("scans");
        assert_eq!(biggest.linear_layers.len(), 45);
    }

    /// The load shape states every field, and `mamba_groups` is the one
    /// no tensor can be measured for.
    #[test]
    fn the_load_shape_states_the_number_no_tensor_carries() {
        let s = row("nemotron-h-8b").load_shape();
        assert_eq!(s.layers, 52);
        assert_eq!(s.head_dim, 128, "the checkpoint's own head width, unpadded");
        assert_eq!(s.n_experts, 0, "every published Nemotron-H is dense");
        assert_eq!(s.mamba_groups, 8);
        assert_eq!(s.kv_shared_layers, 0);
        assert!(!s.tied_embeddings);

        for v in VARIANTS {
            let s = v.load_shape();
            assert_eq!(
                s.mamba_groups, 8,
                "'{}' states a group count of its own",
                v.id
            );
            assert_eq!(s.layers, v.shape.layers());
            assert_eq!(s.head_dim, v.shape.attn.head_dim);
            assert_eq!(s.n_experts, 0);
        }
    }

    /// Every row satisfies the manifest it states.
    #[test]
    fn every_row_matches_the_checkpoint_it_describes() {
        for v in VARIANTS {
            let m = v.manifest();
            assert!(m.layers > 0, "'{}' claims a stack of no layers", v.id);
            let seen = implied(v);
            assert!(
                m.check(&seen).is_ok(),
                "'{}': {}",
                v.id,
                m.check(&seen).unwrap_err()
            );
        }
    }

    /// No two rows claim one checkpoint.
    ///
    /// The 4B and the 8B run the IDENTICAL 52-layer schedule and differ
    /// only in width, which is exactly the pair a manifest has to keep
    /// apart — and it does, because every extent it states is derived
    /// from `hidden`.
    #[test]
    fn the_two_52_layer_rows_do_not_claim_one_another() {
        let four = row("nemotron-h-4b");
        let eight = row("nemotron-h-8b");
        assert_eq!(
            four.shape.layers(),
            eight.shape.layers(),
            "one schedule, two widths"
        );
        assert!(
            four.manifest().check(&implied(eight)).is_err(),
            "the 4B row claims an 8B checkpoint, and a stack of the wrong width would load"
        );
        assert!(
            eight.manifest().check(&implied(four)).is_err(),
            "and the 8B row claims a 4B one"
        );
    }

    /// Both fire classes trace, and the dense MLP is the leg that used
    /// to be unreachable.
    ///
    /// The deleted derivation read this generation's MLP width from
    /// `moe_intermediate_size`, which none of these checkpoints states,
    /// so every one of them would have traced a router over zero
    /// experts. Tracing all three rows here is what holds the repair:
    /// the dense path is now the one the published stacks take.
    #[test]
    fn every_row_traces_both_fire_classes_through_its_dense_mlp() {
        use model_compiler::trace::FireClass;
        for v in VARIANTS {
            assert!(!v.shape.is_mixture(), "'{}' should be dense", v.id);
            for class in [FireClass::Prefill, FireClass::Decode] {
                assert!(
                    v.trace(class, Deployed::single()).is_ok(),
                    "'{}' cannot trace {class:?}, and a row that deploys must fire",
                    v.id
                );
            }
        }
    }

    /// The template opens the assistant turn already inside a
    /// `<think>`.
    ///
    /// The one thing that separates this row's template from qwen's,
    /// and the reason the row states it rather than falling through:
    /// served plain ChatML, a Nemotron-H answers without entering the
    /// reasoning block it was tuned to write — which is not a crash,
    /// just a model that is quietly worse.
    #[cfg(feature = "chat")]
    #[test]
    fn the_assistant_turn_opens_inside_a_think_block() {
        use std::sync::Arc;
        // `<think>\n` is ONE entry, because the suffix is encoded whole:
        // `Tokenizer::from_vocab` is a raw-char table and a fixture that
        // spelled `<think>` and `\n` separately would encode the suffix
        // to nothing and pass this test by dropping the thing it checks.
        let vocab: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<think>\n",
            "\n",
            "assistant",
            "user",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        let tok = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        let cue = row("nemotron-h-8b").chat(tok.clone()).cue();
        assert_eq!(tok.decode(&cue, false), "<|im_start|>assistant\n<think>\n");
    }

    /// A METAL load is refused BY NAME rather than traced as a llama.
    ///
    /// The guard that replaces `driver-metal`'s `LLAMA_LIKE` table. That
    /// table answered "does this build serve you" from an architecture
    /// STRING reduced by `canonical()`, in a driver, before any text was
    /// traced — so it could say yes to a row this build cannot resolve
    /// (it listed `gpt_oss`, whose every publication either fails this
    /// crate's manifest or names tensors `driver-metal` has no handle
    /// for) and no to one whose text it models (it omitted `gemma3`). The row answers now, and what it answers with is a
    /// sentence naming what is missing.
    ///
    /// The comparison is against [`project::NO_METAL`] itself and not a
    /// paraphrase, so the sentence a caller is shown is the sentence
    /// this test pins — `csm`'s `NO_TRACE` sets the same shape.
    #[test]
    fn a_metal_load_is_refused_by_name_and_not_traced_as_a_llama() {
        use crate::catalog::{Backend, Deployed, MetalBinding};
        use crate::deployment::Refusal;
        use model_compiler::trace::FireClass;

        let bind = MetalBinding {
            quant_group: 64,
            quant_bits: 4,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
        };
        assert!(!VARIANTS.is_empty());
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let err = v
                    .trace(class, Deployed::metal(&bind))
                    .expect_err("this build has no Metal text for this generation");
                assert_eq!(
                    err,
                    Refusal::Unsupported(super::project::NO_METAL),
                    "`{}` refused a Metal load with a sentence that is not the \
                     one the row states",
                    v.id
                );
            }
        }
        // And the refusal is about the BACKEND and nothing else: the
        // same rows keep answering a CUDA load exactly as they did.
        for v in VARIANTS {
            assert!(
                v.trace(FireClass::Decode, Deployed::single()).is_ok(),
                "`{}` stopped serving CUDA",
                v.id
            );
        }
        // A `Backend::Cuda` is what `Deployed::single()` states, so the
        // arm above is reached by every existing caller unchanged.
        assert!(matches!(Deployed::single().backend, Backend::Cuda));
    }
}
