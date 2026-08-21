//! Gemma 3n — the generation with a per-layer embedding table.
//!
//! Two rows, one shape, and one number that used to be zero. `gemma3n`
//! reached the deployment through `("gemma3n", gemma3n_facts_from_hf)`
//! in `deployment_cuda::FACTS_ROWS` and an `impl PlannedFamily` that
//! overrode four methods and let the fifth — `tables()` — default. The
//! default says a family has no per-layer embeddings and no logit cap.
//! Gemma 3n has both, and it is the only generation in this catalog with
//! the first. See [`project`]'s module doc: the correction lives there
//! because it is a fact of the projection, and both rows inherit it.
//!
//! The other thing that used to be here was a `_ =>` arm. `instruct.rs`
//! matched `architectures[0]`, `Gemma3nForConditionalGeneration` was not
//! one of its arms, and the fallback was ChatML — whose `<|im_end|>` a
//! gemma vocabulary does not contain, so the turns it ended were turns
//! it was not having. The row below states `Gemma3nText` because that is
//! what a gemma-3n speaks.

pub mod forward;

/// The three projections a row of this generation makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;

/// The SHAPE — ungated, because a row is written in it and a row must
/// exist under every aspect.
pub mod spec;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use self::spec::{Gemma3nAltUpFacts, Gemma3nAttnFacts, Gemma3nFacts, window_schedule};

/// RMSNorm epsilon, shared by the whole generation.
///
/// Stated once rather than per row because every published config of
/// this generation carries the same `1e-6` — a generation-level
/// constant is an honest way to say that, where a copy per row invites
/// one of them to drift.
const NORM_EPS: f32 = 1e-6;

/// The family label a GUEST PROGRAM matches on.
///
/// `gemma3n`, which is what both published checkpoints state
/// (`model_type: "gemma3n"`) and what the boundary derives:
/// `architectures[0]` is `Gemma3nForConditionalGeneration`, and the
/// worker's stem heuristic strips `ForConditionalGeneration` — the whole
/// suffix, which is why `Gemma4ForConditionalGeneration` used to reach a
/// table row it did not belong in — before checking the rest against
/// [`crate::catalog::arches`].
///
/// The `n` is part of the family and not a size: a gemma-3n is not a
/// gemma-3 with extra tables, it is the generation with per-layer
/// embeddings, and a program that matched `gemma3` on it would select an
/// image front-end built for the other stack.
const ARCH: &str = "gemma3n";

/// The published context ceiling, shared by both rows.
///
/// One constant because both corpus configs —
/// `google--gemma-3n-E2B-it.json` and `google--gemma-3n-E4B-it.json` —
/// state `max_position_embeddings: 32768` under `text_config`, which is
/// where a `*ForConditionalGeneration` package puts its decoder's facts.
/// The E4B is five layers deeper and twice the MLP; the ceiling is not
/// one of the numbers that moved.
///
/// Shorter than gemma-3's 131 072 and gemma-4's, which is worth stating
/// because the three generations sit beside each other: 32 768 is this
/// generation's own word, read from its own configs, not a value carried
/// down from a neighbour.
const MAX_MODEL_LEN: u32 = 32_768;

/// One Gemma 3n checkpoint.
///
/// The two rope bases are the row's rather than the shape's, for the
/// reason gemma-3's schedule is: a base frequency is not something a
/// checkpoint's tensors can be measured against, and the shape is the
/// thing a manifest is a projection of.
pub struct Gemma3n {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: Gemma3nFacts,
    /// `rope_theta` — what the full-attention layers rotate on.
    pub rope_theta_global: f32,
    /// `rope_local_base_freq` — what the sliding layers rotate on. The
    /// old derivation parsed this field and then broadcast the global
    /// base over every layer anyway.
    pub rope_theta_local: f32,
}

/// E2B's window schedule: thirty layers, every fifth one full.
///
/// Expanded from the rule at compile time rather than transcribed from
/// `layer_types`, so the thirty entries and the sentence "four sliding
/// then one full" cannot come apart. `google/gemma-3n-E2B-it` lists
/// `full_attention` at 4, 9, 14, 19, 24 and 29; the test below checks
/// exactly that.
const E2B_WINDOWS: [i32; 30] = window_schedule(5, 512);

/// E4B's: the same rule over thirty-five layers, so the last layer is a
/// full one here and the schedule closes on a boundary.
const E4B_WINDOWS: [i32; 35] = window_schedule(5, 512);

/// The generation's rows.
///
/// `const`, so identity is in the binary. Both rows state every field,
/// including the ones they share, because a row is a MEASUREMENT of a
/// checkpoint and "these two agree about the residual width" is worth
/// reading off two lines rather than inferring from one.
pub const VARIANTS: &[Gemma3n] = &[
    // google/gemma-3n-E2B-it. The "E" is EFFECTIVE parameters: the
    // per-layer embedding table is most of the weight and is GATHERED
    // rather than multiplied, so a five-billion-parameter checkpoint
    // runs like a two-billion one. That is the fact `ple_dim` carries
    // into the deployment, and the fact the old default erased.
    Gemma3n {
        id: "gemma-3n-e2b",
        shape: Gemma3nFacts {
            vocab: 262_400,
            hidden: 2048,
            // Thirty layers at a uniform 8192. The config states this as
            // a LIST — `intermediate_size` is an array for this family —
            // and the loader refuses one whose length is not the layer
            // count, which is why this field is also where the layer
            // count comes from.
            per_layer_intermediate: &[8192; 30],
            laurel_rank: 64,
            ple_width: 256,
            // `activation_sparsity_pattern` is 0.95 for the first ten
            // layers and 0.0 for the rest, in BOTH published
            // checkpoints — a leading run, which is what
            // `is_sparse(l) = l < sparsity_layers` encodes.
            sparsity_layers: 10,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
            window_left: &E2B_WINDOWS,
        },
        rope_theta_global: 1_000_000.0,
        rope_theta_local: 10_000.0,
    },
    // google/gemma-3n-E4B-it. Five more layers and twice the MLP; the
    // residual, the heads and both embedding tables are E2B's exactly.
    // Two rows and not one, because those are the two numbers a
    // checkpoint gets measured against.
    Gemma3n {
        id: "gemma-3n-e4b",
        shape: Gemma3nFacts {
            vocab: 262_400,
            hidden: 2048,
            per_layer_intermediate: &[16384; 35],
            laurel_rank: 64,
            ple_width: 256,
            // Ten again, and not a fraction of the depth: the leading
            // run is the same length in a stack five layers deeper.
            sparsity_layers: 10,
            altup: Gemma3nAltUpFacts {
                num_streams: 4,
                active: 0,
            },
            attn: Gemma3nAttnFacts {
                heads: 8,
                kv_heads: 2,
                head_dim: 256,
            },
            window_left: &E4B_WINDOWS,
        },
        rope_theta_global: 1_000_000.0,
        rope_theta_local: 10_000.0,
    },
];

crate::rows_of!(Gemma3n);

impl Variant for Gemma3n {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// Dense, and `kv_shared_layers: 0` DESPITE the config's
    /// `num_kv_shared_layers: 10`.
    ///
    /// That field tells an authoring pass which k/v projections are dead
    /// weight it must not declare. Gemma-3n's traced text writes a k/v
    /// plane for every layer — see the note on `kv_source` in
    /// [`project::deployment`] — so none of them are dead, and declaring
    /// otherwise fails the load on a tensor the forward then asks for.
    /// The number in the config is real; honouring it is a change to the
    /// forward, not a change to this row.
    fn load_shape(&self) -> LoadShape {
        // Tied: gemma-3n ships no `lm_head`, which the manifest states
        // as an absence.
        LoadShape::dense(self.shape.layers(), self.shape.attn.head_dim, true)
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        // No host scalars and nothing sharded per rank: gemma-4 is the
        // generation that reads `layer_scalars`.
        let _ = load;
        let mut deployment = project::deployment(
            &self.shape,
            self.rope_theta_global,
            self.rope_theta_local,
            NORM_EPS,
        );
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // FALSE, and this is the row where that is a decision rather
            // than an observation. Both published packages DO ship
            // encoders — a USM conformer under `audio_config` and a
            // MobileNet-style vision block under `vision_config` — so
            // "does this checkpoint have a tower" and "does this row
            // advertise the encode entry point" are different questions
            // here.
            //
            // The second one is what the field asks, and it is answered
            // by what the entry point can serve. `Deployment::towers` is
            // empty on both rows because no gemma-3n tower is
            // transcribed anywhere in this crate, and `driver-cuda`'s
            // `pie_cuda_encode` binds gemma-4's kernels only —
            // `pie_k_vision_gemma4_audio_encode` and its vision twin —
            // refusing on an absent tower rather than defaulting to a
            // plausible one. A `true` here would advertise a capability
            // whose every call refuses, which is the bug this field
            // exists for pointed the other way: gemma-4 said `false`
            // while its towers worked.
            //
            // The day a gemma-3n tower is transcribed into
            // `Deployment::towers`, this becomes a derivation of that
            // field, as gemma-4's is.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The generic dense contract, which is what `HF_ROWS` gave this
    /// family under both `"gemma3n"` and `"gemma3n_text"`: a gemma-3n
    /// checkpoint already publishes the names the bind path reads, so
    /// the authoring pass has nothing to rename.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_dense(builder)
            }
            // The registry this replaced held NO MLX row for gemma-3n, and
            // the absence was a silence the caller read as "no
            // contract". Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "gemma-3n: no MLX authoring pass exists for this generation, \
                 so there is no name layout to author against",
            ),
        }
    }

    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        // METAL, refused by name. `llama_like_metal` is the only Metal
        // text in this build and it is not this model's — see
        // [`project::NO_METAL`] for what it states instead and why
        // reaching for it would trace a different model under this
        // row's id. The refusal is stated HERE, at the row, rather than
        // consulted from a list of architecture strings a driver keeps:
        // a list is a fourth place for the answer to live and a fourth
        // place for it to be wrong.
        if let crate::catalog::Backend::Metal(_) = load.backend {
            return Err(crate::deployment::Refusal::Unsupported(project::NO_METAL));
        }
        Ok(project::trace(
            &self.shape,
            class,
            load,
            NORM_EPS,
            self.rope_theta_global,
            self.rope_theta_local,
        ))
    }

    /// Gemma's `<start_of_turn>` template.
    ///
    /// Reached through `shared/` rather than through `gemma_3::chat`,
    /// because a generation module may not name a sibling — see
    /// `tests/sibling_isolation.rs`, whose own message prescribes
    /// `shared/` for exactly this. The template is one template; what
    /// two generations must not share is a MODULE.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::gemma_chat::Gemma3Instruct::for_variant(
            tokenizer,
            crate::shared::gemma_chat::Gemma3Variant::Gemma3nText,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::AttnOutput;

    fn row(id: &str) -> &'static Gemma3n {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The ids an operator types, and nothing else in the generation.
    #[test]
    fn the_generation_has_exactly_two_rows() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids, vec!["gemma-3n-e2b", "gemma-3n-e4b"]);
        assert_eq!(rows().len(), 2);
        assert_eq!(rows()[0].id(), "gemma-3n-e2b");
    }

    /// The layer count has ONE source, so a row cannot say thirty in one
    /// field and thirty-five in another — which is the failure mode a
    /// per-layer list invites.
    #[test]
    fn the_layer_count_has_exactly_one_source() {
        assert_eq!(row("gemma-3n-e2b").shape.layers(), 30);
        assert_eq!(row("gemma-3n-e4b").shape.layers(), 35);
        for v in VARIANTS {
            assert_eq!(
                v.shape.per_layer_intermediate.len(),
                v.shape.window_left.len()
            );
            assert_eq!(v.manifest().layers, v.shape.layers());
            assert_eq!(v.load_shape().layers, v.shape.layers());
            assert_eq!(
                v.deployment(Deployed::single()).expect("deploys").layers,
                v.shape.layers()
            );
        }
    }

    /// The window schedule each row carries is the one its published
    /// `layer_types` lists — expanded from the rule, not transcribed.
    #[test]
    fn the_window_schedule_is_the_published_one() {
        let full = |v: &Gemma3n| -> Vec<usize> {
            (0..v.shape.window_left.len())
                .filter(|&l| v.shape.window_left[l] == -1)
                .collect()
        };
        assert_eq!(full(row("gemma-3n-e2b")), vec![4, 9, 14, 19, 24, 29]);
        assert_eq!(full(row("gemma-3n-e4b")), vec![4, 9, 14, 19, 24, 29, 34]);
        for v in VARIANTS {
            assert!(v.shape.window_left.iter().all(|&w| w == -1 || w == 512));
        }
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` in
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and
    /// whether a tower is present — and the third is the one this
    /// generation makes interesting, so it has a test of its own below.
    /// Both rows are asserted; both advertise the same three answers.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("deploys")
                .advertised;
            assert_eq!(
                a.arch, "gemma3n",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 32_768,
                "{}: both corpus configs state it under `text_config`",
                v.id
            );
            assert_ne!(
                a.max_model_len, 0,
                "{}: a ceiling of 0 is 'the row does not say'",
                v.id
            );
        }
    }

    /// A CHECKPOINT WITH TOWERS THAT ADVERTISES NO ENCODE ENTRY, on
    /// purpose.
    ///
    /// Both published gemma-3n packages ship an audio encoder and a
    /// vision encoder. Neither is transcribed into `Deployment::towers`
    /// here, and `driver-cuda`'s encode entry binds gemma-4's kernels
    /// and refuses on an absent tower — so the honest answer to "does
    /// this row ship a tower the encode entry serves" is no. The
    /// assertion is written as the pair, because the pair is the
    /// invariant: no towers, no advertisement. If a gemma-3n tower is
    /// ever stated, this test fails and the field becomes a derivation
    /// of `towers`, as gemma-4's already is.
    #[test]
    fn no_row_advertises_an_encoder_the_encode_entry_cannot_serve() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("deploys");
            assert!(
                d.towers.audio.is_none(),
                "{}: no audio tower is stated here",
                v.id
            );
            assert!(
                d.towers.vision.is_none(),
                "{}: no vision tower is stated here",
                v.id
            );
            assert_eq!(
                d.advertised.media_encode,
                d.towers.audio.is_some() || d.towers.vision.is_some(),
                "{}: what a row advertises and what it carries must be one statement",
                v.id
            );
        }
    }

    /// The label is the stem the WORKER derives, and the whole suffix
    /// comes off: `read_hf_config_defaults` strips
    /// `ForConditionalGeneration` before checking the result against
    /// [`crate::catalog::arches`], and a partial strip is how
    /// `Gemma4ForConditionalGeneration` once reached a row it did not
    /// belong in.
    #[test]
    fn the_label_is_what_architectures_reduces_to() {
        let stem = "Gemma3nForConditionalGeneration"
            .to_lowercase()
            .strip_suffix("forconditionalgeneration")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert_ne!(ARCH, "gemma3", "the `n` is the family, not a size");
    }

    /// THE CLAIM THIS GENERATION IS NAMED FOR. `ple_dim` is nonzero on
    /// both rows, and the assertion goes through `Variant::deployment`
    /// — the method a driver calls — because the trait method that
    /// answered zero was the one nobody wrote.
    #[test]
    fn every_row_deploys_a_nonzero_ple_dim() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("deploys");
            assert_eq!(d.ple_dim, 256, "{}", v.id);
            assert_eq!(d.logit_softcap, 30.0, "{}", v.id);
        }
    }

    /// And the SSA-args exception is gemma-4's alone: these rows pin the
    /// driver's landing buffer, as every non-gemma-4 row does.
    #[test]
    fn the_attention_output_is_driver_pinned() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("deploys");
            assert_eq!(d.attn_output, AttnOutput::DriverPinned, "{}", v.id);
        }
    }

    /// Every trait method answers on every row, which is what having no
    /// default bodies buys: a generation cannot be half-stated.
    #[test]
    fn every_row_answers_every_question() {
        for v in VARIANTS {
            assert!(!v.id().is_empty());
            let m = v.manifest();
            assert!(m.layers > 0 && !m.tensors.is_empty());
            let ls = v.load_shape();
            assert_eq!(ls.head_dim, 256);
            assert_eq!(ls.n_experts, 0, "gemma-3n is dense");
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(
                ls.kv_shared_layers, 0,
                "stated where it is implemented, which is gemma-4"
            );
            assert!(ls.tied_embeddings);
            let d = v.deployment(Deployed::single()).expect("deploys");
            assert_eq!(d.attention.len() as u32, d.layers);
            assert_eq!(d.shape.hidden, 2048);
        }
    }

    /// The two rows are DISTINGUISHABLE by their manifests alone, or
    /// `catalog::identify` would call a real checkpoint ambiguous. They
    /// agree on every extent but the MLP's and the per-layer embedding
    /// table's, and those two are enough.
    #[test]
    fn the_two_rows_are_told_apart_by_their_tensors() {
        let ext = |v: &Gemma3n, n: &str| {
            v.manifest()
                .tensors
                .iter()
                .find(|t| t.name == n)
                .expect("stated")
                .extents
                .clone()
        };
        let (a, b) = (row("gemma-3n-e2b"), row("gemma-3n-e4b"));
        assert_ne!(
            ext(a, "layer.{}.mlp.gate_proj"),
            ext(b, "layer.{}.mlp.gate_proj")
        );
        assert_ne!(
            ext(a, "embed_tokens_per_layer"),
            ext(b, "embed_tokens_per_layer")
        );
        assert_eq!(ext(a, "embed_tokens"), ext(b, "embed_tokens"));
    }

    /// The template is gemma's, and the assertion is the bug itself: a
    /// gemma vocabulary has no `<|im_end|>`, so the row that fell
    /// through to ChatML sealed its turns with a token that does not
    /// exist and therefore never sealed them.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_gemmas_own_and_not_chatml() {
        let vocab: Vec<String> = [
            "<start_of_turn>",
            "<end_of_turn>",
            "<eos>",
            "<bos>",
            "user",
            "model",
            "\n",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let chat = v.chat(tok.clone());
            assert!(
                chat.seal().contains(&1),
                "{} does not seal with <end_of_turn>",
                v.id
            );
            assert!(
                chat.seal().contains(&2),
                "{} does not seal with <eos>",
                v.id
            );
            assert!(
                chat.cue().starts_with(&[0]),
                "{} cues with <start_of_turn>",
                v.id
            );
        }
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
        use model_ir::trace::FireClass;

        let bind = MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 64,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            fuse_residual_gemv: true,
            paged_multi_batch: true,
            qmm_multi_batch: true,
            add_bias: false,
            fused_qk_rope: false,
        };
        assert!(!VARIANTS.is_empty());
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let err = v
                    .trace(class, Deployed::metal(&bind))
                    .expect_err("this build has no Metal text for this generation");
                assert_eq!(
                    err,
                    Refusal::Unsupported(project::NO_METAL),
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

/// This generation's tensor names, in every vocabulary that spells them.
#[cfg(feature = "contract")]
pub mod import;
