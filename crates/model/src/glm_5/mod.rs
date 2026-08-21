//! GLM-5 — MLA attention with a DSA lightning indexer, over a
//! dense-prefix MoE stack.
//!
//! One generation, one directory, and now one ROW. What this generation
//! was before is worth stating because it is the defect the merge fixes:
//! there were TWO directories. `glm5/` held the forward text and the
//! shape it traces; `glm_5/` held the load contract. Nothing named both,
//! so the two spellings drifted — `deployment_cuda::FACTS_ROWS` keyed
//! `"glm_moe_dsa"` at a `glm5_facts_from_hf` that re-parsed
//! `config.json` on every load, `contract::HF_ROWS` keyed the same
//! string at `glm_5::contract::author_glm5`, and `instruct::create` keyed
//! it a third time at ChatML. Three keys, three tables, two directories,
//! nothing holding any of them to each other.
//!
//! Read `VARIANTS` and then `impl Variant`. What this generation adds
//! over its shape is what a shape cannot state: which pass authors it and
//! which template speaks for it.

#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward — MLA plus a DSA indexer over a dense-prefix MoE
/// stack.
pub mod forward;

/// The SHAPE, ungated: a catalog row is written in these words.
pub mod spec;

/// The three projections a row makes — its manifest, its deployment,
/// its text.
pub mod project;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::Advertised;
use crate::manifest::Manifest;
use spec::Glm5Facts;

/// The coarse family label a GUEST PROGRAM matches on.
///
/// The one place a family string survives the merge, and it is no longer
/// a dispatch key: `"glm_moe_dsa"` reached `FACTS_ROWS`, `HF_ROWS` and
/// `instruct::create` as three separate keys into three separate tables,
/// and is now one label a row hands out. Stated rather than derived from
/// `architectures[0]`, whose derivation stripped `ForCausalLM` off a
/// lowercased name and is how a checkpoint reached a row it did not
/// belong in.
const ARCH: &str = "glm_moe_dsa";

/// One GLM-5 checkpoint.
///
/// A newtype over the shape for the exemplar's reason: `chat` and
/// `author` are answers the numbers cannot give, and this generation's
/// two are ChatML with the role markers in its stop set, and the MLA
/// authoring pass that dequantizes an FP8 `kv_b_proj` on the way in.
pub struct Glm5 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: Glm5Facts,
    /// Rope's base frequency. NOT in the shape, because the tracer takes
    /// it per-layer through `Deployment` rather than through the shape.
    pub rope_theta: f32,
    /// RMSNorm epsilon — a constant of the checkpoint that no tensor
    /// extent carries, so a row must state it.
    pub norm_eps: f32,
    /// The published context ceiling, `0` where this tree has no
    /// published config to read one from.
    pub max_model_len: u32,
    /// The head owns its own table.
    pub tied_embeddings: bool,
}

/// The generation's rows.
///
/// ONE, and the one is a MEASUREMENT: `zai-org/GLM-5-106B-A12B` is the
/// checkpoint `Glm5Facts::glm5_106b_a12b()` was transcribed from, and it
/// is the only GLM-5 geometry written down anywhere in this tree. A
/// second row here would be a model nobody measured — it would state
/// extents no checkpoint has, `identify` would match nothing against it,
/// and the first person to find out would be whoever tried to serve it.
pub const VARIANTS: &[Glm5] = &[
    // zai-org/GLM-5-106B-A12B. 106B total, 12B active: 128 experts at
    // 1408 with one shared, eight routed per token, over a 46-layer MLA
    // stack whose first three layers are dense.
    Glm5 {
        id: "glm-5-106b-a12b",
        shape: Glm5Facts {
            layers: 46,
            vocab: 151_552,
            hidden: 4096,
            dense_intermediate: 10_944,
            dense_layers: 3,
            attn: spec::Glm5MlaFacts {
                hidden: 4096,
                heads: 96,
                q_lora_rank: 1536,
                kv_lora_rank: 512,
                qk_nope_head_dim: 128,
                qk_rope_head_dim: 64,
                v_head_dim: 128,
                output_gate: false,
            },
            dsa: spec::Glm5DsaFacts {
                index_n_heads: 64,
                index_head_dim: 128,
                index_topk: 2048,
            },
            moe: spec::Glm5MoeFacts {
                hidden: 4096,
                num_experts: 128,
                top_k: 8,
                norm_topk_prob: true,
                routed_scaling: 2.5,
                moe_intermediate: 1408,
                shared_intermediate: 1408,
                aligned_block: 16,
            },
        },
        // Neither theta nor epsilon is a tensor extent, so neither can be
        // checked against a checkpoint and both have to be STATED. These
        // two are what a GLM-5 config that omits them reads as — the
        // normalizer's defaults, and `synthetic--glm-moe-dsa.json`, the
        // only glm-5 config committed here, omits both. That is the
        // honest reading available in this tree and the first thing to
        // correct against a published `config.json`.
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        // Not stated, because nothing in this tree states it: the only
        // GLM-5 config committed here is a synthetic whose 2048 is a
        // fixture's ceiling and not this checkpoint's. `0` is the field's
        // own word for "the row does not say", which a driver reports as
        // no advertised ceiling rather than as a two-thousand-token one.
        max_model_len: 0,
        tied_embeddings: false,
    },
];

crate::rows_of!(Glm5);

impl Glm5 {
    /// What a driver ADVERTISES about this row, as distinct from what it
    /// needs to fire it.
    ///
    /// These were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda`: a whole parsed `config.json` stayed alive for the
    /// life of a load so the driver could answer `model_type`,
    /// `max_position_embeddings` and "does this ship a tower" at
    /// capability time. They are facts about the MODEL, so the row
    /// answers them.
    ///
    /// Stated here rather than inline in [`Variant::deployment`] for a
    /// reason particular to this generation: that method REFUSES, always
    /// — no build in this tree provisions an MLA store — so a statement
    /// written inside it would be a capability answer no test could
    /// reach and no operator could ever read back. A row's identity does
    /// not depend on being servable, and neither does this.
    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // No tower. This generation is text-only, and the field it
            // replaces was hardwired `false` for every family — which is
            // how gemma-4's ported vision tower became unreachable
            // through the engine while four GPU tests that call the
            // encode entry point directly kept passing.
            media_encode: false,
        }
    }
}

impl Variant for Glm5 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    /// The head dim an AUTHORING pass needs, and for MLA that is the
    /// latent page row rather than a per-head width: the projections a
    /// tensor-parallel split cuts are `q_b_proj` and `kv_b_proj`, whose
    /// rows are heads of `qk_nope + qk_rope` and `qk_nope + v` — and
    /// both of those are multiples of this. A split that lands
    /// mid-latent produces a contract that compiles and a model that is
    /// wrong.
    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.attn.kv_a_width(),
            self.shape.moe.num_experts,
            self.tied_embeddings,
        )
    }

    /// # Errors
    ///
    /// [`crate::deployment::Refusal::Unsupported`] — this build
    /// provisions no MLA store, and the row says so at the door.
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        project::deployment(
            &self.shape,
            self.rope_theta,
            self.norm_eps,
            self.advertised(),
        )
    }

    /// # Errors
    ///
    /// The checkpoint contradicts a shape the author asserts, or the
    /// naming is one this generation has no pass for.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => contract::author_glm5(builder),
            // `MLX_ROWS` never held this generation, and the absence was
            // an `Ok(None)` the caller read as "no contract" — a silence
            // shaped exactly like a family nobody had written yet.
            // Stated as the refusal it always was: there is no published
            // MLX conversion of GLM-5, so there is no name layout to
            // author against.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "glm-5: no MLX authoring pass exists for this generation, so \
                 there is no name layout to author against",
            ),
        }
    }

    /// # Errors
    ///
    /// Whatever [`Self::deployment`] refuses, because it is asked first
    /// and the two questions are one question: "does this build serve
    /// this row". The text itself traces for both fire classes.
    ///
    /// It has to be asked here rather than assumed. This doc used to say
    /// the refusal happened "before anything reaches here" — but nothing
    /// sequenced the two calls, so a build with no MLA latent store
    /// refused at the door and handed out a fire anyway.
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
        // The plan this row WOULD fire. It cannot run in this build --
        // `deployment` refuses for a store nothing provisions -- and it is
        // written anyway, because the alternative is a row that stops being
        // able to serve the day one is. `tests/a_store_this_build_does_not
        // _provision.rs` holds the refusal so the day it changes is a test
        // failure naming this line rather than a silent revival.
        self.deployment(load)
            .map(|_| project::trace(&self.shape, class, self.norm_eps, self.rope_theta))
    }

    /// ChatML, with the two role markers in the stop set beside
    /// `<|im_end|>`. Stated by the row; the arm that used to state it
    /// keyed on the `model_type` string `"glm_moe_dsa"`, in a table that
    /// could disagree with the other two.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::GLM_CHATML,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Glm5 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. `Glm5Facts::glm5_106b_a12b()` was
    /// committed as a measurement of `zai-org/GLM-5-106B-A12B`; so is
    /// the row, and the row is now the only one a driver reads.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("glm-5-106b-a12b").shape, Glm5Facts::glm5_106b_a12b());
    }

    /// One row, one id, and the id is unique and non-empty. A duplicate
    /// here is two rows a checkpoint could match, which the catalog
    /// reports as ambiguous rather than resolving by order.
    #[test]
    fn the_rows_ids_are_unique_and_non_empty() {
        let mut seen = std::collections::BTreeSet::new();
        for v in VARIANTS {
            assert!(
                !v.id.is_empty(),
                "a row with no id is a row nothing can ask for"
            );
            assert!(seen.insert(v.id), "{} is stated twice", v.id);
        }
        assert_eq!(rows().len(), VARIANTS.len(), "one dyn entry per row");
        for (dynamic, row) in rows().iter().zip(VARIANTS) {
            assert_eq!(dynamic.id(), row.id, "the widening reordered the table");
        }
    }

    /// The id is what a boundary carries, so it says the MODEL and not
    /// an encoding of it: this generation's checkpoint ships bf16 and
    /// FP8 experts, and both are the same model.
    #[test]
    fn the_id_names_a_model_and_not_a_packing() {
        for v in VARIANTS {
            assert_eq!(v.id(), v.id);
            assert!(
                v.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{} is not lowercase-hyphenated",
                v.id,
            );
            for banned in ["fp8", "int4", "bf16", "awq", "w4a16"] {
                assert!(!v.id.contains(banned), "{} names a packing", v.id);
            }
        }
    }

    /// Every row answers every question, which is what having no default
    /// bodies buys — and for this generation one of the answers is a
    /// refusal, which is still an answer.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let m = v.manifest();
            assert_eq!(m.layers, v.shape.layers);
            assert!(!m.tensors.is_empty());
            assert!(
                matches!(
                    v.deployment(Deployed::single()),
                    Err(crate::deployment::Refusal::Unsupported(_))
                ),
                "no MLA store is built here",
            );
        }
    }

    /// The generation's family label is a CONST and not a per-row
    /// string, because every row of a generation shares it: `arch_name()`
    /// is a host function inferlets call, and a program asking "is this
    /// a GLM" wants the coarse answer.
    #[test]
    fn the_generation_states_one_family_label() {
        assert_eq!(
            ARCH, "glm_moe_dsa",
            "the string three tables used to key on"
        );
        assert!(!ARCH.is_empty());
    }

    /// The load shape, field by field. `head_dim` is the sharp one: an
    /// authoring pass splits `q_b_proj` and `kv_b_proj` by it, and the
    /// latent page row is what both are multiples of.
    #[test]
    fn the_load_shape_states_the_latent_page_row() {
        let ls = row("glm-5-106b-a12b").load_shape();
        assert_eq!(ls.layers, 46);
        assert_eq!(ls.head_dim, 576, "the latent plus the shared rope half");
        assert_eq!(ls.n_experts, 128);
        assert_eq!(ls.mamba_groups, 0, "no mixer in this stack");
        assert_eq!(ls.kv_shared_layers, 0, "every layer owns its pages");
        assert!(!ls.tied_embeddings, "this generation unties its head");
    }

    /// A row that LOADS and does not SERVE says so at the door. The
    /// manifest is complete, the load shape is complete — and the
    /// deployment refuses, which is the whole reason the trait's
    /// signature carries a `Result`.
    #[test]
    fn the_row_loads_and_refuses_to_serve_in_this_build() {
        let v = row("glm-5-106b-a12b");
        assert!(
            v.manifest()
                .tensors
                .iter()
                .any(|t| t.name.contains("kv_b_proj"))
        );
        assert!(
            matches!(
                v.deployment(Deployed::single()),
                Err(crate::deployment::Refusal::Unsupported(_))
            ),
            "no MLA store is built here, and that is not a property of GLM-5",
        );
    }

    /// The trace is still owed and still given: a build that cannot
    /// serve the row can still compile its text, which is what keeps the
    /// goldens honest.
    ///
    /// Asked of [`project::trace`] and not of [`Variant::trace`], and the
    /// distinction is the same one `plan` draws against `deployment`: the
    /// projection is TOTAL, a fact about the row, while the trait method
    /// is capability-gated and must refuse exactly when the door does.
    /// Asking the gated one here is what let this test and
    /// `catalog::tests::a_row_that_cannot_deploy_cannot_trace_either`
    /// state opposite things about the same call.
    #[test]
    fn every_row_traces_both_fire_classes() {
        use model_ir::trace::FireClass;
        for v in VARIANTS {
            for class in [FireClass::Prefill, FireClass::Decode] {
                let plan = project::trace(&v.shape, class);
                assert!(!plan.ops.is_empty(), "{class:?} traced nothing");
            }
        }
    }

    /// The authoring pass is chosen by NAMING, and the naming this
    /// generation has no pass for is refused rather than silently
    /// skipped. `MLX_ROWS` expressed that absence as a missing row and
    /// the caller as an `Ok(None)`, which reads the same as "authored
    /// nothing, successfully".
    #[cfg(feature = "contract")]
    #[test]
    fn an_mlx_checkpoint_is_refused_rather_than_authored_empty() {
        use crate::encoding::Encoding as StoredEncoding;
        use crate::shared::policy::{Naming, Policy};
        use model_loader::checkpoint::CheckpointMetadata;
        use model_loader::plan::StorageTarget;

        let v = row("glm-5-106b-a12b");
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let target = StorageTarget::default();
        let encoding = StoredEncoding::dense();

        let mlx = Policy {
            naming: Naming::Mlx,
            ..Policy::default()
        };
        let mut b = crate::shared::builder::Builder::new(
            &meta,
            v.id,
            v.load_shape(),
            &encoding,
            &target,
            &mlx,
        );
        assert_eq!(b.naming(), Naming::Mlx);
        assert!(
            v.author(&mut b).is_err(),
            "there is no MLX pass to run, and answering `Ok` would publish a \
             contract with nothing in it",
        );

        // And the naming that DOES have a pass reaches it: an empty
        // checkpoint authors an empty contract without refusing.
        let hf = Policy::default();
        let mut b = crate::shared::builder::Builder::new(
            &meta,
            v.id,
            v.load_shape(),
            &encoding,
            &target,
            &hf,
        );
        assert_eq!(b.naming(), Naming::Hf);
        assert!(
            v.author(&mut b).is_ok(),
            "the HF pass is the one this row has"
        );
    }

    /// The chat answer is GLM's own stop set, and the two role markers
    /// in it are the whole reason a row has to state this: the arm this
    /// replaces ended in `_ => QwenInstruct` with the plain two-token
    /// set, so a GLM that emitted `<|assistant|>` mid-stream was not
    /// stopped and kept generating the next turn on its own behalf.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question_with_glms_own_stop_set() {
        use tokenizer::Tokenizer;

        let words: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            // GLM's two extra stop markers, which are the whole
            // difference between this template and the plain one.
            "<|user|>",
            "<|assistant|>",
            "system",
            "user",
            "assistant",
            "\n",
            "<think>",
            "</think>",
            "<tool_response>",
            "</tool_response>",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&words));
        for v in VARIANTS {
            let chat = v.chat(tok.clone());
            assert_eq!(
                chat.seal().len(),
                4,
                "{}: the two role markers ride in the stop set beside <|im_end|>",
                v.id,
            );
            assert_eq!(
                tok.decode(&chat.user("Hi"), false),
                "<|im_start|>user\nHi<|im_end|>\n",
                "{}: a turn this checkpoint's vocabulary cannot close is a hang",
                v.id,
            );
        }
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` in
    /// `driver-cuda`: a driver kept a whole parsed `config.json` alive
    /// for the life of a load so it could answer `model_type`,
    /// `max_position_embeddings` and "does this ship a tower" at
    /// capability time.
    ///
    /// Read off `advertised()` rather than off a served `Deployment`,
    /// because this generation's `deployment` refuses in
    /// every build here — no MLA store is provisioned — and a capability
    /// statement that only exists once a build can serve the model is a
    /// statement no operator can read while deciding whether to try.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v.advertised();
            assert_eq!(
                a.arch, "glm_moe_dsa",
                "{}: the family label a guest program sees",
                v.id
            );
            assert!(
                !a.arch.is_empty(),
                "{}: an empty label matches no guest predicate",
                v.id
            );
            assert_eq!(
                a.max_model_len, 0,
                "{}: state a ceiling only from a published config — the only GLM-5 \
                 config in this tree is a parser fixture, and its ceiling is the fixture's",
                v.id,
            );
            assert!(
                !a.media_encode,
                "{}: no GLM-5 row ships a tower the encode entry serves, and a true here \
                 has the worker build an encode executor for a model with nothing to encode",
                v.id,
            );
        }
    }

    /// The label is a FAMILY and the id is a MODEL, and the difference
    /// is the reason the catalog exists.
    #[test]
    fn one_family_label_covers_every_row_and_the_ids_stay_distinct() {
        let labels: std::collections::BTreeSet<&str> =
            VARIANTS.iter().map(|v| v.advertised().arch).collect();
        assert_eq!(labels.len(), 1, "one family label");
        let ids: std::collections::BTreeSet<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), VARIANTS.len(), "every row is its own model");
        assert!(
            ids.len() >= labels.len(),
            "a label can name many rows and never fewer; the moment it names fewer, \
             something is dispatching on the label again",
        );
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
        // And the refusal is about the BACKEND and nothing else.
        //
        // Stated as "not the Metal sentence" rather than `is_ok()`,
        // because this family's CUDA answer is not `Ok` either: MLA has
        // no store in ANY build, `deployment` has always said so, and
        // `trace` now asks it first instead of firing a text for a row
        // the door already turned away. Asserting `is_ok()` here would
        // make this test a second, quieter claim that the MLA store is
        // built -- which is the coupling `Deployed::metal` exists to
        // avoid. What "unchanged" means is that no CUDA caller ever sees
        // a sentence about Metal.
        for v in VARIANTS {
            let cuda = v.trace(FireClass::Decode, Deployed::single());
            assert_ne!(
                cuda.as_ref().err(),
                Some(&Refusal::Unsupported(project::NO_METAL)),
                "`{}` answered a CUDA load with a sentence about Metal",
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
