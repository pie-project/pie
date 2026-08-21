//! Gemma 2 — the generation whose per-layer table was a rule.
//!
//! Three rows and one shape. What used to be here was two entries in two
//! tables keyed on the string `"gemma2"` — one in `contract::HF_ROWS`
//! saying who authors the load, one in `deployment_cuda::FACTS_ROWS`
//! saying what to deploy — plus an arm of `instruct::create` keyed on an
//! architecture name saying which template speaks. Nothing held the
//! three to each other.
//!
//! Read [`VARIANTS`] and then `impl Variant`. The numbers are the
//! published `config.json`s (2B, 9B, 27B); the 9B row is checked against
//! the committed forward fixture, because a row and a fixture that
//! disagree are two measurements of one checkpoint.

#[cfg(feature = "chat")]
pub mod chat;

/// The declared forward — plain attention with a norm PAIR per block, an
/// alternating sliding window, and softcaps.
pub mod forward;

/// The SHAPE: the numbers a gemma-2 checkpoint has.
///
/// Ungated. A catalog row is written in these words, and a row must
/// exist under every aspect.
pub mod spec;

/// The three projections a row of this generation makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use self::spec::{Gemma2AttnFacts, Gemma2Facts};

/// RMSNorm epsilon, shared by the whole generation.
///
/// Stated once rather than per row because every published gemma-2
/// config carries the same `1e-6` — a generation-level constant is an
/// honest way to say that, where three copies of the same literal
/// invite one of them to drift.
const NORM_EPS: f32 = 1e-6;

/// The family label a GUEST PROGRAM matches on.
///
/// `gemma2`, which is both what the checkpoints state (`model_type:
/// "gemma2"`) and what the boundary derives: `architectures[0]` is
/// `Gemma2ForCausalLM`, and the worker's stem heuristic strips the task
/// suffix off a lowercased name and then CHECKS the result against
/// [`crate::catalog::arches`] — so the label a row hands out and the
/// label a checkpoint reduces to have to be the same string or a real
/// gemma-2 is refused at the boundary.
///
/// Deliberately coarser than an id: `gemma2` names three checkpoints of
/// three shapes, and a program asking "is this a gemma" wants the coarse
/// answer. It is NOT a dispatch key — that is what the string `"gemma2"`
/// was in three tables, and nothing in this crate matches on it now.
const ARCH: &str = "gemma2";

/// The published context ceiling, shared by the whole generation.
///
/// One constant because all three published gemma-2 configs state
/// `max_position_embeddings: 8192` — the 2B, the 9B and the 27B alike.
/// It is the shortest ceiling in this catalog and it is not a mistake:
/// gemma-2 predates the long-context releases, and gemma-3 next door
/// states 32 768 and 131 072 on the same family's rows. The `2048` in
/// `corpus/synthetic--gemma2.json` is a PARSER fixture, not a
/// checkpoint, and is not what this states.
///
/// A training-time fact rather than a deployment one: nothing in a fire
/// reads it, and a driver serving a shorter context is serving
/// correctly.
const MAX_MODEL_LEN: u32 = 8_192;

/// One Gemma 2 checkpoint.
///
/// A newtype over the shape rather than an `impl Variant for
/// Gemma2Facts`, for the exemplar's reason: `chat` and `author` are
/// questions a set of numbers cannot answer, and the shape is shared
/// with a tracer that must not learn what a chat template is.
pub struct Gemma2 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: Gemma2Facts,
    /// Rope's base frequency. NOT in the shape, because the tracer takes
    /// it per layer through `Deployment` rather than through the facts —
    /// and every published gemma-2 states the same 10 000.
    pub rope_theta: f32,
}

/// The generation's rows.
///
/// `const`, which is the whole architecture: identity is in the binary,
/// so the three questions have one answer and the answer is linked.
/// Every field is stated even when it is false, for the reason the
/// fixtures give — a row is a MEASUREMENT of a real checkpoint, and
/// "this one has no q-norm" is part of the measurement.
pub const VARIANTS: &[Gemma2] = &[
    // google/gemma-2-2b-it. `head_dim` 256 while `hidden / heads` is
    // 288: the config states it, and a derivation that divided would
    // size every projection wrong.
    Gemma2 {
        id: "gemma-2-2b",
        shape: Gemma2Facts {
            layers: 26,
            vocab: 256_000,
            hidden: 2304,
            intermediate: 9216,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 8,
                kv_heads: 4,
                head_dim: 256,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },
    // google/gemma-2-9b-it — the geometry the committed forward fixture
    // measures.
    Gemma2 {
        id: "gemma-2-9b",
        shape: Gemma2Facts {
            layers: 42,
            vocab: 256_000,
            hidden: 3584,
            intermediate: 14336,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 16,
                kv_heads: 8,
                head_dim: 256,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },
    // google/gemma-2-27b-it — the only gemma-2 whose heads are 128 wide,
    // and the only one whose `query_pre_attn_scalar` (144) is neither
    // the head dim nor the hidden size over the head count.
    Gemma2 {
        id: "gemma-2-27b",
        shape: Gemma2Facts {
            layers: 46,
            vocab: 256_000,
            hidden: 4608,
            intermediate: 36864,
            tied_embeddings: true,
            final_logit_softcap: true,
            sliding_window: 4096,
            full_attn_interval: 2,
            attn: Gemma2AttnFacts {
                heads: 32,
                kv_heads: 16,
                head_dim: 128,
                attn_logit_softcap: true,
            },
        },
        rope_theta: 10_000.0,
    },
];

crate::rows_of!(Gemma2);

impl Variant for Gemma2 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The STATED head dim, never `hidden / heads`: gemma-2-2b is 2304
    /// over 8 heads and its heads are 256 wide, not 288, so a
    /// tensor-parallel row split that divided would cut a head in half.
    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(
            self.shape.layers,
            self.shape.attn.head_dim,
            self.shape.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        // Gemma-2 reads nothing from the load: it ships no host scalars
        // and its text binds nothing per rank.
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.rope_theta, NORM_EPS);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // No tower. Gemma-2 is text-only — `PaliGemma` is the vision
            // package of this era and a different checkpoint — so
            // `Deployment::towers` stays empty and there is nothing for
            // the driver's encode entry to serve. Gemma-4 is the one
            // generation in this catalog that answers otherwise, and it
            // derives the answer from the towers it carries.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The generic dense contract. Gemma-2's checkpoint already uses the
    /// names the bind path reads, so there is no gemma-2 authoring pass
    /// and the row says whose it borrows — the same N:1 the `HF_ROWS`
    /// column expressed, spelled as a call.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_dense(builder)
            }
            // The registry this replaced held NO MLX row for gemma-2, and
            // the absence was a silence the caller read as "no
            // contract". Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "gemma-2: no MLX authoring pass exists for this generation, \
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
            self.rope_theta,
        ))
    }

    /// Gemma's own `<start_of_turn>` template, stated rather than fallen
    /// through to. The old registry ended in `_ => QwenInstruct`, and a
    /// gemma that seals a turn with `<|im_end|>` emits a token its
    /// vocabulary does not contain.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::gemma_chat::Gemma3Instruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{AttnOutput, NormPlacement};

    fn row(id: &str) -> &'static Gemma2 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement of
    /// `google/gemma-2-9b-it`, so if they disagree one of them is wrong.
    #[test]
    fn the_9b_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("gemma-2-9b").shape, Gemma2Facts::gemma_2_9b());
    }

    /// Every row answers every question, which is what having no default
    /// bodies buys.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("gemma-2 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.vocab, v.shape.vocab);
            assert_eq!(d.norm, NormPlacement::Pre);
            assert_eq!(
                d.attn_output,
                AttnOutput::DriverPinned,
                "gemma-4 is the exception"
            );
            assert_eq!(d.ple_dim, 0, "per-layer embeddings belong to 3n and 4");

            let m = v.manifest();
            assert_eq!(m.layers, v.shape.layers);
            assert!(
                m.tensors
                    .iter()
                    .any(|t| t.name == "layer.{}.pre_feedforward_layernorm")
            );

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, v.shape.attn.head_dim, "the TRUE head dim");
            assert!(ls.tied_embeddings, "every gemma-2 ties");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0, "KV sharing is gemma-4's");
        }
    }

    /// The cap on the FINAL logits reaches the deployment on every row.
    /// It is the launch at the end of the pass, and a driver reads it
    /// from here rather than from a config it kept resident.
    #[test]
    fn the_final_logit_softcap_reaches_the_deployment() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            assert_eq!(d.logit_softcap, 30.0, "{}", v.id);
        }
    }

    /// And the cap on the ATTENTION scores, which is a DIFFERENT number
    /// in a different place, reaches it too.
    ///
    /// Two caps is the whole point: `attn_logit_softcapping: 50.0` runs
    /// inside the attention kernel over the scores,
    /// `final_logit_softcapping: 30.0` runs once at the readout. The
    /// shape had measured both from the start and only one of them had
    /// anywhere to go — `AttnCtx::logits_soft_cap` was the literal
    /// `0.0`, so a capped gemma-2 and an uncapped one attended
    /// identically. Asserting they DIFFER is the point: one field
    /// carrying both, or a projection that copied the readout's number
    /// across, would pass a test that only checked for non-zero.
    #[test]
    fn the_attention_softcap_reaches_it_too_and_is_not_the_readout_s() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            assert_eq!(d.attn_logit_softcap, 50.0, "{}", v.id);
            assert_ne!(d.attn_logit_softcap, d.logit_softcap, "{}", v.id);
        }
    }

    /// The alternation is a rule on every row, and it produces the same
    /// per-layer list the shape used to carry as a `Vec`.
    #[test]
    fn every_row_alternates_local_and_global_attention() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            let windows: Vec<i32> = d.attention.iter().map(|a| a.window).collect();
            let longhand: Vec<i32> = (0..v.shape.layers)
                .map(|l| if l % 2 == 1 { -1 } else { 4096 })
                .collect();
            assert_eq!(windows, longhand, "{}", v.id);
        }
    }

    /// Three rows, three checkpoints nothing can confuse: the manifests
    /// differ in extents, which is how `identify` tells them apart
    /// without reading a `model_type`. Two rows no checkpoint can
    /// distinguish are one row, and `Unmatched::Ambiguous` is what the
    /// catalog says about a table that has both.
    #[test]
    fn no_two_rows_describe_the_same_checkpoint() {
        let fingerprint = |v: &Gemma2| -> Vec<String> {
            v.manifest()
                .tensors
                .iter()
                .map(|t| format!("{}:{:?}:{:?}", t.name, t.extents, t.presence))
                .collect()
        };
        for (i, a) in VARIANTS.iter().enumerate() {
            for b in &VARIANTS[i + 1..] {
                assert_ne!(fingerprint(a), fingerprint(b), "{} and {}", a.id, b.id);
            }
        }
    }

    /// The row's template is gemma's, and this is the test the old
    /// registry could not have written: `instruct::create` ended in
    /// `_ => QwenInstruct`, so "which template does this model get" had
    /// an answer for strings nothing enumerated. A gemma sealed with
    /// ChatML's `<|im_end|>` — a token gemma's vocabulary does not
    /// contain — and generated fluently past the end of every turn.
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

    /// The ids an operator types, and nothing in them about how a
    /// checkpoint was quantized or packaged.
    #[test]
    fn the_ids_are_the_ones_an_operator_types() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids, vec!["gemma-2-2b", "gemma-2-9b", "gemma-2-27b"]);
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and "does
    /// this ship a tower" — and holding a parsed `config.json` open for
    /// the life of a load to answer them is what this replaces. All three
    /// rows are asserted because all three advertise.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("gemma-2 deploys")
                .advertised;
            assert_eq!(
                a.arch, "gemma2",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 8_192,
                "{}: the 2B, 9B and 27B configs all state 8192",
                v.id
            );
            assert!(
                !a.media_encode,
                "{}: gemma-4 is the only generation whose towers the encode entry serves",
                v.id
            );
        }
    }

    /// This generation's ceiling is SHORTER than its successor's, and
    /// the test says so out loud: a shared `8192` next to gemma-3's
    /// `131072` looks like a row someone forgot to update, and it is
    /// what all three published gemma-2 configs state. The `2048` in the
    /// synthetic corpus config is a parser fixture and is not it either.
    ///
    /// The comparison AGAINST gemma-3 is not here. It was, briefly, as
    /// `assert!(MAX_MODEL_LEN < 32_768)` — a second spelling of the line
    /// above it, true by constant folding and unable to fail. Reading
    /// gemma-3's row instead made it a live assertion and a sibling
    /// edge, which `tests/sibling_isolation.rs` refuses. It lives in
    /// `tests/neighbouring_generations.rs` now, beside the OLMo pair
    /// that made that file necessary for the same reason.
    #[test]
    fn the_generation_states_a_shorter_ceiling_than_its_successors_on_purpose() {
        assert_eq!(MAX_MODEL_LEN, 8_192);
        assert_ne!(
            MAX_MODEL_LEN, 2_048,
            "the synthetic corpus config is not a checkpoint"
        );
        let ceilings: std::collections::BTreeSet<u32> = VARIANTS
            .iter()
            .map(|v| {
                v.deployment(Deployed::single())
                    .expect("deploys")
                    .advertised
                    .max_model_len
            })
            .collect();
        assert_eq!(
            ceilings.len(),
            1,
            "the generation agrees with itself: {ceilings:?}"
        );
    }

    /// The label is the stem the WORKER derives, which is what lets the
    /// boundary CHECK it: `read_hf_config_defaults` lowercases
    /// `architectures[0]`, strips the task suffix, and refuses a family
    /// [`crate::catalog::arches`] does not list. `gemma-2` or `gemma_2`
    /// here would refuse `Gemma2ForCausalLM`, which is all three rows.
    #[test]
    fn the_label_is_what_architectures_reduces_to() {
        let stem = "Gemma2ForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert!(
            !ARCH.contains('-') && !ARCH.contains('_'),
            "the stem carries no separator"
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
