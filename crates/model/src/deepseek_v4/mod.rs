//! DeepSeek-V4 — a rank-K residual over compressed attention and a
//! dense-prefix mixture.
//!
//! One generation, one directory, and now one ROW. What this generation
//! was before is the three-table shape the catalog exists to remove:
//! `deployment_cuda::FACTS_ROWS` held `"deepseek_v4"` against a
//! `dsv4_facts_from_hf` that re-parsed `config.json` on every load,
//! `contract::HF_ROWS` held the same string against
//! `author_deepseek_v4`, and `instruct::create` held it a third time,
//! pointing at a template that lived in a SIBLING generation's
//! directory. Three keys, three tables, and one of them reaching across
//! an edge the isolation rule forbids — the template is
//! [`crate::shared::deepseek`] now, so this row names a family and not
//! a sibling.
//!
//! The three tables also disagreed with the model. `dsv4_facts_from_hf`
//! filled `o_lora_rank: 0`, `o_groups: 1`, `hc: { mult: 1 }` and
//! `swiglu_limit_milli: 0` — a stack with an ordinary residual, an
//! ungrouped output projection and an unclamped activation, which is not
//! this generation at all. It parsed a config into a shape the config
//! does not state, and the forward text traced whatever it was handed.
//!
//! Read `VARIANTS` and then `impl Variant`. What this generation adds
//! over its shape is what a shape cannot state: which pass authors it,
//! which template speaks for it, and that no build here provisions the
//! store it needs.

#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward — hyper-connections over compressed attention and
/// an MoE stack.
#[cfg(feature = "forward")]
pub mod forward;

/// The SHAPE, ungated: a catalog row is written in these words.
pub mod spec;

/// The three projections a row makes — its manifest, its deployment,
/// its text.
pub mod project;

use std::sync::{Arc, OnceLock};

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::deployment::Advertised;
use crate::manifest::Manifest;
use spec::Dsv4Facts;

/// The coarse family label a GUEST PROGRAM matches on.
///
/// The one place a family string survives the refactor: `engine`'s
/// `model.arch_name()` is a host function inferlets call. What it is NOT
/// any more is a dispatch key — nothing in this crate matches on it.
const ARCH: &str = "deepseek_v4";

/// One DeepSeek-V4 checkpoint.
///
/// A newtype over the shape for the exemplar's reason: `chat` and
/// `author` are answers the numbers cannot give, and this generation's
/// two are DeepSeek's fullwidth-delimiter template and an authoring pass
/// with its own tensor-parallel shard rule — the only family in the tree
/// that splits an expert's intermediate dim WITHIN the expert.
pub struct Dsv4 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: Dsv4Facts,
    /// Rope's base frequency. The rope is PARTIAL — the last
    /// `qk_rope_head_dim` channels of each head — and which channels
    /// turn is [`project::deployment`]'s statement; this is only the
    /// base they turn at.
    pub rope_theta: f32,
    /// RMSNorm epsilon — a constant of the checkpoint that no tensor
    /// extent carries, so a row must state it.
    pub norm_eps: f32,
    /// The published context ceiling, `0` where this tree has no
    /// published config to read one from.
    pub max_model_len: u32,
    /// The head shares the embedding table.
    pub tied_embeddings: bool,
}

/// The generation's rows.
///
/// ONE, and it is the only V4 geometry written down anywhere in this
/// tree: `Dsv4Facts::dsv4_synthetic()`, which the goldens, the arena
/// check, the lowering test and the executor bind all trace against. It
/// is a FIXTURE and not a published checkpoint — no `config.json` for a
/// real V4 exists here, and
/// `driver-cuda/tests/hf_config_dump/corpus/synthetic--deepseek-v4.json`
/// is a hand-written toy at 128 hidden whose only number this row takes
/// is the compression schedule.
///
/// Stated as a row anyway, and the reason is the redesign's own: a row
/// has to exist under every aspect or the generation is unanswerable in
/// a build that does not compile the tracer. What a row must NOT do is
/// invent numbers, so this states the measurement that exists and says
/// plainly which one it is. The first published V4 config replaces these
/// numbers and nothing else.
pub const VARIANTS: &[Dsv4] = &[Dsv4 {
    id: "deepseek-v4",
    shape: Dsv4Facts {
        layers: 6,
        vocab: 129_280,
        hidden: 2048,
        dense_intermediate: 5632,
        dense_layers: 1,
        // The per-layer schedule, which the derivation this replaces
        // never carried at all — see [`project::deployment`].
        ratios: &[1, 2, 4],
        attn: spec::Dsv4AttnFacts {
            hidden: 2048,
            heads: 16,
            head_dim: 128,
            q_lora_rank: 768,
            qk_rope_head_dim: 64,
            sliding_window: 2048,
            o_lora_rank: 512,
            o_groups: 4,
        },
        hc: spec::Dsv4HcFacts { mult: 4 },
        moe: spec::Dsv4MoeFacts {
            num_experts: 64,
            top_k: 6,
            norm_topk_prob: false,
            routed_scaling: 2.5,
            moe_intermediate: 1024,
            swiglu_limit_milli: 7000,
            hash_routed: false,
        },
    },
    // Neither theta nor epsilon is a tensor extent, so neither can be
    // checked against a checkpoint and both have to be STATED. These are
    // what a V4 config that omits them reads as — the normalizer's
    // defaults — and `synthetic--deepseek-v4.json`, the only V4 config
    // committed here, omits both. It states a `compress_rope_theta` of
    // 10000.0, which belongs to the COMPRESSED pass and is a different
    // rope; it is not read as this one. The honest reading available in
    // this tree, and the first thing to correct against a published
    // config.
    rope_theta: 10_000.0,
    norm_eps: 1e-5,
    // Not stated: the 2048 in the synthetic is a toy's ceiling and not a
    // checkpoint's. `0` is the field's own word for "the row does not
    // say".
    max_model_len: 0,
    // The committed contract fixture loads this generation at
    // `LoadShape::dense(1, 0, true)` and ships no `lm_head.weight`,
    // which is the only statement about the head anywhere in the tree.
    tied_embeddings: true,
}];

/// This generation's contribution to [`crate::catalog::catalog`].
///
/// The `OnceLock` is only the widening from `&Dsv4` to `&dyn Variant`;
/// the rows themselves are `const` and in `.rodata`.
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Dsv4 {
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
    /// — no build in this tree provisions the compressed cache — so a
    /// statement written inside it would be a capability answer no test
    /// could reach and no operator could ever read back. A row's
    /// identity does not depend on being servable, and neither does
    /// this.
    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // No tower: this generation is text in, text out. The field
            // it replaces was hardwired `false` for every family, which
            // is how gemma-4's ported vision and audio towers became
            // unreachable through the engine while four GPU tests that
            // call the encode entry point directly kept passing — so
            // `false` here is a STATEMENT and not the old silence.
            media_encode: false,
        }
    }
}

impl Variant for Dsv4 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    /// The head dim an AUTHORING pass needs, and for this generation it
    /// is the head's own width — K and V are one straight projection, so
    /// there is no latent row to state instead. `n_experts` is what
    /// makes the two expert passes walk anything at all: both key on
    /// `.ffn.experts.<e>.` and the walk is over this count.
    fn load_shape(&self) -> LoadShape {
        LoadShape::mixture(
            self.shape.layers,
            self.shape.attn.head_dim,
            self.shape.moe.num_experts,
            self.tied_embeddings,
        )
    }

    /// # Errors
    ///
    /// [`crate::deployment::Refusal::Unsupported`] — this build
    /// provisions no compressed KV store, and the row says so at the
    /// door rather than at its first fire.
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.rope_theta, self.norm_eps)?;
        deployment.advertised = self.advertised();
        Ok(deployment)
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
            crate::shared::policy::Naming::Hf => contract::author_deepseek_v4(builder),
            // `MLX_ROWS` never held this generation, and the absence was
            // an `Ok(None)` the caller read as "no contract" — a silence
            // shaped exactly like a generation nobody had written yet.
            // Stated as the refusal it always was: an MLX conversion
            // would spell the experts some third way, and the shard rule
            // this generation carries is written against `.ffn.experts.`.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "deepseek-v4: no MLX authoring pass exists for this generation, \
                 so there is no name layout to author against",
            ),
        }
    }

    /// # Errors
    ///
    /// Whatever [`Self::deployment`] refuses, because it is asked first
    /// and the two questions are one question: "does this build serve
    /// this row". The text itself traces for both fire classes.
    ///
    /// It has to be asked here rather than assumed. This doc used to
    /// claim "a row whose store is unbuilt is turned away at load, so
    /// this is only ever reached by a caller that wants the declaration
    /// itself" — but nothing sequenced the two calls, and the caller it
    /// described was hypothetical. The row refused at the door and
    /// handed out a fire anyway.
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
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
        self.deployment(load)?;
        Ok(project::trace(&self.shape, class))
    }

    /// DeepSeek's own template, with the FULLWIDTH role delimiters.
    ///
    /// Stated by the row; the arm that used to state it was a
    /// `"deepseek_v4"` key in `instruct::create` pointing into
    /// `deepseek_r1`'s directory. The distinction is not cosmetic: the
    /// `_ =>` arm beside it handed unknown architectures ChatML, and
    /// `<|im_end|>` is not `<｜end▁of▁sentence｜>` — a V4 sealed with the
    /// wrong one generates past the end of its turn.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::deepseek::R1Instruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Dsv4 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. `Dsv4Facts::dsv4_synthetic()` is
    /// what every declared V4 plan in this tree is traced from; so is
    /// the row, and the row is now the only one a driver reads.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("deepseek-v4").shape, Dsv4Facts::dsv4_synthetic());
    }

    /// The row states the stack, where the derivation this replaces
    /// stated something simpler than the stack. Held here as literals
    /// rather than by calling `dsv4_facts_from_hf` — that function is
    /// deleted, and these four numbers are part of why it had to be.
    #[test]
    fn the_row_states_the_stack_the_old_derivation_flattened() {
        let s = &row("deepseek-v4").shape;
        assert!(s.hc.mult > 1, "the derivation read 1: an ordinary residual");
        assert!(
            s.attn.o_lora_rank > 0,
            "the derivation read 0: an ungrouped output"
        );
        assert!(s.attn.o_groups > 1, "the derivation read 1: a single group");
        assert!(
            s.moe.swiglu_limit_milli > 0,
            "the derivation read 0: no clamp"
        );
        assert!(
            !s.ratios.is_empty(),
            "the derivation passed no schedule at all"
        );
    }

    /// One row, one id, and the id is unique and non-empty. A duplicate
    /// here is two rows a checkpoint could match, which the catalog
    /// answers as `Ambiguous` — a load failure with two right answers.
    #[test]
    fn the_ids_are_unique_and_non_empty() {
        let mut seen = std::collections::BTreeSet::new();
        for v in VARIANTS {
            assert!(
                !v.id.is_empty(),
                "a row with no id cannot be asked for by name"
            );
            assert!(seen.insert(v.id), "{} is stated twice", v.id);
        }
    }

    /// An id is what an operator TYPES, so it is lowercase and
    /// hyphenated — never the `model_type` string, which is what the
    /// three tables keyed on and which spells this generation with an
    /// underscore.
    #[test]
    fn the_ids_are_lowercase_and_hyphenated() {
        for v in VARIANTS {
            assert!(
                v.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{} is not the shape an id is written in",
                v.id,
            );
            assert!(
                !v.id.contains('_'),
                "{}: `_` is a model_type's spelling",
                v.id
            );
            assert_ne!(v.id, ARCH, "an id names a MODEL and an arch names a family");
        }
    }

    /// Every row reaches the catalog. A row in `VARIANTS` that `rows()`
    /// does not widen is a model the tree claims to serve and no lookup
    /// can find.
    #[test]
    fn every_row_reaches_the_catalog() {
        assert_eq!(rows().len(), VARIANTS.len());
        for (widened, v) in rows().iter().zip(VARIANTS) {
            assert_eq!(widened.id(), v.id);
        }
        // The `OnceLock` hands back the same slab twice rather than
        // rebuilding it, which is what makes `&'static dyn` honest.
        assert_eq!(rows().as_ptr(), rows().as_ptr());
    }

    /// The load shape, field by field. `head_dim` is the sharpest of
    /// them: `[2048, 2048]` is 16 heads of 128 or 32 of 64, the tensor
    /// cannot say which, and a TP row split that guesses wrong cuts a
    /// head in half.
    #[test]
    fn the_load_shape_states_what_no_extent_can() {
        let s = row("deepseek-v4").load_shape();
        assert_eq!(s.layers, 6);
        assert_eq!(s.head_dim, 128);
        assert_eq!(s.n_experts, 64, "the expert passes walk this count");
        assert_eq!(s.mamba_groups, 0, "nothing here is a Mamba mixer");
        assert_eq!(s.kv_shared_layers, 0, "every layer owns its pages");
        assert!(s.tied_embeddings);
    }

    /// A row answers the identity questions whether or not this build
    /// can serve it, which is the whole point of splitting identity from
    /// deployment: `pie model convert` compiles neither tracer nor
    /// driver and still has to know what a V4 ships.
    #[test]
    fn identity_does_not_depend_on_being_servable() {
        let v = row("deepseek-v4");
        let m = v.manifest();
        assert_eq!(m.layers, v.shape.layers);
        assert!(
            m.tensors.len() > 10,
            "a stack this deep names more than a handful"
        );
        assert_eq!(v.id(), "deepseek-v4");
        assert!(matches!(
            v.deployment(Deployed::single()),
            Err(crate::deployment::Refusal::Unsupported(_)),
        ));
    }

    /// The text traces for both fire classes even though the deployment
    /// refuses, and the two answers are about different things: a
    /// declaration exists, and this build cannot provision the store it
    /// needs. The family string is what the goldens are keyed on, so a
    /// rename here is a rename of every recorded plan.
    ///
    /// That distinction is real, and the call it belongs to is
    /// [`project::trace`] — TOTAL, like `plan`, a fact about the row.
    /// [`Variant::trace`] is the other thing: capability-gated, and it
    /// must refuse exactly where the door does. This test used to ask
    /// the gated one, which is how "the declaration exists" turned into
    /// "the serving path hands out a fire for a row it just turned
    /// away".
    #[cfg(feature = "forward")]
    #[test]
    fn the_declaration_traces_where_the_deployment_refuses() {
        use model_compiler::trace::FireClass;
        let v = row("deepseek-v4");
        assert!(
            v.deployment(Deployed::single()).is_err(),
            "this test is about a row this build refuses; that is the premise",
        );
        for (class, suffix) in [
            (FireClass::Decode, "decode"),
            (FireClass::Prefill, "prefill"),
        ] {
            let plan = project::trace(&v.shape, class);
            assert_eq!(plan.family, format!("deepseek_v4.cuda.{suffix}"));
            assert!(
                v.trace(class, Deployed::single()).is_err(),
                "the serving path must refuse where the door refused",
            );
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

        let v = row("deepseek-v4");
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

    /// The chat answer is DeepSeek's own and not ChatML. The `_ =>` arm
    /// this replaces handed every unrecognized architecture
    /// `<|im_start|>`, and these delimiters are FULLWIDTH — a different
    /// token entirely, so a turn opened one way cannot be closed the
    /// other.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question_in_deepseeks_own_delimiters() {
        use crate::instruct::Instruct;
        use tokenizer::Tokenizer;

        let words: Vec<String> = [
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜end▁of▁sentence｜>",
            "<|EOT|>",
            "<｜begin▁of▁sentence｜>",
            "<think>",
            "</think>",
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "<｜tool▁call▁end｜>",
            "<｜tool▁outputs▁begin｜>",
            "<｜tool▁outputs▁end｜>",
            "<｜tool▁output▁begin｜>",
            "<｜tool▁output▁end｜>",
            "\n",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&words));
        for v in VARIANTS {
            let chat = v.chat(tok.clone());
            let turn = tok.decode(&chat.user("Hi"), false);
            assert_eq!(turn, "<｜User｜>Hi", "{}", v.id);
            assert!(
                !turn.contains("<|im_start|>"),
                "{}: the ChatML fallback is gone",
                v.id
            );
            assert_eq!(
                tok.decode(&chat.cue(), false),
                "<｜Assistant｜>",
                "{}: the cue opens the turn the checkpoint was tuned on",
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
    /// every build here — no compressed cache is provisioned — and a capability
    /// statement that only exists once a build can serve the model is a
    /// statement no operator can read while deciding whether to try.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v.advertised();
            assert_eq!(
                a.arch, "deepseek_v4",
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
                "{}: state a ceiling only from a published config — the only DeepSeek-V4 \
                 config in this tree is a parser fixture, and its ceiling is the fixture's",
                v.id,
            );
            assert!(
                !a.media_encode,
                "{}: no DeepSeek-V4 row ships a tower the encode entry serves, and a true here \
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
    #[cfg(feature = "forward")]
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
