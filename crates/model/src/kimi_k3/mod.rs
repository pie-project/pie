//! Kimi K3 — an MLA / KDA hybrid over an MXFP4 mixture.
//!
//! One generation, one directory, and now one ROW. What this generation
//! was before is the three-table shape the catalog exists to remove:
//! `deployment_cuda::FACTS_ROWS` held `"kimi_k3"` against a
//! `kimi_k3_facts_from_hf` that re-parsed `config.json` on every load,
//! `contract::HF_ROWS` held the same string against `author_kimi_k3`,
//! and `instruct::create` held it a third time, in kimi-k2's arm,
//! pointing at a template that lived in kimi-k2's directory. Three keys,
//! three tables, and one of them reaching into a sibling for its answer.
//!
//! The template is shared and stays shared — `shared::kimi` is where
//! it went, so this row names a FAMILY rather than a sibling generation.
//!
//! Read `VARIANTS` and then `impl Variant`. What this generation adds
//! over its shape is what a shape cannot state: which pass authors it,
//! which template speaks for it, and — here uniquely — that its text
//! cannot yet trace the model the row states.

#[cfg(feature = "contract")]
pub mod contract;

/// The declared forward -- an MLA / KDA hybrid over an MXFP4 MoE stack.
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
use spec::KimiK3Facts;

/// The coarse family label a GUEST PROGRAM matches on.
///
/// The one place a family string survives the refactor: `engine`'s
/// `model.arch_name()` is a host function inferlets call. What it is NOT
/// any more is a dispatch key — nothing in this crate matches on it.
const ARCH: &str = "kimi_k3";

/// One Kimi K3 checkpoint.
///
/// A newtype over the shape for the exemplar's reason: `chat` and
/// `author` are answers the numbers cannot give, and this generation's
/// two are kimi's shared `<|im_middle|>` protocol and an authoring pass
/// that bands a padded gate bank and dequantizes MXFP4 experts.
pub struct KimiK3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: KimiK3Facts,
    /// Rope's base frequency, for the KDA layers. The MLA layers rotate
    /// nothing at all — see [`project::deployment`], which is where that
    /// stops being a comment and becomes a per-layer statement.
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
/// ONE, and it is the only K3 geometry written down anywhere in this
/// tree: `KimiK3Facts::kimi_k3_synthetic()`, which the goldens, the
/// arena check, the lowering test and the executor bind all trace
/// against. It is a FIXTURE and not a published checkpoint — no
/// `config.json` for a real K3 exists here, and
/// `driver-cuda/tests/hf_config_dump/corpus/synthetic--kimi-k3.json` is
/// a hand-written toy at 128 hidden.
///
/// Stated as a row anyway, and the reason is the redesign's own: a row
/// has to exist under every aspect or the generation is unanswerable in
/// a build that does not compile the tracer. What a row must NOT do is
/// invent numbers, so this states the measurement that exists and says
/// plainly which one it is. The first published K3 config replaces these
/// eleven numbers and nothing else.
pub const VARIANTS: &[KimiK3] = &[KimiK3 {
    id: "kimi-k3",
    shape: KimiK3Facts {
        layers: 8,
        vocab: 163_840,
        hidden: 2048,
        dense_intermediate: 5632,
        dense_layers: 1,
        full_attn_interval: 4,
        attn_res_block: 4,
        attn: spec::KimiK3MlaFacts {
            hidden: 2048,
            heads: 16,
            q_lora_rank: 768,
            kv_lora_rank: 256,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            // TRUE on the row, where the fixture says false. They are
            // answering different questions: the fixture states what
            // `forward::kimi_k3_cuda` can DECLARE — the text asserts on
            // this and would panic — and a row states what the model IS.
            // `deployment_cuda::kimi_k3_facts_from_hf` set it from the
            // config and got `true`, which is the same reading this
            // makes. The disagreement surfaces as `trace()` refusing,
            // out loud, instead of as a panic inside a walk.
            output_gate: true,
        },
        kda: spec::KimiK3KdaFacts {
            value_heads: 16,
            value_head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound_milli: 0,
            norm_eps_micro: 10,
        },
        moe: spec::KimiK3MoeFacts {
            num_experts: 64,
            top_k: 6,
            // As `spec.rs` -- inherited from K2.
            norm_topk_prob: false,
            routed_scaling: 2.0,
            moe_intermediate: 1024,
            shared_intermediate: 1024,
        },
    },
    // Neither theta nor epsilon is a tensor extent, so neither can be
    // checked against a checkpoint and both have to be STATED. These are
    // what a K3 config that omits them reads as — the normalizer's
    // defaults — and `synthetic--kimi-k3.json`, the only K3 config
    // committed here, omits both. The honest reading available in this
    // tree, and the first thing to correct against a published config.
    rope_theta: 10_000.0,
    norm_eps: 1e-5,
    // Not stated: the 2048 in the synthetic is a fixture's ceiling and
    // not a checkpoint's. `0` is the field's own word for "the row does
    // not say".
    max_model_len: 0,
    tied_embeddings: false,
}];

crate::rows_of!(KimiK3);

impl KimiK3 {
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
    /// — no build in this tree provisions a KDA state store beside an
    /// MLA cache — so a statement written inside it would be a
    /// capability answer no test could reach and no operator could ever
    /// read back. A row's identity does not depend on being servable,
    /// and neither does this.
    #[must_use]
    pub fn advertised(&self) -> Advertised {
        Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // No tower. The `language_model.` prefix `author_kimi_k3`
            // probes for is the multimodal PACKAGING of a text stack —
            // the decoder nests whether or not an encoder ships beside
            // it — so nesting is not evidence of a tower and this does
            // not read it as one. Guessing `true` here would have the
            // worker build an encode executor for a model with nothing
            // to encode.
            media_encode: false,
        }
    }
}

impl Variant for KimiK3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape, self.tied_embeddings)
    }

    /// The head dim an AUTHORING pass needs, and for this hybrid that is
    /// the MLA page row: `author_kimi_k3` reads `b.shape().layers` to
    /// walk the stack and the head dim to split `q_b_proj` and
    /// `kv_b_proj`, whose rows are heads of `qk_nope + qk_rope` and
    /// `qk_nope + v` — both multiples of the latent row. The KDA half is
    /// split by the head count it reads off `b_proj` instead, which is
    /// the only place its unpadded head count is written down.
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
            crate::shared::policy::Naming::Hf => contract::author_kimi_k3(builder),
            // `MLX_ROWS` never held this generation, and the absence was
            // an `Ok(None)` the caller read as "no contract" — a silence
            // shaped exactly like a generation nobody had written yet.
            // Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "kimi-k3: no MLX authoring pass exists for this generation, so \
                 there is no name layout to author against",
            ),
        }
    }

    /// # Errors
    ///
    /// Whatever [`Self::deployment`] refuses — asked first, because the
    /// door's question and the fire's are one question — and then
    /// [`crate::deployment::Refusal::Unsupported`] from
    /// [`project::trace`] when this row states the gated MLA output the
    /// generation ships and the text cannot declare it. That second
    /// refusal is where the assertion inside the text stops being a
    /// panic; the first is why a row with nowhere to put its KV never
    /// gets a fire to begin with.
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
            .and_then(|_| project::trace(&self.shape, class))
    }

    /// Kimi's own `<|im_middle|>` protocol, shared with kimi-k2 and
    /// stated from `shared/` rather than reached for across a sibling
    /// directory. The arm that used to state it keyed on `"kimi_k2" |
    /// "kimi_k25" | "kimi_k3"` — three strings for one answer, in a
    /// table that could disagree with the other two.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::kimi::KimiInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static KimiK3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The row and the traced fixture are the same measurement, and they
    /// differ in exactly ONE field. Anything else drifting apart means a
    /// golden is a trace of a model no row states.
    #[test]
    fn the_row_differs_from_the_traced_fixture_in_the_output_gate_alone() {
        let mut fixture = KimiK3Facts::kimi_k3_synthetic();
        let row = &row("kimi-k3").shape;
        assert_ne!(*row, fixture, "the row states the gate the text refuses");
        fixture.attn.output_gate = true;
        assert_eq!(
            *row, fixture,
            "one field apart: the row is the model, the fixture is what the \
             text can declare",
        );
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
    /// an encoding of it: this generation ships MXFP4 experts, and an
    /// id that said so would make a bf16 republish a different model.
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
            for banned in ["mxfp4", "fp8", "int4", "bf16", "awq"] {
                assert!(!v.id.contains(banned), "{} names a packing", v.id);
            }
        }
    }

    /// The generation's family label is a CONST and not a per-row
    /// string, because every row of a generation shares it.
    #[test]
    fn the_generation_states_one_family_label() {
        assert_eq!(ARCH, "kimi_k3", "the string three tables used to key on");
        assert!(!ARCH.is_empty());
    }

    /// Every row answers every question, which is what having no default
    /// bodies buys — and for this generation two of the answers are
    /// refusals, which are still answers.
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

    /// The load shape, field by field. `head_dim` is the sharp one: an
    /// authoring pass splits `q_b_proj` and `kv_b_proj` by it, and the
    /// latent page row is what both are multiples of.
    #[test]
    fn the_load_shape_states_the_latent_page_row() {
        let ls = row("kimi-k3").load_shape();
        assert_eq!(ls.layers, 8);
        assert_eq!(ls.head_dim, 320, "the latent plus the shared rope half");
        assert_eq!(ls.n_experts, 64);
        assert_eq!(ls.mamba_groups, 0, "the KDA state is not a mamba mixer's");
        assert_eq!(ls.kv_shared_layers, 0, "every paging layer owns its own");
        assert!(!ls.tied_embeddings, "this generation unties its head");
    }

    /// A row that LOADS and does not SERVE says so at the door — and
    /// this one cannot TRACE either, which is a second refusal and a
    /// different question. The manifest and the load shape are complete
    /// regardless: a checkpoint can be identified and authored by a
    /// build that could never fire it.
    #[test]
    fn the_row_is_complete_where_it_is_unservable() {
        let v = row("kimi-k3");
        assert!(
            v.manifest().tensors.len() > 10,
            "identity does not depend on serving"
        );
        assert_eq!(v.load_shape().layers, v.shape.layers);
        assert!(matches!(
            v.deployment(Deployed::single()),
            Err(crate::deployment::Refusal::Unsupported(_)),
        ));
    }

    /// The trace refuses, because the row states a gate the text asserts
    /// against. A `Result` here is what turns that assertion from a
    /// panic inside a walk into an answer at the boundary.
    #[test]
    fn the_row_refuses_to_trace_the_gate_its_text_cannot_state() {
        use model_ir::trace::FireClass;
        for class in [FireClass::Decode, FireClass::Prefill] {
            let err = row("kimi-k3")
                .trace(class, Deployed::single())
                .expect_err("the text asserts on the gate this row states");
            assert!(matches!(err, crate::deployment::Refusal::Unsupported(_)));
        }
    }

    /// The chat answer is the FAMILY's, not a sibling's. Naming
    /// `crate::kimi_k2::chat` here would be the reach the templates were
    /// moved to `shared/` to remove — and it is kimi's own
    /// `<|im_middle|>` protocol, which READS like ChatML and is not: the
    /// `_ =>` arm this replaces handed every unknown architecture
    /// `<|im_start|>`, and a K3 served that way emits a turn its
    /// checkpoint never learned to close.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question_in_kimis_own_protocol() {
        use tokenizer::Tokenizer;

        let words: Vec<String> = ["<|im_user|>", "<|im_middle|>", "<|im_end|>", "user", "Hi"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&words));
        for v in VARIANTS {
            let turn = tok.decode(&v.chat(tok.clone()).user("Hi"), false);
            assert_eq!(turn, "<|im_user|>user<|im_middle|>Hi<|im_end|>", "{}", v.id);
            assert!(
                !turn.contains("<|im_start|>"),
                "the `_ =>` ChatML arm is gone"
            );
        }
    }

    /// Both authoring arms are stated, and the MLX one is a refusal
    /// rather than a silence: `MLX_ROWS` held no K3 row, and the caller
    /// read that absence as "no contract".
    #[cfg(feature = "contract")]
    #[test]
    fn an_mlx_naming_is_refused_out_loud_and_an_hf_naming_is_authored() {
        use crate::encoding::Encoding as StoredEncoding;
        use crate::shared::policy::{Naming, Policy};
        use model_loader::checkpoint::CheckpointMetadata;
        use model_loader::plan::StorageTarget;

        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let target = StorageTarget::default();
        let encoding = StoredEncoding::dense();
        let v = row("kimi-k3");

        let mlx = Policy {
            naming: Naming::Mlx,
            ..Policy::default()
        };
        let mut builder = crate::shared::builder::Builder::new(
            &meta,
            v.id,
            v.load_shape(),
            &encoding,
            &target,
            &mlx,
        );
        assert_eq!(builder.naming(), Naming::Mlx);
        assert!(
            v.author(&mut builder).is_err(),
            "no MLX layout exists to author against"
        );

        let hf = Policy::default();
        let mut builder = crate::shared::builder::Builder::new(
            &meta,
            v.id,
            v.load_shape(),
            &encoding,
            &target,
            &hf,
        );
        assert_eq!(builder.naming(), Naming::Hf);
        // An empty checkpoint has no expert stacks and no gate banks to
        // band, so the HF pass runs to its `publish_remaining` and
        // reports what it found — which is nothing, and not an error.
        assert!(
            v.author(&mut builder).is_ok(),
            "the HF pass must reach a checkpoint that simply has no tensors",
        );
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
    /// every build here — no KDA state store is provisioned beside the
    /// MLA cache — and a capability
    /// statement that only exists once a build can serve the model is a
    /// statement no operator can read while deciding whether to try.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v.advertised();
            assert_eq!(
                a.arch, "kimi_k3",
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
                "{}: state a ceiling only from a published config — the only Kimi-K3 \
                 config in this tree is a parser fixture, and its ceiling is the fixture's",
                v.id,
            );
            assert!(
                !a.media_encode,
                "{}: no Kimi-K3 row ships a tower the encode entry serves, and a true here \
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
        // CUDA is refused too, and on a DIFFERENT ground. This block
        // used to be an `assert_ne!` against the Metal sentence, which
        // was the weakest of the four MLA families' versions of this
        // test and is why one behaviour change produced three failures
        // and one pass: `deepseek_v4`, `glm_5` and `kimi_k2` asserted
        // that CUDA still traced, this asserted only that the refusal
        // was not the Metal one, and all four had taken the same new
        // line in `trace`.
        //
        // THIS ROW HAS TWO REASONS and they are now ORDERED. The store
        // refusal fires first, because `trace` asks `deployment` before
        // it traces; the output-gate refusal `project::trace` states —
        // the semantic `SigmoidGateMul` wants equal shapes and MLA's
        // absorb is rank-3 — is reached only once a store exists. The
        // prose here used to warn that a gate firing first "would
        // swallow the more specific refusal and report the wrong missing
        // thing", and the answer is that it reports the FIRST missing
        // thing: a store has to land before a text can be asked for, so
        // the refusals are staged rather than lost.
        //
        // What is asserted is the invariant that the ordering created:
        // the door and the fire refuse with ONE sentence. That is the
        // gap the new line closed — `deployment()` consulted the store
        // and `trace()` did not, so a row refused at the door and handed
        // out a fire anyway.
        for v in VARIANTS {
            let at_the_door = v
                .deployment(Deployed::single())
                .expect_err("this build provisions no store for this row's KvStyle");
            let at_the_fire = v
                .trace(FireClass::Decode, Deployed::single())
                .expect_err("a row refused at the door must not hand out a fire");
            assert_eq!(
                at_the_fire, at_the_door,
                "`{}` refuses a CUDA load at the door and at the fire with two \
                 different sentences; they are one question and must not have \
                 two answers",
                v.id
            );
            assert_ne!(
                at_the_fire,
                Refusal::Unsupported(project::NO_METAL),
                "`{}` answered a CUDA load with the METAL refusal",
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
