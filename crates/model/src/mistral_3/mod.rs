//! The Mistral lineage.
//!
//! Two rows and a chat template. The authoring pass is
//! `crate::shared::llama_like::contract::author_dense` — Mistral's block is the
//! llama block, and saying so by CALLING the pass is what the old
//! `HF_ROWS` said by repeating a function pointer in a column.
//!
//! # What the string used to decide
//!
//! `"mistral3" | "ministral3"` selected the template and `"mistral"`
//! selected the facts, two spellings of one lineage in two tables, and
//! `instruct::create`'s `_ =>` arm meant a third spelling produced
//! ChatML rather than an error. Mistral's protocol is not ChatML and not
//! role headers: it is `[INST] … [/INST]` with no marker at all opening
//! the assistant turn, so a mis-keyed load did not fail — it answered in
//! a format the tune had never seen.
//!
//! # The two rows are not the same architecture
//!
//! Mistral-7B-v0.3 is the 32k-vocabulary dense stack with full attention
//! and `rope_theta: 1e6`. Ministral-8B is a later design: a 131k
//! vocabulary, 36 layers, `rope_theta: 1e8`, and a 32k sliding window
//! that is genuinely on. They share a template and an author and nothing
//! else, which is exactly the kind of split a single `model_type` string
//! could not express.

#[cfg(feature = "chat")]
pub mod chat;

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::project;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use model_ir::facts::{NormPlacement, QkNorm};
use model_ir::trace::{NormVariant, RopeKind};

/// One Mistral checkpoint.
pub struct Mistral3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency, and the two rows disagree about it by two
    /// orders of magnitude — 1e6 against 1e8. A per-row field rather
    /// than a per-generation constant for precisely this reason.
    pub rope_theta: f32,
    /// The RMSNorm epsilon — `1e-5` on both rows, Mistral's since 7B
    /// v0.1. A launch-time constant of the checkpoint that no extent
    /// records, so a row has to state it or a driver has to guess.
    pub norm_eps: f32,
    /// Sliding-window width, `-1` for full attention.
    pub window: i32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `mistral`, with no digit, which is what the boundary derives: both
/// releases state `architectures: ["MistralForCausalLM"]`, and the
/// worker lowercases that and strips `ForCausalLM` before checking the
/// rest against [`crate::catalog::arches`]. The generation is called
/// `mistral_3` in this crate for the release line it covers; the label
/// is not the module's name, and spelling it `mistral3` here would
/// refuse every real Mistral checkpoint at the boundary.
const ARCH: &str = "mistral";

/// The published context ceiling, shared by both rows.
///
/// 32 768, from `mistralai/Mistral-7B-Instruct-v0.3` and from
/// `mistralai/Ministral-8B-Instruct-2410`, which both state
/// `max_position_embeddings: 32768`.
///
/// The Ministral card advertises 128k, and the CONFIG is what this
/// states, deliberately: `tests/catalog_differential.rs` compares this
/// number against a `config.json`'s `max_position_embeddings`, the
/// checked-in corpus is `config.json` files, and the number a guest
/// program gets should be the one the checkpoint it is being served
/// carries. The 128k claim belongs to a serving stack that extends
/// positions past what the config states; nothing here does that, so
/// claiming it would be advertising someone else's deployment.
///
/// One constant because the two configs agree, and they disagree about
/// plenty else — rope theta by two orders of magnitude, the vocabulary
/// by more than 100 000 entries. The agreement is a fact about the two
/// releases rather than an assumption carried across them.
const MAX_MODEL_LEN: u32 = 32_768;

/// The generation's rows.
pub const VARIANTS: &[Mistral3] = &[
    // mistralai/Mistral-7B-Instruct-v0.3 — the committed fixture, and
    // the workspace's fused-QKV-without-qk-norm parity shape. v0.3's
    // one change from v0.2 is in the table below: the vocabulary grew
    // from 32000 to 32768 for the tool-call control tokens.
    Mistral3 {
        id: "mistral-7b-v0.3",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 14_336,
            vocab: 32_768,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-5,
        window: -1,
    },
    // mistralai/Ministral-8B-Instruct-2410. The sliding window here is
    // real — the config states 32768 and nothing switches it off — which
    // is what makes it worth contrasting with `qwen_2`, where the same
    // key ships beside `use_sliding_window: false` and means nothing.
    // (The published model interleaves 32k and 128k window layers; a row
    // states one width for the stack, which is the shape
    // `Deployment::attention` gets from `project::deployment`, and the
    // conservative one is the one that cannot over-attend.)
    Mistral3 {
        id: "ministral-8b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 12_288,
            vocab: 131_072,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 100_000_000.0,
        norm_eps: 1e-5,
        window: 32_768,
    },
];

crate::rows_of!(Mistral3);

impl Mistral3 {
    /// The scalars this row states, read ONCE.
    ///
    /// Both [`Variant::deployment`] and [`Variant::trace`] take it. They
    /// used to read `rope_theta`, `norm_eps` and `window` off `self`
    /// separately — the same three fields, spelled twice, with nothing
    /// holding the two spellings together.
    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            rope_theta: self.rope_theta,
            norm_eps: self.norm_eps,
            window: self.window,
            rope_rescaled: false,
            // Unread: this generation is dense, so no router is stated.
            norm_topk_prob: true,
        }
    }
}

impl Variant for Mistral3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(
            self.shape.layers,
            self.shape.head_dim,
            self.shape.tied_embeddings,
        )
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.row());
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // Text only. Pixtral and Mistral-Small-3.x carry vision
            // towers, and they are different packages with a different
            // `architectures[0]` and no row here — so nothing this
            // generation deploys has a tower for the driver's encode
            // entry to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// `author_dense` rather than `author_llama_like`: Mistral ships no
    /// attention biases and no q/k norms, so the pass that declares the
    /// plain dense stack is the whole contract.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_dense(builder)
            }
            // The registry this replaced held an MLX row for `mistral`,
            // and a row that states only the HF author hands Metal the
            // checkpoint's own names and its own dtype. See
            // `llama_3::mod`'s `author`.
            crate::shared::policy::Naming::Mlx => {
                crate::shared::llama_like::contract::author_llama_mlx(builder)
            }
        }
    }

    /// This row's text, for whichever backend asked.
    ///
    /// The WINDOW is the field to read twice here. Mistral-7B-v0.3
    /// states `sliding_window: 4096` and the v0.1/v0.2 rows state none,
    /// so this is the family's one generation where the same three
    /// scalars carry a real per-row difference — and a window dropped
    /// is a model attending its whole prefix, fluent about a context it
    /// was never trained to see.
    ///
    /// `rope_rescaled: false`: no Mistral config in this table states
    /// `rope_scaling`; the base alone describes the ladder.
    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// `[INST] … [/INST]`, whose assistant turn opens with NOTHING — the
    /// one protocol here for which the ChatML fallback was not merely
    /// wrong but invisible: a model prompted in ChatML still answers.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::MistralInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Mistral3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and
    /// whether a tower is present. Both rows are asserted because both
    /// advertise, and 0 is checked apart from the value because 0 is
    /// what the DEFAULT carries: it would mean "this row does not say".
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("dense mistral is servable")
                .advertised;
            assert_eq!(
                a.arch, "mistral",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 32_768,
                "{}: both configs state 32768",
                v.id
            );
            assert_ne!(a.max_model_len, 0, "{}: 0 is 'the row does not say'", v.id);
            assert!(
                !a.media_encode,
                "{}: no Mistral row here carries a tower",
                v.id
            );
        }
    }

    /// THE CEILING IS THE CONFIG'S AND NOT THE CARD'S.
    ///
    /// Ministral-8B is advertised as a 128k model and its
    /// `config.json` states `max_position_embeddings: 32768`. The row
    /// states the config, because that is what
    /// `tests/catalog_differential.rs` compares against and what the
    /// checkpoint being served actually carries. This assertion exists
    /// so that a later editor reading the model card does not "correct"
    /// the number upward and start advertising four times the positions
    /// the checkpoint states.
    #[test]
    fn the_ministral_row_states_its_config_rather_than_its_model_card() {
        let a = row("ministral-8b")
            .deployment(Deployed::single())
            .expect("servable")
            .advertised;
        assert_eq!(
            a.max_model_len, 32_768,
            "the config's number, not the card's 131072"
        );
        assert_ne!(
            a.max_model_len, 131_072,
            "the card's claim is not the checkpoint's claim"
        );
    }

    /// The label is the FAMILY and not this module's name.
    ///
    /// `read_hf_config_defaults` lowercases `architectures[0]`, strips
    /// `ForCausalLM`, and refuses any stem [`crate::catalog::arches`]
    /// does not list. `MistralForCausalLM` reduces to `mistral` with no
    /// digit — the release line the module is named for never reaches
    /// the boundary.
    #[test]
    fn the_label_is_the_family_and_not_the_release_line() {
        let stem = "MistralForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert_ne!(
            ARCH, "mistral3",
            "the module's name is not the checkpoint's name"
        );
        assert!(
            ARCH.chars().all(|c| c.is_ascii_lowercase()),
            "letters only, lowercased"
        );
    }

    /// The fixture and the row are the same measurement.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(
            row("mistral-7b-v0.3").shape,
            LlamaLikeFacts::mistral_7b_v03()
        );
    }

    /// Every row answers every question.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("dense mistral is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.vocab, v.shape.vocab);
            assert_eq!(d.norm, crate::deployment::NormPlacement::Pre);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, 128, "{}", v.id);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert!(!ls.tied_embeddings, "no Mistral row ties its head");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.window, v.window, "{} layer {l}", v.id);
                assert_eq!(a.rope_theta, v.rope_theta, "{} layer {l}", v.id);
                assert_eq!(a.head_dim, 128);
                assert_eq!(a.sm_scale, 1.0 / 128.0_f32.sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 2);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(ids, ["mistral-7b-v0.3", "ministral-8b"]);
    }

    /// The two rows share a template and an author and disagree about
    /// everything a checkpoint publishes — which is why they are two
    /// rows and not one row with a knob.
    #[test]
    fn the_two_rows_are_two_architectures() {
        let seven = row("mistral-7b-v0.3");
        let eight = row("ministral-8b");

        assert_eq!(seven.window, -1, "v0.3 attends its whole context");
        assert_eq!(eight.window, 32_768, "Ministral's window is on");
        assert_eq!(seven.rope_theta, 1_000_000.0);
        assert_eq!(eight.rope_theta, 100_000_000.0);
        assert_ne!(seven.shape.vocab, eight.shape.vocab);
        assert_ne!(seven.shape.layers, eight.shape.layers);
        assert_ne!(seven.shape.intermediate, eight.shape.intermediate);
        assert_ne!(
            seven.manifest(),
            eight.manifest(),
            "no checkpoint matches both"
        );
    }

    /// Both rows are GQA at eight KV heads, and the manifest's k/v
    /// extents are that fact rather than a second statement of it.
    #[test]
    fn the_kv_projections_are_a_quarter_of_the_query_one() {
        for v in VARIANTS {
            assert_eq!(v.shape.q_heads / v.shape.kv_heads, 4, "{}", v.id);
            let m = v.manifest();
            let extent = |name: &str| -> Vec<u64> {
                m.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .expect("stated")
                    .extents
                    .clone()
            };
            let hidden = u64::from(v.shape.hidden);
            assert_eq!(extent("layer.{}.self_attn.q_proj"), vec![4096, hidden]);
            assert_eq!(extent("layer.{}.self_attn.k_proj"), vec![1024, hidden]);
            assert_eq!(extent("layer.{}.self_attn.o_proj"), vec![hidden, 4096]);
            assert_eq!(
                extent("layer.{}.mlp.gate_proj"),
                vec![u64::from(v.shape.intermediate), hidden]
            );
        }
    }

    /// Untied throughout: both checkpoints ship `lm_head.weight`, and
    /// the manifest requires it — the fact that separates them from
    /// every tied row in the lineage.
    #[test]
    fn both_rows_ship_their_own_head() {
        for v in VARIANTS {
            assert!(!v.shape.tied_embeddings, "{}", v.id);
            let m = v.manifest();
            let head = m
                .tensors
                .iter()
                .find(|t| t.name == "lm_head")
                .expect("stated");
            assert_eq!(
                head.presence,
                crate::manifest::Presence::Required,
                "{}",
                v.id
            );
            assert_eq!(
                head.extents,
                vec![u64::from(v.shape.vocab), u64::from(v.shape.hidden)],
                "{}",
                v.id
            );
        }
    }

    /// The protocol, exactly: brackets around the user turn and nothing
    /// opening the assistant's. The empty cue is the part a ChatML
    /// fallback got wrong silently.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_the_instruct_bracket_protocol() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]",
            "[SYSTEM_PROMPT]",
            "[/SYSTEM_PROMPT]",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        for v in VARIANTS {
            let inst = v.chat(tok.clone());
            assert_eq!(
                tok.decode(&inst.user("Hi"), false),
                "[INST]Hi[/INST]",
                "{}",
                v.id
            );
            assert_eq!(
                tok.decode(&inst.system("Hi"), false),
                "[SYSTEM_PROMPT]Hi[/SYSTEM_PROMPT]",
                "{}",
                v.id
            );
            assert!(
                inst.cue().is_empty(),
                "{}: the assistant turn opens with nothing",
                v.id
            );
        }
    }

    /// Every row has an author and it runs.
    #[cfg(feature = "contract")]
    #[test]
    fn every_row_authors() {
        use model_loader::checkpoint::CheckpointMetadata;
        use model_loader::plan::StorageTarget;

        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let encoding = crate::encoding::Encoding::dense();
        let target = StorageTarget::default();
        let policy = crate::shared::policy::Policy::default();

        for v in VARIANTS {
            let mut builder = crate::shared::builder::Builder::new(
                &metadata,
                v.id(),
                v.load_shape(),
                &encoding,
                &target,
                &policy,
            );
            v.author(&mut builder)
                .unwrap_or_else(|e| panic!("{} refused to author: {e:?}", v.id));
        }
    }

    #[test]
    fn every_row_traces_both_fire_classes() {
        use model_ir::trace::FireClass;

        for v in VARIANTS {
            for class in [FireClass::Decode, FireClass::Prefill] {
                let plan = v
                    .trace(class, Deployed::single())
                    .expect("llama-like traces");
                assert!(
                    plan.family.contains("llama_like"),
                    "{}: {}",
                    v.id,
                    plan.family
                );
                assert!(!plan.ops.is_empty(), "{}", v.id);
            }
        }
    }
}
