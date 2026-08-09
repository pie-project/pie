//! OLMo 2.
//!
//! Three rows, the lineage's dense author, and OLMo's own chat markers.
//! This is the generation that made two of the old derivation's four
//! tensor probes necessary, and both are rows in the manifest now.
//!
//! # Post-norm, and how it used to be discovered
//!
//! OLMo 2 norms each sub-layer's OUTPUT rather than its input: the
//! checkpoint ships `post_attention_layernorm` and
//! `post_feedforward_layernorm` and NO `input_layernorm`, each sub-layer
//! reads the residual stream raw, and a separate residual add lands the
//! normed result. `llama_like_facts_from_hf` found this out by asking
//! whether the `layer.0.attn_norm` alias happened to end in
//! `input_layernorm.weight` — a probe of the LOAD, made to answer a
//! question about the MODEL, whose "no" was also the answer for a
//! checkpoint that simply had not been aliased yet. The rows state
//! [`NormPlacement::Post`], and
//! [`project::manifest`](crate::shared::llama_like::project::manifest)
//! turns it into the pair of expectations the probe was groping for.
//!
//! # Global q/k norm, and how THAT used to be discovered
//!
//! OLMo 2's `q_norm`/`k_norm` are one RMSNorm over the whole flattened
//! projection — `[q_heads * head_dim]`, 2048 for the 1B — where Qwen 3's
//! are per-head `[head_dim]`. The derivation told them apart by dividing
//! a byte count from the safetensors header by two and comparing against
//! `head_dim`. The extent IS the fact, so it is an extent here, and a
//! checkpoint whose q-norm is the wrong width is a different variant
//! rather than the same variant lowered to the wrong kernel.
//!
//! Neither fact came from `model_type` — but the string is what selected
//! the code that went looking, through `model_type.starts_with("olmo")`.

#[cfg(feature = "chat")]
pub mod chat;

// `Arc` is the chat aspect's alone: it is the tokenizer a template
// is handed and the `dyn Instruct` it is returned as. `OnceLock`
// widens this generation's rows and every aspect reads that.
#[cfg(feature = "chat")]
use std::sync::Arc;
use std::sync::OnceLock;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;
use crate::shared::llama_like::project;
use crate::shared::llama_like::spec::LlamaLikeFacts;

use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::trace::{NormVariant, RopeKind};

/// One OLMo 2 checkpoint.
pub struct Olmo2 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency — 5e5 across the generation.
    pub rope_theta: f32,
    /// The norm's epsilon, stated per row rather than shared, because
    /// it is a measurement of a checkpoint and the next release that
    /// changes it must change a number rather than break a rule.
    pub norm_eps: f32,
    /// Sliding-window width, `-1` for full attention. OLMo 2 ships
    /// `sliding_window: null`; the interleaved windows are OLMo 3's.
    pub window: i32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `olmo2`, which is what the boundary derives: every published OLMo 2
/// config states `architectures: ["Olmo2ForCausalLM"]`, and the worker
/// lowercases that and strips `ForCausalLM` before checking the rest
/// against [`crate::catalog::arches`]. A label this crate spells any
/// other way — `olmo`, `olmo-2`, `OLMo2` — turns a real OLMo 2
/// checkpoint away at the boundary, because the check is an equality
/// against the set this generation contributes to.
///
/// It is also the one place OLMo 2 and OLMo 3 are told apart by name:
/// they share `LlamaLikeFacts`, they share `project::deployment`, and
/// what separates them for a program choosing a chat template is
/// exactly this string.
const ARCH: &str = "olmo2";

/// The published context ceiling, shared by all three rows.
///
/// 4096, from `allenai/OLMo-2-0425-1B-Instruct`,
/// `allenai/OLMo-2-1124-7B-Instruct` and
/// `allenai/OLMo-2-1124-13B-Instruct`, which all state
/// `max_position_embeddings: 4096`. One constant because the generation
/// does not disagree with itself — the 13B is the 7B with eight more
/// layers and wider heads, not a longer-context release.
///
/// Short by 2025 standards, and that is the point of stating it rather
/// than defaulting it: an OLMo 2 is a 4k model, OLMo 3 next door is a
/// 64k one, and a caller that assumed the newer number for both would
/// hand this generation sixteen times the context it was trained on.
/// Nothing in a fire reads this — a driver serving fewer positions is
/// serving correctly — but a guest program sizing a conversation
/// against it would be sizing against fiction.
const MAX_MODEL_LEN: u32 = 4_096;

/// The generation's rows.
///
/// Multi-head attention throughout — `kv_heads == q_heads`, no grouping
/// — which is unusual enough in 2024 to be worth stating three times
/// rather than defaulting once, and which the manifest carries as
/// `k_proj` being exactly as wide as `q_proj`.
pub const VARIANTS: &[Olmo2] = &[
    // allenai/OLMo-2-0425-1B-Instruct — the committed fixture.
    Olmo2 {
        id: "olmo-2-1b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 16,
            q_heads: 16,
            kv_heads: 16,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 100_352,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // allenai/OLMo-2-1124-7B-Instruct.
    Olmo2 {
        id: "olmo-2-7b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 11_008,
            vocab: 100_352,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // allenai/OLMo-2-1124-13B-Instruct.
    Olmo2 {
        id: "olmo-2-13b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 40,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 13_824,
            vocab: 100_352,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
];

/// This generation's contribution to [`crate::catalog::catalog`].
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Olmo2 {
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

impl Variant for Olmo2 {
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
            // Text only. OLMo 2 ships no encoder in any of the three
            // releases, so `Deployment::towers` is empty and the
            // driver's encode entry has nothing here to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// `author_dense`: the q/k norms and the post-norm pair are ordinary
    /// per-layer tensors as far as the CONTRACT is concerned — where
    /// they sit in the block is the trace's business, and the trace
    /// learns it from `norm_placement` on the same row.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_dense(builder)
            }
            // The registry this replaced held NO MLX row for olmo-2, and
            // the absence was a silence the caller read as "no
            // contract". Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "olmo-2: no MLX authoring pass exists for this family, so \
                 there is no name layout to author against",
            ),
        }
    }

    /// This row's text, for whichever backend asked.
    ///
    /// `rope_rescaled: false`: OLMo 2 states a plain `rope_theta` of
    /// 500 000 and no `rope_scaling` in any of the three releases. Its
    /// successor does state one, which is why the flag is a row's
    /// answer and not a generation-family assumption — see
    /// [`crate::olmo_3`].
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// `<|user|>\n … \n`, with `<|endoftext|>` closing the assistant
    /// turn — Tulu's format, which OLMo 2 Instruct is tuned on.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::Olmo2Instruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Olmo2 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("olmo-2-1b").shape, LlamaLikeFacts::olmo2_1b());
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and
    /// whether a tower is present — and every row is asserted because
    /// every row advertises. A ceiling of 0 is the shape the DEFAULT
    /// has, so it is checked separately: it would mean "this row does
    /// not say", and `tests/catalog_differential.rs` compares this
    /// number against the corpus's `max_position_embeddings`.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("dense olmo2 is servable")
                .advertised;
            assert_eq!(
                a.arch, "olmo2",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 4_096,
                "{}: all three releases state 4096",
                v.id
            );
            assert_ne!(a.max_model_len, 0, "{}: 0 is 'the row does not say'", v.id);
            assert!(!a.media_encode, "{}: OLMo 2 ships no encoder tower", v.id);
        }
    }

    /// The label is what `architectures[0]` reduces to under the
    /// worker's heuristic, which is the check a real checkpoint has to
    /// pass: `read_hf_config_defaults` lowercases the string, strips
    /// `ForCausalLM`, and refuses anything [`crate::catalog::arches`]
    /// does not list.
    #[test]
    fn the_label_is_what_architectures_reduces_to() {
        let stem = "Olmo2ForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert_eq!(
            ARCH,
            ARCH.to_lowercase(),
            "the worker compares against a lowercased stem"
        );
        assert!(!ARCH.contains('-'), "no separator survives the reduction");
    }

    /// Every row answers every question.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("dense olmo2 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.vocab, 100_352);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, 128, "{}", v.id);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert!(!ls.tied_embeddings, "no OLMo 2 row ties its head");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.window, -1, "{} layer {l} attends its whole context", v.id);
                assert_eq!(a.rope_theta, 500_000.0, "{}", v.id);
                assert_eq!(a.head_dim, 128);
                assert_eq!(a.kv_source, l as u32);
                assert_eq!(a.sm_scale, 1.0 / 128.0_f32.sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 3);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(ids, ["olmo-2-1b", "olmo-2-7b", "olmo-2-13b"]);
    }

    /// Post-norm reaches the deployment, and the manifest says the same
    /// thing in the checkpoint's own vocabulary: the pair that follows
    /// each sub-layer is required and `input_layernorm` is FORBIDDEN, so
    /// no pre-norm checkpoint can match an OLMo row and no OLMo
    /// checkpoint can match a llama one.
    #[test]
    fn the_norm_follows_the_sublayer_and_the_manifest_states_both_halves() {
        for v in VARIANTS {
            assert_eq!(v.shape.norm_placement, NormPlacement::Post, "{}", v.id);

            let d = v.deployment(Deployed::single()).expect("servable");
            assert_eq!(d.norm, crate::deployment::NormPlacement::Post, "{}", v.id);

            let m = v.manifest();
            let spec = |name: &str| {
                m.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .expect("stated one way or the other")
            };
            assert_eq!(
                spec("layer.{}.post_attention_layernorm").presence,
                crate::manifest::Presence::Required,
                "{}",
                v.id
            );
            assert_eq!(
                spec("layer.{}.post_attention_layernorm").extents,
                vec![u64::from(v.shape.hidden)],
                "{}",
                v.id
            );
            assert_eq!(
                spec("layer.{}.input_layernorm").presence,
                crate::manifest::Presence::Absent,
                "{} reads the residual stream raw",
                v.id
            );
        }
    }

    /// The q-norm's WIDTH is the whole projection, not one head — the
    /// fact the old derivation reconstructed by dividing a byte count.
    /// For every row here the two candidate widths differ, so the
    /// expectation genuinely discriminates.
    #[test]
    fn the_qk_norm_spans_the_projection_rather_than_one_head() {
        for v in VARIANTS {
            assert_eq!(v.shape.qk_norm, QkNorm::Global, "{}", v.id);
            let m = v.manifest();
            let q_norm = m
                .tensors
                .iter()
                .find(|t| t.name == "layer.{}.self_attn.q_norm")
                .expect("global qk-norm is a required tensor");
            assert_eq!(
                q_norm.presence,
                crate::manifest::Presence::Required,
                "{}",
                v.id
            );
            assert_eq!(
                q_norm.extents,
                vec![u64::from(v.shape.q_width())],
                "{}",
                v.id
            );
            assert_ne!(
                q_norm.extents,
                vec![u64::from(v.shape.head_dim)],
                "{}: a per-head gamma would be a different variant",
                v.id
            );
        }
    }

    /// Multi-head attention: as many KV heads as query heads, so `k_proj`
    /// is as wide as `q_proj` — the opposite of every GQA row in the
    /// lineage, and a fact a checkpoint publishes.
    #[test]
    fn attention_is_multi_head_rather_than_grouped() {
        for v in VARIANTS {
            assert_eq!(v.shape.kv_heads, v.shape.q_heads, "{}", v.id);
            assert_eq!(v.shape.kv_width(), v.shape.q_width(), "{}", v.id);
            let m = v.manifest();
            let extent = |name: &str| -> Vec<u64> {
                m.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .expect("stated")
                    .extents
                    .clone()
            };
            assert_eq!(
                extent("layer.{}.self_attn.k_proj"),
                extent("layer.{}.self_attn.q_proj")
            );
        }
    }

    /// `fused_qkv: false` is the binding fact: the dense join would
    /// re-fuse the raw projections, but OLMo's binder reads the
    /// per-projection views, so three GEMMs is what runs.
    #[test]
    fn the_projections_are_bound_unfused() {
        for v in VARIANTS {
            assert!(!v.shape.fused_qkv, "{}", v.id);
            assert!(!v.shape.qkv_bias, "{}", v.id);
        }
    }

    /// The template, exactly, including the part that is easy to get
    /// wrong: the user turn ends with a bare newline and no marker, and
    /// the assistant turn is what `<|endoftext|>` closes.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_the_tulu_marker_protocol() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|endoftext|>",
            "\n",
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
                "<|user|>\nHi\n",
                "{}",
                v.id
            );
            assert_eq!(
                tok.decode(&inst.cue(), false),
                "<|assistant|>\n",
                "{}",
                v.id
            );
            assert_eq!(
                tok.decode(&inst.assistant("Hi"), false),
                "<|assistant|>\nHi<|endoftext|>",
                "{}",
                v.id
            );
            assert!(
                !tok.decode(&inst.user("Hi"), false).contains("<|im_start|>"),
                "{}",
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

    #[cfg(feature = "forward")]
    #[test]
    fn every_row_traces_both_fire_classes() {
        use model_compiler::trace::FireClass;

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
