//! The Phi lineage.
//!
//! Two rows, an authoring pass of its own in `contract`, and a chat
//! template of its own in `chat`. Phi-3 is the generation that most
//! rewards having the three answers on ONE row, because two of them are
//! surprising and the third depends on both.
//!
//! # The head that is not a power of two
//!
//! Phi-3-mini is 3072 hidden over 32 heads, so its head is 96 wide, and
//! this build instantiates 64/128/256/512. The row states 96 — the
//! checkpoint's truth, and the width a tensor-parallel split must
//! respect — and [`project::round_up_attn_head_dim`] pads it to 128 for
//! the kernel. The two are separate answers to separate questions
//! (`load_shape().head_dim` is 96; `Deployment::attention[l].head_dim`
//! is 128), and the softmax scale follows the PADDED width because that
//! is the width the kernel reduces over.
//!
//! # The projection that ships fused and is bound unfused
//!
//! The checkpoint publishes one `qkv_proj`, and `fused_qkv` is still
//! `false`. That is not a contradiction: `contract::author_phi3`
//! splits the shipped tensor into banded q/k/v views, and the CUDA dense
//! join only re-fuses RAW source tensors — a contract-derived band is
//! not one — so the deployment binds three projections and the trace
//! writes three matmuls. `fused_qkv` is a fact about the BINDING, which
//! is why it belongs beside the author on one row rather than in a
//! second table keyed on the same string.

#[cfg(feature = "chat")]
pub mod chat;
#[cfg(feature = "contract")]
pub mod contract;

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

/// One Phi-3 checkpoint.
pub struct Phi3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers, with `head_dim` the checkpoint's own 96 for mini —
    /// never the padded 128.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency. Phi-3 kept Llama 2's 10k.
    pub rope_theta: f32,
    /// The RMSNorm epsilon — `1e-5`, Llama 2's, which is the lineage
    /// Phi-3's `LlamaForCausalLM`-shaped stack came out of. No tensor
    /// extent records it, so the row is the only place it can live once
    /// `config.json` stops being resident at launch.
    pub norm_eps: f32,
    /// Sliding-window width. 2047 for the 4k releases, and it is real:
    /// the config states it and nothing switches it off.
    pub window: i32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `phi3`, which is what the boundary derives: both releases state
/// `architectures: ["Phi3ForCausalLM"]` — the corpus's own
/// `microsoft--Phi-3-mini-4k-instruct.json` included — and the worker
/// lowercases that and strips `ForCausalLM` before checking the rest
/// against [`crate::catalog::arches`]. A label spelled `phi-3` or
/// `phi_3` turns a real Phi-3 checkpoint away at the boundary, because
/// the check is an equality against the set this generation contributes
/// to and no hyphen or underscore survives the reduction.
const ARCH: &str = "phi3";

/// The published context ceiling, shared by both rows.
///
/// 4096, from `microsoft/Phi-3-mini-4k-instruct` and
/// `microsoft/Phi-3-medium-4k-instruct`, which both state
/// `max_position_embeddings: 4096`.
///
/// One constant only because of what the row list below is: BOTH rows
/// are 4k releases. The 128k siblings exist and publish 131 072, and
/// they have no row here — not because their ceiling is unknown but
/// because their tensors are identical to the 4k ones and their rope is
/// LongRoPE, which [`RopeKind`] has no value for. So this constant is
/// safe today for the reason the rows are two rather than four, and the
/// day a 128k row can be identified it must state its own number rather
/// than inherit this one. The test below holds the pairing.
const MAX_MODEL_LEN: u32 = 4_096;

/// The generation's rows.
///
/// Two, and the absence of a third is deliberate. `Phi-3-mini-128k`
/// publishes the SAME tensors at the same extents as `phi-3-mini-4k` —
/// it differs in `sliding_window` and in shipping LongRoPE's
/// per-dimension `short_factor`/`long_factor` ladders — so a row for it
/// could not be identified apart from the 4k row, and its rope is a kind
/// [`RopeKind`] has no value for. A row that stated `Standard` or `Yarn`
/// for it would be stating something false about how its positions are
/// computed, and a wrong constant is a silently wrong model.
pub const VARIANTS: &[Phi3] = &[
    // microsoft/Phi-3-mini-4k-instruct — the corpus's own
    // `microsoft--Phi-3-mini-4k-instruct.json`, the committed fixture,
    // and the row `tests/catalog_differential.rs` pins. MHA: 32 query
    // heads and 32 KV heads, no grouping at all.
    Phi3 {
        id: "phi-3-mini-4k",
        shape: LlamaLikeFacts {
            hidden: 3072,
            layers: 32,
            q_heads: 32,
            kv_heads: 32,
            head_dim: 96,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 32_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        window: 2047,
    },
    // microsoft/Phi-3-medium-4k-instruct. GQA at last (40 q / 10 kv),
    // and a head that needs no padding: 5120 / 40 = 128.
    Phi3 {
        id: "phi-3-medium-4k",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 10,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 17_920,
            vocab: 32_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        window: 2047,
    },
];

/// This generation's contribution to [`crate::catalog::catalog`].
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Phi3 {
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

impl Variant for Phi3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// 96 for mini, and this is the method where that matters most: a
    /// tensor-parallel row split computed against the padded 128 lands
    /// mid-head and produces a contract that compiles and a model that
    /// is wrong.
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
            // Text only. Phi-3-vision is a different package with a
            // different `architectures[0]` and no row here, so nothing
            // this generation deploys carries a tower for the driver's
            // encode entry to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The generation's own pass, because the checkpoint's fused
    /// `qkv_proj` has to be banded into three views before anything else
    /// can bind — the one place in this lineage where the author is not
    /// the shared dense one.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        contract::author_phi3(builder)
    }

    /// This row's text, for whichever backend asked.
    ///
    /// `rope_rescaled: false`, and the reason is which ROWS are here:
    /// both are the 4k releases (`phi-3-mini-4k`, `phi-3-medium-4k`),
    /// whose configs state a plain `rope_theta` and no scaling. The
    /// 128k releases extend by LongRoPE — a per-channel factor table
    /// that is a different statement entirely — and they have no row in
    /// this table, so nothing here may claim to serve them.
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// Phi-3's own markers: `<|user|>\n … <|end|>\n`. Close enough to
    /// ChatML to be mistaken for it by a reader, and different enough to
    /// derail the tune — the `<|end|>` the model was trained to emit is
    /// not `<|im_end|>`, so the ChatML fallback also broke the STOP set.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::Phi3Instruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Phi3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and
    /// whether a tower is present. Both rows are asserted because both
    /// advertise, and a ceiling of 0 is checked apart from the value
    /// because 0 is what the DEFAULT carries: it would mean "this row
    /// does not say", and `tests/catalog_differential.rs` compares this
    /// number against the corpus's `max_position_embeddings`.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("dense phi-3 is servable")
                .advertised;
            assert_eq!(
                a.arch, "phi3",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(a.max_model_len, 4_096, "{}: a 4k release states 4096", v.id);
            assert_ne!(a.max_model_len, 0, "{}: 0 is 'the row does not say'", v.id);
            assert!(!a.media_encode, "{}: no Phi-3 row carries a tower", v.id);
        }
    }

    /// THE SHARED CEILING IS SAFE ONLY BECAUSE OF WHICH RELEASES HAVE
    /// ROWS, and this is where that dependency is written down.
    ///
    /// Every row here is a `-4k` release. Their `-128k` siblings
    /// publish 131 072 and cannot be told apart from these by tensor
    /// extents, which is why they have no row (see [`VARIANTS`]). If
    /// one ever gains a row, this test fails first, and the fix is a
    /// per-row `max_model_len` field rather than a second constant —
    /// the ceiling would then be the one thing separating two rows with
    /// identical shapes.
    #[test]
    fn every_row_that_shares_the_ceiling_is_a_4k_release() {
        for v in VARIANTS {
            assert!(
                v.id.ends_with("-4k"),
                "{}: a row that is not a 4k release cannot share MAX_MODEL_LEN",
                v.id
            );
            assert_eq!(
                v.window, 2047,
                "{}: the 4k releases are the ones with a real sliding window; the 128k \
                 siblings switch it off",
                v.id
            );
        }
        assert_eq!(
            VARIANTS.len(),
            2,
            "two 4k rows, and the 128k pair deliberately absent"
        );
    }

    /// The label is what `architectures[0]` reduces to under the
    /// worker's heuristic, which is the check a real checkpoint has to
    /// pass: `read_hf_config_defaults` lowercases the string, strips
    /// `ForCausalLM`, and refuses any stem [`crate::catalog::arches`]
    /// does not list.
    #[test]
    fn the_label_is_what_architectures_reduces_to() {
        let stem = "Phi3ForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert!(
            !ARCH.contains('-') && !ARCH.contains('_'),
            "no separator survives it"
        );
        assert_eq!(
            ARCH,
            ARCH.to_lowercase(),
            "the worker compares against a lowercased stem"
        );
    }

    /// The fixture and the row are the same measurement — including the
    /// 96, which the fixture's own doc calls "the logical 96".
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("phi-3-mini-4k").shape, LlamaLikeFacts::phi3_mini());
    }

    /// Every row answers every question.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("dense phi3 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.vocab, 32_064);
            assert_eq!(d.norm, crate::deployment::NormPlacement::Pre);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert!(!ls.tied_embeddings, "no Phi-3 row ties its head");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.window, 2047, "{} layer {l} slides", v.id);
                assert_eq!(a.rope_theta, 10_000.0, "{}", v.id);
                assert_eq!(a.kv_source, l as u32);
                assert_eq!(a.sm_scale, 1.0 / (a.head_dim as f32).sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 2);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(ids, ["phi-3-mini-4k", "phi-3-medium-4k"]);
    }

    /// The row keeps 96 and the kernel gets 128, and both are stated in
    /// the place that needs them. The medium row is the control: its 128
    /// pads to itself, so the two answers coincide there and the
    /// difference below is the padding and nothing else.
    #[test]
    fn the_mini_head_is_ninety_six_and_pads_to_one_twenty_eight() {
        let mini = row("phi-3-mini-4k");
        assert_eq!(
            mini.shape.hidden / mini.shape.q_heads,
            96,
            "96 is the checkpoint's own"
        );
        assert_eq!(
            mini.load_shape().head_dim,
            96,
            "the contract gets the true width"
        );

        let d = mini.deployment(Deployed::single()).expect("servable");
        assert_eq!(d.shape.head_dim, 96, "the geometry still states the truth");
        assert_eq!(
            d.shape.head_dim_kernel, 128,
            "the kernel gets the instantiated width"
        );
        assert_eq!(d.attention[0].head_dim, 128);
        assert_eq!(
            d.attention[0].sm_scale,
            1.0 / 128.0_f32.sqrt(),
            "the scale follows the width the kernel reduces over"
        );

        let medium = row("phi-3-medium-4k")
            .deployment(Deployed::single())
            .expect("servable");
        assert_eq!(medium.shape.head_dim, 128);
        assert_eq!(medium.shape.head_dim_kernel, 128, "nothing to pad");
        assert_eq!(medium.attention[0].head_dim, 128);
    }

    /// The manifest's q/k/v extents are the row's own arithmetic at the
    /// TRUE head width, which is what a checkpoint publishes: mini's
    /// `q_proj` is `[32 * 96, 3072]` = `[3072, 3072]`, not `[4096, …]`.
    /// A manifest written against the padded width would reject every
    /// Phi-3 checkpoint there is.
    #[test]
    fn the_manifest_is_written_at_the_unpadded_width() {
        let mini = row("phi-3-mini-4k");
        let m = mini.manifest();
        let extent = |name: &str| -> Vec<u64> {
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(mini.shape.q_width(), 3072);
        assert_eq!(
            mini.shape.kv_width(),
            3072,
            "MHA: as many KV heads as query heads"
        );
        assert_eq!(extent("layer.{}.self_attn.q_proj"), vec![3072, 3072]);
        assert_eq!(extent("layer.{}.self_attn.k_proj"), vec![3072, 3072]);
        assert_eq!(extent("layer.{}.self_attn.o_proj"), vec![3072, 3072]);
        assert_eq!(extent("lm_head"), vec![32_064, 3072]);

        let medium = row("phi-3-medium-4k");
        assert_eq!(medium.shape.q_width(), 5120);
        assert_eq!(medium.shape.kv_width(), 1280, "GQA: 10 KV heads of 128");
    }

    /// `fused_qkv: false` on a checkpoint that ships a fused
    /// projection — the binding fact, and the one a shape-only table
    /// could never have held next to the author that causes it.
    #[test]
    fn the_binding_is_unfused_although_the_checkpoint_is_not() {
        for v in VARIANTS {
            assert!(!v.shape.fused_qkv, "{}", v.id);
            let m = v.manifest();
            for name in [
                "layer.{}.self_attn.q_proj",
                "layer.{}.self_attn.k_proj",
                "layer.{}.self_attn.v_proj",
            ] {
                let spec = m.tensors.iter().find(|t| t.name == name).expect("stated");
                assert_eq!(
                    spec.presence,
                    crate::manifest::Presence::Required,
                    "{} {name}",
                    v.id
                );
            }
        }
    }

    /// The template, exactly, and the stop set with it: `<|end|>` is
    /// what the tune emits, and it is not `<|im_end|>`.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_phi3s_own_markers() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|end|>",
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
                "<|user|>\nHi<|end|>\n",
                "{}",
                v.id
            );
            assert_eq!(
                tok.decode(&inst.cue(), false),
                "<|assistant|>\n",
                "{}",
                v.id
            );
            let stops = inst.seal();
            assert!(
                !stops.is_empty(),
                "{}: <|end|> is in the vocabulary above",
                v.id
            );
            assert!(
                !tok.decode(&stops, false).contains("<|im_end|>"),
                "{}: the ChatML fallback's stop set was the wrong one",
                v.id
            );
        }
    }

    /// Every row has an author, and it is Phi-3's own rather than the
    /// lineage's dense one.
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
