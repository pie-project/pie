//! Qwen 3 — the exemplar row.
//!
//! Every other llama-lineage generation in this tree is this file with
//! different numbers and a different chat template, which is the claim
//! the catalog makes: a generation is not a code path, it is a set of
//! ROWS plus the two answers a shape cannot give.
//!
//! Read `VARIANTS` and then `impl Variant` below; there is nothing else
//! in the module, and that is the point. What used to be here was
//! nothing at all — `qwen_3` was an absent module, because "everything
//! it held was ChatML". Its numbers lived in `deployment_cuda`'s
//! `FACTS_ROWS` under the string `"qwen3"`, its authoring lived in
//! `contract::HF_ROWS` under the same string, and its chat lived in
//! `instruct::create` under `"Qwen3ForCausalLM"`. Three tables, three
//! keys, no one holding them together.

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

/// One Qwen 3 checkpoint.
///
/// A newtype over the family shape rather than an `impl Variant for
/// LlamaLikeFacts`, and the reason is `chat` and `author`: twelve
/// generations share `LlamaLikeFacts`, and a blanket impl would have to
/// answer "which template speaks for this?" from numbers that do not
/// know. The shape is stated once; the generation adds what the shape
/// cannot hold.
pub struct Qwen3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency. NOT in `LlamaLikeFacts`, because the
    /// tracer takes it per-layer through `Deployment` rather than
    /// through the shape.
    pub rope_theta: f32,
    /// Sliding-window width, `-1` for full attention. Qwen 3 dense
    /// ships `sliding_window: null`.
    pub window: i32,
}

/// RMSNorm epsilon, shared by the whole generation.
///
/// Stated once rather than per row because every published Qwen 3
/// config carries the same `1e-6` — a generation-level constant is an
/// honest way to say that, where eight copies of a literal invite one
/// of them to drift. The qwen-2 generation before it used `1e-5`, and
/// that generation states its own.
const NORM_EPS: f32 = 1e-6;

/// The family label a GUEST PROGRAM matches on.
///
/// Coarser than an id on purpose, and that is the one place a family
/// string survives the refactor: `engine`'s `model.arch_name()` is a
/// host function inferlets call, and `VisionArch::from_arch_name`
/// selects an image front-end from it. What it is NOT any more is a
/// dispatch key — nothing in this crate matches on it — so the string
/// that reached three registries and disagreed with itself is now a
/// label a row hands out.
///
/// This generation is TEXT-ONLY, and this label is why that used to be
/// untrue in practice: the front-end table asked whether an arch
/// *contained* `qwen3`, which every row here satisfies, so all eight
/// sizes claimed the Qwen3-VL image front-end that lives one directory
/// over in `qwen_3_5`. It now matches the whole label and this one
/// selects nothing, which is the honest answer for a stack with no
/// vision tower.
///
/// Stated rather than derived from `architectures[0]`. That derivation
/// stripped `ForCausalLM` and `ForConditionalGeneration` from the front
/// of a lowercase name and is why `Gemma4ForConditionalGeneration`
/// reached a table row it did not belong in.
const ARCH: &str = "qwen3";

/// The published context ceiling, shared by every Qwen 3 release.
///
/// One constant because every published config states `40960`,
/// including the two MoE sizes. A training-time fact rather than a
/// deployment one — nothing in a fire reads it, and a driver serving a
/// shorter context is serving correctly.
const MAX_MODEL_LEN: u32 = 40_960;

/// The generation's rows.
///
/// `const`, which is the whole architecture: identity is in the binary,
/// so the three questions have one answer and the answer is linked.
/// Every field is stated even when it is zero, for the reason the
/// fixtures give — a row is a MEASUREMENT of a real checkpoint, and
/// "this one has no experts" is part of the measurement.
pub const VARIANTS: &[Qwen3] = &[
    // Qwen/Qwen3-0.6B. head_dim 128 is stated by the config and is NOT
    // hidden/heads (1024/16 = 64): the config is the authority and the
    // derivation that assumed the quotient was wrong for this row.
    Qwen3 {
        id: "qwen3-0.6b",
        shape: LlamaLikeFacts {
            hidden: 1024,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 3072,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-1.7B — the same 28-layer geometry at twice the width.
    Qwen3 {
        id: "qwen3-1.7b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 28,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 6144,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-4B.
    Qwen3 {
        id: "qwen3-4b",
        shape: LlamaLikeFacts {
            hidden: 2560,
            layers: 36,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 9728,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-8B — the first untied head in the generation.
    Qwen3 {
        id: "qwen3-8b",
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
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-14B.
    Qwen3 {
        id: "qwen3-14b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 40,
            q_heads: 40,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 17_408,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-32B.
    Qwen3 {
        id: "qwen3-32b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 64,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 25_600,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-30B-A3B — a MIXTURE, and still llama-like: the
    // attention is unchanged and only the block between the two norms
    // differs. This is the row `deployment_cuda` got wrong. Its author
    // column said `author_qwen3_5_moe` (a gated-delta-net mixture with
    // `GATE_SECOND`), its facts column said `llama_like_facts_from_hf`
    // (dense), and both were reached by the string `"qwen3_moe"`. One
    // row cannot hold two answers to one question.
    Qwen3 {
        id: "qwen3-30b-a3b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 48,
            q_heads: 32,
            kv_heads: 4,
            head_dim: 128,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 768,
            shared_intermediate: 0,
            intermediate: 6144,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
    // Qwen/Qwen3-235B-A22B.
    Qwen3 {
        id: "qwen3-235b-a22b",
        shape: LlamaLikeFacts {
            hidden: 4096,
            layers: 94,
            q_heads: 64,
            kv_heads: 4,
            head_dim: 128,
            n_experts: 128,
            experts_per_token: 8,
            moe_intermediate: 1536,
            shared_intermediate: 0,
            intermediate: 12_288,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::PerHead,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 1_000_000.0,
        window: -1,
    },
];

/// This generation's contribution to [`crate::catalog::catalog`].
///
/// The `OnceLock` is only the widening from `&Qwen3` to `&dyn Variant`;
/// the rows themselves are `const` and in `.rodata`.
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Qwen3 {
    /// The scalars this row states, read ONCE.
    ///
    /// Both [`Variant::deployment`] and [`Variant::trace`] take it. They
    /// used to read `rope_theta`, `norm_eps` and `window` off `self`
    /// separately — the same three fields, spelled twice, with nothing
    /// holding the two spellings together.
    fn row(&self) -> project::RowScalars {
        project::RowScalars {
            rope_theta: self.rope_theta,
            norm_eps: NORM_EPS,
            window: self.window,
            rope_rescaled: false,
            // TRUE for every qwen3 row, dense and routed. `Qwen3-30B-A3B`
            // and `Qwen3-235B-A22B` both publish `"norm_topk_prob": true`,
            // against a `Qwen3MoeConfig` class default of false.
            norm_topk_prob: true,
        }
    }
}

impl Variant for Qwen3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The unpadded head dim, which is the one an authoring pass needs:
    /// a tensor-parallel row split that lands mid-head produces a
    /// contract that compiles and a model that is wrong.
    fn load_shape(&self) -> LoadShape {
        if self.shape.n_experts == 0 {
            LoadShape::dense(
                self.shape.layers,
                self.shape.head_dim,
                self.shape.tied_embeddings,
            )
        } else {
            LoadShape::mixture(
                self.shape.layers,
                self.shape.head_dim,
                self.shape.n_experts,
                self.shape.tied_embeddings,
            )
        }
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
            // No tower. Qwen3-VL has one, and it is deliberately not
            // this: its tower writes into the fire's hidden rows rather
            // than handing host rows back, so it is an in-fire path and
            // not an encode one.
            media_encode: false,
        };
        Ok(deployment)
    }

    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_llama_like(builder)
            }
            // The registry this replaced held an MLX row for `qwen3` and `qwen3_moe`,
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
    /// The epsilon is the generation CONSTANT and not a row field, for
    /// the reason [`NORM_EPS`] gives: every published Qwen-3 config
    /// states `1e-6`, and a row that could hold a different one would
    /// be a row that can disagree with the generation for no reason a
    /// checkpoint gave it.
    ///
    /// `rope_rescaled: false`: Qwen 3 publishes no `rope_scaling` in
    /// any release this table holds, dense or mixture. Its long-context
    /// story is YaRN applied at serve time, which is not a fact the
    /// weights carry.
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// ChatML, and stated rather than fallen through to.
    ///
    /// The old `instruct::create` reached the same constructor by a
    /// `_ =>` arm, so EVERY unlisted architecture got it too. The arm
    /// is gone; this row says ChatML because Qwen 3 is ChatML.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(crate::shared::chatml::QwenInstruct::new(
            tokenizer,
            crate::shared::chatml::QWEN_CHATML,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. `qwen3_0_6b()` was committed as a
    /// measurement of `Qwen/Qwen3-0.6B`; so is the row.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        let row = VARIANTS
            .iter()
            .find(|v| v.id == "qwen3-0.6b")
            .expect("row present");
        assert_eq!(row.shape, LlamaLikeFacts::qwen3_0_6b());
    }

    /// The mixture row is a mixture. This is the disagreement the
    /// catalog exists to make impossible: one string reached an author
    /// that wrote a gated-delta-net mixture and a derivation that
    /// produced dense facts.
    #[test]
    fn the_mixture_row_is_a_mixture_in_the_only_place_it_is_stated() {
        let moe = VARIANTS
            .iter()
            .find(|v| v.id == "qwen3-30b-a3b")
            .expect("row present");
        assert_ne!(moe.shape.n_experts, 0);
        assert_ne!(moe.shape.moe_intermediate, 0);
        // And the manifest agrees, because it is a projection of the
        // same field: a mixture ships a router, a dense block does not.
        let binding = moe.manifest();
        let names: Vec<&str> = binding.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("mlp.gate")), "{names:?}");
    }

    /// Every row answers every question, which is what having no
    /// default bodies buys.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("dense qwen3 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(v.manifest().layers, v.shape.layers);
            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert_eq!(ls.tied_embeddings, v.shape.tied_embeddings);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);
        }
    }

    /// Qwen 3 states `head_dim: 128` while `hidden / q_heads` is 64 for
    /// the 0.6B — the config is the authority and a derivation that
    /// assumed the quotient was wrong for this row. The row keeps the
    /// stated value and `Deployment` keeps the kernel's.
    #[test]
    fn head_dim_is_the_stated_one_not_the_quotient() {
        let row = VARIANTS
            .iter()
            .find(|v| v.id == "qwen3-0.6b")
            .expect("row present");
        assert_eq!(row.shape.hidden / row.shape.q_heads, 64);
        assert_eq!(row.load_shape().head_dim, 128);
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` in
    /// `driver-cuda`: a driver kept a whole parsed `config.json` alive
    /// for the life of a load so it could answer `model_type`,
    /// `max_position_embeddings` and "does this ship a tower" at
    /// capability time.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("qwen3 deploys")
                .advertised;
            assert_eq!(
                a.arch, "qwen3",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 40_960,
                "{}: every published Qwen 3 states it",
                v.id
            );
            assert!(
                !a.media_encode,
                "{}: no Qwen 3 ships a tower the encode entry serves",
                v.id
            );
        }
    }

    /// The label is a FAMILY and the id is a MODEL, and the difference
    /// is the reason the catalog exists.
    #[test]
    fn one_family_label_covers_eight_distinct_rows() {
        let labels: std::collections::BTreeSet<&str> = VARIANTS
            .iter()
            .map(|v| v.deployment(Deployed::single()).unwrap().advertised.arch)
            .collect();
        assert_eq!(labels.len(), 1, "one label");
        let ids: std::collections::BTreeSet<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), VARIANTS.len(), "eight ids");
        assert!(
            ids.len() > labels.len(),
            "`qwen3` named twelve checkpoints of six shapes; that is what made it \
             unusable as a dispatch key and is why nothing dispatches on it now"
        );
    }
}
