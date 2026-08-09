//! The Qwen3.5 lineage: the GATED-DELTA-NET hybrid — full attention
//! interleaved with linear attention — dense and MoE.
//!
//! This is where the disagreement that motivated the catalog lands.
//! `"qwen3_moe"` was one string reaching two tables: `contract::HF_ROWS`
//! sent it to `contract::author_qwen3_5_moe`, which writes a
//! gated-delta-net mixture with `GATE_SECOND`, and
//! `deployment_cuda::FACTS_ROWS` sent it to `llama_like_facts_from_hf`,
//! which produces a DENSE llama-like stack with no GDN block anywhere in
//! it. Both answers were reachable, neither was checked against the
//! other, and the checkpoint that would have exposed it is a Qwen3
//! mixture — whose row now lives in [`crate::qwen_3`] and is llama-like,
//! because that is what it is.
//!
//! What is here is the family that author was actually written for. A
//! row states its MLP kind, and the same statement chooses the author,
//! sizes the manifest's router and fills the deployment's expert width —
//! so there is no second place for a different answer to come from.
//!
//! Chat: the qwen3 lineage's ChatML, stated per row rather than reached
//! through `instruct::create`'s `_ =>` arm.

// `Arc` reaches this module only through `Variant::chat`, so the
// import carries that method's gate. It used to ride along with
// `OnceLock`, which `rows()` needed unconditionally until
// `rows_of!` absorbed it.
#[cfg(feature = "chat")]
use std::sync::Arc;

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::manifest::Manifest;

use spec::{
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MlpKind, Qwen35MoeMlpFacts,
};

use model_compiler::trace::NormVariant;

#[cfg(feature = "contract")]
pub mod contract;

/// The hybrid's forward pass: GDN layers, full-attention layers, MoE or
/// dense MLP, composed by a static layer schedule.
///
/// Written in `model-compiler`'s tracing eDSL: ordinary Rust that runs at
/// model-load time with the checkpoint's facts in hand and records what one
/// pass computes. The traced form is what a driver executes.
pub mod forward;

/// What a Qwen3.5 checkpoint IS — ungated, because a row is written in
/// these words and a row must answer under every aspect.
pub mod spec;

/// What those numbers imply: a manifest, a deployment, a trace.
pub mod project;

/// One Qwen3.5 checkpoint.
///
/// A newtype over [`Qwen35HybridFacts`] for [`crate::qwen_3::Qwen3`]'s
/// reason — the shape cannot say which template speaks for it — plus one
/// of its own: rope's base frequency is carried per LAYER through
/// [`crate::deployment::Deployment`] and never entered the shape.
pub struct Qwen35 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: Qwen35HybridFacts,
    /// Rope's base frequency (`rope_parameters.rope_theta`); `1e7`
    /// across the whole lineage, and stated per row anyway because a
    /// value every row shares today is still a measurement of each.
    pub rope_theta: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-6` across the generation.
    /// A CONSTANT of the checkpoint that no tensor extent carries, so
    /// there is nowhere but a row for it to live — the launch path used
    /// to read it off a resident `HfConfig`.
    pub norm_eps: f32,
}

impl Qwen35 {
    /// Whether this row's MLP is a mixture.
    ///
    /// ONE predicate, read three times — by [`Variant::author`], by
    /// [`Variant::load_shape`] and by the manifest's router row. That is
    /// the repair: the defect this catalog exists for was two tables
    /// answering "is `qwen3_moe` a mixture" differently, and two
    /// readings of one field cannot.
    #[must_use]
    pub const fn is_mixture(&self) -> bool {
        matches!(self.shape.mlp, Qwen35MlpKind::Moe(_))
    }

    /// Routed experts, 0 when dense.
    #[must_use]
    pub const fn experts(&self) -> u32 {
        match &self.shape.mlp {
            Qwen35MlpKind::Moe(moe) => moe.num_experts,
            Qwen35MlpKind::Dense { .. } => 0,
        }
    }
}

/// The full-attention block every Qwen3.5 row shares but its widths.
///
/// A `const fn` rather than five repetitions, and it takes exactly the
/// three numbers that differ. `head_dim` is 256 in every published
/// config; the rotary width is 64 in every one (`partial_rotary_factor
/// = 0.25` of 256, RESOLVED — a row states the channel count, not the
/// factor, because the factor is config parsing); the output-gated q
/// bank is the family's shape; and every norm in the block folds
/// Gemma's `(1 + w)`.
const fn attn(hidden: u32, q_heads: u32, kv_heads: u32) -> Qwen35FullAttnFacts {
    Qwen35FullAttnFacts {
        hidden,
        q_heads,
        kv_heads,
        head_dim: 256,
        rotary_dim: 64,
        // The default binding: `PIE_QWEN35_FUSED_FULL_ATTN_QGKV` unset,
        // so the load binds three projections and the trace writes three
        // matmuls.
        fused_qkv: false,
        norm_variant: NormVariant::Gemma,
    }
}

/// The GDN block every Qwen3.5 row shares but its head counts.
///
/// `linear_key_head_dim` and `linear_value_head_dim` are 128 in every
/// published config and `linear_conv_kernel_dim` is 4; what a row varies
/// is how many key heads and how many value heads share them — 16:16 on
/// the 0.8B, 16:32 in the middle of the range, 16:48 on Qwen3.6-27B.
const fn gdn(hidden: u32, key_heads: u32, value_heads: u32) -> Qwen35GdnFacts {
    Qwen35GdnFacts {
        hidden,
        key_heads,
        value_heads,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel: 4,
        // `PIE_QWEN35_FUSED_GDN_PROJ` is off by default: the checkpoint
        // ships four projections and the trace writes four matmuls.
        fused_in_proj: false,
        norm_variant: NormVariant::Gemma,
    }
}

/// The family label a GUEST PROGRAM matches on.
///
/// `qwen3_5`, underscore and all, because that is what the checkpoints
/// state and what the boundary derives: `model_type` is `qwen3_5` (and
/// `qwen3_5_text` / `qwen3_5_moe_text` for the nested towers), and
/// `architectures[0]` is `Qwen3_5ForConditionalGeneration`, which the
/// worker's stem heuristic reduces to exactly this before checking it
/// against [`crate::catalog::arches`]. Prettifying it to `qwen3.5` would
/// make that check refuse every checkpoint this table claims.
///
/// It also carries a second decision: `VisionArch::from_arch_name`
/// selects the Qwen3-VL image front-end on this label WHOLE — so
/// Qwen3.6-27B, whose row lives here because it is a Qwen3.5 by shape,
/// reaches the same front-end its siblings do, and the text-only `qwen3`
/// generation next door does not.
const ARCH: &str = "qwen3_5";

/// The published context ceiling, shared by every row of the lineage.
///
/// One constant because all five committed corpus configs state
/// `max_position_embeddings: 262144` under `text_config` — the 0.8B pair,
/// the 4B, the 9B, the 35B-A3B mixture and both Qwen3.6-27B builds — so
/// there is nothing here for a row field to vary. Read from the corpus
/// rather than from a model card, which is why this is the one number in
/// the table `tests/catalog_differential.rs` can check against the
/// checkpoint's own words.
///
/// A training-time fact rather than a deployment one: nothing in a fire
/// reads it, and a driver serving a shorter context is serving
/// correctly.
const MAX_MODEL_LEN: u32 = 262_144;

/// The generation's rows.
///
/// `const`, so the identity is in `.rodata` and the three questions have
/// one answer each. Every number below is read from a committed corpus
/// config (`crates/driver-cuda/tests/hf_config_dump/corpus/`) under
/// `text_config` — these are `*ForConditionalGeneration` checkpoints
/// whose text tower is nested, which is a PACKAGING fact the manifest's
/// prefix stripping already divides out.
pub const VARIANTS: &[Qwen35] = &[
    // Qwen/Qwen3.5-0.8B-Base — AND Qwen/Qwen3.5-0.8B, the instruct tune.
    //
    // ONE ROW, because `Qwen--Qwen3.5-0.8B.json` and
    // `Qwen--Qwen3.5-0.8B-Base.json` are byte-identical: same widths,
    // same schedule, same tie, so the two checkpoints publish the same
    // tensors at the same extents and NOTHING a manifest can ask tells
    // them apart. Two rows would make every 0.8B checkpoint
    // `Unmatched::Ambiguous` — the honest report of a table defect, and
    // it would be a table defect: the difference between these two is
    // their WEIGHTS, and a row is a shape. The base model names it.
    Qwen35 {
        id: "qwen3.5-0.8b-base",
        shape: Qwen35HybridFacts {
            layers: 24,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Gemma,
            attn: attn(1024, 8, 2),
            gdn: gdn(1024, 16, 16),
            mlp: Qwen35MlpKind::Dense { intermediate: 3584 },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
    // Qwen/Qwen3.5-4B. The first row where the value heads outnumber the
    // key heads — 32 over 16, a 2:1 GDN share.
    Qwen35 {
        id: "qwen3.5-4b",
        shape: Qwen35HybridFacts {
            layers: 32,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: true,
            norm_variant: NormVariant::Gemma,
            attn: attn(2560, 16, 4),
            gdn: gdn(2560, 16, 32),
            mlp: Qwen35MlpKind::Dense { intermediate: 9216 },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
    // Qwen/Qwen3.5-9B — the 4B's layer and head counts at a wider
    // hidden, and the first UNTIED head in the generation.
    Qwen35 {
        id: "qwen3.5-9b",
        shape: Qwen35HybridFacts {
            layers: 32,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: attn(4096, 16, 4),
            gdn: gdn(4096, 16, 32),
            mlp: Qwen35MlpKind::Dense {
                intermediate: 12_288,
            },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
    // Qwen/Qwen3.5-35B-A3B (`model_type: qwen3_5_moe_text`) — the
    // mixture this generation's MoE author was written for. It ships NO
    // `intermediate_size` at all: there is no dense block to size, only
    // 256 experts of 512 and one shared expert of the same width.
    Qwen35 {
        id: "qwen3.5-35b-a3b",
        shape: Qwen35HybridFacts {
            layers: 40,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: attn(2048, 16, 2),
            gdn: gdn(2048, 16, 32),
            mlp: Qwen35MlpKind::Moe(Qwen35MoeMlpFacts {
                hidden: 2048,
                num_experts: 256,
                top_k: 8,
                moe_intermediate: 512,
                shared_expert_intermediate: 512,
                norm_variant: NormVariant::Gemma,
            }),
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
    // Qwen/Qwen3.6-27B — AND Qwen/Qwen3.6-27B-FP8.
    //
    // ONE ROW. `Qwen--Qwen3.6-27B-FP8.json` differs from
    // `Qwen--Qwen3.6-27B.json` by exactly one key: an added
    // `quantization_config`. Every width, the layer count, the schedule
    // and the tie are identical, because an FP8 build is the same model
    // stored differently — the manifest compares LOGICAL extents with
    // the packing undone, and `crate::shared::policy` is where a decision about
    // how to READ those weights belongs. A second row would record the
    // file format in the identity.
    //
    // Qwen3.6 is a Qwen3.5 by shape (`model_type: qwen3_5_text`), so it
    // is a row of this generation and not a module of its own.
    Qwen35 {
        id: "qwen3.6-27b",
        shape: Qwen35HybridFacts {
            layers: 64,
            full_attn_interval: 4,
            vocab: 248_320,
            tied_embeddings: false,
            norm_variant: NormVariant::Gemma,
            attn: attn(5120, 24, 4),
            gdn: gdn(5120, 16, 48),
            mlp: Qwen35MlpKind::Dense {
                intermediate: 17_408,
            },
        },
        rope_theta: 1e7,
        norm_eps: 1e-6,
    },
];

crate::rows_of!(Qwen35);

impl Variant for Qwen35 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The UNPADDED head dim (256 here, which is also the instantiated
    /// one), and `mamba_groups: 0` — a gated-delta-net block has no
    /// Mamba B/C bands to fuse. Its conv bank packs `[K | K | V]`, and
    /// that the shard must fall on those block boundaries is a fact the
    /// contract states directly (`gdn_kkv_blocked_shards`).
    fn load_shape(&self) -> LoadShape {
        if self.is_mixture() {
            LoadShape::mixture(
                self.shape.layers,
                self.shape.attn.head_dim,
                self.experts(),
                self.shape.tied_embeddings,
            )
        } else {
            LoadShape::dense(
                self.shape.layers,
                self.shape.attn.head_dim,
                self.shape.tied_embeddings,
            )
        }
    }

    /// Servable: the paged KV store this build has, plus the recurrent
    /// slabs `RecurrentStateCache` allocates. No refusal — the GDN store
    /// is BUILT, which is the whole difference between this family and
    /// the MLA lineage's, and it is now a difference the two rows STATE
    /// rather than one a driver discovers at its first fire.
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment = project::deployment(&self.shape, self.rope_theta, self.norm_eps);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // No tower the ENCODE entry serves. These are
            // `*ForConditionalGeneration` packages and the label above
            // does select a Qwen3-VL image front-end — but that tower
            // writes into the fire's hidden rows rather than handing host
            // rows back, so it is an in-fire path. `Deployment::towers`
            // is empty here and `driver-cuda`'s encode entry binds
            // gemma-4's kernels only; advertising `true` would promise an
            // entry point that refuses.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The author the row's own MLP kind chooses.
    ///
    /// This is the seam the catalog closes. `author_qwen3_5_moe` writes
    /// the per-expert stacks and the shared-expert join with
    /// `GATE_SECOND = true`, and it used to be reachable from
    /// `"qwen3_moe"` — a string whose FACTS came from the dense
    /// llama-like derivation. Here the mixture that selects the author
    /// is the mixture the manifest expects and the mixture the
    /// deployment sizes, because it is one field read three times.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf if self.is_mixture() => {
                contract::author_qwen3_5_moe(builder)
            }
            crate::shared::policy::Naming::Hf => contract::author_qwen3_5(builder),
            // ONE MLX author for both MLP kinds, which is what the
            // registry this replaced stated: every `qwen3_5*` row, dense
            // and routed alike, mapped to `author_qwen3_5_mlx`.
            crate::shared::policy::Naming::Mlx => contract::author_qwen3_5_mlx(builder),
        }
    }

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
        Ok(project::trace(&self.shape, class, load))
    }

    /// ChatML, stated rather than fallen through to.
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

    fn row(id: &str) -> &'static Qwen35 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The three capability answers come off the ROW.
    ///
    /// The last three reads of a resident `HfConfig` in `driver-cuda`:
    /// `model_type`, `max_position_embeddings` and "does this ship a
    /// tower", answered here by five rows that are `const` rather than by
    /// a parsed `config.json` kept alive for the life of a load. Every
    /// row is asserted because every row advertises, mixture included.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("the hybrid deploys")
                .advertised;
            assert_eq!(
                a.arch, "qwen3_5",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 262_144,
                "{}: every committed corpus config states it under `text_config`",
                v.id
            );
            assert!(
                !a.media_encode,
                "{}: the Qwen3-VL tower writes into the fire's hidden rows and is not \
                 an encode entry",
                v.id
            );
            assert!(
                v.deployment(Deployed::single())
                    .expect("deploys")
                    .towers
                    .vision
                    .is_none(),
                "{}: a row that advertised no encoder must not carry one",
                v.id
            );
        }
    }

    /// The label survives the boundary check the worker performs, and IS
    /// the label the image front-end is chosen by.
    ///
    /// Two consumers, one string. `read_hf_config_defaults` reduces
    /// `Qwen3_5ForConditionalGeneration` to a stem and refuses anything
    /// [`crate::catalog::arches`] does not list; `VisionArch::from_arch_name`
    /// matches this label WHOLE. A prettier `qwen3.5` would fail the
    /// first, and a bare `qwen35` would pass it while silently losing the
    /// second — which is why both are asserted.
    ///
    /// It used to be enough that the label *contained* `qwen3`. That is
    /// exactly what made the front-end table wrong: the text-only Qwen 3
    /// rows advertise `qwen3`, and containment cannot tell a prefix from
    /// a family.
    #[test]
    fn the_label_is_the_stem_the_boundary_derives_and_the_front_end_matches() {
        let stem = "Qwen3_5ForConditionalGeneration"
            .to_lowercase()
            .strip_suffix("forconditionalgeneration")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(
            stem, ARCH,
            "the worker would refuse a real Qwen3.5 checkpoint"
        );
        // The front-end table is the chat aspect's, so a build without
        // it makes no claim about which decoder a label selects.
        #[cfg(feature = "chat")]
        assert_eq!(
            crate::multimodal::VisionArch::from_arch_name(ARCH),
            Some(crate::multimodal::VisionArch::Qwen36),
            "this generation's label selects the Qwen3-VL front-end"
        );
        assert_eq!(ARCH, ARCH.to_lowercase(), "a family label is lowercase");
    }

    /// Qwen3.6-27B rides this generation's label, which is the same
    /// claim its row makes by being here: `model_type: qwen3_5_text` is
    /// what the checkpoint says about itself, so a 3.6 is a 3.5 by shape
    /// and by family, and only its id says 3.6.
    #[test]
    fn the_three_point_six_row_advertises_the_three_point_five_family() {
        let d = row("qwen3.6-27b")
            .deployment(Deployed::single())
            .expect("deploys");
        assert_eq!(d.advertised.arch, ARCH);
        assert_eq!(
            d.advertised.max_model_len, 262_144,
            "its corpus config states it too"
        );
        let labels: std::collections::BTreeSet<&str> = VARIANTS
            .iter()
            .map(|v| {
                v.deployment(Deployed::single())
                    .expect("deploys")
                    .advertised
                    .arch
            })
            .collect();
        assert_eq!(labels.len(), 1, "one family over five rows: {labels:?}");
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong. Both committed hybrid fixtures
    /// have a row here.
    #[test]
    fn the_rows_agree_with_the_committed_fixtures() {
        assert_eq!(
            row("qwen3.5-0.8b-base").shape,
            Qwen35HybridFacts::qwen3_5_0_8b()
        );
        assert_eq!(row("qwen3.6-27b").shape, Qwen35HybridFacts::qwen3_6_27b());
        // The MoE block is stated inline in the row because a `const`
        // cannot call a fixture, so the two are held together here.
        let Qwen35MlpKind::Moe(moe) = &row("qwen3.5-35b-a3b").shape.mlp else {
            panic!("the 35B-A3B row is a mixture");
        };
        assert_eq!(*moe, Qwen35MoeMlpFacts::qwen3_5_35b_a3b());
    }

    /// THE DISAGREEMENT, closed. `author_qwen3_5_moe` is selected by the
    /// same field the manifest's router and the deployment's expert
    /// width are projections of — so an author that writes a mixture
    /// cannot be reached by a row that deploys as dense, which is
    /// exactly what `"qwen3_moe"` did across `HF_ROWS` and `FACTS_ROWS`.
    #[test]
    fn the_author_and_the_deployment_read_one_mixture_field() {
        for v in VARIANTS {
            let manifest = v.manifest();
            let router = manifest
                .tensors
                .iter()
                .find(|t| t.name.ends_with("mlp.gate"))
                .expect("every row states whether it routes");
            assert_eq!(
                router.presence == crate::manifest::Presence::Required,
                v.is_mixture(),
                "{}: the router row and the author's field disagree",
                v.id,
            );
            assert_eq!(v.load_shape().n_experts != 0, v.is_mixture(), "{}", v.id);
            let d = v.deployment(Deployed::single()).expect("servable");
            // A mixture's launch width is the per-expert one; a dense
            // row's is its own MLP. `widest_mlp()` is what sizes the one
            // workspace both kinds share, so it is the number to check.
            match &v.shape.mlp {
                Qwen35MlpKind::Moe(m) => {
                    assert_eq!(d.shape.moe_intermediate, m.moe_intermediate);
                    assert_eq!(d.shape.intermediate, 0, "{}: no dense block", v.id);
                }
                Qwen35MlpKind::Dense { intermediate } => {
                    assert_eq!(d.shape.intermediate, *intermediate);
                    assert_eq!(d.shape.moe_intermediate, 0, "{}: no experts", v.id);
                }
            }
            assert_eq!(
                d.norm_eps, v.norm_eps,
                "{}: stated once, projected once",
                v.id
            );
        }
    }

    /// Exactly one row is a mixture, and it is the 35B-A3B — the
    /// checkpoint `author_qwen3_5_moe` was written for.
    #[test]
    fn the_mixture_row_is_a_mixture_in_the_only_place_it_is_stated() {
        let moe: Vec<&str> = VARIANTS
            .iter()
            .filter(|v| v.is_mixture())
            .map(|v| v.id)
            .collect();
        assert_eq!(moe, vec!["qwen3.5-35b-a3b"]);
        let m = row("qwen3.5-35b-a3b");
        assert_eq!(m.experts(), 256);
        assert_eq!(m.load_shape().n_experts, 256);
        let names: Vec<String> = m
            .manifest()
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .collect();
        assert!(names.iter().any(|n| n.ends_with("mlp.gate")), "{names:?}");
        // And the dense rows state the absence, so a dense checkpoint
        // cannot match the mixture row by accident.
        assert_eq!(row("qwen3.5-4b").experts(), 0);
    }

    /// An FP8 build and a bf16 build are ONE ROW. `Qwen3.6-27B-FP8`
    /// differs from `Qwen3.6-27B` by an added `quantization_config` and
    /// nothing else, so the two publish the same LOGICAL tensors — and
    /// the id carries no encoding word, which is the catalog's own rule
    /// re-asserted here for the checkpoint that motivated it.
    #[test]
    fn an_fp8_build_and_a_bf16_build_are_one_row() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert!(ids.contains(&"qwen3.6-27b"));
        assert_eq!(
            ids.iter()
                .filter(|id| id.starts_with("qwen3.6-27b"))
                .count(),
            1
        );
        for id in &ids {
            assert!(!id.contains("fp8"), "'{id}' names an encoding");
        }
        // The mechanism and not just the naming: nothing in the shape
        // records how a weight is STORED, so there is no field an FP8
        // row could differ in — and the manifest names no scale plane.
        let m = row("qwen3.6-27b").manifest();
        assert!(m.tensors.iter().all(|t| !t.name.contains("scale")), "{m:?}");
    }

    /// The base model and its instruct tune are one row, because a
    /// manifest compares TENSORS and theirs are identical. The
    /// alternative is two rows every 0.8B checkpoint matches equally
    /// well, which `Unmatched::Ambiguous` reports as the table defect it
    /// would be.
    #[test]
    fn the_instruct_tune_and_the_base_share_a_row() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert!(ids.contains(&"qwen3.5-0.8b-base"));
        assert!(
            !ids.contains(&"qwen3.5-0.8b"),
            "a second row nothing could tell apart"
        );
    }

    /// The ids an operator types, and nothing in them about how a
    /// checkpoint was quantized or packaged.
    #[test]
    fn the_ids_are_the_ones_an_operator_types() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(
            ids,
            vec![
                "qwen3.5-0.8b-base",
                "qwen3.5-4b",
                "qwen3.5-9b",
                "qwen3.5-35b-a3b",
                "qwen3.6-27b",
            ],
        );
    }

    /// Every row answers every question, which is what having no default
    /// bodies buys.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("the GDN store is built");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, 256, "the TRUE head dim, never a padded one");
            assert_eq!(ls.tied_embeddings, v.shape.tied_embeddings);
            assert_eq!(ls.mamba_groups, 0, "GDN is not Mamba");
            assert_eq!(ls.kv_shared_layers, 0);
        }
    }

    /// Every row is a HYBRID: it states which of its layers hold no
    /// pages and the recurrent slabs those need. A Qwen3.5 whose
    /// deployment said `recurrent: None` would be the dense llama-like
    /// answer the old table gave for this lineage's mixture key.
    #[test]
    fn every_row_carries_recurrent_state() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            let r = d
                .recurrent
                .as_ref()
                .expect("a gated-delta-net stack carries state");
            assert!(!r.linear_layers.is_empty(), "{}", v.id);
            assert_eq!(
                r.linear_layers.len() as u32,
                v.shape.layers - v.shape.layers / v.shape.full_attn_interval,
                "{}: three linear layers per full-attention one",
                v.id,
            );
            assert_eq!(r.state_elem, 2, "{}", v.id);
            assert_eq!(d.kv, crate::deployment::KvStyle::Paged, "{}", v.id);
        }
    }

    /// Rope's partial rotation is the row's, per layer, and 64 of 256 —
    /// the resolved channel count, not `partial_rotary_factor`. A driver
    /// that rotated the whole head would rotate four times too much.
    #[test]
    fn every_row_states_its_partial_rotation() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            assert_eq!(
                d.rotary_by_layer(),
                vec![64; v.shape.layers as usize],
                "{}",
                v.id
            );
            assert!(d.theta_by_layer().is_empty(), "one theta, so no table");
            assert_eq!(v.rope_theta, 1e7);
        }
    }

    /// Every row traces, for every class a fire can carry.
    #[test]
    fn every_row_traces() {
        use model_compiler::trace::FireClass;
        for v in VARIANTS {
            for class in [FireClass::Decode, FireClass::Prefill] {
                let plan = v.trace(class, Deployed::single()).expect("the text exists");
                assert!(!plan.ops.is_empty(), "{} {class:?}", v.id);
            }
        }
    }

    /// Every row speaks ChatML, stated rather than reached through a
    /// `_ =>` arm that also served models with no template at all.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_states_its_template() {
        let vocab: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "system",
            "\n",
            "user",
            "assistant",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let chat = v.chat(tok.clone());
            assert!(
                chat.user("Hi").starts_with(&[0]),
                "{} opens a turn wrong",
                v.id
            );
            assert!(
                chat.seal().contains(&1),
                "{} does not seal with <|im_end|>",
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
        use model_compiler::trace::FireClass;

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
