//! Llama 3 — the 3.1, 3.2 and 3.3 releases.
//!
//! One generation, one directory: the rows below and the header-marked
//! chat template in `chat`.
//!
//! The authoring passes most of the llama lineage calls are NOT here.
//! They were, because llama-3 wrote them down first, and ten generations
//! reached across a sibling edge to call them. They live in
//! [`crate::shared::llama_like::contract`] now — being written first is a
//! fact about who wrote it, not a claim of ownership.
//!
//! # The three keys this generation was reached by
//!
//! `FACTS_ROWS` derived the shape under `"llama"` AND `"llama3"`,
//! `HF_ROWS` chose the author under `"llama3"`, and `instruct::create`
//! chose the template under `"llama3" | "l4ma"`. The last two keys are
//! pie's own nicknames — no `config.json` contains either; a Llama 3
//! checkpoint says `model_type: "llama"` — so whether a load got
//! Llama's header protocol or ChatML depended on a rename performed in
//! another crate, and the `_ =>` arm at the bottom of `instruct::create`
//! meant a rename that drifted produced a *working* boot that spoke the
//! wrong dialect. That failure has no spelling here: the row states the
//! template, and a checkpoint reaches the row by matching its tensors.
//!
//! # One generation, five rows, two geometries
//!
//! 3.1-70B and 3.3-70B are the same architecture — 3.3 is a retrain, not
//! a reshape — so their manifests are identical and
//! [`identify`](crate::catalog::identify) cannot tell a 3.3 checkpoint
//! from a 3.1 one. Both rows are here anyway, because both names are
//! ones an operator types and each is truthfully stateable; what
//! distinguishes them is [`Override::Id`](crate::catalog::Override),
//! whose path still checks the manifest it names. A row that lied about
//! its shape to become unambiguous would be the `config.json` problem
//! again, wearing a different hat.

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

/// One Llama 3 checkpoint.
///
/// The same newtype-over-the-shape the exemplar
/// ([`crate::qwen_3::Qwen3`]) argues for: twelve generations share
/// `LlamaLikeFacts`, and a blanket `impl Variant for LlamaLikeFacts`
/// would have to answer "which template speaks for this?" out of numbers
/// that do not know.
pub struct Llama3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency. Not in `LlamaLikeFacts`, because the
    /// tracer takes it per-layer through `Deployment`. 500k across the
    /// whole generation — Llama 2's 10k moved once, at 3.0.
    pub rope_theta: f32,
    /// The RMSNorm epsilon every norm of the stack carries — `1e-5`
    /// across Llama 3, which is the value the reference implementation
    /// trained with and not a tolerance anyone is free to round. It sits
    /// on the row rather than in `LlamaLikeFacts` because the facts are
    /// what the TRACER needs (extents, placements, kinds) and this is
    /// what the LAUNCH needs; `Deployment::norm_eps` is where it lands,
    /// and the launch path used to read it off a resident `HfConfig`.
    pub norm_eps: f32,
    /// Sliding-window width, `-1` for full attention. Llama 3 attends
    /// its whole context at every layer.
    pub window: i32,
    /// The divisor Llama 3's piecewise rope rescaling applies to the
    /// low-frequency end.
    ///
    /// THE ONE NUMBER OF THE RESCALING THAT MOVES. Every row below
    /// states `low_freq_factor: 1.0`, `high_freq_factor: 4.0` and an
    /// original context of 8192 — so those three are
    /// [`ROPE_LOW_FREQ_FACTOR`], [`ROPE_HIGH_FREQ_FACTOR`] and
    /// [`ROPE_ORIGINAL_MAX`] — while this one is `32.0` for the 3.2
    /// pair and `8.0` for the 3.1 and 3.3 rows. A generation constant
    /// would have to be wrong for one of those groups, which is the
    /// same test `max_model_len` failed and the reason it is a field.
    pub rope_factor: f32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `llama` and not `llama3`, which is the generation's own name and
/// would have been the tidier-looking string. Two things state this one
/// and neither states the other: a Llama 3 `config.json` says
/// `model_type: "llama"`, and `architectures[0]` is `LlamaForCausalLM`,
/// which the worker's stem heuristic (strip the task suffix off a
/// lowercased name — see `embedded_driver::read_hf_config_defaults`)
/// reduces to exactly `llama` and then CHECKS against
/// [`crate::catalog::arches`]. A row advertising `llama3` would make
/// that check refuse a real Llama 3 checkpoint for not being in a
/// catalog that holds five of them.
///
/// `"llama3"` and `"l4ma"` — the keys `HF_ROWS` and `instruct::create`
/// used — were pie's own nicknames, carried by no checkpoint anywhere;
/// see the module doc for what a drifting rename did to the chat
/// template. This label is not a dispatch key and nothing matches on it.
const ARCH: &str = "llama";

/// The wavelength cut BELOW which Llama 3 rescales nothing, expressed
/// as a divisor of [`ROPE_ORIGINAL_MAX`].
///
/// `1.0` in every published Llama 3 `config.json`, including the
/// corpus's own `meta-llama--Llama-3.2-1B-Instruct.json`. A constant
/// rather than a field because, unlike [`Llama3::rope_factor`], no
/// release has ever moved it.
const ROPE_LOW_FREQ_FACTOR: f32 = 1.0;

/// The wavelength cut ABOVE which Llama 3 rescales fully. `4.0`
/// throughout the generation.
const ROPE_HIGH_FREQ_FACTOR: f32 = 4.0;

/// The context Llama 3 was TRAINED at, which is what the rescaling is
/// measured against.
///
/// `8192` — deliberately not [`MAX_MODEL_LEN`], which is the 131_072 the
/// rescaling BUYS. Conflating the two is the arithmetic error the whole
/// piecewise scheme exists to avoid: the ladder is built for 8192 and
/// stretched, not built for 131_072.
const ROPE_ORIGINAL_MAX: u32 = 8_192;

/// The published context ceiling, shared by every row here.
///
/// One constant because 3.1 moved it ONCE — 3.0's 8192 to 131 072 — and
/// nothing in this table predates that move: 3.1, 3.2 and 3.3 all state
/// `max_position_embeddings: 131072`, and the corpus's own
/// `meta-llama--Llama-3.2-1B-Instruct.json` is one of them. A Llama 3.0
/// row would state 8192 and break this constant into a row field, which
/// is the correct outcome and the reason this comment names the number
/// it is NOT.
///
/// A training-time fact rather than a deployment one: nothing in a fire
/// reads it, and a driver serving a shorter context is serving
/// correctly. What it must not be is 0 — the field's word for "the row
/// does not say" — for a generation whose ceiling five published
/// configs state.
const MAX_MODEL_LEN: u32 = 131_072;

/// The generation's rows.
///
/// Every field is stated even when it is zero, for the reason the
/// fixtures give: a row is a MEASUREMENT of a real checkpoint, and "this
/// one has no experts" is part of the measurement.
///
/// [`RopeKind::Yarn`] on every row is the one value here that the old
/// derivation did not produce — `llama_like_facts_from_hf` wrote
/// `RopeKind::Standard` for all twelve generations, unconditionally,
/// having never looked at `rope_scaling`. Every checkpoint below ships
/// `rope_scaling.rope_type: "llama3"`, the low/high-frequency split that
/// stretches 8k pretraining positions over a 128k window, and `Yarn` is
/// this tree's tag for "the frequency ladder is scaled" (the enum has
/// two values, and `Standard` is the other one). Stating it is the
/// point of moving identity into the binary: the row says what the
/// checkpoint is, not what a derivation happened to fill in.
pub const VARIANTS: &[Llama3] = &[
    // meta-llama/Llama-3.2-1B-Instruct — the corpus's own
    // `meta-llama--Llama-3.2-1B-Instruct.json`, and the row
    // `tests/catalog_differential.rs` pins against the old derivation.
    // head_dim 64 here (2048 / 32) rather than the generation's usual
    // 128, which is why the 1B is the row worth pinning: it is the only
    // one whose attention kernel dispatches to a different width.
    Llama3 {
        id: "llama-3.2-1b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 16,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 64,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,
        // the corpus config states it.
        rope_factor: 32.0,
    },
    // meta-llama/Llama-3.2-3B-Instruct. Fewer query heads than the 1B
    // (24 against 32) at a wider head, which is the distillation's
    // doing and exactly the kind of fact a "derive it from hidden /
    // heads" shortcut gets wrong.
    Llama3 {
        id: "llama-3.2-3b",
        shape: LlamaLikeFacts {
            hidden: 3072,
            layers: 28,
            q_heads: 24,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8192,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,
        // same release as the 1B.
        rope_factor: 32.0,
    },
    // meta-llama/Llama-3.1-8B-Instruct — the first untied head in the
    // generation. The 1B and 3B are distillations of it and tie;
    // everything at 8B and up ships its own `lm_head`.
    Llama3 {
        id: "llama-3.1-8b",
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
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,
        // the 3.1 line.
        rope_factor: 8.0,
    },
    // meta-llama/Llama-3.1-70B-Instruct.
    Llama3 {
        id: "llama-3.1-70b",
        shape: LlamaLikeFacts {
            hidden: 8192,
            layers: 80,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 28_672,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,
        // the 3.1 line.
        rope_factor: 8.0,
    },
    // meta-llama/Llama-3.3-70B-Instruct — 3.1-70B's geometry exactly,
    // retrained. See the module doc: the two rows are indistinguishable
    // by manifest, and that is a fact about the release rather than
    // something the catalog can resolve by inspecting harder.
    Llama3 {
        id: "llama-3.3-70b",
        shape: LlamaLikeFacts {
            hidden: 8192,
            layers: 80,
            q_heads: 64,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 28_672,
            vocab: 128_256,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
        window: -1,
        // a 3.1 refresh, same rescaling.
        rope_factor: 8.0,
    },
];

/// This generation's contribution to [`crate::catalog::catalog`].
///
/// The `OnceLock` is only the widening from `&Llama3` to `&dyn Variant`;
/// the rows themselves are `const` and in `.rodata`.
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Llama3 {
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
            rope_rescaled: true,
            // Unread: this generation is dense, so no router is stated.
            norm_topk_prob: true,
        }
    }
}

impl Variant for Llama3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The unpadded head dim, which is the one an authoring pass needs:
    /// a tensor-parallel row split that lands mid-head produces a
    /// contract that compiles and a model that is wrong. Every row here
    /// is dense, so the mixture fields are not merely zero by default —
    /// there is no Llama 3 mixture to state.
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
        // THE RESCALING, which nothing carried for the length of this
        // refactor. `driver-metal` read these four numbers off the
        // `pie.model/1` descriptor and built its decode ladder from
        // them; deleting the descriptor left `DecodeGeometry` holding
        // four zeroes, and a zero factor reads as "no rescaling". Every
        // row here would have attended past 8192 with the wrong
        // wavelengths — degrading rather than failing, which is the
        // shape of defect this catalog exists to make impossible.
        deployment.rope_scaling = Some(crate::deployment::RopeScaling::Llama3 {
            factor: self.rope_factor,
            low_freq_factor: ROPE_LOW_FREQ_FACTOR,
            high_freq_factor: ROPE_HIGH_FREQ_FACTOR,
            original_max_position: ROPE_ORIGINAL_MAX,
        });
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // No tower. Llama 3 is text-only in every release this table
            // holds — the vision line is Llama 3.2's `Mllama*` package, a
            // different checkpoint with a different stack and no row
            // here — so there is nothing for the driver's encode entry to
            // serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The lineage's own pass, called rather than tabulated.
    ///
    /// `author_llama_like` lives in [`crate::shared::llama_like::contract`]
    /// and is what `HF_ROWS` named in a dozen rows' author column; the N:1 is
    /// unchanged, but it is a call now and cannot fall out of step with
    /// the shape stated three fields above it.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_llama_like(builder)
            }
            // A DIFFERENT BACKEND'S READING of the same checkpoint, and
            // the reason this is a `match` rather than a call: the MLX
            // author renames every tensor to the binder's own vocabulary
            // and states the dtype its kernels read. Handing Metal the
            // HF contract published the checkpoint's raw names AND left
            // MLX's fp16 sidecars uncast, so a bf16 kernel read a scale
            // of 6e-3 as 1e-20 and every logit came out zero.
            crate::shared::policy::Naming::Mlx => {
                crate::shared::llama_like::contract::author_llama_mlx(builder)
            }
        }
    }

    /// This row's text, for whichever backend asked.
    ///
    /// `rope_rescaled: TRUE`, and it is the one row field here with
    /// teeth: llama-3 rotates on a piecewise-rescaled ladder
    /// ([`Self::deployment`] states the four numbers as
    /// `RopeScaling::Llama3`), and NO `rope_theta` expresses it. The
    /// Metal text therefore reads a frequency TABLE the driver derived
    /// at load rather than deriving a ladder from the base, and this
    /// flag is how it knows which.
    ///
    /// Nothing carried that for the length of this refactor:
    /// `driver-metal` read the four numbers off the deleted
    /// `pie.model/1` descriptor, and a factor of zero reads as "no
    /// rescaling", so every llama-3 attended past its trained 8192 with
    /// the wrong wavelengths — degrading rather than failing.
    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// The header protocol, which is NOT ChatML.
    ///
    /// `<|start_header_id|>role<|end_header_id|>\n\n` … `<|eot_id|>`,
    /// and it matters that the row says so: the string that used to
    /// reach this template was one no checkpoint carries, and the arm
    /// that caught a misspelling handed back ChatML — a model that
    /// answers, fluently, having never seen the turn boundaries it was
    /// tuned on.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::LlamaInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Llama3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// Every row answers every question, which is what having no default
    /// method bodies buys — and the answers are the row's own numbers
    /// rather than a second statement of them.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("dense llama is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.q_heads, v.shape.q_heads);
            assert_eq!(d.shape.kv_heads, v.shape.kv_heads);
            assert_eq!(d.shape.vocab, v.shape.vocab);
            assert_eq!(d.norm, crate::deployment::NormPlacement::Pre);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert_eq!(ls.tied_embeddings, v.shape.tied_embeddings);
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.window, -1, "{} layer {l} attends its whole context", v.id);
                assert_eq!(a.rope_theta, 500_000.0, "{}", v.id);
                assert_eq!(a.kv_source, l as u32, "no KV sharing in this lineage");
                assert_eq!(a.sm_scale, 1.0 / (a.head_dim as f32).sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 5);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    /// The catalog's rows are the ones an operator names, so the ids are
    /// distinct and stated once.
    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(
            ids,
            [
                "llama-3.2-1b",
                "llama-3.2-3b",
                "llama-3.1-8b",
                "llama-3.1-70b",
                "llama-3.3-70b"
            ]
        );
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` in
    /// `driver-cuda`: a whole parsed `config.json` was kept alive for the
    /// life of a load so a capability could answer `model_type`,
    /// `max_position_embeddings` and "does this ship a tower". All five
    /// rows are asserted because all five advertise, and a row that
    /// answered `0` would be telling a guest program this generation has
    /// no published ceiling.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("dense llama deploys")
                .advertised;
            assert_eq!(
                a.arch, "llama",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 131_072,
                "{}: 3.1 raised the ceiling and 3.2 and 3.3 kept it",
                v.id
            );
            assert!(
                !a.media_encode,
                "{}: no row of this generation ships a tower the encode entry serves",
                v.id
            );
        }
    }

    /// The label is the one the WORKER's stem heuristic produces, which
    /// is what makes the check at the boundary a check and not a second
    /// opinion. `read_hf_config_defaults` lowercases `architectures[0]`,
    /// strips the task suffix and refuses anything
    /// [`crate::catalog::arches`] does not list — so `llama3` here would
    /// refuse `LlamaForCausalLM`, which is every checkpoint in this
    /// table.
    #[test]
    fn the_label_is_what_architectures_reduces_to_and_not_the_generations_name() {
        let stem = "LlamaForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(
            stem, ARCH,
            "the worker would refuse a real Llama 3 checkpoint"
        );
        assert_ne!(
            ARCH, "llama3",
            "pie's own nickname, which no checkpoint carries"
        );
    }

    /// One label over five rows, which is what a FAMILY means: `llama`
    /// names two geometries and three release trains, and a program
    /// asking "is this a llama" wants the coarse answer while a driver
    /// loading one needs the id.
    #[test]
    fn one_family_label_covers_five_distinct_rows() {
        let labels: std::collections::BTreeSet<&str> = VARIANTS
            .iter()
            .map(|v| {
                v.deployment(Deployed::single())
                    .expect("deploys")
                    .advertised
                    .arch
            })
            .collect();
        assert_eq!(labels.len(), 1, "one label: {labels:?}");
        let ids: std::collections::BTreeSet<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(ids.len(), VARIANTS.len(), "five ids");
        assert!(
            ids.len() > labels.len(),
            "a label per row is not a family label"
        );
    }

    /// The scaled rope ladder is a fact about the checkpoint, and the
    /// derivation this replaces did not have it: it wrote
    /// `RopeKind::Standard` for every llama-like model without reading
    /// `rope_scaling` at all.
    #[test]
    fn every_row_states_the_llama3_rope_scaling() {
        for v in VARIANTS {
            assert_eq!(v.shape.rope, RopeKind::Yarn, "{}", v.id);
            assert_eq!(v.rope_theta, 500_000.0, "{}", v.id);
        }
    }

    /// The rescaling reaches a `Deployment`, and the factor is the ROW's.
    ///
    /// `RopeKind::Yarn` above says only THAT the ladder is rescaled. The
    /// four numbers that say HOW travelled on the `pie.model/1` descriptor
    /// and stopped travelling when it was deleted: `driver-metal`'s
    /// `DecodeGeometry` kept `rope_freq_factor` and nothing filled it, and
    /// a zero factor reads as "no rescaling". A llama-3 does not fail that
    /// way — it attends past 8192 with the wrong wavelengths and degrades
    /// fluently.
    #[test]
    fn the_rescaling_reaches_the_deployment_with_the_rows_own_factor() {
        for v in VARIANTS {
            let d = v
                .deployment(Deployed::single())
                .expect("every llama-3 row deploys");
            let Some(crate::deployment::RopeScaling::Llama3 {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            }) = d.rope_scaling
            else {
                panic!("{} states no llama3 rescaling", v.id);
            };
            assert_eq!(factor, v.rope_factor, "{} carries its own factor", v.id);
            assert_eq!(low_freq_factor, ROPE_LOW_FREQ_FACTOR, "{}", v.id);
            assert_eq!(high_freq_factor, ROPE_HIGH_FREQ_FACTOR, "{}", v.id);
            assert_eq!(original_max_position, ROPE_ORIGINAL_MAX, "{}", v.id);
            // The TRAINED context, not the advertised one. Conflating them
            // is the arithmetic error the piecewise scheme exists to avoid.
            assert!(
                original_max_position < d.advertised.max_model_len,
                "{}: the rescaling is measured against the context the checkpoint was \
                 trained at, which is shorter than the one it buys",
                v.id
            );
        }
    }

    /// The factor is not one number, which is why it is a field.
    ///
    /// Llama 3.2's pair rescale by 32 and the 3.1/3.3 rows by 8, from the
    /// same `rope_theta` and the same 8192. A generation constant would
    /// have to be wrong for one group — silently, since nothing fires on
    /// it and only long contexts would show it.
    #[test]
    fn the_factor_differs_by_release_and_that_is_not_a_transcription_slip() {
        let f = |id: &str| row(id).rope_factor;
        assert_eq!(f("llama-3.2-1b"), 32.0, "the corpus config states 32.0");
        assert_eq!(f("llama-3.2-3b"), 32.0);
        assert_eq!(f("llama-3.1-8b"), 8.0);
        assert_eq!(f("llama-3.1-70b"), 8.0);
        assert_eq!(f("llama-3.3-70b"), 8.0);
        let distinct: std::collections::BTreeSet<u32> =
            VARIANTS.iter().map(|v| v.rope_factor.to_bits()).collect();
        assert_eq!(
            distinct.len(),
            2,
            "two values, so a shared const cannot express them"
        );
    }

    /// The tie is the ABSENCE of `lm_head`, which is the only way a
    /// manifest can tell a tied variant from an untied one: every extent
    /// they share agrees.
    #[test]
    fn the_distilled_pair_ties_its_head_and_the_rest_do_not() {
        for (id, tied) in [
            ("llama-3.2-1b", true),
            ("llama-3.2-3b", true),
            ("llama-3.1-8b", false),
            ("llama-3.1-70b", false),
            ("llama-3.3-70b", false),
        ] {
            let v = row(id);
            assert_eq!(v.shape.tied_embeddings, tied, "{id}");
            let m = v.manifest();
            let head = m
                .tensors
                .iter()
                .find(|t| t.name == "lm_head")
                .unwrap_or_else(|| panic!("{id} states lm_head one way or the other"));
            let want = if tied {
                crate::manifest::Presence::Absent
            } else {
                crate::manifest::Presence::Required
            };
            assert_eq!(head.presence, want, "{id}");
        }
    }

    /// GQA at eight KV heads throughout, and the manifest carries the
    /// consequence: `k_proj` is `kv_heads * head_dim` rows, not
    /// `hidden`. This is the arithmetic that makes a manifest a CHECK
    /// rather than a restatement.
    #[test]
    fn the_kv_projections_are_narrower_than_the_query_ones() {
        for v in VARIANTS {
            assert_eq!(v.shape.kv_heads, 8, "{}", v.id);
            assert!(v.shape.q_heads > v.shape.kv_heads, "{}", v.id);
            assert_eq!(v.shape.q_heads % v.shape.kv_heads, 0, "{}", v.id);

            let m = v.manifest();
            let extent = |name: &str| -> Vec<u64> {
                m.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .unwrap_or_else(|| panic!("{} states {name}", v.id))
                    .extents
                    .clone()
            };
            let hidden = u64::from(v.shape.hidden);
            let q = u64::from(v.shape.q_width());
            let kv = u64::from(v.shape.kv_width());
            assert_eq!(extent("layer.{}.self_attn.q_proj"), vec![q, hidden]);
            assert_eq!(extent("layer.{}.self_attn.k_proj"), vec![kv, hidden]);
            assert_eq!(extent("layer.{}.self_attn.v_proj"), vec![kv, hidden]);
            assert_eq!(
                extent("embed_tokens"),
                vec![u64::from(v.shape.vocab), hidden]
            );
        }
    }

    /// No attention bias anywhere in the lineage, stated rather than
    /// inherited: Qwen2 next door sets the same field the other way, and
    /// the manifest row it adds is the difference.
    #[test]
    fn the_projections_carry_no_bias() {
        for v in VARIANTS {
            assert!(!v.shape.qkv_bias, "{}", v.id);
            let m = v.manifest();
            assert!(
                !m.tensors.iter().any(|t| t.name.ends_with("q_proj.bias")),
                "{} declares a bias it does not ship",
                v.id
            );
        }
    }

    /// The two 70B rows are one geometry, and the test says so rather
    /// than pretending otherwise. A checkpoint satisfying one satisfies
    /// the other; `identify` will report them as ambiguous, and the
    /// resolution is the operator's `Override::Id`.
    #[test]
    fn the_two_70b_rows_are_indistinguishable_by_checkpoint() {
        let a = row("llama-3.1-70b");
        let b = row("llama-3.3-70b");
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.manifest(), b.manifest());
        assert_ne!(a.id(), b.id());
    }

    /// The 1B's head is 64 wide where the rest of the generation is 128,
    /// and both are widths this build instantiates — so neither pads and
    /// the softmax scale differs between the two.
    #[test]
    fn the_1b_dispatches_to_a_narrower_attention_head() {
        let small = row("llama-3.2-1b")
            .deployment(Deployed::single())
            .expect("servable");
        let large = row("llama-3.1-8b")
            .deployment(Deployed::single())
            .expect("servable");
        assert_eq!(small.attention[0].head_dim, 64);
        assert_eq!(large.attention[0].head_dim, 128);
        assert_eq!(
            small.shape.head_dim_kernel, 64,
            "64 is instantiated; nothing pads"
        );
        assert_eq!(large.shape.head_dim_kernel, 128);
        assert!(small.attention[0].sm_scale > large.attention[0].sm_scale);
    }

    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_the_header_protocol_and_not_the_chatml_fallback() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "\n\n",
            "\n",
            "system",
            "user",
            "assistant",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        let inst = row("llama-3.1-8b").chat(tok.clone());
        let turn = tok.decode(&inst.user("Hi"), false);
        assert_eq!(
            turn,
            "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        );
        assert!(
            !turn.contains("<|im_start|>"),
            "the `_ =>` ChatML arm is gone"
        );
        assert_eq!(
            tok.decode(&inst.cue(), false),
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "the cue is the assistant header with nothing after it"
        );
        assert!(
            !inst.seal().is_empty(),
            "<|eot_id|> is in the vocabulary above"
        );
    }

    /// Every row's chat answer is the same template, reached without a
    /// string: five calls, five `LlamaInstruct`s, no fallback in sight.
    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "\n\n",
            "user",
            "Hi",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let text = tok.decode(&v.chat(tok.clone()).user("Hi"), false);
            let want = "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>";
            assert_eq!(text, want, "{}", v.id);
        }
    }

    /// The author is reached from the ROW, so this exercises the
    /// dispatch the catalog replaced rather than the pass itself: an
    /// empty checkpoint publishes nothing, and `author_llama_like`
    /// declaring nothing over it is the honest outcome. What is under
    /// test is that every row HAS an author and that it runs.
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

    /// Both fire classes trace, for every row, and the text that comes
    /// back is the llama-like one — the family a row states by which
    /// projection it calls, no longer by which string reached which
    /// table.
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
                assert!(!plan.values.is_empty(), "{}", v.id);
            }
        }
        // A deeper stack is a longer text, which is the one relation
        // between a row's numbers and its trace worth asserting here.
        let trace_of = |id: &str| {
            row(id)
                .trace(FireClass::Decode, Deployed::single())
                .expect("traces")
        };
        let small = trace_of("llama-3.2-1b");
        let large = trace_of("llama-3.1-70b");
        assert!(large.ops.len() > small.ops.len(), "80 layers out-op 16");
    }
}
