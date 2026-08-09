//! Gemma 3 — a llama-like shape on a schedule of its own.
//!
//! Five rows, one shape struct shared with a dozen other generations,
//! and a small projection wrapper for the two things that shape cannot
//! hold: a rope base that differs per layer and a window that is not the
//! same on every layer. See [`project`] for why both are row fields.
//!
//! What used to be here was `("gemma3", llama_like_facts_from_hf)` in
//! `deployment_cuda::FACTS_ROWS` — the GENERIC llama-like derivation,
//! which read gemma-3's norms as Plain/Pre, broadcast the config's
//! single `sliding_window` to every layer and its single `rope_theta`
//! along with it. None of that failed loudly. A model whose every sixth
//! layer attends 1024 tokens instead of its whole context still writes
//! fluent sentences.

#[cfg(feature = "chat")]
pub mod chat;

/// The three projections a row of this generation makes: its tensor
/// manifest, its `Deployment`, and its traced text.
pub mod project;

use std::sync::{Arc, OnceLock};

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::shared::llama_like::spec::LlamaLikeFacts;
use crate::manifest::Manifest;

use self::project::Schedule;

use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::trace::{NormVariant, RopeKind};

/// RMSNorm epsilon, shared by the whole generation.
///
/// Stated once rather than per row because every published config of
/// this generation carries the same `1e-6` — a generation-level
/// constant is an honest way to say that, where a copy per row invites
/// one of them to drift.
const NORM_EPS: f32 = 1e-6;

/// The family label a GUEST PROGRAM matches on.
///
/// `gemma3` for all five rows including `embeddinggemma-300m`, which is
/// a gemma-3 text tower and says so in `model_type: "gemma3_text"`. The
/// boundary derives the same string for the rows that matter most:
/// `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration` both reduce
/// to `gemma3` under the worker's stem heuristic, which then checks the
/// result against [`crate::catalog::arches`].
///
/// The one checkpoint that does NOT reduce to it is embeddinggemma,
/// whose `architectures[0]` is `Gemma3TextModel` — no task suffix to
/// strip, so the heuristic yields `gemma3textmodel` and the boundary
/// refuses it. That is a gap in the heuristic and not a reason for this
/// row to advertise a second family label: the checkpoint IS a gemma-3,
/// an operator states `arch_name` for it explicitly, and a per-row label
/// would put the packaging back into the identity.
const ARCH: &str = "gemma3";

/// One Gemma 3 checkpoint.
///
/// The shape is the family's; what the generation adds is the schedule,
/// the template that speaks for it and the author that writes its
/// contract — which are exactly the things a set of widths cannot say.
pub struct Gemma3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers the family shares.
    pub shape: LlamaLikeFacts,
    /// The numbers gemma-3 states per LAYER.
    pub schedule: Schedule,
    /// `max_position_embeddings`, advertised and never fired on.
    ///
    /// A row's field and not a generation constant, because the rows
    /// DISAGREE and the disagreement is real: the multimodal releases
    /// state 131 072, the text-only 1B states 32 768, and
    /// `embeddinggemma-300m` states 2048 — three ceilings spanning six
    /// doublings inside one family. One constant here would be a lie
    /// about two rows out of five, and a shared 131 072 would tell a
    /// guest program the 300M can take sixty-four times the context its
    /// own config publishes.
    pub max_model_len: u32,
}

/// The layer schedule every published gemma-3 states, hoisted because
/// all five rows agree on it: five sliding layers then one full, local
/// rope at 10 000 and global at 1 000 000.
///
/// The two window widths and the one scalar that differ are stated per
/// row, as they must be — a constant that is nearly always the same is
/// still not a default.
const fn gemma_3_schedule(sliding_window: i32, query_pre_attn_scalar: u32) -> Schedule {
    Schedule {
        sliding_window,
        full_attn_interval: 6,
        rope_theta_local: 10_000.0,
        rope_theta_global: 1_000_000.0,
        query_pre_attn_scalar,
    }
}

/// The shape every gemma-3 row shares apart from its widths.
///
/// `NormVariant::Gemma` is the `(1 + w)` rmsnorm, `Sandwich` is the
/// four-norm block, `PerHead` is the q/k norm gemma-3 added over
/// gemma-2 — and all three are places the generic derivation answered
/// wrong for this generation because the row it filled had no way to
/// say otherwise.
const fn gemma_3_shape(
    hidden: u32,
    layers: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    vocab: u32,
) -> LlamaLikeFacts {
    LlamaLikeFacts {
        hidden,
        layers,
        q_heads,
        kv_heads,
        head_dim,
        // DENSE: no mixture anywhere in gemma-3.
        n_experts: 0,
        experts_per_token: 0,
        moe_intermediate: 0,
        shared_intermediate: 0,
        intermediate,
        vocab,
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Gemma,
        norm_placement: NormPlacement::Sandwich,
        qk_norm: QkNorm::PerHead,
        fused_qkv: true,
        tied_embeddings: true,
        qkv_bias: false,
    }
}

/// The generation's rows.
///
/// `const`, which is the whole architecture: identity is in the binary,
/// so the three questions have one answer and the answer is linked.
pub const VARIANTS: &[Gemma3] = &[
    // google/gemma-3-1b-it. The only text-only gemma-3 and the only one
    // whose vocab is 262 144 rather than 262 208 — the multimodal rows
    // carry 64 extra image tokens, and that difference is what tells
    // this row's checkpoint from a 4B's if every other extent somehow
    // agreed.
    Gemma3 {
        id: "gemma-3-1b",
        shape: gemma_3_shape(1152, 26, 4, 1, 256, 6912, 262_144),
        schedule: gemma_3_schedule(512, 256),
        // 32 768, a QUARTER of its siblings'. The 1B is the text-only
        // release and the only one Google published at the short
        // ceiling; every other gemma-3 states 131 072.
        max_model_len: 32_768,
    },
    // google/gemma-3-4b-it (its text tower; the vision encoder is
    // `crate::multimodal`'s and no part of this row).
    Gemma3 {
        id: "gemma-3-4b",
        shape: gemma_3_shape(2560, 34, 8, 4, 256, 10_240, 262_208),
        schedule: gemma_3_schedule(1024, 256),
        // Stated under `text_config` in a `*ForConditionalGeneration`
        // package, which is where this generation's decoder facts live.
        max_model_len: 131_072,
    },
    // google/gemma-3-12b-it.
    Gemma3 {
        id: "gemma-3-12b",
        shape: gemma_3_shape(3840, 48, 16, 8, 256, 15_360, 262_208),
        schedule: gemma_3_schedule(1024, 256),
        max_model_len: 131_072,
    },
    // google/gemma-3-27b-it — 128-wide heads and a
    // `query_pre_attn_scalar` of 168, which is neither the head dim nor
    // the hidden size over the head count. HF is unambiguous that the
    // scale is `1/sqrt` of the scalar, so this row is the reason the
    // scale is not derived from the head dim.
    Gemma3 {
        id: "gemma-3-27b",
        shape: gemma_3_shape(5376, 62, 32, 16, 128, 21_504, 262_208),
        schedule: gemma_3_schedule(1024, 168),
        max_model_len: 131_072,
    },
    // google/embeddinggemma-300m — a gemma-3 text tower at 768 wide,
    // and a row for the reason every row exists: it is a checkpoint
    // this build can be handed. Its `use_bidirectional_attention` is
    // real and `Deployment` has nowhere to put it, so what this row
    // serves is the causal reading of it.
    Gemma3 {
        id: "embeddinggemma-300m",
        shape: gemma_3_shape(768, 24, 3, 1, 256, 1152, 262_144),
        schedule: gemma_3_schedule(512, 256),
        // 2048, and read straight out of the corpus's own
        // `google--embeddinggemma-300m.json` rather than inferred from
        // the family. An embedding model is trained on documents that
        // fit, not on long contexts, and this is the row
        // `tests/catalog_differential.rs` would fail loudest if a
        // generation-wide constant had been used instead: it compares
        // this field against `max_position_embeddings` in that file.
        max_model_len: 2_048,
    },
];

/// This generation's contribution to [`crate::catalog::catalog`].
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: OnceLock<Vec<&'static dyn Variant>> = OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

impl Variant for Gemma3 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// The stated head dim: gemma-3-1b is 1152 over 4 heads and its
    /// heads are 256 wide, not 288.
    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(self.shape.layers, self.shape.head_dim, self.shape.tied_embeddings)
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        // Gemma-3 ships no host scalars and binds nothing per rank.
        let _ = load;
        let mut deployment = project::deployment(&self.shape, &self.schedule, NORM_EPS);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: self.max_model_len,
            // No tower the ENCODE entry serves. The 4B, 12B and 27B
            // packages do ship a SigLIP vision encoder — but it is not
            // in `Deployment::towers`, `crate::multimodal` computes its
            // patch grid host-side, and `driver-cuda`'s encode entry
            // binds gemma-4's kernels and refuses on absent towers. A
            // `true` here would advertise an entry point that has no
            // encoder to run, which is the mirror of the bug this field
            // exists for: gemma-4 advertised `false` while its towers
            // worked.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The generic dense contract: gemma-3's checkpoint already uses the
    /// names the bind path reads.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => crate::shared::llama_like::contract::author_dense(builder),
            // The registry this replaced held NO MLX row for gemma-3, and
            // the absence was a silence the caller read as "no
            // contract". Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "gemma-3: no MLX authoring pass exists for this generation, \
                 so there is no name layout to author against",
            ),
        }
    }

    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, &self.schedule, NORM_EPS, class, load)
    }

    /// Gemma's `<start_of_turn>` template, which lives in `shared/`
    /// because gemma-3n binds it too.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(self::chat::Gemma3Instruct::for_variant(
            tokenizer,
            self::chat::Gemma3Variant::Gemma3Text,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{AttnOutput, NormPlacement as DeployNorm};

    fn row(id: &str) -> &'static Gemma3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda`, and the ceiling is the one that had to become a
    /// row FIELD: this generation's five rows publish three different
    /// numbers, so there is no constant to read.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for (id, want) in [
            ("gemma-3-1b", 32_768),
            ("gemma-3-4b", 131_072),
            ("gemma-3-12b", 131_072),
            ("gemma-3-27b", 131_072),
            ("embeddinggemma-300m", 2_048),
        ] {
            let a = row(id).deployment(Deployed::single()).expect("gemma-3 deploys").advertised;
            assert_eq!(a.arch, "gemma3", "{id}: the family label a guest program sees");
            assert_eq!(a.max_model_len, want, "{id}: its own config states {want}");
            assert!(
                !a.media_encode,
                "{id}: the SigLIP tower is not one the driver's encode entry serves"
            );
        }
        // And no row is left at the field's word for "does not say",
        // which is what `Default::default()` would have given all five.
        for v in VARIANTS {
            assert_ne!(v.max_model_len, 0, "{}: an unstated ceiling is not a ceiling", v.id);
        }
    }

    /// THE ROWS DISAGREE, and the disagreement is the measurement.
    ///
    /// Three ceilings inside one generation — 2048, 32 768, 131 072 —
    /// and the temptation to fold them into one `MAX_MODEL_LEN` is
    /// exactly what this test exists to stop. The 1B is the text-only
    /// release and states a quarter of its siblings'; embeddinggemma is
    /// an embedding tower and states a sixty-fourth of them. Both
    /// numbers are their own checkpoints' words.
    #[test]
    fn the_ceilings_differ_by_row_and_that_is_not_a_transcription_slip() {
        let len = |id: &str| {
            row(id).deployment(Deployed::single()).expect("deploys").advertised.max_model_len
        };
        assert!(len("gemma-3-1b") < len("gemma-3-4b"), "the 1B publishes the shorter ceiling");
        assert_eq!(len("gemma-3-1b") * 4, len("gemma-3-4b"), "32 768 against 131 072");
        assert!(
            len("embeddinggemma-300m") < len("gemma-3-1b"),
            "an embedding tower is trained on documents that fit"
        );
        let distinct: std::collections::BTreeSet<u32> =
            VARIANTS.iter().map(|v| v.max_model_len).collect();
        assert_eq!(
            distinct.len(),
            3,
            "three ceilings over five rows; one shared constant would misstate two of them: \
             {distinct:?}"
        );
        // The multimodal three agree with each other, which is why the
        // field is per row and not per SIZE.
        for id in ["gemma-3-4b", "gemma-3-12b", "gemma-3-27b"] {
            assert_eq!(len(id), 131_072, "{id}");
        }
    }

    /// One family label over five rows, and it is the stem the WORKER
    /// derives for the four whose `architectures[0]` carries a task
    /// suffix. `read_hf_config_defaults` strips that suffix and refuses
    /// a family [`crate::catalog::arches`] does not list; `gemma3` is
    /// what `Gemma3ForCausalLM` and `Gemma3ForConditionalGeneration`
    /// both reduce to.
    #[test]
    fn one_family_label_covers_five_rows_and_survives_the_boundary_check() {
        for name in ["Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"] {
            let lower = name.to_lowercase();
            let stem = lower
                .strip_suffix("forconditionalgeneration")
                .or_else(|| lower.strip_suffix("forcausallm"))
                .expect("one of the two suffixes the worker strips");
            assert_eq!(stem, ARCH, "{name} must reduce to the label the row hands out");
        }
        let labels: std::collections::BTreeSet<&str> = VARIANTS
            .iter()
            .map(|v| v.deployment(Deployed::single()).expect("deploys").advertised.arch)
            .collect();
        assert_eq!(labels.len(), 1, "a label is a FAMILY: {labels:?}");
        assert_eq!(VARIANTS.len(), 5, "and five rows share it");
    }

    /// Every row answers every question.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("gemma-3 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.norm, DeployNorm::Pre);
            assert_eq!(d.attn_output, AttnOutput::DriverPinned);
            assert_eq!(d.logit_softcap, 0.0, "gemma-3 caps nothing");
            assert_eq!(d.ple_dim, 0);

            let m = v.manifest();
            assert_eq!(m.layers, v.shape.layers);
            assert!(m.tensors.iter().any(|t| t.name == "layer.{}.post_feedforward_layernorm"));

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, v.shape.head_dim);
            assert!(ls.tied_embeddings, "every gemma-3 ties");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);
        }
    }

    /// Five local layers then one global, on every row — the pattern the
    /// generic derivation could not express and therefore flattened.
    #[test]
    fn every_row_runs_the_five_to_one_window() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            for (l, a) in d.attention.iter().enumerate() {
                let full = (l + 1) % 6 == 0;
                assert_eq!(a.window == -1, full, "{} layer {l}", v.id);
                if !full {
                    assert_eq!(a.window, v.schedule.sliding_window, "{} layer {l}", v.id);
                }
            }
        }
    }

    /// Two rope bases, two orders of magnitude apart, each on the layers
    /// that rotate at it.
    #[test]
    fn every_row_carries_both_rope_bases() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("servable");
            let local = d.attention[0].rope_theta;
            let global = d.attention[5].rope_theta;
            assert_eq!(local, 10_000.0, "{}", v.id);
            assert_eq!(global, 1_000_000.0, "{}", v.id);
            for (l, a) in d.attention.iter().enumerate() {
                let expected = if (l + 1) % 6 == 0 { global } else { local };
                assert_eq!(a.rope_theta, expected, "{} layer {l}", v.id);
            }
        }
    }

    /// The 27B is the row that makes `query_pre_attn_scalar` a field
    /// rather than a synonym for the head dim.
    #[test]
    fn the_27b_scales_by_its_own_scalar_and_not_by_its_head_dim() {
        let v = row("gemma-3-27b");
        assert_eq!(v.shape.head_dim, 128);
        assert_eq!(v.schedule.query_pre_attn_scalar, 168);
        let d = v.deployment(Deployed::single()).expect("servable");
        assert!((d.attention[0].sm_scale - 1.0 / 168f32.sqrt()).abs() < 1e-6);
        // Every other row states a scalar that happens to equal its head
        // dim, which is why this one is the test.
        for other in VARIANTS.iter().filter(|o| o.id != "gemma-3-27b") {
            assert_eq!(other.schedule.query_pre_attn_scalar, other.shape.head_dim, "{}", other.id);
        }
    }

    /// The three facts the generic derivation got wrong are stated on
    /// every row: gemma's `(1 + w)` norm, the four-norm block, and the
    /// per-head q/k norm.
    #[test]
    fn the_norms_are_stated_rather_than_derived_from_a_checkpoints_shape() {
        for v in VARIANTS {
            assert_eq!(v.shape.norm_variant, NormVariant::Gemma, "{}", v.id);
            assert_eq!(v.shape.norm_placement, NormPlacement::Sandwich, "{}", v.id);
            assert_eq!(v.shape.qk_norm, QkNorm::PerHead, "{}", v.id);
        }
    }

    /// Five rows, five checkpoints nothing can confuse.
    #[test]
    fn no_two_rows_describe_the_same_checkpoint() {
        let fingerprint = |v: &Gemma3| -> Vec<String> {
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

    /// The template is gemma's, from `shared/` — where it lives
    /// because gemma-3n binds the same one, and a generation may not
    /// reach into a sibling for it.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_gemmas_own_and_not_chatml() {
        let vocab: Vec<String> =
            ["<start_of_turn>", "<end_of_turn>", "<eos>", "<bos>", "user", "model", "\n", "Hi"]
                .iter()
                .map(|s| (*s).to_string())
                .collect();
        let tok = Arc::new(tokenizer::Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let chat = v.chat(tok.clone());
            assert!(chat.seal().contains(&1), "{} does not seal with <end_of_turn>", v.id);
            assert!(chat.seal().contains(&2), "{} does not seal with <eos>", v.id);
            assert!(chat.cue().starts_with(&[0]), "{} cues with <start_of_turn>", v.id);
        }
    }

    /// The ids an operator types.
    #[test]
    fn the_ids_are_the_ones_an_operator_types() {
        let ids: Vec<&str> = VARIANTS.iter().map(|v| v.id).collect();
        assert_eq!(
            ids,
            vec!["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b", "embeddinggemma-300m"],
        );
    }
}
