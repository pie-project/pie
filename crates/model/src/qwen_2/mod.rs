//! Qwen 2 — the 2.5 release.
//!
//! One generation, one directory: the rows below and the ChatML
//! constructor in [`chat`]. The authoring pass is the lineage's, next
//! door in `crate::llama_3::contract`, which is what "one family, many
//! rows" means in this crate — a generation adds the two answers a shape
//! cannot give, and borrows the rest.
//!
//! # The attention bias is the generation's signature
//!
//! Qwen2 is the llama block plus `{q,k,v}_proj.bias`, and that one bit
//! travelled badly. It arrived as `attention_bias` out of a
//! `config.json` and was believed; a checkpoint whose converter dropped
//! the key deployed as an unbiased model that loaded, ran, and answered
//! subtly wrongly. Here it is a manifest ROW — `layer.{}.q_proj.bias`,
//! required, `[q_heads * head_dim]` — so a checkpoint without it is not
//! this variant rather than being this variant deployed wrong.
//!
//! # The window that was read but never switched on
//!
//! Every Qwen2.5 config carries `sliding_window` (32768 or 131072) AND
//! `use_sliding_window: false`. The old derivation passed the width
//! through because it read the key; the switch beside it decided
//! nothing. These rows state `window: -1`, full attention, which is what
//! the checkpoints actually do — and what the committed
//! [`LlamaLikeFacts::qwen2_5_1_5b`] fixture already says by calling the
//! window "unused".

#[cfg(feature = "chat")]
pub mod chat;

use std::sync::{Arc, OnceLock};

use crate::catalog::{Deployed, LoadShape, Variant};
use crate::families::llama_like::project;
use crate::families::llama_like::spec::LlamaLikeFacts;
use crate::manifest::Manifest;

use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::trace::{NormVariant, RopeKind};

/// One Qwen 2.5 checkpoint.
pub struct Qwen2 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency — 1e6 across the generation.
    pub rope_theta: f32,
    /// The RMSNorm epsilon — `1e-6` for every Qwen2/2.5 release, an
    /// order of magnitude tighter than Llama 3's `1e-5`. Stated on the
    /// row because no tensor extent carries it and the manifest
    /// therefore cannot check it: get it wrong and the model still
    /// loads, still runs, and quietly denormalises its activations.
    pub norm_eps: f32,
    /// Sliding-window width, `-1` for full attention. See the module
    /// doc: the configs ship a width behind a switch that is off.
    pub window: i32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `qwen2` for the 2.5 release, which looks like an off-by-one and is
/// not: `model_type` is `qwen2` in every published Qwen2.5 config, and
/// `architectures[0]` is `Qwen2ForCausalLM` — the stem the worker's
/// heuristic reduces it to and then checks against
/// [`crate::catalog::arches`] (see
/// `embedded_driver::read_hf_config_defaults`). "2.5" is a RELEASE and
/// the ids carry it; the family the checkpoints name is Qwen2, and a row
/// advertising `qwen2.5` would make that check refuse all seven of them.
const ARCH: &str = "qwen2";

/// The published context ceiling, shared by the seven Instruct rows.
///
/// One constant because these seven checkpoints are the INSTRUCT tunes
/// (see the table below) and every one of them states
/// `max_position_embeddings: 32768`, from the 0.5B to the 72B. The
/// number that is NOT here is 131 072: the BASE checkpoints of five of
/// these sizes state it — 1.5B, 7B, 14B, 32B and 72B — and four of the
/// Instruct configs put it on `sliding_window` instead. A row is a
/// shape, and no shape tells a base tune from an instruct one — so a
/// checkpoint whose weights are the base model matches this row and is
/// advertised the instruct ceiling.
/// That is a known and stated over-narrow claim: 32 768 is the length
/// Qwen documents these tunes for without YaRN, and advertising the
/// longer one would promise positions the released config does not.
///
/// A training-time fact rather than a deployment one — nothing in a fire
/// reads it, and a driver serving a shorter context is serving
/// correctly.
const MAX_MODEL_LEN: u32 = 32_768;

/// The generation's rows.
///
/// The seven Instruct sizes, which are the seven a boundary names.
/// `qkv_bias: true` throughout is not a default reaching down the
/// column: it is a measurement, repeated because each row is its own
/// measurement, and the row that can be checked against a committed
/// fixture ([`LlamaLikeFacts::qwen2_5_1_5b`]) is checked below.
///
/// The vocabulary splits the generation in two, which is the fact most
/// worth reading off the table: 151_936 for the sizes with a tied head,
/// 152_064 for the ones with their own `lm_head`. Both are the same
/// tokenizer padded differently, and a row that guessed one would fail
/// its manifest against half the family.
pub const VARIANTS: &[Qwen2] = &[
    // Qwen/Qwen2.5-0.5B-Instruct. GQA at 14 q / 2 kv heads, head_dim 64
    // (896 / 14) — the only row here narrower than 128.
    Qwen2 {
        id: "qwen2.5-0.5b",
        shape: LlamaLikeFacts {
            hidden: 896,
            layers: 24,
            q_heads: 14,
            kv_heads: 2,
            head_dim: 64,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 4864,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-1.5B-Instruct — the committed fixture, and the
    // workspace's llama_like-with-biases parity shape.
    Qwen2 {
        id: "qwen2.5-1.5b",
        shape: LlamaLikeFacts {
            hidden: 1536,
            layers: 28,
            q_heads: 12,
            kv_heads: 2,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 8960,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-3B-Instruct.
    Qwen2 {
        id: "qwen2.5-3b",
        shape: LlamaLikeFacts {
            hidden: 2048,
            layers: 36,
            q_heads: 16,
            kv_heads: 2,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 11_008,
            vocab: 151_936,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: true,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-7B-Instruct — the first untied head, and with it the
    // wider vocabulary.
    Qwen2 {
        id: "qwen2.5-7b",
        shape: LlamaLikeFacts {
            hidden: 3584,
            layers: 28,
            q_heads: 28,
            kv_heads: 4,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 18_944,
            vocab: 152_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-14B-Instruct.
    Qwen2 {
        id: "qwen2.5-14b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 48,
            q_heads: 40,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 13_824,
            vocab: 152_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-32B-Instruct — the 14B's width run deeper, with twice
    // the FFN.
    Qwen2 {
        id: "qwen2.5-32b",
        shape: LlamaLikeFacts {
            hidden: 5120,
            layers: 64,
            q_heads: 40,
            kv_heads: 8,
            head_dim: 128,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            intermediate: 27_648,
            vocab: 152_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
        window: -1,
    },
    // Qwen/Qwen2.5-72B-Instruct.
    Qwen2 {
        id: "qwen2.5-72b",
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
            intermediate: 29_568,
            vocab: 152_064,
            rope: RopeKind::Standard,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Pre,
            qk_norm: QkNorm::Off,
            fused_qkv: true,
            tied_embeddings: false,
            qkv_bias: true,
        },
        rope_theta: 1_000_000.0,
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

impl Variant for Qwen2 {
    fn id(&self) -> &'static str {
        self.id
    }

    fn manifest(&self) -> Manifest {
        project::manifest(&self.shape)
    }

    /// Dense throughout — the Qwen mixtures are `qwen_3`'s and their own
    /// generations', not this one's.
    fn load_shape(&self) -> LoadShape {
        LoadShape::dense(self.shape.layers, self.shape.head_dim, self.shape.tied_embeddings)
    }

    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal> {
        let _ = load;
        let mut deployment =
            project::deployment(&self.shape, self.rope_theta, self.norm_eps, self.window);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // No tower. The Qwen2 line's vision releases are Qwen2-VL and
            // Qwen2.5-VL, separate checkpoints with a `visual` block and
            // no row here; every row in this table is a text stack.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The lineage's pass, which handles the biases: the dense join
    /// re-fuses the q/k/v WEIGHTS and leaves the bias vectors as three
    /// tensors added after the split, which is the hand-written order.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        crate::llama_3::contract::author_llama_like(builder)
    }

    #[cfg(feature = "forward")]
    fn trace(
        &self,
        class: model_compiler::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_compiler::trace::ForwardPlan, crate::deployment::Refusal> {
        Ok(project::trace(&self.shape, class, load))
    }

    /// ChatML with tools and WITHOUT thinking, which is why this names
    /// the generation's own constructor rather than one of the four
    /// configurations in [`crate::families::chatml`]: `QWEN_CHATML`
    /// turns thinking on (Qwen 3's), `PLAIN_CHATML` turns tools off.
    /// Qwen 2.5 is neither, and the difference is visible — a replayed
    /// assistant turn keeps its `<think>` block here because there is no
    /// thinking protocol to strip it for.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Qwen2 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// The fixture and the row are the same measurement, so if they
    /// disagree one of them is wrong.
    #[test]
    fn the_row_agrees_with_the_committed_fixture() {
        assert_eq!(row("qwen2.5-1.5b").shape, LlamaLikeFacts::qwen2_5_1_5b());
    }

    /// Every row answers every question.
    #[test]
    fn every_row_projects() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("dense qwen2 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.intermediate, v.shape.intermediate);
            assert_eq!(d.norm, crate::deployment::NormPlacement::Pre);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, v.shape.head_dim, "the TRUE head dim, never a padded one");
            assert_eq!(ls.tied_embeddings, v.shape.tied_embeddings);
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.rope_theta, 1_000_000.0, "{}", v.id);
                assert_eq!(a.kv_source, l as u32);
                assert_eq!(a.sm_scale, 1.0 / (a.head_dim as f32).sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 7);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(
            ids,
            [
                "qwen2.5-0.5b",
                "qwen2.5-1.5b",
                "qwen2.5-3b",
                "qwen2.5-7b",
                "qwen2.5-14b",
                "qwen2.5-32b",
                "qwen2.5-72b",
            ]
        );
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` in
    /// `driver-cuda` — `model_type`, `max_position_embeddings`, and
    /// "does this ship a tower" — and keeping a parsed `config.json`
    /// alive for the life of a load to answer them is what this replaces.
    /// All seven rows are asserted, because all seven advertise.
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v.deployment(Deployed::single()).expect("dense qwen2 deploys").advertised;
            assert_eq!(a.arch, "qwen2", "{}: the family label a guest program sees", v.id);
            assert_eq!(
                a.max_model_len, 32_768,
                "{}: every Qwen2.5 Instruct config states it, 0.5B through 72B",
                v.id
            );
            assert!(!a.media_encode, "{}: the VL releases are other checkpoints", v.id);
        }
    }

    /// The label is the stem the WORKER derives, which is the whole
    /// reason it can be checked at the boundary rather than believed:
    /// `read_hf_config_defaults` lowercases `architectures[0]`, strips
    /// the task suffix, and refuses what [`crate::catalog::arches`] does
    /// not list. `qwen2.5` here would refuse `Qwen2ForCausalLM`, which is
    /// every checkpoint this table claims.
    #[test]
    fn the_label_is_the_family_the_checkpoints_name_not_the_release() {
        let stem = "Qwen2ForCausalLM"
            .to_lowercase()
            .strip_suffix("forcausallm")
            .expect("the suffix the worker strips")
            .to_string();
        assert_eq!(stem, ARCH);
        assert!(!ARCH.contains("2.5"), "the release rides in the id, not in the family label");
        for v in VARIANTS {
            assert!(v.id.starts_with("qwen2.5-"), "{}: the id is where the release is", v.id);
        }
    }

    /// The generation's signature, as an expectation a checkpoint either
    /// meets or does not: the bias is `q_heads * head_dim` wide and the
    /// manifest requires it on every row.
    #[test]
    fn the_attention_bias_is_a_manifest_row_rather_than_a_config_key() {
        for v in VARIANTS {
            assert!(v.shape.qkv_bias, "{}", v.id);
            let m = v.manifest();
            let bias = m
                .tensors
                .iter()
                .find(|t| t.name == "layer.{}.self_attn.q_proj.bias")
                .unwrap_or_else(|| panic!("{} requires the bias it ships", v.id));
            assert_eq!(bias.presence, crate::manifest::Presence::Required, "{}", v.id);
            assert_eq!(bias.extents, vec![u64::from(v.shape.q_width())], "{}", v.id);
        }
    }

    /// The switch beside the width is off, so the width is not a fact
    /// about the deployment. Stating `-1` is a deliberate departure from
    /// the derivation, which passed `sliding_window` through because it
    /// read that key and not the one next to it.
    #[test]
    fn the_window_is_full_because_the_switch_is_off() {
        for v in VARIANTS {
            assert_eq!(v.window, -1, "{}", v.id);
            let d = v.deployment(Deployed::single()).expect("servable");
            assert!(d.attention.iter().all(|a| a.window == -1), "{}", v.id);
        }
    }

    /// Tie and vocabulary move together in this generation, and the
    /// manifest states both: the head's absence and the embedding
    /// table's width.
    #[test]
    fn the_small_sizes_tie_their_head_and_carry_the_narrower_vocabulary() {
        for (id, tied, vocab) in [
            ("qwen2.5-0.5b", true, 151_936u32),
            ("qwen2.5-1.5b", true, 151_936),
            ("qwen2.5-3b", true, 151_936),
            ("qwen2.5-7b", false, 152_064),
            ("qwen2.5-14b", false, 152_064),
            ("qwen2.5-32b", false, 152_064),
            ("qwen2.5-72b", false, 152_064),
        ] {
            let v = row(id);
            assert_eq!(v.shape.tied_embeddings, tied, "{id}");
            assert_eq!(v.shape.vocab, vocab, "{id}");

            let m = v.manifest();
            let head = m.tensors.iter().find(|t| t.name == "lm_head").expect("stated either way");
            let want = if tied {
                crate::manifest::Presence::Absent
            } else {
                crate::manifest::Presence::Required
            };
            assert_eq!(head.presence, want, "{id}");
            let embed = m.tensors.iter().find(|t| t.name == "embed_tokens").expect("stated");
            assert_eq!(embed.extents, vec![u64::from(vocab), u64::from(v.shape.hidden)], "{id}");
        }
    }

    /// No q/k norm anywhere in Qwen 2.5 — Qwen 3 added it — and the
    /// manifest says so by FORBIDDING the tensor, which is what keeps a
    /// Qwen3 checkpoint from matching a Qwen2 row.
    #[test]
    fn the_generation_predates_qk_norm() {
        for v in VARIANTS {
            assert_eq!(v.shape.qk_norm, QkNorm::Off, "{}", v.id);
            let m = v.manifest();
            let q_norm = m
                .tensors
                .iter()
                .find(|t| t.name == "layer.{}.self_attn.q_norm")
                .expect("stated as an absence");
            assert_eq!(q_norm.presence, crate::manifest::Presence::Absent, "{}", v.id);
        }
    }

    /// ChatML, with the thinking protocol OFF — which is the difference
    /// between this generation's constructor and Qwen 3's, and it is
    /// observable rather than a comment: a replayed assistant turn keeps
    /// its `<think>` block, because there is none to strip.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_chatml_without_the_thinking_protocol() {
        use crate::instruct::Instruct;
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "\n",
            "system",
            "user",
            "assistant",
            "<think>",
            "</think>",
            "Hi",
            "Bye",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));

        let inst = row("qwen2.5-7b").chat(tok.clone());
        assert_eq!(tok.decode(&inst.user("Hi"), false), "<|im_start|>user\nHi<|im_end|>\n");
        assert_eq!(tok.decode(&inst.cue(), false), "<|im_start|>assistant\n");

        let replayed = tok.decode(&inst.assistant("<think>Hi</think>Bye"), false);
        assert!(
            replayed.contains("<think>"),
            "Qwen 2.5 has no thinking protocol, so nothing is stripped on replay: {replayed}"
        );
    }

    #[cfg(feature = "chat")]
    #[test]
    fn every_row_answers_the_chat_question() {
        use crate::instruct::Instruct;
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = ["<|im_start|>", "<|im_end|>", "\n", "user", "Hi"]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let tok = Arc::new(Tokenizer::from_vocab(&vocab));
        for v in VARIANTS {
            let text = tok.decode(&v.chat(tok.clone()).user("Hi"), false);
            assert_eq!(text, "<|im_start|>user\nHi<|im_end|>\n", "{}", v.id);
        }
    }

    /// Every row has an author and it runs. The pass is the lineage's;
    /// what is under test is the row's answer to the question.
    #[cfg(feature = "contract")]
    #[test]
    fn every_row_authors() {
        use model_loader::checkpoint::CheckpointMetadata;
        use model_loader::plan::StorageTarget;

        let metadata = CheckpointMetadata { files: Vec::new(), tensors: Vec::new() };
        let encoding = crate::encoding::Encoding::dense();
        let target = StorageTarget::default();
        let policy = crate::policy::Policy::default();

        for v in VARIANTS {
            let mut builder = crate::builder::Builder::new(
                &metadata,
                v.id(),
                v.load_shape(),
                &encoding,
                &target,
                &policy,
            );
            v.author(&mut builder).unwrap_or_else(|e| panic!("{} refused to author: {e:?}", v.id));
        }
    }

    #[cfg(feature = "forward")]
    #[test]
    fn every_row_traces_both_fire_classes() {
        use model_compiler::trace::FireClass;

        for v in VARIANTS {
            for class in [FireClass::Decode, FireClass::Prefill] {
                let plan = v.trace(class, Deployed::single()).expect("llama-like traces");
                assert!(plan.family.contains("llama_like"), "{}: {}", v.id, plan.family);
                assert!(!plan.ops.is_empty(), "{}", v.id);
            }
        }
    }
}
