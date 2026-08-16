//! OLMo 3.
//!
//! Two rows. The block is OLMo 2's — post-norm, global q/k norm, no
//! attention bias — and the module doc next door in [`crate::olmo_2`]
//! explains what those two facts replaced. What is new here is the
//! attention schedule and the rope.
//!
//! # The window, and what a row can honestly say about it
//!
//! Olmo-3 ships `layer_types`: three `sliding_attention` layers then one
//! `full_attention`, repeating for the whole stack, with the sliding
//! ones at 4096. A row states ONE `window`, because
//! [`project::deployment`](crate::shared::llama_like::project::deployment)
//! writes one `LayerAttention` per layer from one width — the per-layer
//! table is `window_by_layer`'s, which gemma-2 and gemma-4 override and
//! the llama lineage does not.
//!
//! So `window: 4096` here says what the old derivation said —
//! `deployment_of` read `hf.sliding_window` for every layer of a
//! llama-like family, the interleave never having reached it — and it is
//! the conservative direction: every layer attends at most what its own
//! `layer_types` entry allows, and the full-attention layers attend less
//! than they could. Making the interleave expressible is a change to the
//! shared projection, not to a row, which is why it is not made here.
//!
//! # Yarn, stated
//!
//! `rope_scaling.rope_type: "yarn"`, factor 8 over an 8192-position
//! original window. `llama_like_facts_from_hf` wrote
//! [`RopeKind::Standard`] for every checkpoint it ever saw, having never
//! read `rope_scaling` at all; the row says what the checkpoint is.

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

/// One OLMo 3 checkpoint.
pub struct Olmo3 {
    /// The stable name a boundary carries.
    pub id: &'static str,
    /// The numbers.
    pub shape: LlamaLikeFacts,
    /// Rope's base frequency, under the yarn scaling the shape states.
    pub rope_theta: f32,
    /// The norm's epsilon, stated per row rather than shared, because
    /// it is a measurement of a checkpoint and the next release that
    /// changes it must change a number rather than break a rule.
    pub norm_eps: f32,
    /// Sliding-window width. See the module doc: one width for the
    /// stack, which is the shape a `Deployment` of this family has.
    pub window: i32,
}

/// The family label a GUEST PROGRAM matches on.
///
/// `olmo3`, which is what the boundary derives from
/// `architectures: ["Olmo3ForCausalLM"]` — the corpus's own
/// `allenai--Olmo-3-1025-7B.json` states exactly that — once the worker
/// has lowercased it and stripped `ForCausalLM`.
///
/// The digit is the whole difference from OLMo 2, and it carries real
/// weight: the two generations share `LlamaLikeFacts` and share
/// `project::deployment`, so a program that saw `olmo2` here would
/// select a chat template and a context budget built for a 4k model
/// while being served a 64k one.
const ARCH: &str = "olmo3";

/// The published context ceiling, shared by both rows.
///
/// 65 536, from the corpus's `allenai--Olmo-3-1025-7B.json` and from
/// `allenai/Olmo-3-1125-32B`, which both state
/// `max_position_embeddings: 65536`. One constant because the two rows
/// agree, and the 32B disagrees with the 7B about plenty else — it is
/// the first grouped attention in the OLMo line — so the agreement here
/// is a fact about the generation rather than an assumption carried
/// across it.
///
/// Sixteen times OLMo 2's 4096, which is the number this generation is
/// interesting for and the reason the two are not collapsed into one
/// shared constant somewhere above them both.
const MAX_MODEL_LEN: u32 = 65_536;

/// OLMo 3's YaRN rescaling, stated once because both rows share it.
///
/// Read off the corpus's `allenai--Olmo-3-1025-7B.json`, which states
/// all five numbers explicitly — including the `attention_factor` that
/// most configs omit. That makes this generation the place the formula
/// behind an omitted one is CHECKED rather than assumed: HF computes
/// `0.1 * ln(factor) + 1` when a config is silent, and
/// `0.1 * ln(8.0) + 1` is `1.2079441541679836`, which is exactly what
/// OLMo 3 writes down. See
/// `an_omitted_attention_factor_is_the_formula_olmo_3_states`.
///
/// The 32B row shares it. There is no `Olmo-3-1125-32B` config in the
/// corpus, so this is the same latitude [`MAX_MODEL_LEN`] takes — a
/// generation constant the 7B pins and the 32B inherits — and the
/// manifest match still refuses a checkpoint whose shape disagrees.
const ROPE_SCALING: crate::deployment::RopeScaling = crate::deployment::RopeScaling::Yarn {
    factor: 8.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    attention_factor: 1.207_944_2,
    original_max_position: 8_192,
    // OMITTED by the config, which is HF's default.
    truncate: true,
};

/// The generation's rows.
///
/// The vocabulary moved from OLMo 2's 100_352 to 100_278 — a different
/// tokenizer, not a different padding — so no OLMo 2 row and no OLMo 3
/// row can match the same checkpoint's `embed_tokens`. That is the
/// manifest doing the work a `model_type` of `"olmo2"` versus `"olmo3"`
/// used to do, except that it is checked against the checkpoint rather
/// than believed.
pub const VARIANTS: &[Olmo3] = &[
    // allenai/Olmo-3-1025-7B — the corpus's own
    // `allenai--Olmo-3-1025-7B.json`, and the row
    // `tests/catalog_differential.rs` pins against the old derivation.
    // MHA, like the whole of OLMo 2.
    Olmo3 {
        id: "olmo-3-7b",
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
            vocab: 100_278,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: 4096,
    },
    // allenai/Olmo-3-1125-32B — the first grouped attention in the OLMo
    // line: 40 query heads over 8 KV heads, where every OLMo 2 row and
    // the 7B above are multi-head.
    Olmo3 {
        id: "olmo-3-32b",
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
            vocab: 100_278,
            rope: RopeKind::Yarn,
            norm_variant: NormVariant::Plain,
            norm_placement: NormPlacement::Post,
            qk_norm: QkNorm::Global,
            fused_qkv: false,
            tied_embeddings: false,
            qkv_bias: false,
            o_bias: false,
            router_bias: false,
        },
        rope_theta: 500_000.0,
        norm_eps: 1e-6,
        window: 4096,
    },
];

crate::rows_of!(Olmo3);

impl Olmo3 {
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

impl Variant for Olmo3 {
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
        deployment.rope_scaling = Some(ROPE_SCALING);
        deployment.advertised = crate::deployment::Advertised {
            arch: ARCH,
            max_model_len: MAX_MODEL_LEN,
            // Text only, as OLMo 2 is. Neither release ships an
            // encoder, so `Deployment::towers` is empty and the
            // driver's encode entry has nothing here to serve.
            media_encode: false,
        };
        Ok(deployment)
    }

    /// The lineage's dense pass, as OLMo 2 uses: nothing about the norm
    /// placement or the q/k norms changes which tensors a contract
    /// declares.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error> {
        match builder.naming() {
            crate::shared::policy::Naming::Hf => {
                crate::shared::llama_like::contract::author_dense(builder)
            }
            // The registry this replaced held NO MLX row for olmo-3, and
            // the absence was a silence the caller read as "no
            // contract". Stated as the refusal it always was.
            crate::shared::policy::Naming::Mlx => crate::shared::builder::fail(
                "olmo-3: no MLX authoring pass exists for this family, so \
                 there is no name layout to author against",
            ),
        }
    }

    /// This row's text, for whichever backend asked.
    ///
    /// `rope_rescaled: TRUE`: OLMo 3 states YaRN, all five numbers,
    /// including the `attention_factor` most configs omit
    /// ([`ROPE_SCALING`]). No base expresses a YaRN ladder, so the
    /// Metal text reads the driver's derived frequency table instead —
    /// and a row that said otherwise would rotate every channel but the
    /// first at the wrong wavelength, at every position but zero.
    ///
    /// Its predecessor states none, which is why this is a row's answer
    /// rather than something a family assumes — see [`crate::olmo_2`].
    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal> {
        project::trace(&self.shape, self.row(), class, load)
    }

    /// ChatML-SHAPED, and its own template all the same: OLMo 3 moved
    /// from OLMo 2's `<|user|>` markers to `<|im_start|>role`, keeps
    /// `<|endoftext|>` as the stop, and opens the assistant turn already
    /// inside a `<think>`. Two of those three are what the ChatML
    /// fallback would have got wrong on a template that LOOKS like the
    /// one it hands out.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct> {
        Arc::new(chat::OlmoInstruct::new(tokenizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str) -> &'static Olmo3 {
        VARIANTS.iter().find(|v| v.id == id).expect("row present")
    }

    /// OLMo 3 states the `attention_factor` that most configs omit — and
    /// that makes it the oracle for every row that omits it.
    ///
    /// HF computes `0.1 * ln(factor) + 1` when a config is silent about
    /// `attention_factor`. `gpt_oss`'s rows depend on that formula, since
    /// `openai--gpt-oss-20b.json` states four of YaRN's five numbers and
    /// not the fifth. A constant copied from a formula nobody checked is
    /// exactly the kind of digit this catalog is supposed to make
    /// impossible — so it is checked HERE, against a published config
    /// that writes the answer down.
    ///
    /// `allenai--Olmo-3-1025-7B.json` states `1.2079441541679836` at
    /// `factor: 8.0`, and `0.1 * ln(8) + 1` is that number to every digit
    /// the file carries. The formula is therefore not an assumption.
    #[test]
    fn an_omitted_attention_factor_is_the_formula_olmo_3_states() {
        let crate::deployment::RopeScaling::Yarn {
            factor,
            attention_factor,
            ..
        } = ROPE_SCALING
        else {
            panic!("OLMo 3 rescales by YaRN");
        };
        assert_eq!(factor, 8.0, "the corpus config's factor");
        // The corpus's own digits, and the formula's value at that factor.
        let published = 1.207_944_154_167_983_6_f64;
        let formula = 0.1 * f64::from(factor).ln() + 1.0;
        assert!(
            (formula - published).abs() < 1e-12,
            "HF's default `attention_factor` formula does not reproduce the number \
             OLMo 3 publishes: formula {formula}, config {published}"
        );
        // And the row carries it, at f32 precision.
        assert!(
            (f64::from(attention_factor) - published).abs() < 1e-6,
            "the row states {attention_factor}, the config states {published}"
        );
    }

    /// The rescaling reaches a `Deployment`, on both rows.
    ///
    /// A YaRN ladder is not something a `rope_theta` expresses, and
    /// `driver-metal` REFUSES a row that states one rather than serving
    /// it with the rescaling flattened away — which is what the old
    /// reader did to every kind that was not `llama3`.
    #[test]
    fn both_rows_carry_the_yarn_rescaling_into_the_deployment() {
        for v in VARIANTS {
            let d = v.deployment(Deployed::single()).expect("OLMo 3 deploys");
            assert_eq!(
                d.rope_scaling,
                Some(ROPE_SCALING),
                "{} must state its ladder, or a driver serves it unrescaled",
                v.id
            );
        }
    }

    /// The three capability answers come off the ROW.
    ///
    /// They were the last three reads of a resident `HfConfig` inside
    /// `driver-cuda` — `model_type`, `max_position_embeddings` and
    /// whether a tower is present. Both rows are asserted because both
    /// advertise; a ceiling of 0 is checked apart from the value
    /// because 0 is what the DEFAULT carries and it would mean "this
    /// row does not say".
    #[test]
    fn the_row_answers_what_the_driver_advertises() {
        for v in VARIANTS {
            let a = v
                .deployment(Deployed::single())
                .expect("dense olmo3 is servable")
                .advertised;
            assert_eq!(
                a.arch, "olmo3",
                "{}: the family label a guest program sees",
                v.id
            );
            assert_eq!(
                a.max_model_len, 65_536,
                "{}: the corpus config and the 32B release both state 65536",
                v.id
            );
            assert_ne!(a.max_model_len, 0, "{}: 0 is 'the row does not say'", v.id);
            assert!(!a.media_encode, "{}: OLMo 3 ships no encoder tower", v.id);
        }
    }

    /// The label is what `architectures[0]` reduces to under the
    /// worker's heuristic — the corpus's own `allenai--Olmo-3-1025-7B`
    /// states `Olmo3ForCausalLM` — and the worker refuses any stem
    /// [`crate::catalog::arches`] does not list.
    #[test]
    fn the_label_is_what_architectures_reduces_to() {
        let stem = "Olmo3ForCausalLM"
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
                .expect("dense olmo3 is servable");
            assert_eq!(d.layers, v.shape.layers);
            assert_eq!(d.attention.len() as u32, v.shape.layers);
            assert_eq!(d.shape.hidden, v.shape.hidden);
            assert_eq!(d.shape.vocab, 100_278);
            assert_eq!(d.norm, crate::deployment::NormPlacement::Post);
            assert_eq!(v.manifest().layers, v.shape.layers);
            assert_eq!(v.id(), v.id);

            let ls = v.load_shape();
            assert_eq!(ls.layers, v.shape.layers);
            assert_eq!(ls.head_dim, 128, "{}", v.id);
            assert_eq!(
                ls.head_dim, v.shape.head_dim,
                "the TRUE head dim, never a padded one"
            );
            assert!(!ls.tied_embeddings, "no OLMo 3 row ties its head");
            assert_eq!(ls.n_experts, 0);
            assert_eq!(ls.mamba_groups, 0);
            assert_eq!(ls.kv_shared_layers, 0);

            for (l, a) in d.attention.iter().enumerate() {
                assert_eq!(a.window, 4096, "{} layer {l}", v.id);
                assert_eq!(a.rope_theta, 500_000.0, "{}", v.id);
                assert_eq!(a.head_dim, 128);
                assert_eq!(a.kv_source, l as u32);
                assert_eq!(a.sm_scale, 1.0 / 128.0_f32.sqrt());
            }
        }
        assert_eq!(VARIANTS.len(), 2);
        assert_eq!(rows().len(), VARIANTS.len());
    }

    #[test]
    fn the_ids_are_the_ones_the_generation_ships() {
        let ids: Vec<&str> = rows().iter().map(|v| v.id()).collect();
        assert_eq!(ids, ["olmo-3-7b", "olmo-3-32b"]);
    }

    /// The window is stated, uniform, and the same one the derivation
    /// applied — `deployment_of` gave every layer of a llama-like family
    /// `hf.sliding_window`, so this row deploys what that derived for
    /// the corpus config it is pinned against.
    #[test]
    fn the_window_is_one_width_for_the_whole_stack() {
        for v in VARIANTS {
            assert_eq!(v.window, 4096, "{}", v.id);
            let d = v.deployment(Deployed::single()).expect("servable");
            assert!(d.attention.iter().all(|a| a.window == 4096), "{}", v.id);
            assert!(
                !d.attention.iter().any(|a| a.window == -1),
                "{}: no layer is deployed with full attention, which is the \
                 conservative half of the interleave the row cannot state",
                v.id
            );
        }
    }

    /// The rope is scaled, and saying so is the departure from the
    /// derivation this replaces: it wrote `Standard` for every
    /// llama-like checkpoint, this one included.
    #[test]
    fn the_rope_is_the_scaled_ladder_rather_than_the_plain_one() {
        for v in VARIANTS {
            assert_eq!(v.shape.rope, RopeKind::Yarn, "{}", v.id);
            assert_ne!(v.shape.rope, RopeKind::Standard, "{}", v.id);
            assert_eq!(v.rope_theta, 500_000.0, "{}", v.id);
        }
    }

    /// OLMo 3 keeps OLMo 2's block, and the manifest carries both halves
    /// of it: the post-norm pair required, `input_layernorm` forbidden,
    /// and a q-norm spanning the whole projection.
    #[test]
    fn the_block_is_still_post_norm_with_a_global_qk_norm() {
        for v in VARIANTS {
            assert_eq!(v.shape.norm_placement, NormPlacement::Post, "{}", v.id);
            assert_eq!(v.shape.qk_norm, QkNorm::Global, "{}", v.id);

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
                spec("layer.{}.input_layernorm").presence,
                crate::manifest::Presence::Absent,
                "{}",
                v.id
            );
            let q_norm = spec("layer.{}.self_attn.q_norm");
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
        }
    }

    /// The 32B is grouped where the 7B is multi-head, which the manifest
    /// states as `k_proj` being a fifth of `q_proj` — the two rows are
    /// not one shape at two sizes.
    #[test]
    fn the_32b_is_grouped_where_the_7b_is_multi_head() {
        let seven = row("olmo-3-7b");
        let thirty_two = row("olmo-3-32b");

        assert_eq!(seven.shape.kv_heads, seven.shape.q_heads);
        assert_eq!(seven.shape.kv_width(), 4096);
        assert_eq!(thirty_two.shape.q_heads / thirty_two.shape.kv_heads, 5);
        assert_eq!(thirty_two.shape.q_width(), 5120);
        assert_eq!(thirty_two.shape.kv_width(), 1024);

        let m = thirty_two.manifest();
        let extent = |name: &str| -> Vec<u64> {
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .expect("stated")
                .extents
                .clone()
        };
        assert_eq!(extent("layer.{}.self_attn.k_proj"), vec![1024, 5120]);
        assert_eq!(extent("layer.{}.self_attn.q_proj"), vec![5120, 5120]);
        assert_ne!(seven.manifest(), thirty_two.manifest());
    }

    /// A different tokenizer from OLMo 2, so no checkpoint can satisfy
    /// both generations' embedding expectation. The manifest is where
    /// that separation lives now.
    #[test]
    fn the_vocabulary_separates_this_generation_from_olmo_2() {
        for v in VARIANTS {
            assert_eq!(v.shape.vocab, 100_278, "{}", v.id);
            assert_ne!(v.shape.vocab, 100_352, "{}: that is OLMo 2's", v.id);
            let m = v.manifest();
            let embed = m
                .tensors
                .iter()
                .find(|t| t.name == "embed_tokens")
                .expect("stated");
            assert_eq!(
                embed.extents,
                vec![100_278, u64::from(v.shape.hidden)],
                "{}",
                v.id
            );
        }
    }

    /// The template, exactly: ChatML's markers, OLMo's stop token, and a
    /// cue that opens the thinking block.
    #[cfg(feature = "chat")]
    #[test]
    fn the_template_is_chatml_shaped_with_olmos_own_stop_and_cue() {
        use tokenizer::Tokenizer;

        let vocab: Vec<String> = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "\n",
            "system",
            "user",
            "assistant",
            "environment",
            "<think>",
            "</think>",
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
                "<|im_start|>user\nHi<|im_end|>\n",
                "{}",
                v.id
            );
            let cue = tok.decode(&inst.cue(), false);
            assert!(
                cue.starts_with("<|im_start|>assistant\n"),
                "{}: {cue}",
                v.id
            );
            assert!(
                cue.contains("<think>"),
                "{}: the turn opens inside a think block",
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
        let seven = row("olmo-3-7b")
            .trace(model_ir::trace::FireClass::Decode, Deployed::single())
            .expect("traces");
        let thirty_two = row("olmo-3-32b")
            .trace(model_ir::trace::FireClass::Decode, Deployed::single())
            .expect("traces");
        assert!(
            thirty_two.ops.len() > seven.ops.len(),
            "64 layers out-op 32"
        );
    }
}
