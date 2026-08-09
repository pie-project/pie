//! Which text the loaded checkpoint is.
//!
//! # Selecting is not choosing
//!
//! "Nothing in the driver may choose a kernel" is the crate's governing rule,
//! and a reader could mistake this module for a breach of it. It is not, and
//! the distinction is worth stating precisely because it will be reached for
//! again when the other three families land.
//!
//! A *choice* is the driver deciding what to run. What happens here is a
//! **lookup**: the checkpoint states its architecture, and this answers with
//! the text written for it. Nothing about which kernels fire is decided here —
//! the text names every symbol, the lowering flattens it, and the executor
//! walks the result. Remove this module and the same kernels would run; you
//! would simply have no way to say which model you loaded.
//!
//! The test is the one `metal.md` gives for the whole crate: *does removing it
//! change which kernels fire?* It does not.
//!
//! # Why the driver and not the engine
//!
//! It could sit in the seam instead. It sits here because running the model it
//! loaded is the driver's job, and because `driver-cuda-new`'s shell does the
//! same in `pie_cuda_launch` for the same reason. A seam that selected texts
//! would have to know every family, and it would learn a new one every time a
//! driver did.

use model_compiler::trace::{FireClass, ForwardPlan};

/// Why no text could be selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unfamiliar {
    /// No text has been written for this architecture.
    ///
    /// Three of four families are in this state today (`gemma4`, `gpt-oss`,
    /// `qwen`), and it is the honest report: their driver code still exists
    /// but the text that would replace it does not.
    NoText {
        /// What the checkpoint called itself.
        arch: String,
        /// The architectures a text does exist for.
        known: Vec<&'static str>,
    },
}

/// The architectures `llama_like`'s text serves.
///
/// One text covers many architectures — that is what makes it a *family* — so
/// this is a list rather than a name. Every entry is a deployment the family's
/// facts can describe.
const LLAMA_LIKE: &[&str] = &[
    "llama", "llama3", "llama4", "mistral", "phi3", "olmo2", "qwen2", "qwen3",
];

/// Every architecture some text serves.
#[must_use]
pub fn known() -> Vec<&'static str> {
    LLAMA_LIKE.to_vec()
}

/// The text for `arch`, traced for `class`.
///
/// `facts` and `metal` are the deployment's, and they are the caller's to
/// supply because they come from the checkpoint's descriptor — not from
/// anything this module could derive.
///
/// # Errors
///
/// [`Unfamiliar::NoText`], naming the architecture and what is known. A driver
/// that guessed a text here would run a different model's program against this
/// checkpoint's weights, which is fluent nonsense rather than a failure.
pub fn plan_for(
    arch: &str,
    class: FireClass,
    facts: &model::families::llama_like::forward::facts::LlamaLikeFacts,
    metal: &model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) -> Result<ForwardPlan, Unfamiliar> {
    if LLAMA_LIKE.contains(&arch) {
        return Ok(model::families::llama_like::forward::llama_like_metal(
            facts, metal, class,
        ));
    }
    Err(Unfamiliar::NoText {
        arch: arch.to_string(),
        known: known(),
    })
}

/// Whether any text serves `arch`.
#[must_use]
pub fn serves(arch: &str) -> bool {
    LLAMA_LIKE.contains(&arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_text_serves_many_architectures_which_is_what_a_family_is() {
        assert!(serves("qwen3"));
        assert!(serves("llama3"));
        assert!(serves("mistral"));
    }

    #[test]
    fn an_architecture_with_no_text_says_so_and_says_what_is_known() {
        // Three of four families are here today. A driver that guessed would
        // run another model's program against these weights, which is fluent
        // nonsense rather than a failure.
        assert!(!serves("gemma4"));
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        match plan_for("gemma4", FireClass::Decode, &facts, &metal) {
            Err(Unfamiliar::NoText { arch, known }) => {
                assert_eq!(arch, "gemma4");
                assert!(known.contains(&"qwen3"), "and what IS served: {known:?}");
            }
            Ok(_) => panic!("gemma4 has no metal text yet"),
        }
    }

    #[test]
    fn a_served_architecture_traces_a_plan_of_that_family() {
        let facts = model::families::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b();
        let metal = model::families::llama_like::forward::facts::LlamaLikeMetalFacts::synthetic();
        let plan = plan_for("qwen3", FireClass::Decode, &facts, &metal).expect("qwen3 is served");
        assert!(
            plan.family.starts_with("llama_like"),
            "the plan states its family: {}",
            plan.family
        );
    }
}

/// The deployment's facts, derived from what the descriptor says.
///
/// # Why this is not a fixture
///
/// It replaced one. The seam's `launch` took `LlamaLikeFacts::qwen3_0_6b()` —
/// a TEST fixture — for every checkpoint, so a deployment with different head
/// counts would have traced another model's program against its weights and
/// answered fluent nonsense. A fixture standing where a fact belongs is the
/// exact defect this crate keeps finding, and it is worth naming when it is
/// one's own.
///
/// # What is still assumed, and why each is stated rather than hidden
///
/// `DecodeGeometry` carries the shape; three facts it has no field for are
/// stated here with the reason:
///
/// * `qk_norm` — read from the checkpoint's own tensors, because a per-head
///   Q/K norm exists exactly when the weights for it do.
/// * `fused_qkv` — whether the deployment binds ONE packed projection. This
///   asks the tensors too: a checkpoint with `self_attn.qkv_proj` fused it,
///   one with `q_proj`/`k_proj`/`v_proj` did not, and the text traces
///   differently for each.
/// * `qkv_bias` — the qwen-2 attention biases, again by tensor.
///
/// Asking the checkpoint rather than the config is deliberate: the config
/// states an architecture and the tensors state a BINDING, and every one of
/// these three is a binding fact.
#[must_use]
pub fn facts_from(
    geometry: &crate::batch::DecodeGeometry,
    has_tensor: impl Fn(&str) -> bool,
) -> (
    model::families::llama_like::forward::facts::LlamaLikeFacts,
    model::families::llama_like::forward::facts::LlamaLikeMetalFacts,
) {
    use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model_compiler::dsl::{ScaleLayout, WeightRepr};

    let facts = LlamaLikeFacts {
        hidden: geometry.hidden,
        layers: geometry.n_layers,
        q_heads: geometry.n_q_heads,
        kv_heads: geometry.n_kv_heads,
        head_dim: geometry.head_dim,
        intermediate: geometry.intermediate,
        vocab: geometry.vocab,
        rope: model_compiler::trace::RopeKind::Standard,
        norm_variant: model_compiler::trace::NormVariant::Plain,
        norm_placement: model::families::llama_like::forward::facts::NormPlacement::Pre,
        qk_norm: if has_tensor("layers.0.self_attn.q_norm.weight") {
            model_compiler::facts::QkNorm::PerHead
        } else {
            model_compiler::facts::QkNorm::Off
        },
        fused_qkv: has_tensor("layers.0.self_attn.qkv_proj.weight"),
        tied_embeddings: geometry.tied_embeddings,
        qkv_bias: has_tensor("layers.0.self_attn.q_proj.bias"),
        // Asked of the TENSORS, like the other binding facts: a router weight
        // in the checkpoint is a mixture and its absence is a dense FFN. The
        // counts come from the geometry, which the fire already states.
        n_experts: if has_tensor("layers.0.mlp.gate.weight") {
            geometry.n_experts
        } else {
            0
        },
        experts_per_token: geometry.experts_per_token,
        moe_intermediate: geometry.moe_intermediate,
        // A shared expert is its own tensor, and qwen3-moe has none.
        shared_intermediate: if has_tensor("layers.0.mlp.shared_expert.up_proj.weight") {
            geometry.moe_intermediate
        } else {
            0
        },
    };
    let metal = LlamaLikeMetalFacts {
        fuse_residual_gemv: true,
        paged_multi_batch: true,
        qmm_multi_batch: true,
        // The checkpoint's own affine format. `AffineFormat` is what the
        // descriptor stated; a pipeline built for the wrong point returns
        // fluent nonsense rather than failing, which is why neither number is
        // inferred from a tensor's shape (g64/b8 and g128/b4 pack identically).
        proj_repr: WeightRepr::Scaled {
            layout: ScaleLayout::PerGroup,
            group: geometry.quant.group,
            axis: 0,
            zero_point: true,
        },
        affine_bits: geometry.quant.bits,
        // The narrowest rung, which is what a short window fires; `bn = 32` is
        // the only column tile the residual GEMM is instantiated at.
        qmm_tile: (16, 32),
        // Asked of the tensors, like the other binding facts. It is false for
        // every MLX checkpoint -- `compile_load_plan` authors with
        // `Projections::InPlace` and the join declines under it -- but asking
        // is what makes that a finding rather than an assumption.
        gate_up_fused: has_tensor("layers.0.mlp.gate_up_proj.fused.weight"),
        // The checkpoint's own `rms_norm_eps`. A norm handed zero divides by
        // the root of the mean square alone, which for a near-zero row is an
        // infinity the next kernel spreads everywhere.
        rms_eps: geometry.eps,
        // The checkpoint's own rotary base.
        rope_theta: geometry.rope_theta,
        // Whether the ladder is RESCALED, in which case no base expresses it
        // and the driver hands over a table instead.
        rope_freq_table: geometry.rope_freq_factor > 0.0,
        // gemma's side network, asked of the TENSORS: a checkpoint that ships
        // a per-layer embedding table is a deployment that has one.
        per_layer_emb_dim: 0,
        per_layer_scalar: false,
        dense_beside_moe: false,
        // The CONFIG's, not a tensor's. Asking layer 0 was the first draft
        // and it is a SLIDING layer, which ships its `v_proj` — only the full
        // ones do not, and a fact derived from the wrong layer is false for
        // exactly the deployment it describes.
        v_from_k: geometry.attention_k_eq_v,
        kv_shared_layers: 0,
        // gemma's readout cap. Zero is "none" and the text names nothing.
        logit_softcap: geometry.final_logit_softcap,
        // Asked of the TENSORS: a sink is a weight, and a checkpoint that
        // ships one is a deployment that has them.
        attn_sinks: has_tensor("layers.0.self_attn.sinks"),
        // WHICH activation. A swiglu limit of zero is "this deployment is not
        // gpt-oss" and not a clamp at zero, which would zero the gate branch
        // entirely. `Geglu` is gemma's and reaches here when a gemma text does.
        activation: if geometry.swiglu_limit > 0.0 {
            model::families::llama_like::forward::facts::Activation::SwiGlu {
                limit: geometry.swiglu_limit,
                alpha: geometry.swiglu_alpha,
            }
        } else {
            model::families::llama_like::forward::facts::Activation::SiluMul
        },
        // Empty is every layer attending the whole context, which is what a
        // llama-like deployment does. `DecodeGeometry` carries no window at
        // all, so this is the honest answer and not a default: the families
        // that alternate (gemma4, gpt-oss) will need the geometry to state one
        // before their texts can, and stating it here from nothing would make
        // that a silent wrong answer instead of a missing one.
        // Which layers slide. A gemma4 stack alternates: every
        // `full_attn_every`-th layer attends everything and the rest attend a
        // window. Empty for a deployment that does not alternate, which is
        // every llama-like one.
        window_left: if geometry.full_attn_every > 1 && geometry.sliding_window > 0 {
            (0..geometry.n_layers)
                .map(|l| {
                    if (l + 1).is_multiple_of(geometry.full_attn_every) {
                        -1
                    } else {
                        i32::try_from(geometry.sliding_window).unwrap_or(-1)
                    }
                })
                .collect()
        } else {
            Vec::new()
        },
    };
    (facts, metal)
}

#[cfg(test)]
mod facts_tests {
    use super::*;

    fn geometry() -> crate::batch::DecodeGeometry {
        crate::batch::DecodeGeometry {
            hidden: 1024,
            n_layers: 28,
            vocab: 151_936,
            n_q_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate: 3072,
            tied_embeddings: true,
            quant: crate::batch::AffineFormat {
                bits: 4,
                group: 64,
            },
            ..crate::batch::DecodeGeometry::default()
        }
    }

    #[test]
    fn the_shape_comes_from_the_descriptor_and_not_from_a_fixture() {
        // This replaced `LlamaLikeFacts::qwen3_0_6b()` standing in the seam's
        // `launch` for every checkpoint — a fixture where a fact belongs, and
        // a deployment with other head counts would have traced another
        // model's program against its weights.
        let (facts, metal) = facts_from(&geometry(), |_| false);
        assert_eq!(facts.hidden, 1024);
        assert_eq!(facts.layers, 28);
        assert_eq!(facts.q_heads, 16);
        assert_eq!(facts.kv_heads, 8);
        assert_eq!(facts.intermediate, 3072);
        assert!(facts.tied_embeddings);
        assert_eq!(metal.affine_bits, 4, "the checkpoint's own affine format");
    }

    #[test]
    fn the_binding_facts_ask_the_tensors_rather_than_the_config() {
        // A config states an architecture; a tensor states a BINDING. Whether
        // this deployment fused its QKV is answerable only by looking.
        let split = facts_from(&geometry(), |_| false).0;
        assert!(!split.fused_qkv, "no fused tensor, so the text traces three");
        assert_eq!(split.qk_norm, model_compiler::facts::QkNorm::Off);
        assert!(!split.qkv_bias);

        let fused = facts_from(&geometry(), |name| {
            name == "layers.0.self_attn.qkv_proj.weight"
                || name == "layers.0.self_attn.q_norm.weight"
                || name == "layers.0.self_attn.q_proj.bias"
        })
        .0;
        assert!(fused.fused_qkv);
        assert_eq!(fused.qk_norm, model_compiler::facts::QkNorm::PerHead);
        assert!(fused.qkv_bias);
    }

    #[test]
    fn the_affine_point_follows_the_checkpoints_group_and_width() {
        // g64/b8 and g128/b4 pack to identical shapes, so neither number can
        // be inferred from a tensor — a pipeline built for the wrong point
        // returns fluent nonsense rather than failing.
        let g = crate::batch::DecodeGeometry {
            quant: crate::batch::AffineFormat {
                bits: 8,
                group: 128,
            },
            ..geometry()
        };
        let (_, metal) = facts_from(&g, |_| false);
        assert_eq!(metal.affine_bits, 8);
        assert_eq!(
            model_compiler::dsl::metal::affine_point(metal.proj_repr, metal.affine_bits),
            "_bfloat16_gs_128_b_8"
        );
    }
}
