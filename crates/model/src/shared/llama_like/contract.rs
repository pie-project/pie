//! Llama 3's load contract — and the dense schema most generations share.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/llama_like/llama_like_contract.hpp`.
//! Two of the three authors here are bound by generations that ship no
//! contract of their own: `author_llama_like` serves Llama 3, Mistral and
//! Qwen2/3, and `author_dense` serves a dozen `model_type`s across five
//! vendors whose checkpoints already use the names the bind path reads.
//! Those are rows in `model::contract::HF_ROWS`, not dependencies — a
//! generation with nothing of its own to say names the one that says it.
//!
//! Phi-3 used to live here, because it is dense with two source-side
//! fusions to undo. It has its own crate now (`pie-model-phi-3`): the
//! splits are Phi-3's alone, and what it shares is the three-pass dense
//! tail, which every generation spells out for itself.

use model_loader::error::Error;

use crate::shared::builder::Builder;
use crate::shared::mlx;

/// llama, llama3, mistral, qwen2, qwen3: the checkpoint already uses the
/// names the bind path reads. BF16 -> FP8/INT8 runtime quant is wired for
/// these.
pub fn author_llama_like(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// mistral3, ministral3, olmo2, olmo3: dense, nothing beyond the
/// name-pattern rules.
pub fn author_dense(b: &mut Builder<'_>) -> Result<(), Error> {
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// The Metal lowering of the llama-shaped families: rename for MLX's binder,
/// bind in place. Ported from
/// `crates/driver-metal/csrc/src/model/llama/llama_contract.hpp` — what is
/// family-specific is the tensor NAMES, and the mechanics are
/// [`mlx::author_mlx_file`].
pub fn author_llama_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    // `tie_word_embeddings` is what the config SAYS; a shipped `lm_head` is
    // what the checkpoint DOES. When they disagree the tensors win, because
    // mapping a real `lm_head` onto `shared_embedding` declares that name
    // twice and the load fails with a duplicate rather than with the
    // disagreement.
    let has_lm_head = b
        .tensors()
        .iter()
        .any(|raw| raw.name.starts_with("lm_head."));
    let tied = b.shape().tied_embeddings && !has_lm_head;
    mlx::author_mlx_file(b, "llama", &move |_, raw_name| {
        llama_mlx_name(raw_name, tied)
    })
}

/// The runtime name for a checkpoint tensor, or `None` to skip it. Every
/// name is either mapped or explicitly skipped; an unrecognised one is an
/// error, because a tensor declared under its checkpoint name would never be
/// found by the binder.
fn llama_mlx_name(raw_name: &str, tied: bool) -> Result<Option<String>, Error> {
    // A multimodal release wraps the decoder. Text decode binds none of the
    // towers.
    for skip in [
        "model.visual.",
        "model.vision_tower.",
        "model.audio_tower.",
        "visual.",
    ] {
        if raw_name.starts_with(skip) {
            return Ok(None);
        }
    }
    // Rotary inverse frequencies are recomputed on the GPU from `rope_theta`,
    // so a checkpoint that persists them is shipping a derived tensor.
    if raw_name.contains("rotary_emb.inv_freq") {
        return Ok(None);
    }

    if let Some(tail) = raw_name.strip_prefix("lm_head.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("lm_head.{tail}")
        }));
    }

    // Its own output is a valid input: see `mlx::already_lowered`. After the
    // `lm_head.` arm above, because that one is not an identity when tied.
    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }

    // Some releases nest the decoder one level deeper. Accept either.
    let rest = raw_name
        .strip_prefix("model.language_model.")
        .or_else(|| raw_name.strip_prefix("model."));
    let Some(rest) = rest else {
        return mlx::fail(format!(
            "Metal llama schema has no declared mapping or skip for '{raw_name}'"
        ));
    };

    if let Some(tail) = rest.strip_prefix("embed_tokens.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("embed_tokens.{tail}")
        }));
    }
    if rest == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }

    let (layer, member) = mlx::layer_member(rest, "llama", raw_name)?;
    // The mixture's naming is the same rule in every routed family, so it is
    // asked of one place rather than restated here.
    if let Some(renamed) = mlx::routed_expert_member(raw_name, member, "llama", false)? {
        return Ok(Some(format!("layers.{layer}.{renamed}")));
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(raw: &str, tied: bool) -> Option<String> {
        llama_mlx_name(raw, tied).unwrap_or_else(|e| panic!("'{raw}' is refused: {e:?}"))
    }

    /// Every arm of the rename, stated as the pair it maps.
    ///
    /// This is the function that decides what the Metal binder will look
    /// up. A wrong answer is not a crash: `Store::checkpoint_names` returns
    /// an empty candidate list for a name it does not know, so a tensor
    /// renamed wrongly is a tensor silently absent from the forward pass.
    #[test]
    fn every_name_the_metal_binder_reads_is_the_one_this_maps_to() {
        for (raw, tied, expected) in [
            ("model.embed_tokens.weight", false, "embed_tokens.weight"),
            ("model.embed_tokens.weight", true, "shared_embedding.weight"),
            ("lm_head.weight", false, "lm_head.weight"),
            ("lm_head.weight", true, "shared_embedding.weight"),
            ("model.norm.weight", false, "final_norm.weight"),
            (
                "model.layers.3.self_attn.q_proj.weight",
                false,
                "layers.3.self_attn.q_proj.weight",
            ),
            (
                "model.language_model.layers.3.mlp.down_proj.weight",
                false,
                "layers.3.mlp.down_proj.weight",
            ),
        ] {
            assert_eq!(
                mapped(raw, tied).as_deref(),
                Some(expected),
                "'{raw}' with tied={tied}"
            );
        }
    }

    /// A tied checkpoint publishes ONE embedding under one name.
    ///
    /// `embed_tokens` and `lm_head` both become `shared_embedding` when the
    /// weights are tied, which is what makes the tie real rather than a
    /// claim: two names would be two allocations of the largest tensor in a
    /// small model.
    #[test]
    fn tying_makes_the_two_embedding_names_one_name() {
        assert_eq!(
            mapped("model.embed_tokens.weight", true),
            mapped("lm_head.weight", true),
            "tied, the input and output embeddings are the same tensor"
        );
        assert_ne!(
            mapped("model.embed_tokens.weight", false),
            mapped("lm_head.weight", false),
            "untied, they are two"
        );
    }

    /// The tensors a text decode does not bind, and why each is skipped.
    ///
    /// Skipping is not the same as failing to map: a tower's weights are
    /// real tensors that this deployment has no use for, and `inv_freq` is
    /// DERIVED -- recomputed on the GPU from `rope_theta`, so a checkpoint
    /// that persists it is shipping a number the kernel already has.
    #[test]
    fn the_towers_and_the_derived_frequencies_are_skipped_not_mapped() {
        for raw in [
            "model.visual.blocks.0.attn.qkv.weight",
            "model.vision_tower.encoder.layer.0.mlp.fc1.weight",
            "model.audio_tower.layers.0.conv.weight",
            "visual.merger.mlp.0.weight",
            "model.layers.0.self_attn.rotary_emb.inv_freq",
        ] {
            assert_eq!(
                mapped(raw, false),
                None,
                "'{raw}' is not bound by a text decode"
            );
        }
    }

    /// A name with no rule is refused, and the refusal names it.
    ///
    /// This is the arm the whole function is shaped around. Falling through
    /// -- declaring the tensor under its checkpoint name -- produces a
    /// contract that loads, an allocation that happens, and a binder that
    /// never finds it. The tensor is simply not in the forward pass, and
    /// the model generates.
    #[test]
    fn a_name_with_no_rule_is_refused_by_name() {
        let refused = llama_mlx_name("encoder.block.0.weird.weight", false)
            .expect_err("a name outside every rule is an error");
        let message = format!("{refused:?}");
        assert!(
            message.contains("encoder.block.0.weird.weight"),
            "the refusal says which tensor it could not place: {message}"
        );
    }

    /// The function's own output is a valid input.
    ///
    /// An already-lowered name passes through unchanged, which is what lets
    /// a contract be re-authored over its own output without every name
    /// growing a second `layers.` prefix.
    #[test]
    fn an_already_lowered_name_passes_through_unchanged() {
        for raw in ["layers.0.self_attn.q_proj.weight", "final_norm.weight"] {
            assert_eq!(
                mapped(raw, false).as_deref(),
                Some(raw),
                "'{raw}' is already what the binder reads"
            );
        }
    }

    /// The tied fold and the pass-through claim disjoint names.
    ///
    /// The source orders the `lm_head.` arm before `already_lowered` and
    /// says why: the head "is not an identity when tied". That reason is
    /// sound and currently inoperative -- `already_lowered` does not accept
    /// `lm_head.` at all, so swapping the two arms changes nothing today.
    ///
    /// What this states instead is the condition under which the ordering
    /// WOULD start mattering. Adding `lm_head.` to `already_lowered`'s
    /// table is a one-line change in another file, made by someone
    /// lowering a family whose head really is pre-named, and it would make
    /// a tied checkpoint publish `lm_head.weight` unchanged: two
    /// allocations of the largest tensor in a small model, and a binder
    /// looking up a name the tied deployment never declares.
    #[test]
    fn nothing_the_tied_fold_claims_is_a_name_the_pass_through_would_take() {
        assert!(
            !mlx::already_lowered("lm_head.weight"),
            "the two arms are disjoint, which is why their order is free"
        );
        assert_eq!(
            mapped("lm_head.weight", true).as_deref(),
            Some("shared_embedding.weight"),
            "and the tied head folds"
        );
    }
}
