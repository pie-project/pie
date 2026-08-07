//! Llama 3's load contract — and the dense schema most generations share.
//!
//! Ported from `driver/cuda/src/model/llama_like/llama_like_contract.hpp`.
//! Two of the three authors here are bound by generations that ship no
//! contract of their own: `author_llama_like` serves Llama 3, Mistral and
//! Qwen2/3, and `author_dense` serves a dozen `model_type`s across five
//! vendors whose checkpoints already use the names the bind path reads.
//! Those are rows in `pie_model::contract::HF_ROWS`, not dependencies — a
//! generation with nothing of its own to say names the one that says it.
//!
//! Phi-3 used to live here, because it is dense with two source-side
//! fusions to undo. It has its own crate now (`pie-model-phi-3`): the
//! splits are Phi-3's alone, and what it shares is the three-pass dense
//! tail, which every generation spells out for itself.

use pie_loader::error::Error;

use pie_model_common::builder::Builder;
use pie_model_common::mlx;

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
/// `driver/metal/src/model/llama/llama_contract.hpp` — what is
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
    let tied = b.facts().tied_embeddings && !has_lm_head;
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
