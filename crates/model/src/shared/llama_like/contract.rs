use model_loader::error::Error;

use crate::shared::builder::Builder;
use crate::shared::mlx;

pub fn author_llama_like(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

pub fn author_dense(b: &mut Builder<'_>) -> Result<(), Error> {

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

pub fn author_llama_mlx(b: &mut Builder<'_>) -> Result<(), Error> {

    let has_lm_head = b
        .tensors()
        .iter()
        .any(|raw| raw.name.starts_with("lm_head."));
    let tied = b.shape().tied_embeddings && !has_lm_head;
    mlx::author_mlx_file(b, "llama", &move |_, raw_name| {
        llama_mlx_name(raw_name, tied)
    })
}

fn llama_mlx_name(raw_name: &str, tied: bool) -> Result<Option<String>, Error> {

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

    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }

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

    if let Some(renamed) = mlx::routed_expert_member(raw_name, member, "llama", false)? {
        return Ok(Some(format!("layers.{layer}.{renamed}")));
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}
