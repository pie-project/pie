use model_loader::contract::Expr;
use model_loader::error::Error;

use crate::shared::builder::Builder;
use crate::shared::mlx;

pub fn author_gemma4(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_encode_scope()?;

    b.decoder_layer_prefix("model.language_model.layers.");
    fold_router_scale(b)?;

    const GATE_SECOND: bool = true;
    b.fused_moe_gate_up_tp_slices(GATE_SECOND)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

fn fold_router_scale(b: &mut Builder<'_>) -> Result<(), Error> {
    const SUFFIX: &str = ".router.scale";

    let layers = b.source_name(b.decoder_layer_prefix_value());
    for raw in b.tensors().to_vec() {
        if !raw.name.starts_with(&layers) || !raw.name.ends_with(SUFFIX) {
            continue;
        }

        if raw.shape.len() != 1 || raw.shape[0] <= 0 {
            return mlx::fail(format!(
                "gemma_4 router scale: '{}' is {:?}, and the fold needs the \
                 [hidden] vector the forward's rmsnorm reads",
                raw.name, raw.shape
            ));
        }

        let inv_sqrt_h = 1.0f32 / (raw.shape[0] as f32).sqrt();
        b.define(
            b.output_name(&raw.name),
            Expr::src(&raw.name).scale(inv_sqrt_h),
            raw.encoding.clone(),
            Some(raw.shape.clone()),
        );
        b.consume(raw.id);
    }
    Ok(())
}

pub fn author_gemma4_mlx(b: &mut Builder<'_>) -> Result<(), Error> {

    let first_shared = if b.shape().kv_shared_layers > 0 {
        i64::from(b.shape().layers) - i64::from(b.shape().kv_shared_layers)
    } else {
        -1
    };
    mlx::author_mlx_file(b, "Gemma4", &move |_, raw_name| {
        gemma4_mlx_name(raw_name, first_shared)
    })
}

fn gemma4_mlx_name(raw_name: &str, first_shared_layer: i64) -> Result<Option<String>, Error> {

    for skip in [
        "audio_tower.",
        "vision_tower.",
        "embed_audio.",
        "embed_vision.",
    ] {
        if mlx::has_wrapper_member(raw_name, skip) {
            return Ok(None);
        }
    }

    if let Some(tail) = raw_name.strip_prefix("lm_head.") {
        return Ok(Some(format!("shared_embedding.{tail}")));
    }

    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }
    let Some(rest) = mlx::decoder_member(raw_name) else {
        return mlx::fail(format!(
            "Metal Gemma4 schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    if let Some(tail) = rest.strip_prefix("embed_tokens.") {
        return Ok(Some(format!("shared_embedding.{tail}")));
    }

    for direct in [
        "embed_tokens_per_layer.",
        "per_layer_model_projection.",
        "per_layer_projection_norm.",
    ] {
        if rest.starts_with(direct) {
            return Ok(Some(rest.to_string()));
        }
    }
    if rest == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }
    let (layer, member) = mlx::layer_member(rest, "Gemma4", raw_name)?;

    let index: i64 = layer.parse().expect("validated digits");
    if first_shared_layer >= 0 && index >= first_shared_layer {
        for unused in [
            "self_attn.k_proj.",
            "self_attn.v_proj.",
            "self_attn.k_norm.",
        ] {
            if member.starts_with(unused) {
                return Ok(None);
            }
        }
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}
