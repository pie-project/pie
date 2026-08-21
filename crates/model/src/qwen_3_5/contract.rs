use model_loader::checkpoint::RawTensor;
use model_loader::contract::Expr;
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantScheme};

use crate::shared::builder::{Builder, is_raw};
use crate::shared::mlx;
use crate::shared::moe::hf_moe_expert_stacks;

pub fn author_qwen3_5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();

    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fp32_parameters(b)?;

    b.also_join_module("mtp.layers.0.");
    mtp_int8_lm_head(b)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

pub fn author_qwen3_5_moe(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fp32_parameters(b)?;

    const GATE_SECOND: bool = true;
    b.fused_moe_gate_up_tp_slices(GATE_SECOND)?;
    shared_expert_gate_up_joins(b);
    hf_moe_expert_stacks(b, GATE_SECOND, false)?;
    b.publish_remaining()
}

fn gdn_kkv_blocked(b: &Builder<'_>, raw: &RawTensor, k_dim: i64, v_dim: i64) -> (Expr, Vec<i64>) {
    let src = || Expr::src(&raw.name);
    let (key_lo, key_rows) = b.band(src(), 0, 0, k_dim);
    let (key_hi, _) = b.band(src(), 0, k_dim, k_dim);
    let (value, value_rows) = b.band(src(), 0, 2 * k_dim, v_dim);
    let mut shape = raw.shape.clone();
    shape[0] = 2 * key_rows + value_rows;
    (Expr::concat(0, vec![key_lo, key_hi, value]), shape)
}

fn gdn_kkv_blocked_shards(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.target().tp_size <= 1 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
        let la = format!("{}{layer}.linear_attn.", b.decoder_layer_prefix_value());
        let (Some(qkv), Some(z)) = (
            b.find(&b.source_name(&format!("{la}in_proj_qkv.weight"))),
            b.find(&b.source_name(&format!("{la}in_proj_z.weight"))),
        ) else {
            continue;
        };
        if qkv.shape.is_empty() || z.shape.is_empty() {
            continue;
        }
        let v_dim = z.shape[0];
        let conv_dim = qkv.shape[0];
        if conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0 {
            continue;
        }
        let k_dim = (conv_dim - v_dim) / 2;
        for leaf in ["in_proj_qkv.weight", "conv1d.weight"] {
            let Some(raw) = b.find(&b.source_name(&format!("{la}{leaf}"))) else {
                continue;
            };
            if raw.shape.is_empty() || raw.shape[0] != conv_dim {
                continue;
            }
            let (expr, shape) = gdn_kkv_blocked(b, raw, k_dim, v_dim);
            let id = raw.id;
            let encoding = raw.encoding.clone();
            b.define(b.output_name(&raw.name), expr, encoding, Some(shape));
            b.consume(id);
        }
    }
    Ok(())
}

fn gdn_fp32_parameters(b: &mut Builder<'_>) -> Result<(), Error> {
    for raw in b.tensors().to_vec() {
        if ![".linear_attn.A_log", ".linear_attn.norm.weight"]
            .iter()
            .any(|tail| raw.name.ends_with(tail))
        {
            continue;
        }
        let bf16 = is_raw(&raw.encoding, DType::BF16);
        if !bf16 && !is_raw(&raw.encoding, DType::F32) {
            continue;
        }
        let axis = b.shard_axis(&raw.name)?;
        let (expr, local) = b.shard(Expr::src(&raw.name), raw.shape.clone(), axis);
        let f32enc = Encoding::Raw(DType::F32);
        let expr = if bf16 {
            expr.cast(f32enc.clone())
        } else {
            expr
        };
        b.define(b.output_name(&raw.name), expr, f32enc, Some(local));
        b.consume(raw.id);
    }
    Ok(())
}

fn mtp_int8_lm_head(b: &mut Builder<'_>) -> Result<(), Error> {
    if !b.knobs().qwen35_mtp_int8_lm_head || b.find("mtp.fc.weight").is_none() {
        return Ok(());
    }

    let head = b.find("lm_head.weight").or_else(|| {
        b.tensors()
            .iter()
            .copied()
            .find(|raw| raw.name.ends_with(".embed_tokens.weight"))
    });

    let Some(head) = head else {
        return Ok(());
    };
    if !is_raw(&head.encoding, DType::BF16) {
        return Ok(());
    }
    let name = head.name.clone();
    b.quantized_view(&name, "mtp.lm_head".to_string(), QuantScheme::Int8Symmetric)?;
    Ok(())
}

fn shared_expert_gate_up_join(b: &mut Builder<'_>, layer_prefix: &str) {
    let lp = format!("{layer_prefix}mlp.shared_expert");
    let (Some(gate), Some(up)) = (
        b.find(&b.source_name(&format!("{lp}.gate_proj.weight"))),
        b.find(&b.source_name(&format!("{lp}.up_proj.weight"))),
    ) else {
        return;
    };
    if !is_raw(&gate.encoding, DType::BF16) || !is_raw(&up.encoding, DType::BF16) {
        return;
    }
    if gate.shape.len() != 2 || up.shape.len() != 2 || gate.shape[1] != up.shape[1] {
        return;
    }

    let gate_local = b.split(Expr::src(&gate.name), 0);
    let up_local = b.split(Expr::src(&up.name), 0);
    let rows = b.local_extent(gate.shape[0]) + b.local_extent(up.shape[0]);

    b.define(
        b.output_name(&format!("{lp}.gate_up_proj.weight")),
        Expr::concat(0, vec![gate_local, up_local]),
        gate.encoding.clone(),
        Some(vec![rows, gate.shape[1]]),
    );
}

fn shared_expert_gate_up_joins(b: &mut Builder<'_>) {
    for layer in 0..b.shape().layers {
        let prefix = format!("{}{layer}.", b.decoder_layer_prefix_value());
        shared_expert_gate_up_join(b, &prefix);
    }
    shared_expert_gate_up_join(b, "mtp.layers.0.");
}

pub fn author_qwen3_5_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    let has_lm_head = b.tensors().iter().any(|raw| {
        raw.name.starts_with("lm_head.") || raw.name.starts_with("language_model.lm_head.")
    });
    let tied = b.shape().tied_embeddings && !has_lm_head;
    mlx::author_mlx_file(b, "Qwen3.5", &move |_, raw_name| {
        qwen3_5_mlx_name(raw_name, tied)
    })?;
    gdn_metal_operands(b)
}

fn gdn_metal_operands(b: &mut Builder<'_>) -> Result<(), Error> {

    let module = |b: &Builder<'_>, name: &str, drop: &str| -> Result<String, Error> {
        let base = &name[..name.len() - drop.len()];
        match qwen3_5_mlx_name(base, false)? {
            Some(m) => Ok(m),
            None => mlx::fail(format!(
                "Metal Qwen3.5 has no name for the module '{base}' holds, so \
                 it cannot publish what the gated-DeltaNet shaders read \
                 beside it (of {} decoder tensors)",
                b.tensors().len()
            )),
        }
    };
    for raw in b.tensors().to_vec() {
        if raw.name.ends_with(".linear_attn.A_log") {
            let f32enc = Encoding::Raw(DType::F32);
            let expr = if is_raw(&raw.encoding, DType::BF16) {
                Expr::src(&raw.name).cast(f32enc.clone())
            } else if is_raw(&raw.encoding, DType::F32) {
                Expr::src(&raw.name)
            } else {
                return mlx::fail(format!(
                    "Metal Qwen3.5 needs '{}' as bf16 or f32; the shader reads \
                     it through a `float*`",
                    raw.name
                ));
            };
            let out = module(b, &raw.name, ".A_log")?;
            b.define(
                format!("{out}.A_log"),
                expr,
                f32enc,
                Some(raw.shape.clone()),
            );
            b.consume(raw.id);
        } else if raw.name.ends_with(".linear_attn.conv1d.weight") {

            let conv_dim = raw.shape.first().copied().unwrap_or_default();
            if conv_dim <= 0 {
                return mlx::fail(format!(
                    "Metal Qwen3.5 read '{}' with no output channel count, so \
                     it cannot state the width of the zero bias the \
                     convolution kernel reads",
                    raw.name
                ));
            }
            let bf16 = Encoding::Raw(DType::BF16);
            let out = module(b, &raw.name, ".conv1d.weight")?;
            b.define(
                format!("{out}.conv1d.bias"),
                Expr::fill(
                    0.0,
                    model_loader::contract::TensorType::raw(vec![conv_dim], DType::BF16),
                ),
                bf16,
                Some(vec![conv_dim]),
            );
        }
    }
    Ok(())
}

fn qwen3_5_mlx_name(raw_name: &str, tied: bool) -> Result<Option<String>, Error> {

    if raw_name.ends_with(".linear_attn.A_log") {
        return Ok(None);
    }

    for skip in ["visual.", "vision_tower.", "mtp."] {
        if mlx::has_wrapper_member(raw_name, skip) {
            return Ok(None);
        }
    }

    for head in ["lm_head.", "language_model.lm_head."] {
        if let Some(tail) = raw_name.strip_prefix(head) {
            return Ok(Some(if tied {
                format!("shared_embedding.{tail}")
            } else {
                format!("lm_head.{tail}")
            }));
        }
    }

    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }

    let Some(decoder) = mlx::decoder_member(raw_name).or_else(|| raw_name.strip_prefix("model."))
    else {
        return mlx::fail(format!(
            "Metal Qwen3.5 schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    if let Some(tail) = decoder.strip_prefix("embed_tokens.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("embed_tokens.{tail}")
        }));
    }
    if decoder == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }
    let (layer, member) = mlx::layer_member(decoder, "Qwen3.5", raw_name)?;
    if let Some(renamed) = mlx::routed_expert_member(
        raw_name, member, "Qwen3.5",  true,
    )? {
        return Ok(Some(format!("layers.{layer}.{renamed}")));
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}
