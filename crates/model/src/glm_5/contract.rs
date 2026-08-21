use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantScheme, QuantSpec};

use crate::shared::builder::{Builder, is_raw};
use crate::shared::moe::hf_moe_expert_stacks;

pub fn author_glm5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.shard_embed_tokens();

    bf16_kv_b_proj(b)?;
    b.allow_bf16_runtime_quant();
    b.allow_mxfp4_runtime_quant();

    hf_moe_expert_stacks(b,  true,  true)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

fn bf16_kv_b_proj(b: &mut Builder<'_>) -> Result<(), Error> {
    let f32enc = Encoding::Raw(DType::F32);
    for raw in b.tensors().to_vec() {
        if !raw.name.ends_with(".self_attn.kv_b_proj.weight") {
            continue;
        }

        if !is_raw(&raw.encoding, DType::F8E4M3) {
            continue;
        }
        let weight_name = raw.name.clone();
        let mut factors = None;
        for suffix in ["_scale_inv", "_scale"] {
            factors = b.find(&format!("{weight_name}{suffix}"));
            if factors.is_some() {
                break;
            }
        }
        if factors.is_none() {

            factors = b.find(&format!(
                "{}scale",
                &weight_name[..weight_name.len() - "weight".len()]
            ));
        }
        let Some(factors) = factors else {
            continue;
        };
        let weight_shape = raw.shape.clone();
        if weight_shape.len() != 2 {
            continue;
        }
        let mut factor_shape = factors.shape.clone();
        if !is_raw(&factors.encoding, DType::F32) || factor_shape.is_empty() {
            continue;
        }

        let axis = b.shard_axis(&raw.name)?;
        let mut factor_expr = Expr::src(&factors.name);
        if factor_shape.len() == 1 {
            factor_shape = vec![factor_shape[0], 1];
            factor_expr =
                factor_expr.transmute(TensorType::new(factor_shape.clone(), f32enc.clone()));
        }
        if factor_shape.len() != 2 {
            continue;
        }

        let (scale_local, scale_shape) = b.shard(factor_expr, factor_shape, axis);
        let scale_name = b.output_name(&factors.name);
        let declared = b.define(
            scale_name.clone(),
            scale_local,
            f32enc.clone(),
            Some(scale_shape),
        );
        b.mark_internal(declared);

        let packed_encoding = Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::BF16,
            bits_per_element: 0,
            group_size: 0,
            channel_axis: None,
        });
        let (packed, local_shape) = b.shard(
            Expr::src(&weight_name)
                .transmute(TensorType::new(weight_shape.clone(), packed_encoding)),
            weight_shape,
            axis,
        );
        b.define(
            b.output_name(&weight_name),
            packed.scale_per_block(Expr::out(&scale_name)),
            Encoding::Raw(DType::BF16),
            Some(local_shape),
        );
        b.consume(raw.id);
        b.consume(factors.id);
    }
    Ok(())
}
