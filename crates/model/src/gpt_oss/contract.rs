use model_dsl::axes::DtypeAxis;
use model_loader::contract::{Expr, GroupContract, Scales, TensorContract};
use model_loader::error::Error;
use model_loader::types::{
    BackendKind, DType, Encoding, QuantGranularity, RepackLayout, ScaleForm, TensorId,
};

use crate::shared::builder::{Builder, align_up, is_raw, mxfp4_encoding};
use crate::shared::mlx;
use crate::shared::policy::Mxfp4MoePolicy;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

const _EXPERTS_ARE_MXFP4: () = {
    match <super::forward::ShippedW2 as DtypeAxis>::REPR {
        model_dsl::WeightRepr::Mxfp4Marlin => (),
        _ => panic!(
            "gpt-oss's catalogued expert axis moved off MXFP4-Marlin; \
             every mxfp4 arm of this contract spells that axis"
        ),
    }
};

pub fn author_gpt_oss(b: &mut Builder<'_>) -> Result<(), Error> {
    shipped_axis_check(b)?;
    mxfp4_groups(b)?;
    b.fused_moe_gate_up_tp_slices(false)?;
    b.publish_remaining()
}

fn shipped_axis_check(b: &Builder<'_>) -> Result<(), Error> {
    if b.target().backend != BackendKind::Cuda {
        return Ok(());
    }
    if let Some(t) = b.tensors().iter().find(|t| {
        t.name.ends_with("mlp.experts.gate_up_proj") || t.name.ends_with("mlp.experts.down_proj")
    }) {
        return fail(format!(
            "gpt-oss's catalogued CUDA SKU ships `{}` experts (axis W2, \
             repr Mxfp4Marlin: `_blocks`/`_scales` triplets), but '{}' is \
             the dequantized bf16 bank — the same model in a repr this \
             build's CUDA text has no routed leg for",
            <super::forward::ShippedW2 as DtypeAxis>::NAME,
            t.name,
        ));
    }
    Ok(())
}

fn mxfp4_block_scales(weight: String) -> Scales {
    Scales {
        of: weight,
        granularity: QuantGranularity::PerGroup,
        group_size: 32,
        channel_axis: 1,
        form: ScaleForm::RawE8M0,
    }
}

fn mxfp4_groups(b: &mut Builder<'_>) -> Result<(), Error> {
    let native = b.mxfp4_moe() == Mxfp4MoePolicy::NativeGemm;
    if native && !b.target().native_mxfp4_moe {
        return fail(
            "GPT-OSS native MXFP4 requested, but target does not support native MXFP4 MoE",
        );
    }
    if !native && b.stream_routed_experts() {
        return streamed_expert_groups(b);
    }
    for raw in b.tensors().to_vec() {
        let Some(base) = raw.name.strip_suffix("_blocks").map(str::to_string) else {
            continue;
        };
        let block = raw;
        let (Some(scale), Some(bias)) = (
            b.find(&format!("{base}_scales")),
            b.find(&format!("{base}_bias")),
        ) else {
            continue;
        };
        if native {
            if base.ends_with("gate_up_proj") {
                native_gate_up(b, block, scale, bias, &base)?;
            } else if base.ends_with("down_proj") {
                native_down(b, block, scale, bias, &base)?;
            } else {
                return fail(format!(
                    "GPT-OSS MXFP4 tensor '{}' is not gate_up_proj or down_proj",
                    block.name
                ));
            }
        } else {
            b.push_direct(block, format!("{base}.weight"), None)?;
            let scales = b.push_direct(scale, format!("{base}.weight_scale"), None)?;

            b.set_scales(scales, mxfp4_block_scales(format!("{base}.weight")));
            b.push_direct(bias, format!("{base}.bias"), None)?;
        }
        b.consume(block.id);
        b.consume(scale.id);
        b.consume(bias.id);
    }
    Ok(())
}

fn native_gate_up(
    b: &mut Builder<'_>,
    block: &model_loader::checkpoint::RawTensor,
    scale: &model_loader::checkpoint::RawTensor,
    bias: &model_loader::checkpoint::RawTensor,
    base: &str,
) -> Result<(), Error> {
    if block.shape.len() != 4 || scale.shape.len() != 3 || bias.shape.len() != 2 {
        return fail(format!(
            "GPT-OSS native gate/up '{base}' has an unsupported block/scale/bias rank"
        ));
    }
    let experts = block.shape[0];
    let fused_rows = block.shape[1];
    let groups = block.shape[2];
    if fused_rows % 2 != 0 || block.shape[3] != 16 {
        return fail(format!(
            "GPT-OSS native gate/up '{base}' expected [E, 2I, H/32, 16]"
        ));
    }
    if scale.shape != [experts, fused_rows, groups] || bias.shape != [experts, fused_rows] {
        return fail(format!(
            "GPT-OSS native gate/up '{base}' scale/bias shape mismatch"
        ));
    }
    let full_intermediate = fused_rows / 2;
    let hidden = groups * 32;
    let local_intermediate = b.local_extent(full_intermediate);
    let intermediate_native = align_up(local_intermediate, 128)?;
    let prefix = &base[..base.len() - "gate_up_proj".len()];

    for (half, first_row) in [("gate_proj", 0i64), ("up_proj", 1i64)] {
        let out_base = format!("{prefix}{half}");

        let rows = |b: &Builder<'_>, name: &str| {
            b.split(Expr::src(name), 1)
                .stride(1, first_row, local_intermediate, 2)
        };
        let block_rows = rows(b, &block.name);
        let scale_rows = rows(b, &scale.name);
        let bias_rows = rows(b, &bias.name);

        b.push_repack(
            format!("{out_base}.weight"),
            block_rows,
            RepackLayout::MarlinMxfp4Weight,
            mxfp4_encoding(1),
            vec![experts, intermediate_native, hidden],
        );

        let scales = b.push_repack(
            format!("{out_base}.weight_scale"),
            scale_rows,
            RepackLayout::MarlinMxfp4Scale,
            Encoding::Raw(DType::U8),
            vec![experts, intermediate_native, groups],
        );
        b.set_scales(scales, mxfp4_block_scales(format!("{out_base}.weight")));

        b.push_expr(
            format!("{out_base}.bias"),
            bias,
            vec![experts, local_intermediate],
            bias_rows,
        );
    }
    Ok(())
}

fn native_down(
    b: &mut Builder<'_>,
    block: &model_loader::checkpoint::RawTensor,
    scale: &model_loader::checkpoint::RawTensor,
    bias: &model_loader::checkpoint::RawTensor,
    base: &str,
) -> Result<(), Error> {
    if block.shape.len() != 4 || scale.shape.len() != 3 || bias.shape.len() != 2 {
        return fail(format!(
            "GPT-OSS native down '{base}' has an unsupported block/scale/bias rank"
        ));
    }
    let experts = block.shape[0];
    let hidden = block.shape[1];
    let groups = block.shape[2];
    if block.shape[3] != 16 {
        return fail(format!(
            "GPT-OSS native down '{base}' expected [E, H, I/32, 16]"
        ));
    }
    if scale.shape != [experts, hidden, groups] || bias.shape != [experts, hidden] {
        return fail(format!(
            "GPT-OSS native down '{base}' scale/bias shape mismatch"
        ));
    }
    let local_intermediate = b.local_extent(groups * 32);
    if local_intermediate % 32 != 0 {
        return fail(format!(
            "GPT-OSS native down '{base}' TP shard must align to MXFP4 32-wide groups"
        ));
    }
    let intermediate_native = align_up(local_intermediate, 128)?;

    b.push_repack(
        format!("{base}.weight"),
        b.split(Expr::src(&block.name), 2),
        RepackLayout::MarlinMxfp4Weight,
        mxfp4_encoding(2),
        vec![experts, hidden, intermediate_native],
    );

    let scales = b.push_repack(
        format!("{base}.weight_scale"),
        b.split(Expr::src(&scale.name), 2),
        RepackLayout::MarlinMxfp4Scale,
        Encoding::Raw(DType::U8),
        vec![experts, hidden, intermediate_native / 32],
    );
    b.set_scales(scales, mxfp4_block_scales(format!("{base}.weight")));

    b.push_direct(bias, format!("{base}.bias"), None)?;
    Ok(())
}

fn streamed_expert_groups(b: &mut Builder<'_>) -> Result<(), Error> {
    let experts = i64::from(b.shape().n_experts);
    if experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
        let bound = format!("model.layers.{layer}.mlp.experts.");
        let prefix = b.source_name(&bound);

        let mut tensors: Vec<TensorContract> = Vec::new();
        let mut consumed: Vec<TensorId> = Vec::new();
        let mut grouped: Vec<&str> = Vec::new();
        for half in ["gate_up_proj", "down_proj"] {
            let (Some(block), Some(scale)) = (
                b.find(&format!("{prefix}{half}_blocks")),
                b.find(&format!("{prefix}{half}_scales")),
            ) else {
                continue;
            };
            if block.shape.first() != Some(&experts) || scale.shape.first() != Some(&experts) {
                return fail(format!(
                    "GPT-OSS expert group '{half}' is not stacked over {experts} experts"
                ));
            }

            let band = |raw: &model_loader::checkpoint::RawTensor| {
                let mut shape = raw.shape.clone();
                shape[0] = 1;
                (Expr::src(&raw.name).select(0, 1, 1), shape)
            };
            let (block_node, block_shape) = band(block);
            let (scale_node, scale_shape) = band(scale);

            tensors.push(TensorContract::new(
                format!("{half}.weight"),
                block_node,
                block_shape,
                Encoding::Raw(DType::U8),
            ));

            tensors.push(
                TensorContract::new(
                    format!("{half}.weight_scale"),
                    scale_node,
                    scale_shape,
                    Encoding::Raw(DType::U8),
                )
                .scaling(mxfp4_block_scales(format!("{half}.weight"))),
            );

            consumed.push(block.id);
            consumed.push(scale.id);
            grouped.push(half);
        }
        if tensors.is_empty() {
            continue;
        }
        b.push_group(GroupContract {
            name: bound[..bound.len() - 1].to_string(),
            arity: experts as u32,
            tensors,
        });

        for half in grouped {
            let Some(bias) = b.find(&format!("{prefix}{half}_bias")) else {
                return fail(format!(
                    "GPT-OSS expert bank '{prefix}{half}' is streamed but \
                     '{prefix}{half}_bias' is not in the checkpoint; the bind \
                     reads the bias resident beside the group"
                ));
            };
            let id = bias.id;
            b.push_direct(bias, format!("{bound}{half}.bias"), None)?;
            b.consume(id);
        }
        for id in consumed {
            b.consume(id);
        }
    }
    Ok(())
}

pub fn author_gpt_oss_mlx(b: &mut Builder<'_>) -> Result<(), Error> {

    mlx::int4_requested(b, "GptOss")?;
    let mut declared = 0usize;
    for raw in b.tensors().to_vec() {

        if raw.name.ends_with("_scales")
            || raw.name.ends_with("_blocks")
            || raw.name.ends_with("_bias")
        {
            continue;
        }
        let output = gptoss_mlx_name(&raw.name)?;
        if raw.name.ends_with(".weight") && is_raw(&raw.encoding, DType::U32) {
            let base = &raw.name[..raw.name.len() - ".weight".len()];
            let scales = b.find(&format!("{base}.scales"));
            let biases = b.find(&format!("{base}.biases"));
            let Some(scales) = scales else {
                return fail(format!(
                    "Metal GptOss: '{}' is a packed weight with no scales, which no \
                     scheme here describes",
                    raw.name
                ));
            };
            let Some(biases) = biases else {

                mlx::push_mlx_mxfp4_stacked(b, raw, scales, output)?;
                declared += 1;
                continue;
            };

            let packed_cols = *raw.shape.last().unwrap_or(&0);
            let groups = *scales.shape.last().unwrap_or(&0);
            if groups <= 0 || packed_cols % (2 * groups) != 0 {
                return fail(format!(
                    "Metal GptOss: '{}' is not quantized in groups of 64, which is \
                     what these kernels read",
                    raw.name
                ));
            }
            let bits = packed_cols / (2 * groups);
            if bits != 4 && bits != 8 {
                return fail(format!(
                    "Metal GptOss: '{}' is {bits}-bit, and only 4 and 8 are described here",
                    raw.name
                ));
            }
            mlx::push_mlx_affine_stacked(b, raw, scales, biases, bits, 64, output)?;
        } else if raw.name.ends_with(".weight")
            && raw.shape.len() == 2
            && is_raw(&raw.encoding, DType::BF16)
        {

            mlx::push_encoded_affine(b, Expr::src(&raw.name), raw.shape[0], raw.shape[1], output)?;
        } else {
            mlx::push_direct(b, raw, output);
        }
        declared += 1;
    }
    declare_mxfp4_experts_mlx(b, &mut declared)?;
    if declared == 0 {
        return fail("Metal GptOss schema found no decoder tensors");
    }
    Ok(())
}

fn gptoss_mlx_name(raw_name: &str) -> Result<String, Error> {

    if raw_name.starts_with("lm_head.") {
        return Ok(raw_name.to_string());
    }

    if mlx::already_lowered(raw_name) {
        return Ok(raw_name.to_string());
    }
    let Some(rest) = raw_name.strip_prefix("model.") else {
        return fail(format!(
            "Metal GptOss schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    if rest.starts_with("embed_tokens.") {
        return Ok(rest.to_string());
    }
    if rest == "norm.weight" {
        return Ok("final_norm.weight".to_string());
    }
    let (layer, member) = mlx::layer_member(rest, "GptOss", raw_name)?;
    Ok(format!("layers.{layer}.{member}"))
}

fn declare_mxfp4_experts_mlx(b: &mut Builder<'_>, declared: &mut usize) -> Result<(), Error> {
    const BLOCKS: &str = "_blocks";
    for blocks in b.tensors().to_vec() {
        let Some(base) = blocks.name.strip_suffix(BLOCKS).map(str::to_string) else {
            continue;
        };
        let Some(scales) = b.find(&format!("{base}_scales")) else {
            return fail(format!(
                "Metal GptOss: '{}' is an MXFP4 block tensor with no '_scales' \
                 beside it",
                blocks.name
            ));
        };

        if blocks.shape.len() != 4 || blocks.shape[3] != 16 || blocks.shape[..3] != scales.shape[..]
        {
            return fail(format!(
                "Metal GptOss: MXFP4 tensor '{}' is not shaped \
                 [experts, rows, groups, 16] against its scales",
                blocks.name
            ));
        }
        let experts = blocks.shape[0];
        let stored_rows = blocks.shape[1];
        let groups = blocks.shape[2];
        let cols = groups * 32;

        let mapped = gptoss_mlx_name(&base)?;
        let fused = base.ends_with("gate_up_proj");
        if !fused && !base.ends_with("down_proj") {
            return fail(format!(
                "Metal GptOss: MXFP4 tensor '{}' is neither the fused gate/up \
                 projection nor the down projection",
                blocks.name
            ));
        }

        let prefix = &mapped[..mapped.len() - if fused { 12 } else { 9 }];
        let bias = b.find(&format!("{base}_bias")).cloned();

        let halves: &[(&str, i64)] = if fused {
            &[("gate_proj", 0), ("up_proj", 1)]
        } else {
            &[("down_proj", 0)]
        };
        let rows = if fused { stored_rows / 2 } else { stored_rows };

        for (half, first_row) in halves {
            let name = format!("{prefix}{half}");
            let select =
                |b: &mut Builder<'_>, source: &str, shape: Vec<i64>, as_name: String| -> Expr {
                    let expr = if fused {
                        Expr::src(source).stride(1, *first_row, rows, 2)
                    } else {
                        Expr::src(source)
                    };
                    let index =
                        b.define(as_name.clone(), expr, Encoding::Raw(DType::U8), Some(shape));
                    b.mark_internal(index);
                    Expr::out(&as_name)
                };
            let half_blocks = select(
                b,
                &blocks.name,
                vec![experts, rows, groups, 16],
                format!("{name}.mxfp4_blocks"),
            );
            let half_scales = select(
                b,
                &scales.name,
                vec![experts, rows, groups],
                format!("{name}.mxfp4_exponents"),
            );

            let values = mlx::mxfp4_values(
                b,
                half_blocks,
                half_scales,
                experts * rows,
                groups,
                format!("{name}.mxfp4_scales"),
            );
            let index = b.define(
                format!("{name}.dequantized"),
                values,
                Encoding::Raw(DType::BF16),
                Some(vec![experts * rows, cols]),
            );
            b.mark_internal(index);
            mlx::push_encoded_affine(
                b,
                Expr::out(format!("{name}.dequantized")),
                experts * rows,
                cols,
                format!("{name}.weight"),
            )?;
            if let Some(bias) = &bias {
                if bias.shape.len() != 2 || bias.shape != [experts, stored_rows] {
                    return fail(format!(
                        "Metal GptOss: '{}' does not match the projection it biases",
                        bias.name
                    ));
                }
                let expr = if fused {
                    Expr::src(&bias.name).stride(1, *first_row, rows, 2)
                } else {
                    Expr::src(&bias.name)
                };
                b.define(
                    format!("{name}.bias"),
                    expr,
                    bias.encoding.clone(),
                    Some(vec![experts, rows]),
                );
            }
            *declared += 1;
        }
    }
    Ok(())
}
