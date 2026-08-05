//! What GPT-OSS binds.
//!
//! Ported from `driver/cuda/src/model/mixtral/mixtral_contract.hpp` (the
//! Mixtral family header — plain Mixtral needs nothing special, GPT-OSS is
//! the whole file). Its experts ship as an MXFP4 `_blocks`/`_scales`/`_bias`
//! triplet, and the layout the contract asks for depends on whether this
//! device has a native MXFP4 GEMM.

use pie_loader::contract::{Expr, GroupContract, Scales, TensorContract};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, QuantGranularity, RepackLayout, ScaleForm, TensorId};

use pie_model_common::builder::{Builder, align_up, is_raw, mxfp4_encoding};
use pie_model_common::mlx;
use pie_model_common::policy::Mxfp4MoePolicy;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// GPT-OSS. The dense QKV join is deliberately absent: this bind path reads
/// `q_proj`/`k_proj`/`v_proj` individually, so fusing them would consume the
/// three and leave the bind with a missing weight.
pub fn author_gpt_oss(b: &mut Builder<'_>) -> Result<(), Error> {
    mxfp4_groups(b)?;
    b.fused_moe_gate_up_tp_slices(false)?;
    b.publish_remaining()
}

/// State that an MXFP4 scale tensor holds the block scales for `weight`.
///
/// `channel_axis` is 1 for both halves even though `down_proj` is declared
/// with `mxfp4_encoding(2)`. That is what the old name matching produced —
/// it read the axis off the scheme, not off the encoding — and the value is
/// live: it reaches `QuantMeta` and is serialized into the weight-store
/// cache. Changing it is a separate question from moving where it is stated.
fn mxfp4_block_scales(weight: String) -> Scales {
    Scales {
        of: weight,
        granularity: QuantGranularity::PerGroup,
        group_size: 32,
        channel_axis: 1,
        form: ScaleForm::RawE8M0,
    }
}

/// Declare GPT-OSS's MXFP4 expert triplets the way this device wants them.
///
/// `_blocks`/`_scales`/`_bias` either pass through as three plain tensors
/// for the routed-decode path, or get Marlin-repacked into a native MXFP4
/// GEMM layout. Which one it is is the driver's `Mxfp4MoePolicy`, resolved
/// against what the device measured — not a property of the checkpoint.
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
            // The routed-dequant path reads these bytes through quant_meta,
            // exactly as the native path does, so it needs the same pairing
            // stated. Publishing the scale as a plain tensor leaves
            // quant_meta empty and the bind fails.
            if let Some(scales) = scales {
                b.set_scales(scales, mxfp4_block_scales(format!("{base}.weight")));
            }
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
    block: &pie_loader::checkpoint::RawTensor,
    scale: &pie_loader::checkpoint::RawTensor,
    bias: &pie_loader::checkpoint::RawTensor,
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

    // gate and up are interleaved row by row, so this rank's share of the
    // *logical* intermediate axis is `split` on the fused axis — which lands
    // on `[2·local_start, 2·(local_start + local_intermediate))` — followed
    // by the even or odd rows of that band. Both are ordinary nodes, so the
    // rank stays where every other family puts it: in the target, resolved
    // by the loader, never written into the contract.
    for (half, first_row) in [("gate_proj", 0i64), ("up_proj", 1i64)] {
        let out_base = format!("{prefix}{half}");
        // Precomputed rather than closed over `b`: the borrow checker's
        // price for a helper that reads the builder while the pushes below
        // mutate it.
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
        if let Some(scales) = scales {
            b.set_scales(scales, mxfp4_block_scales(format!("{out_base}.weight")));
        }

        // The bias needed a `DenseRowGather` kernel only because the algebra
        // could not say "every other row of this rank's band". It can, so
        // this is a copy.
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
    block: &pie_loader::checkpoint::RawTensor,
    scale: &pie_loader::checkpoint::RawTensor,
    bias: &pie_loader::checkpoint::RawTensor,
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

    // Down is sharded along K, which is the packed axis: one group is 32
    // elements, so this rank's column band is a `split` of the *group* axis
    // and the offset never has to be spelled in elements.
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
    if let Some(scales) = scales {
        b.set_scales(scales, mxfp4_block_scales(format!("{base}.weight")));
    }

    b.push_direct(bias, format!("{base}.bias"), None)?;
    Ok(())
}

/// The same MXFP4 experts, declared as a group instead of a bank.
///
/// GPT-OSS's experts are one tensor per *layer* with the experts stacked
/// along axis 0, so an instance is a band of a bank and the index decides
/// only where the band starts — the reason `select` exists.
///
/// Weights and scales only; the biases are kilobytes against megabytes and
/// the bind de-interleaves the gate/up bias with a kernel, which is host
/// work a group plan has no node for. Rank-blind, correctly: the packed
/// resident path publishes these same blocks unsharded, and streaming is a
/// residency decision, so it inherits that layout. Packed only: the native
/// path Marlin-repacks into a layout whose rows are permuted across the
/// whole bank, so one expert's repacked bytes are not a contiguous band.
fn streamed_expert_groups(b: &mut Builder<'_>) -> Result<(), Error> {
    let experts = i64::from(b.facts().num_experts);
    if experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
        let bound = format!("model.layers.{layer}.mlp.experts.");
        let prefix = b.source_name(&bound);

        let mut tensors: Vec<TensorContract> = Vec::new();
        let mut consumed: Vec<TensorId> = Vec::new();
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
            // One expert's band: `len` 1 along the expert axis, starting at
            // `index * 1`. The leading 1 stays, because a `Select` is a
            // slice and a slice keeps its rank — and the bind reads a slot
            // through a view anyway, exactly as it read the bank through
            // one.
            let band = |raw: &pie_loader::checkpoint::RawTensor| {
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
            // The same pairing the resident path states, for the same
            // reason: the routed-dequant kernel reads the factors through
            // quant_meta, and a plain tensor leaves that empty.
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
        }
        if tensors.is_empty() {
            continue;
        }
        b.push_group(GroupContract {
            name: bound[..bound.len() - 1].to_string(),
            arity: experts as u32,
            tensors,
        });

        // The biases stay resident, under the names the bind already reads.
        for half in ["gate_up_proj", "down_proj"] {
            if let Some(bias) = b.find(&format!("{prefix}{half}_bias")) {
                let id = bias.id;
                b.push_direct(bias, format!("{bound}{half}.bias"), None)?;
                b.consume(id);
            }
        }
        for id in consumed {
            b.consume(id);
        }
    }
    Ok(())
}

/// The Metal lowering. Ported from
/// `driver/metal/src/model/gptoss/gptoss_contract.hpp`: rename for MLX's
/// binder, pair the affine triplets (deriving the width from the group-64
/// kernels' equation), accept shipped MXFP4 by transmute, and — for the
/// projections the published checkpoint left in BF16 — quantize into the
/// affine layout at load, because every matvec here is a quantized one.
pub fn author_gpt_oss_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    // This family encodes its BF16 projections whatever the request says --
    // its matvecs have no unquantized path -- so the answer is discarded and
    // only the refusal is wanted: a request this lowering cannot serve must
    // not be silently ignored here when the other three refuse it.
    mlx::int4_requested(b, "GptOss")?;
    let mut declared = 0usize;
    for raw in b.tensors().to_vec() {
        // The published checkpoint's MXFP4 experts are consumed as a pair
        // and declared from the `_blocks` half below; `runtime_name` is not
        // asked about them, because the names they produce are the split
        // projections', which no per-tensor mapping can state.
        if raw.name.ends_with("_scales")
            || raw.name.ends_with("_blocks")
            || raw.name.ends_with("_bias")
        {
            continue;
        }
        let Some(output) = gptoss_mlx_name(&raw.name)? else {
            continue;
        };
        if raw.name.ends_with(".weight") && is_raw(&raw.encoding, DType::U32) {
            let base = &raw.name[..raw.name.len() - ".weight".len()];
            let scales = b.find(&format!("{base}.scales"));
            let biases = b.find(&format!("{base}.biases"));
            let Some(scales) = scales else {
                return fail(format!(
                    "Metal GptOss: '{}' is a packed weight with no scales, which no                      scheme here describes",
                    raw.name
                ));
            };
            let Some(biases) = biases else {
                // Scales with no zero points is MXFP4, which `mlx_lm convert
                // -q` leaves the MoE experts in. The loader decodes MXFP4
                // already, so it is accepted by transmute rather than
                // refused.
                mlx::push_mlx_mxfp4_stacked(b, raw, scales, output)?;
                declared += 1;
                continue;
            };
            // Width comes from the tensors, not from the config: gpt-oss
            // states no quantization block, and these kernels are group-64,
            // so `bits = packed_cols / (2 * groups)`.
            let packed_cols = *raw.shape.last().unwrap_or(&0);
            let groups = *scales.shape.last().unwrap_or(&0);
            if groups <= 0 || packed_cols % (2 * groups) != 0 {
                return fail(format!(
                    "Metal GptOss: '{}' is not quantized in groups of 64, which is                      what these kernels read",
                    raw.name
                ));
            }
            let bits = packed_cols / (2 * groups);
            if bits != 4 && bits != 8 {
                return fail(format!(
                    "Metal GptOss: '{}' is {bits}-bit, and only 4 and 8 are                      described here",
                    raw.name
                ));
            }
            mlx::push_mlx_affine_stacked(b, raw, scales, biases, bits, 64, output)?;
        } else if raw.name.ends_with(".weight")
            && raw.shape.len() == 2
            && is_raw(&raw.encoding, DType::BF16)
        {
            // A projection the published checkpoint left in BF16, read by a
            // quantized matvec — so the loader quantizes it on the way in.
            // Rank is what separates these from the norms, which must stay
            // values.
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

fn gptoss_mlx_name(raw_name: &str) -> Result<Option<String>, Error> {
    // The head is its own tensor here — NOT the embedding under another name.
    if raw_name.starts_with("lm_head.") {
        return Ok(Some(raw_name.to_string()));
    }
    // Its own output is a valid input: see `mlx::already_lowered`. After the
    // `lm_head.` arm above, which this family answers with an identity anyway.
    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }
    let Some(rest) = raw_name.strip_prefix("model.") else {
        return fail(format!(
            "Metal GptOss schema has no declared mapping or skip for '{raw_name}'"
        ));
    };
    if rest.starts_with("embed_tokens.") {
        return Ok(Some(rest.to_string()));
    }
    if rest == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }
    let (layer, member) = mlx::layer_member(rest, "GptOss", raw_name)?;
    Ok(Some(format!("layers.{layer}.{member}")))
}

/// Declare the experts the way the PUBLISHED checkpoint stores them: the
/// fused `gate_up_proj` holds gate at even rows and up at odd ones, so each
/// half is a strided read; the halves are MXFP4 and are decoded then
/// re-encoded as the affine-U4 the matvecs read — the same two steps
/// `mlx_lm convert --dequantize` then `-q` performs, done at load time.
///
/// The split happens where a stride IS lowered — on the source extent — and
/// each half is published as the plain bytes it is before anything
/// reinterprets it, because a width-changing transmute may only rename a
/// whole tensor. Select, publish, then reinterpret the published name.
fn declare_mxfp4_experts_mlx(b: &mut Builder<'_>, declared: &mut usize) -> Result<(), Error> {
    const BLOCKS: &str = "_blocks";
    for blocks in b.tensors().to_vec() {
        let Some(base) = blocks.name.strip_suffix(BLOCKS).map(str::to_string) else {
            continue;
        };
        let Some(scales) = b.find(&format!("{base}_scales")) else {
            return fail(format!(
                "Metal GptOss: '{}' is an MXFP4 block tensor with no '_scales'                  beside it",
                blocks.name
            ));
        };
        // `[experts, rows, groups, 16]` of nibbles against
        // `[experts, rows, groups]` of exponents.
        if blocks.shape.len() != 4
            || scales.shape.len() != 3
            || blocks.shape[3] != 16
            || blocks.shape[..3] != scales.shape[..]
        {
            return fail(format!(
                "Metal GptOss: MXFP4 tensor '{}' is not shaped                  [experts, rows, groups, 16] against its scales",
                blocks.name
            ));
        }
        let experts = blocks.shape[0];
        let stored_rows = blocks.shape[1];
        let groups = blocks.shape[2];
        let cols = groups * 32;

        let Some(mapped) = gptoss_mlx_name(&base)? else {
            continue;
        };
        let fused = base.ends_with("gate_up_proj");
        if !fused && !base.ends_with("down_proj") {
            return fail(format!(
                "Metal GptOss: MXFP4 tensor '{}' is neither the fused gate/up                  projection nor the down projection",
                blocks.name
            ));
        }
        // `layers.N.mlp.experts.gate_up_proj` → `layers.N.mlp.experts.`
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
                    if let Some(index) =
                        b.define(as_name.clone(), expr, Encoding::Raw(DType::U8), Some(shape))
                    {
                        b.mark_internal(index);
                    }
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
                cols,
                format!("{name}.mxfp4_scales"),
            )?;
            if let Some(index) = b.define(
                format!("{name}.dequantized"),
                values,
                Encoding::Raw(DType::BF16),
                Some(vec![experts * rows, cols]),
            ) {
                b.mark_internal(index);
            }
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
        let (blocks_id, scales_id) = (blocks.id, scales.id);
        b.consume(blocks_id);
        b.consume(scales_id);
        if let Some(bias) = &bias {
            b.consume(bias.id);
        }
    }
    Ok(())
}
