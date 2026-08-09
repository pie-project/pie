//! What GPT-OSS binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/mixtral/mixtral_contract.hpp` (the
//! Mixtral family header — plain Mixtral needs nothing special, GPT-OSS is
//! the whole file). Its experts ship as an MXFP4 `_blocks`/`_scales`/`_bias`
//! triplet, and the layout the contract asks for depends on whether this
//! device has a native MXFP4 GEMM.

use model_loader::contract::{Expr, GroupContract, Scales, TensorContract};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantGranularity, RepackLayout, ScaleForm, TensorId};

use crate::shared::builder::{Builder, align_up, is_raw, mxfp4_encoding};
use crate::shared::mlx;
use crate::shared::policy::Mxfp4MoePolicy;

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
    let experts = i64::from(b.shape().n_experts);
    if experts <= 0 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
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
/// `crates/driver-metal/csrc/src/model/gptoss/gptoss_contract.hpp`: rename for MLX's
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
        //
        // This suffix test is the ONLY thing that keeps the triplets out of
        // the contract. `Builder::consume` would say it more precisely, but
        // `consumed` is read by `publish_remaining` alone and this lowering
        // publishes every tensor itself — so marking them there would be a
        // statement nothing consults.
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
                    "Metal GptOss: '{}' is a packed weight with no scales, which no \
                     scheme here describes",
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
                "Metal GptOss: '{}' is an MXFP4 block tensor with no '_scales' \
                 beside it",
                blocks.name
            ));
        };
        // `[experts, rows, groups, 16]` of nibbles against
        // `[experts, rows, groups]` of exponents.
        //
        // The scales' rank is not asked about separately: with the blocks
        // pinned to rank 4, the slice comparison below is over three
        // extents, and a scale tensor of any other rank fails it on length.
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

        let Some(mapped) = gptoss_mlx_name(&base)? else {
            continue;
        };
        let fused = base.ends_with("gate_up_proj");
        if !fused && !base.ends_with("down_proj") {
            return fail(format!(
                "Metal GptOss: MXFP4 tensor '{}' is neither the fused gate/up \
                 projection nor the down projection",
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::{Mxfp4MoeRequest, Policy, RuntimeQuant};
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, FileId, QuantScheme, Visibility};

    const HIDDEN: i64 = 64;
    // Deliberately not HIDDEN: with the two equal, a `down_proj` bias of
    // [E, INTER] is also a correct [E, HIDDEN], and the shape-mismatch
    // cases below would assert nothing.
    const INTER: i64 = 96;
    const EXPERTS: i64 = 2;
    const E: &str = "model.layers.0.mlp.experts.";

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }
    fn u8e() -> Encoding {
        Encoding::Raw(DType::U8)
    }

    fn tensor(tensors: &mut Vec<RawTensor>, name: String, shape: Vec<i64>, encoding: Encoding) {
        let elements: i64 = shape.iter().product();
        tensors.push(RawTensor {
            id: TensorId(u32::try_from(tensors.len()).expect("a small fixture")),
            name,
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: u64::try_from(elements).unwrap_or(0),
            shape,
            encoding,
        });
    }

    /// The CUDA-side fixture: one layer of MXFP4 expert triplets.
    fn cuda_checkpoint() -> Vec<RawTensor> {
        let mut t = Vec::new();
        tensor(
            &mut t,
            format!("{E}gate_up_proj_blocks"),
            vec![EXPERTS, 2 * INTER, HIDDEN / 32, 16],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{E}gate_up_proj_scales"),
            vec![EXPERTS, 2 * INTER, HIDDEN / 32],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{E}gate_up_proj_bias"),
            vec![EXPERTS, 2 * INTER],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{E}down_proj_blocks"),
            vec![EXPERTS, HIDDEN, INTER / 32, 16],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{E}down_proj_scales"),
            vec![EXPERTS, HIDDEN, INTER / 32],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{E}down_proj_bias"),
            vec![EXPERTS, HIDDEN],
            bf16(),
        );
        tensor(&mut t, "model.norm.weight".into(), vec![HIDDEN], bf16());
        t
    }

    fn reshaped(mut t: Vec<RawTensor>, name: &str, shape: Vec<i64>) -> Vec<RawTensor> {
        let raw = t
            .iter_mut()
            .find(|raw| raw.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not in the fixture"));
        raw.shape = shape;
        t
    }

    fn renamed(mut t: Vec<RawTensor>, name: &str, to: &str) -> Vec<RawTensor> {
        let raw = t
            .iter_mut()
            .find(|raw| raw.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not in the fixture"));
        raw.name = to.to_string();
        t
    }

    fn without(mut t: Vec<RawTensor>, name: &str) -> Vec<RawTensor> {
        let before = t.len();
        t.retain(|raw| raw.name != name);
        assert_eq!(before - 1, t.len(), "'{name}' was not in the fixture");
        t
    }

    fn cuda_target(tp_rank: u32, tp_size: u32) -> StorageTarget {
        StorageTarget {
            backend: BackendKind::Cuda,
            tp_rank,
            tp_size,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 256,
            tile_map_mask: model_loader::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        }
    }

    fn run(
        tensors: Vec<RawTensor>,
        shape: LoadShape,
        target: &StorageTarget,
        policy: &Policy,
        author: impl FnOnce(&mut Builder<'_>) -> Result<(), Error>,
    ) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(&meta, "gpt-oss-test", shape, &enc, target, policy);
        author(&mut b)?;
        b.finish()
    }

    fn cuda(tensors: Vec<RawTensor>, policy: &Policy) -> Result<ModelContract, Error> {
        run(
            tensors,
            LoadShape::mixture(1, 0, 2, true),
            &cuda_target(0, 1),
            policy,
            author_gpt_oss,
        )
    }

    /// The policy a driver with a native MXFP4 GEMM hands over.
    fn native_policy() -> Policy {
        Policy {
            moe_request: Mxfp4MoeRequest::NativeGemm,
            ..Policy::default()
        }
    }

    fn native_target() -> StorageTarget {
        StorageTarget {
            native_mxfp4_moe: true,
            ..cuda_target(0, 1)
        }
    }

    fn refusal(result: Result<ModelContract, Error>) -> String {
        match result {
            Err(Error::Contract(msg)) => msg,
            Err(other) => panic!("expected a contract refusal, got {other:?}"),
            Ok(_) => panic!("expected a refusal, and the author succeeded"),
        }
    }

    fn names(contract: &ModelContract) -> Vec<&str> {
        contract.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    // ─── which lowering, and the target that cannot serve it ─────────

    /// A driver that asks for the native GEMM its device does not have is
    /// refused at authoring, not at bind.
    #[test]
    fn native_mxfp4_on_a_device_without_it_is_refused() {
        let msg = refusal(cuda(cuda_checkpoint(), &native_policy()));
        assert!(msg.contains("does not support native MXFP4 MoE"), "{msg}");
    }

    /// The routed path publishes the triplet as three plain tensors, and
    /// pairs the scale with the weight it scales.
    ///
    /// The pairing is the part that is easy to drop: publishing the scale
    /// as a plain tensor leaves `quant_meta` empty and the bind fails with
    /// a message about something else entirely.
    #[test]
    fn the_routed_path_publishes_a_triplet_and_states_its_pairing() {
        let contract = cuda(cuda_checkpoint(), &Policy::default()).expect("the fixture authors");
        for half in ["gate_up_proj", "down_proj"] {
            let weight = format!("{E}{half}.weight");
            assert!(names(&contract).contains(&weight.as_str()), "{weight}");
            let scale = contract
                .tensors
                .iter()
                .find(|t| t.name == format!("{E}{half}.weight_scale"))
                .unwrap_or_else(|| panic!("{half} has no published scale"));
            let stated = scale
                .scales
                .as_ref()
                .unwrap_or_else(|| panic!("{half}'s scale is not paired with its weight"));
            assert_eq!(stated.of, weight);
            assert_eq!(stated.group_size, 32);
            assert_eq!(stated.form, ScaleForm::RawE8M0);
            assert!(names(&contract).contains(&format!("{E}{half}.bias").as_str()));
        }
    }

    /// A `_blocks` with no `_scales` or no `_bias` beside it is not this
    /// pass's tensor, and is left for the generic publisher.
    #[test]
    fn a_blocks_tensor_with_no_companion_is_left_alone() {
        for missing in ["gate_up_proj_scales", "gate_up_proj_bias"] {
            let contract = cuda(
                without(cuda_checkpoint(), &format!("{E}{missing}")),
                &Policy::default(),
            )
            .unwrap_or_else(|e| panic!("dropping {missing} is not a refusal: {e}"));
            assert!(
                names(&contract).contains(&format!("{E}gate_up_proj_blocks").as_str()),
                "{missing}: the blocks are published under their own name"
            );
        }
    }

    /// The native path knows two projections and refuses a third.
    #[test]
    fn a_native_triplet_that_is_neither_projection_is_refused() {
        let t = cuda_checkpoint();
        let t = renamed(
            t,
            &format!("{E}gate_up_proj_blocks"),
            &format!("{E}mystery_blocks"),
        );
        let t = renamed(
            t,
            &format!("{E}gate_up_proj_scales"),
            &format!("{E}mystery_scales"),
        );
        let t = renamed(
            t,
            &format!("{E}gate_up_proj_bias"),
            &format!("{E}mystery_bias"),
        );
        let msg = refusal(run(
            t,
            LoadShape::mixture(1, 0, 2, true),
            &native_target(),
            &native_policy(),
            author_gpt_oss,
        ));
        assert!(msg.contains("is not gate_up_proj or down_proj"), "{msg}");
    }

    // ─── the native repack, and every shape it insists on ────────────

    fn native(tensors: Vec<RawTensor>) -> Result<ModelContract, Error> {
        run(
            tensors,
            LoadShape::mixture(1, 0, 2, true),
            &native_target(),
            &native_policy(),
            author_gpt_oss,
        )
    }

    #[test]
    fn the_native_path_repacks_both_banks() {
        let contract = native(cuda_checkpoint()).expect("the fixture authors");
        // gate and up are interleaved row by row, so each half is a stride
        // of the fused bank and lands under its own name.
        for name in [
            format!("{E}gate_proj.weight"),
            format!("{E}gate_proj.weight_scale"),
            format!("{E}gate_proj.bias"),
            format!("{E}up_proj.weight"),
            format!("{E}up_proj.weight_scale"),
            format!("{E}up_proj.bias"),
            format!("{E}down_proj.weight"),
            format!("{E}down_proj.weight_scale"),
            format!("{E}down_proj.bias"),
        ] {
            assert!(names(&contract).contains(&name.as_str()), "{name}");
        }
        // The fused bank is gone: the two halves were published apart.
        assert!(
            !names(&contract).contains(&format!("{E}gate_up_proj.weight").as_str()),
            "the fused name must not survive the split"
        );
    }

    #[test]
    fn a_native_gate_up_bank_of_the_wrong_shape_is_refused() {
        let g = format!("{E}gate_up_proj_blocks");
        for (case, tensors, said) in [
            (
                "the blocks are rank 3",
                reshaped(cuda_checkpoint(), &g, vec![EXPERTS, 2 * INTER, HIDDEN / 32]),
                "unsupported block/scale/bias rank",
            ),
            (
                "the scales are rank 2",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}gate_up_proj_scales"),
                    vec![EXPERTS, 2 * INTER],
                ),
                "unsupported block/scale/bias rank",
            ),
            (
                "the bias is rank 1",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}gate_up_proj_bias"),
                    vec![EXPERTS * 2 * INTER],
                ),
                "unsupported block/scale/bias rank",
            ),
            (
                "the rows do not halve",
                reshaped(cuda_checkpoint(), &g, vec![EXPERTS, 3, HIDDEN / 32, 16]),
                "expected [E, 2I, H/32, 16]",
            ),
            (
                "a block is not 16 bytes",
                reshaped(
                    cuda_checkpoint(),
                    &g,
                    vec![EXPERTS, 2 * INTER, HIDDEN / 32, 8],
                ),
                "expected [E, 2I, H/32, 16]",
            ),
            (
                "the scales do not match the blocks",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}gate_up_proj_scales"),
                    vec![EXPERTS, 2 * INTER, 1],
                ),
                "scale/bias shape mismatch",
            ),
            (
                "the bias does not match the blocks",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}gate_up_proj_bias"),
                    vec![EXPERTS, INTER],
                ),
                "scale/bias shape mismatch",
            ),
        ] {
            let msg = refusal(native(tensors));
            assert!(msg.contains(said), "{case}: {msg}");
            assert!(msg.contains("gate/up"), "{case} names the half: {msg}");
        }
    }

    #[test]
    fn a_native_down_bank_of_the_wrong_shape_is_refused() {
        let d = format!("{E}down_proj_blocks");
        for (case, tensors, said) in [
            (
                "the blocks are rank 3",
                reshaped(cuda_checkpoint(), &d, vec![EXPERTS, HIDDEN, INTER / 32]),
                "unsupported block/scale/bias rank",
            ),
            (
                "a block is not 16 bytes",
                reshaped(cuda_checkpoint(), &d, vec![EXPERTS, HIDDEN, INTER / 32, 8]),
                "expected [E, H, I/32, 16]",
            ),
            (
                "the scales do not match the blocks",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}down_proj_scales"),
                    vec![EXPERTS, HIDDEN, 1],
                ),
                "scale/bias shape mismatch",
            ),
            (
                "the bias does not match the blocks",
                reshaped(
                    cuda_checkpoint(),
                    &format!("{E}down_proj_bias"),
                    vec![EXPERTS, INTER],
                ),
                "scale/bias shape mismatch",
            ),
        ] {
            let msg = refusal(native(tensors));
            assert!(msg.contains(said), "{case}: {msg}");
            assert!(msg.contains("native down"), "{case} names the half: {msg}");
        }
    }

    /// Down is sharded along K, which is the packed axis, so a shard that
    /// splits a 32-element group in half is refused rather than rounded.
    ///
    /// 96 columns is three MXFP4 groups; over two ranks that is 48 each,
    /// which is a group and a half.
    #[test]
    fn a_native_down_shard_that_cuts_an_mxfp4_group_is_refused() {
        let msg = refusal(run(
            cuda_checkpoint(),
            LoadShape::mixture(1, 0, 2, true),
            &StorageTarget {
                native_mxfp4_moe: true,
                ..cuda_target(0, 2)
            },
            &native_policy(),
            author_gpt_oss,
        ));
        assert!(msg.contains("align to MXFP4 32-wide groups"), "{msg}");
    }

    // ─── the streamed group ──────────────────────────────────────────

    fn streamed(tensors: Vec<RawTensor>) -> Result<ModelContract, Error> {
        let policy = Policy {
            stream_routed_experts: true,
            ..Policy::default()
        };
        cuda(tensors, &policy)
    }

    /// One group of arity `n_experts`, holding weights and scales only.
    ///
    /// The biases are kilobytes against megabytes and the bind
    /// de-interleaves them with a kernel, which is host work a group plan
    /// has no node for -- so they stay resident under the names the bind
    /// already reads.
    #[test]
    fn the_streamed_path_groups_the_banks_and_leaves_the_biases_resident() {
        let contract = streamed(cuda_checkpoint()).expect("the fixture authors");
        assert_eq!(contract.groups.len(), 1, "one group per layer");
        let group = &contract.groups[0];
        assert_eq!(group.name, "model.layers.0.mlp.experts");
        assert_eq!(group.arity, EXPERTS as u32);
        let held: Vec<&str> = group.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            held,
            vec![
                "gate_up_proj.weight",
                "gate_up_proj.weight_scale",
                "down_proj.weight",
                "down_proj.weight_scale",
            ],
            "weights and scales only"
        );
        // A slice keeps its rank: one expert's band is [1, ...].
        let banded = &group.tensors[0];
        assert_eq!(banded.shape, Some(vec![1, 2 * INTER, HIDDEN / 32, 16]));
        // The same pairing the resident path states.
        assert_eq!(
            group.tensors[1]
                .scales
                .as_ref()
                .expect("the streamed scale is paired too")
                .of,
            "gate_up_proj.weight"
        );
        for half in ["gate_up_proj", "down_proj"] {
            assert!(
                names(&contract).contains(&format!("{E}{half}.bias").as_str()),
                "{half}'s bias stays resident"
            );
        }
    }

    /// A bank that is not stacked over the row's expert count is refused:
    /// the group's arity comes from the row and the band from the tensor,
    /// and a disagreement would band past the end.
    #[test]
    fn a_streamed_bank_that_is_not_stacked_over_the_experts_is_refused() {
        let msg = refusal(streamed(reshaped(
            cuda_checkpoint(),
            &format!("{E}gate_up_proj_blocks"),
            vec![EXPERTS + 1, 2 * INTER, HIDDEN / 32, 16],
        )));
        assert!(
            msg.contains("is not stacked over 2 experts") && msg.contains("gate_up_proj"),
            "{msg}"
        );
    }

    /// A dense row streams nothing.
    #[test]
    fn a_row_with_no_experts_streams_nothing() {
        let policy = Policy {
            stream_routed_experts: true,
            ..Policy::default()
        };
        let contract = run(
            cuda_checkpoint(),
            LoadShape::dense(1, 0, true),
            &cuda_target(0, 1),
            &policy,
            author_gpt_oss,
        )
        .expect("a dense row is not an error");
        assert!(contract.groups.is_empty(), "no groups were declared");
    }

    /// A layer with no expert banks is passed over rather than grouped
    /// empty.
    #[test]
    fn a_layer_with_no_banks_is_passed_over() {
        let mut t = cuda_checkpoint();
        t.retain(|raw| !raw.name.contains("_blocks") && !raw.name.contains("_scales"));
        let contract = streamed(t).expect("a layer with no banks is not an error");
        assert!(contract.groups.is_empty(), "no group was pushed");
    }

    // ─── the Metal lowering ──────────────────────────────────────────

    fn u32e() -> Encoding {
        Encoding::Raw(DType::U32)
    }

    fn metal_target() -> StorageTarget {
        StorageTarget {
            backend: BackendKind::Metal,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 64,
            ..StorageTarget::default()
        }
    }

    fn mlx(tensors: Vec<RawTensor>) -> Result<ModelContract, Error> {
        run(
            tensors,
            LoadShape::mixture(1, 0, 2, true),
            &metal_target(),
            &Policy::default(),
            author_gpt_oss_mlx,
        )
    }

    /// The `mlx_lm convert -q` layout, which the published-checkpoint
    /// golden never reaches: attention already packed into U32 words with
    /// affine `.scales`/`.biases` beside them.
    ///
    /// These kernels are group-64 and gpt-oss states no quantization block,
    /// so the width is read back out of the shapes: 4 bits over 64 columns
    /// is one group, and eight nibbles to a word is 8 words.
    fn converted_checkpoint() -> Vec<RawTensor> {
        let mut t = Vec::new();
        tensor(
            &mut t,
            "model.embed_tokens.weight".into(),
            vec![128, HIDDEN],
            bf16(),
        );
        let p = "model.layers.0.";
        tensor(
            &mut t,
            format!("{p}input_layernorm.weight"),
            vec![HIDDEN],
            bf16(),
        );
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let base = format!("{p}self_attn.{proj}");
            tensor(&mut t, format!("{base}.weight"), vec![HIDDEN, 8], u32e());
            tensor(&mut t, format!("{base}.scales"), vec![HIDDEN, 1], bf16());
            tensor(&mut t, format!("{base}.biases"), vec![HIDDEN, 1], bf16());
        }
        tensor(&mut t, format!("{p}self_attn.sinks"), vec![4], bf16());
        tensor(&mut t, "model.norm.weight".into(), vec![HIDDEN], bf16());
        tensor(&mut t, "lm_head.weight".into(), vec![128, HIDDEN], bf16());
        t
    }

    fn retyped(mut t: Vec<RawTensor>, name: &str, encoding: Encoding) -> Vec<RawTensor> {
        let raw = t
            .iter_mut()
            .find(|raw| raw.name == name)
            .unwrap_or_else(|| panic!("'{name}' was not in the fixture"));
        raw.encoding = encoding;
        t
    }

    const Q: &str = "model.layers.0.self_attn.q_proj";

    /// The width is derived, not declared, and it lands on the tensors as
    /// an affine-U4 group of 64.
    #[test]
    fn a_converted_checkpoint_derives_its_width_from_the_shapes() {
        let contract = mlx(converted_checkpoint()).expect("the fixture authors");
        let q = contract
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj.weight")
            .expect("the packed projection was declared");
        match &q.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(
                    spec.bits_per_element, 4,
                    "8 words / (2 * 1 group) is 4 bits"
                );
                assert_eq!(spec.group_size, 64);
            }
            other => panic!("expected an affine quant encoding, got {other:?}"),
        }
        // The BF16 head is quantized on the way in rather than left a value.
        let head = contract
            .tensors
            .iter()
            .find(|t| t.name == "lm_head.weight")
            .expect("the head was declared");
        assert!(
            matches!(head.encoding, Encoding::Quant(_)),
            "every matvec here is a quantized one"
        );
        // ...and the rank-1 norms are NOT, which is what rank separates.
        let norm = contract
            .tensors
            .iter()
            .find(|t| t.name == "final_norm.weight")
            .expect("the final norm was declared");
        assert_eq!(norm.encoding, bf16());
    }

    /// An 8-bit conversion is the other width these kernels read.
    #[test]
    fn eight_bit_is_the_other_width_these_kernels_read() {
        // 16 words / (2 * 1 group) is 8 bits.
        let contract = mlx(reshaped(
            converted_checkpoint(),
            &format!("{Q}.weight"),
            vec![HIDDEN, 16],
        ))
        .expect("8-bit authors");
        let q = contract
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj.weight")
            .expect("the packed projection was declared");
        match &q.encoding {
            Encoding::Quant(spec) => assert_eq!(spec.bits_per_element, 8),
            other => panic!("expected an affine quant encoding, got {other:?}"),
        }
    }

    #[test]
    fn a_packed_weight_with_no_scales_is_refused() {
        let msg = refusal(mlx(without(converted_checkpoint(), &format!("{Q}.scales"))));
        assert!(
            msg.contains("is a packed weight with no scales") && msg.contains("q_proj"),
            "{msg}"
        );
    }

    /// Scales with no zero points is MXFP4, which `mlx_lm convert -q`
    /// leaves the experts in — accepted by transmute, not refused.
    #[test]
    fn a_packed_weight_with_scales_but_no_zero_points_is_taken_as_mxfp4() {
        let t = without(converted_checkpoint(), &format!("{Q}.biases"));
        // Eight nibbles to a word over 32-element blocks: 8 words is 2
        // blocks, and MXFP4's scales are the U8 E8M0 exponents.
        let t = reshaped(t, &format!("{Q}.scales"), vec![HIDDEN, 2]);
        let t = retyped(t, &format!("{Q}.scales"), u8e());
        let contract = mlx(t).expect("MXFP4 is accepted by transmute");
        let q = contract
            .tensors
            .iter()
            .find(|t| t.name == "layers.0.self_attn.q_proj.weight")
            .expect("the transmuted projection was declared");
        match &q.encoding {
            Encoding::Quant(spec) => {
                assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
                assert_eq!(spec.group_size, 32);
            }
            other => panic!("expected an MXFP4 encoding, got {other:?}"),
        }
    }

    #[test]
    fn a_conversion_that_is_not_in_groups_of_64_is_refused() {
        for (case, shape) in [
            (
                "the words do not divide by twice the groups",
                vec![HIDDEN, 7],
            ),
            ("there are no groups at all", vec![HIDDEN, 8]),
        ] {
            let t = converted_checkpoint();
            let t = if case.contains("no groups") {
                let t = reshaped(t, &format!("{Q}.scales"), vec![HIDDEN, 0]);
                reshaped(t, &format!("{Q}.biases"), vec![HIDDEN, 0])
            } else {
                t
            };
            let msg = refusal(mlx(reshaped(t, &format!("{Q}.weight"), shape)));
            assert!(
                msg.contains("is not quantized in groups of 64"),
                "{case}: {msg}"
            );
        }
    }

    #[test]
    fn a_width_that_is_neither_4_nor_8_is_refused() {
        // 4 words / (2 * 1 group) is 2 bits.
        let msg = refusal(mlx(reshaped(
            converted_checkpoint(),
            &format!("{Q}.weight"),
            vec![HIDDEN, 4],
        )));
        assert!(msg.contains("is 2-bit, and only 4 and 8"), "{msg}");
    }

    /// The lowering encodes its BF16 projections whatever the request says,
    /// so the ANSWER is discarded — but a request it cannot serve must
    /// still be refused here rather than silently ignored, which is the
    /// only reason the call is made at all.
    #[test]
    fn a_runtime_quant_this_lowering_cannot_encode_is_refused() {
        for quant in [RuntimeQuant::Fp8, RuntimeQuant::Int8, RuntimeQuant::Mxfp4] {
            let policy = Policy {
                runtime_quant: quant,
                ..Policy::default()
            };
            let msg = refusal(run(
                converted_checkpoint(),
                LoadShape::mixture(1, 0, 2, true),
                &metal_target(),
                &policy,
                author_gpt_oss_mlx,
            ));
            assert!(
                msg.contains("Metal GptOss") && msg.contains("`int4` is the only request"),
                "{quant:?}: {msg}"
            );
        }
    }

    /// `int4` is served, and asking for it changes nothing: these matvecs
    /// have no unquantized path, so the encoding happens either way.
    #[test]
    fn asking_for_int4_declares_exactly_what_asking_for_nothing_does() {
        let authored = |quant| {
            let policy = Policy {
                runtime_quant: quant,
                ..Policy::default()
            };
            let contract = run(
                converted_checkpoint(),
                LoadShape::mixture(1, 0, 2, true),
                &metal_target(),
                &policy,
                author_gpt_oss_mlx,
            )
            .expect("int4 is served");
            contract
                .tensors
                .iter()
                .map(|t| (t.name.clone(), t.encoding.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(authored(RuntimeQuant::Int4), authored(RuntimeQuant::None));
    }

    #[test]
    fn a_name_the_lowering_has_no_mapping_for_is_refused() {
        let mut t = converted_checkpoint();
        tensor(
            &mut t,
            "transformer.h.0.ln_1.weight".into(),
            vec![HIDDEN],
            bf16(),
        );
        let msg = refusal(mlx(t));
        assert!(
            msg.contains("no declared mapping or skip")
                && msg.contains("transformer.h.0.ln_1.weight"),
            "{msg}"
        );
    }

    #[test]
    fn a_checkpoint_with_no_decoder_tensors_is_refused() {
        let msg = refusal(mlx(Vec::new()));
        assert!(msg.contains("found no decoder tensors"), "{msg}");
    }

    // ─── the published MXFP4 experts ─────────────────────────────────

    /// The published triplets, on top of the converted attention: this is
    /// the half of a real gpt-oss MLX checkpoint that `mlx_lm` leaves
    /// alone.
    fn published_experts() -> Vec<RawTensor> {
        let mut t = converted_checkpoint();
        let p = "model.layers.0.";
        tensor(
            &mut t,
            format!("{p}mlp.router.weight"),
            vec![EXPERTS, HIDDEN],
            bf16(),
        );
        tensor(&mut t, format!("{p}mlp.router.bias"), vec![EXPERTS], bf16());
        let e = format!("{p}mlp.experts.");
        tensor(
            &mut t,
            format!("{e}gate_up_proj_blocks"),
            vec![EXPERTS, 2 * MLX_INTER, HIDDEN / 32, 16],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{e}gate_up_proj_scales"),
            vec![EXPERTS, 2 * MLX_INTER, HIDDEN / 32],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{e}gate_up_proj_bias"),
            vec![EXPERTS, 2 * MLX_INTER],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{e}down_proj_blocks"),
            vec![EXPERTS, HIDDEN, MLX_INTER / 32, 16],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{e}down_proj_scales"),
            vec![EXPERTS, HIDDEN, MLX_INTER / 32],
            u8e(),
        );
        tensor(
            &mut t,
            format!("{e}down_proj_bias"),
            vec![EXPERTS, HIDDEN],
            bf16(),
        );
        t
    }

    const ME: &str = "model.layers.0.mlp.experts.";
    // Not the CUDA fixture's 96: the Metal matvecs are group-64, so the
    // down projection's column count has to be a multiple of 64.
    const MLX_INTER: i64 = 128;

    #[test]
    fn the_published_experts_are_decoded_and_re_encoded_per_half() {
        let contract = mlx(published_experts()).expect("the fixture authors");
        let all = names(&contract);
        for half in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("layers.0.mlp.experts.{half}.weight");
            let w = contract
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{half} was not declared"));
            assert!(
                matches!(w.encoding, Encoding::Quant(_)),
                "{half} is re-encoded as the affine the matvecs read"
            );
            assert!(all.contains(&format!("layers.0.mlp.experts.{half}.bias").as_str()));
            // The intermediate steps exist but are not the driver's to bind.
            for step in ["mxfp4_blocks", "mxfp4_exponents", "dequantized"] {
                let step = format!("layers.0.mlp.experts.{half}.{step}");
                let t = contract
                    .tensors
                    .iter()
                    .find(|t| t.name == step)
                    .unwrap_or_else(|| panic!("{step} was not declared"));
                assert_eq!(t.visibility, Visibility::Internal, "{step}");
            }
        }
        // The source triplets were consumed, not published: nothing under
        // the checkpoint's own `_blocks`/`_scales`/`_bias` names survives.
        assert!(
            !all.iter().any(|n| n.contains("_proj_blocks")
                || n.contains("_proj_scales")
                || n.contains("_proj_bias")),
            "the sources were consumed, and {all:?} still names one"
        );
    }

    /// Each half of the fused bank is half its stored rows.
    #[test]
    fn the_fused_bank_is_split_and_down_is_not() {
        let contract = mlx(published_experts()).expect("the fixture authors");
        let shape = |name: &str| {
            contract
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("{name} was not declared"))
                .shape
                .clone()
        };
        assert_eq!(
            shape("layers.0.mlp.experts.gate_proj.dequantized"),
            Some(vec![EXPERTS * MLX_INTER, HIDDEN]),
            "half the stored rows"
        );
        assert_eq!(
            shape("layers.0.mlp.experts.down_proj.dequantized"),
            Some(vec![EXPERTS * HIDDEN, MLX_INTER]),
            "down keeps all of them"
        );
        assert_eq!(
            shape("layers.0.mlp.experts.gate_proj.bias"),
            Some(vec![EXPERTS, MLX_INTER])
        );
    }

    #[test]
    fn a_block_tensor_with_no_scales_beside_it_is_refused() {
        let msg = refusal(mlx(without(
            published_experts(),
            &format!("{ME}gate_up_proj_scales"),
        )));
        assert!(
            msg.contains("is an MXFP4 block tensor with no '_scales' beside it"),
            "{msg}"
        );
    }

    #[test]
    fn a_block_tensor_that_is_not_shaped_against_its_scales_is_refused() {
        for (case, tensors) in [
            (
                "the blocks are rank 3",
                reshaped(
                    published_experts(),
                    &format!("{ME}gate_up_proj_blocks"),
                    vec![EXPERTS, 2 * MLX_INTER, HIDDEN / 32],
                ),
            ),
            (
                "the scales are rank 4",
                reshaped(
                    published_experts(),
                    &format!("{ME}gate_up_proj_scales"),
                    vec![EXPERTS, 2 * MLX_INTER, HIDDEN / 32, 1],
                ),
            ),
            (
                "a block is not 16 bytes",
                reshaped(
                    published_experts(),
                    &format!("{ME}gate_up_proj_blocks"),
                    vec![EXPERTS, 2 * MLX_INTER, HIDDEN / 32, 8],
                ),
            ),
            (
                "the stacked axes disagree",
                reshaped(
                    published_experts(),
                    &format!("{ME}gate_up_proj_scales"),
                    vec![EXPERTS, INTER, HIDDEN / 32],
                ),
            ),
        ] {
            let msg = refusal(mlx(tensors));
            assert!(
                msg.contains("is not shaped [experts, rows, groups, 16]"),
                "{case}: {msg}"
            );
        }
    }

    #[test]
    fn a_block_tensor_that_is_neither_projection_is_refused() {
        let t = published_experts();
        let t = renamed(
            t,
            &format!("{ME}down_proj_blocks"),
            &format!("{ME}mystery_blocks"),
        );
        let t = renamed(
            t,
            &format!("{ME}down_proj_scales"),
            &format!("{ME}mystery_scales"),
        );
        let msg = refusal(mlx(without(t, &format!("{ME}down_proj_bias"))));
        assert!(
            msg.contains("is neither the fused gate/up projection nor the down projection"),
            "{msg}"
        );
    }

    #[test]
    fn a_bias_that_does_not_match_the_projection_it_biases_is_refused() {
        for (case, shape) in [
            ("it is rank 1", vec![EXPERTS * 2 * INTER]),
            ("it is not the stored rows", vec![EXPERTS, INTER]),
        ] {
            let msg = refusal(mlx(reshaped(
                published_experts(),
                &format!("{ME}gate_up_proj_bias"),
                shape,
            )));
            assert!(
                msg.contains("does not match the projection it biases"),
                "{case}: {msg}"
            );
        }
    }
    // ── The rename the Metal binder looks up ─────────────────────────

    /// `gptoss_mlx_name` maps every checkpoint tensor to the name the
    /// Metal binder asks for. A WRONG answer is silent:
    /// `Store::checkpoint_names` returns an empty candidate list for a
    /// name it does not know, so the tensor is simply absent from the
    /// forward pass and the model still generates -- with a projection
    /// or a norm that was never bound.
    ///
    /// Every pair is stated. A table this small is cheaper to read than
    /// a rule, and the rule is what got the pairs wrong.
    #[test]
    fn every_checkpoint_name_maps_to_the_one_the_binder_asks_for() {
        for (raw, bound) in [
            // The head is its OWN tensor here, not the embedding under
            // another name -- gpt-oss unties them.
            ("lm_head.weight", "lm_head.weight"),
            ("model.embed_tokens.weight", "embed_tokens.weight"),
            ("model.norm.weight", "final_norm.weight"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "model.layers.23.mlp.experts.gate_up_proj",
                "layers.23.mlp.experts.gate_up_proj",
            ),
            (
                "model.layers.7.input_layernorm.weight",
                "layers.7.input_layernorm.weight",
            ),
        ] {
            assert_eq!(
                gptoss_mlx_name(raw).expect("a declared mapping"),
                Some(bound.to_string()),
                "{raw}"
            );
        }
    }

    /// The pass's own output is a valid input.
    ///
    /// A lowered artifact is re-authored through the same function, so
    /// every name it produces has to survive a second pass unchanged.
    /// Without that a re-load renames `final_norm.weight` to nothing and
    /// refuses a file this very code wrote.
    #[test]
    fn what_the_pass_produces_passes_through_it_unchanged() {
        for produced in ["embed_tokens.weight", "final_norm.weight", "lm_head.weight"] {
            assert_eq!(
                gptoss_mlx_name(produced).expect("its own output is accepted"),
                Some(produced.to_string()),
                "a re-load of a lowered artifact refuses {produced}"
            );
        }
    }

    /// A name with no mapping is REFUSED, and the refusal names it.
    ///
    /// This is the whole reason the function returns a `Result` rather
    /// than an `Option`: silently skipping an unknown tensor is exactly
    /// the failure mode -- the load succeeds and a weight is missing.
    #[test]
    fn a_name_with_no_mapping_is_refused_and_the_message_says_which() {
        for unmapped in [
            "visual.blocks.0.attn.qkv.weight",
            "transformer.h.0.attn.c_attn.weight",
            "model.rotary_emb.inv_freq",
        ] {
            let why = gptoss_mlx_name(unmapped).expect_err("no mapping");
            let Error::Contract(why) = why else {
                panic!("expected a contract refusal, got {why:?}")
            };
            assert!(
                why.contains(unmapped) && why.contains("GptOss"),
                "the refusal must name the tensor and the schema: {why}"
            );
        }
    }

    /// A layer tensor whose index is not a number is refused SEPARATELY.
    ///
    /// `layers.foo.weight` is not an unmapped family, it is a malformed
    /// index, and a message that said "no declared mapping" would send
    /// the reader looking for a table entry that could never exist.
    #[test]
    fn a_malformed_layer_index_is_refused_for_being_malformed() {
        let why = gptoss_mlx_name("model.layers.foo.weight").expect_err("not an index");
        let Error::Contract(why) = why else {
            panic!("expected a contract refusal, got {why:?}")
        };
        assert!(
            why.contains("invalid layer index"),
            "a bad index read as a missing mapping: {why}"
        );

        let why = gptoss_mlx_name("model.layers.0").expect_err("no member at all");
        let Error::Contract(why) = why else {
            panic!("expected a contract refusal, got {why:?}")
        };
        assert!(why.contains("malformed"), "{why}");
    }

    /// The `lm_head.` arm comes first, and it has to.
    ///
    /// `mlx::already_lowered` answers `true` for `embed_tokens.` among
    /// others. gpt-oss unties its head, so `lm_head.` must reach an
    /// identity of its own rather than whatever the shared table would
    /// do with it -- and unlike the llama-shaped families, where the two
    /// arms are disjoint and the order is free, this family's `lm_head.`
    /// really is answered twice.
    #[test]
    fn the_untied_head_is_answered_before_the_shared_lowered_table_sees_it() {
        assert_eq!(
            gptoss_mlx_name("lm_head.weight").expect("mapped"),
            Some("lm_head.weight".to_string())
        );
        // Stated so the claim above is checked rather than asserted: if
        // the shared table grows an `lm_head.` entry that answers
        // differently, the arm order stops being free and this test is
        // what says so.
        assert!(
            !crate::shared::mlx::already_lowered("lm_head.weight")
                || gptoss_mlx_name("lm_head.weight").expect("mapped")
                    == Some("lm_head.weight".to_string()),
            "the shared lowered table now answers `lm_head.` too, and the \
             two answers must agree or the arm order is load-bearing"
        );
    }
}
