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

use crate::common::builder::{Builder, align_up, mxfp4_encoding};
use crate::common::policy::Mxfp4MoePolicy;

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
