//! What the Qwen3.5 hybrid families bind.
//!
//! Ported from `driver/cuda/src/model/qwen3_5/qwen3_5_contract.hpp`. The
//! dense hybrid needs one real thing beyond the generic rules: the Gated
//! DeltaNet tensors stack `[K | K | V]` on axis 0, so a uniform row shard
//! cuts across the block boundaries and hands a rank part of K where it
//! needs V. The MoE hybrid adds the shared-expert join and the per-expert
//! stacks.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::Expr;
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, QuantScheme};

use crate::common::builder::{Builder, author_dense_contract, is_raw};
use crate::common::moe::hf_moe_expert_stacks;

/// qwen3_5, qwen3_5_text: a dense hybrid decoder under the usual names.
pub fn author_qwen3_5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    // The vision-language checkpoints nest the decoder; the text-only ones
    // do not. Both are this row, so the prefix is asked for rather than
    // declared.
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fused_in_proj_joins(b)?;
    gdn_fp32_parameters(b)?;
    // The speculative-decoding head is a full-attention layer with the same
    // projection names, so it wants the same join. Checkpoints without one
    // make this a no-op.
    b.also_join_module("mtp.layers.0.");
    mtp_int8_lm_head(b)?;
    author_dense_contract(b)
}

/// qwen3_moe, qwen3_5_moe, qwen3_5_moe_text.
///
/// Deliberately no `dense_fused_projection_joins`: this bind path reads
/// q/k/v separately, and the MLP lives in the experts, so there is no
/// layer-level `gate_proj`/`up_proj` pair to join.
pub fn author_qwen3_5_moe(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    b.decoder_layer_prefix_any_of(&["model.language_model.layers.", "model.layers."]);
    gdn_kkv_blocked_shards(b)?;
    gdn_fused_in_proj_joins(b)?;
    gdn_fp32_parameters(b)?;
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // Both the pre-fused and the per-expert stacking paths publish in the
    // order the bound driver expects.
    let gate_second = b.knobs().qwen35_moe_gate_up_swapped;
    b.fused_moe_gate_up_tp_slices(gate_second)?;
    shared_expert_gate_up_joins(b);
    hf_moe_expert_stacks(b, gate_second, false)?;
    b.publish_remaining()
}

/// This rank's `[K/T | K/T | V/T]` view of a `[2K + V, ...]` tensor.
///
/// Returned with its declared shape, because the extent depends on whether
/// the world divides each block and only `local_extent` knows.
fn gdn_kkv_blocked(b: &Builder<'_>, raw: &RawTensor, k_dim: i64, v_dim: i64) -> (Expr, Vec<i64>) {
    let src = || Expr::src(&raw.name);
    let (key_lo, key_rows) = b.band(src(), 0, 0, k_dim);
    let (key_hi, _) = b.band(src(), 0, k_dim, k_dim);
    let (value, value_rows) = b.band(src(), 0, 2 * k_dim, v_dim);
    let mut shape = raw.shape.clone();
    shape[0] = 2 * key_rows + value_rows;
    (Expr::concat(0, vec![key_lo, key_hi, value]), shape)
}

/// Shard the Gated DeltaNet tensors whose leading axis stacks `[K | K | V]`.
///
/// `linear_attn.in_proj_qkv.weight`, `conv1d.weight` and `conv1d.bias` all
/// stack two key blocks and one value block on axis 0. Take each block's
/// band and join them: every rank gets its own `[K/T | K/T | V/T]`, which is
/// what the GDN kernels address. Without this the loader has no shard axis
/// for these names and leaves them replicated, so every rank loads the whole
/// tensor and the driver slices it afterwards with device-to-device copies.
///
/// `K` and `V` come from the checkpoint, not from a config field: `in_proj_z`
/// is `[V, hidden]`, and `in_proj_qkv` is `[2K + V, hidden]`, so the pair
/// determines both.
fn gdn_kkv_blocked_shards(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.target().tp_size <= 1 {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
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
        for leaf in ["in_proj_qkv.weight", "conv1d.weight", "conv1d.bias"] {
            // The fused layout publishes qkv as the first leg of
            // `in_proj_qkvz` instead; conv1d still wants it under its own
            // name either way.
            if b.knobs().qwen35_fused_gdn_projection && leaf == "in_proj_qkv.weight" {
                continue;
            }
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

/// Join the Gated DeltaNet input projections the forward reads pre-fused.
///
/// `qwen3_5_forward` branches on `la_in_proj_qkvz`: when the fused layout is
/// selected it wants `[2K | V | V]` (qkv then z) and `[H | H]` (b then a) as
/// single tensors. Stating that here rather than concatenating after the
/// load is what makes the layout free — the earlier bind-time version had to
/// hold the sources *and* the join, which on Qwen3.6-35B-A3B meant 1.4 GB of
/// duplicate weights.
///
/// The qkv leg keeps the per-block shard from [`gdn_kkv_blocked_shards`]; z,
/// b and a shard uniformly on axis 0, so a plain `split` is right for them.
fn gdn_fused_in_proj_joins(b: &mut Builder<'_>) -> Result<(), Error> {
    if !b.knobs().qwen35_fused_gdn_projection {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
        let la = format!("{}{layer}.linear_attn.", b.decoder_layer_prefix_value());
        let (Some(qkv), Some(z), Some(bb), Some(aa)) = (
            b.find(&b.source_name(&format!("{la}in_proj_qkv.weight"))),
            b.find(&b.source_name(&format!("{la}in_proj_z.weight"))),
            b.find(&b.source_name(&format!("{la}in_proj_b.weight"))),
            b.find(&b.source_name(&format!("{la}in_proj_a.weight"))),
        ) else {
            continue;
        };
        if qkv.shape.is_empty() || z.shape.is_empty() || bb.shape.is_empty() || aa.shape.is_empty()
        {
            continue;
        }
        let v_dim = z.shape[0];
        let conv_dim = qkv.shape[0];
        if conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0 {
            continue;
        }
        let k_dim = (conv_dim - v_dim) / 2;

        let (blocked, mut qkvz_shape) = gdn_kkv_blocked(b, qkv, k_dim, v_dim);
        let z_local = b.split(Expr::src(&z.name), 0);
        qkvz_shape[0] += b.local_extent(v_dim);
        let encoding = qkv.encoding.clone();
        let (qkv_id, z_id) = (qkv.id, z.id);
        b.define(
            b.output_name(&format!("{la}in_proj_qkvz.weight")),
            Expr::concat(0, vec![blocked, z_local]),
            encoding,
            Some(qkvz_shape),
        );
        b.consume(qkv_id);
        b.consume(z_id);

        let b_local = b.split(Expr::src(&bb.name), 0);
        let a_local = b.split(Expr::src(&aa.name), 0);
        let mut ba_shape = bb.shape.clone();
        ba_shape[0] = b.local_extent(bb.shape[0]) + b.local_extent(aa.shape[0]);
        let encoding = bb.encoding.clone();
        let (b_id, a_id) = (bb.id, aa.id);
        b.define(
            b.output_name(&format!("{la}in_proj_ba.weight")),
            Expr::concat(0, vec![b_local, a_local]),
            encoding,
            Some(ba_shape),
        );
        b.consume(b_id);
        b.consume(a_id);
    }
    Ok(())
}

/// Widen the two gated-delta-net parameters the kernels read as fp32.
///
/// `A_log` and the gated RMSNorm's weight enter the GDN kernels as
/// `const float*`, but HF ships them fp32 on Qwen3.5-4B and **bf16** on
/// Qwen3.6-35B-A3B. Only these two: `dt_bias` sits beside them in the same
/// module and is read as bf16, so a suffix match any looser than this list
/// would silently widen it.
///
/// The `already fp32` branch is not an optimization; it is required. A
/// `Cast` to the encoding its operand already has is refused (a node may not
/// denote exactly its operand), which is what makes the two checkpoint
/// conventions impossible to paper over with one unconditional cast.
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

/// Publish an int8 view of `lm_head` for the speculative head to read.
///
/// The draft step and the main path read the *same* head, at different
/// precisions. So this is not a re-encode: both views are published, and
/// `quantized_view` leaves the bf16 original alone. A tied checkpoint has no
/// `lm_head.weight`; the head is `embed_tokens`, which is what the bind
/// resolves to and therefore what gets quantized.
fn mtp_int8_lm_head(b: &mut Builder<'_>) -> Result<(), Error> {
    if !b.knobs().qwen35_mtp_int8_lm_head || b.find("mtp.fc.weight").is_none() {
        return Ok(());
    }
    // The decoder prefix varies (the VL checkpoints nest it), so the tied
    // fallback matches the suffix.
    let head = b.find("lm_head.weight").or_else(|| {
        b.tensors()
            .iter()
            .copied()
            .find(|raw| raw.name.ends_with(".embed_tokens.weight"))
    });
    // Only a bf16 source: a checkpoint that already ships a quantized head
    // wants that head, not a second encoding of it.
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

/// Join the shared expert's gate and up projections the MoE forward reads
/// pre-fused, and optionally the scalar gate row after them.
///
/// The sources are **not** consumed. Unlike the Gated DeltaNet join, both
/// unfused projections stay live: the fold-into-routed path reads them
/// separately, and which path runs is a per-step decision. So this slab is
/// additive, exactly like the Kimi and DSv4 expert stacks.
///
/// Only bf16 sources, because the bind had exactly one converter and a
/// checkpoint that ships this pair quantized wants the quantized kernels.
/// The scalar gate is replicated, not sharded, so it is named directly while
/// gate and up take the column-parallel split.
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

    let scalar = if b.knobs().qwen35_fused_shared_scalar_gate {
        b.find(&b.source_name(&format!("{lp}_gate.weight")))
            .filter(|scalar| {
                is_raw(&scalar.encoding, DType::BF16)
                    && scalar.shape.len() == 2
                    && scalar.shape[1] == gate.shape[1]
            })
    } else {
        None
    };

    let encoding = gate.encoding.clone();
    let cols = gate.shape[1];
    match scalar {
        Some(scalar) => {
            let shape = vec![rows + scalar.shape[0], cols];
            let expr = Expr::concat(0, vec![gate_local, up_local, Expr::src(&scalar.name)]);
            b.define(
                b.output_name(&format!("{lp}.gate_up_gate_proj.weight")),
                expr,
                encoding,
                Some(shape),
            );
        }
        None => {
            b.define(
                b.output_name(&format!("{lp}.gate_up_proj.weight")),
                Expr::concat(0, vec![gate_local, up_local]),
                encoding,
                Some(vec![rows, cols]),
            );
        }
    }
}

/// Every module that carries a shared expert: the decoder layers and, when
/// the checkpoint ships one, the speculative-decoding block. The MTP layer
/// is not under `decoder_layer_prefix`, and its bind runs the same fusion.
fn shared_expert_gate_up_joins(b: &mut Builder<'_>) {
    for layer in 0..b.facts().num_hidden_layers {
        let prefix = format!("{}{layer}.", b.decoder_layer_prefix_value());
        shared_expert_gate_up_join(b, &prefix);
    }
    shared_expert_gate_up_join(b, "mtp.layers.0.");
}
