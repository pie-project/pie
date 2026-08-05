//! What GLM-5.1 binds.
//!
//! Ported from `driver/cuda/src/model/glm5/glm5_contract.hpp`. MLA attention
//! plus a DSA indexer plus routed and shared MoE. Nothing about the
//! checkpoint's layout is unusual; what is unusual is that this is the one
//! family whose routed experts the CUDA driver re-quantizes at load time,
//! and the one that ships FP8 experts a runtime FP4 request is allowed to
//! consume.

use pie_loader::contract::{Expr, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding, QuantScheme, QuantSpec};

use pie_model_common::builder::{Builder, is_raw};
use pie_model_common::moe::hf_moe_expert_stacks;

/// glm_moe_dsa. `embed_tokens` is sharded on axis 0 under TP to save
/// per-rank memory; the FP4 path touches routed and shared experts only,
/// because there is no FP4 GEMM for the attention projections on this
/// hardware.
pub fn author_glm5(b: &mut Builder<'_>) -> Result<(), Error> {
    b.shard_embed_tokens();
    // Before the runtime-quant flags: this consumes the FP8 pair outright,
    // so there is no `kv_b_proj` left for a bf16 re-quant request to claim.
    bf16_kv_b_proj(b)?;
    b.allow_bf16_runtime_quant();
    b.allow_mxfp4_runtime_quant();
    // GLM-5.2 ships routed experts one tensor per expert; glm5_forward reads
    // the fused 3-D slabs. Float only: this family's quantised checkpoints
    // keep the per-expert layout and take the per-expert forward path.
    //
    // `gate_second` publishes each expert's halves as `[up | gate]`, which
    // flashinfer's CUTLASS grouped GEMM reads fc1 as. Stating it here is the
    // whole point: the alternative is a driver-side block swap over the
    // largest tensor in the model, done after the loader has already placed
    // it. `glm5_moe_gate_up_swapped()` on the forward side is this same
    // constant, and the two have to agree — a load that swaps and a matmul
    // that does not is silently wrong output, not a load error.
    hf_moe_expert_stacks(b, /*gate_second=*/ true, /*float_only=*/ true)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// Publish `kv_b_proj` in BF16, dequantizing the shipped FP8 in the loader.
///
/// The kimi_mla kernels this family shares read `kv_b_proj` as BF16 and
/// there is no FP8 variant, so *something* has to dequantize it. Stated
/// here, the packed source is consumed, the arena accounts for exactly one
/// tensor, and a BF16 checkpoint needs no pass at all because the generic
/// dense path already publishes it.
///
/// **The scale's shape is what says how the weight is blocked.** GLM-5.1
/// ships a DeepSeek-style square block (one factor per 128x128 tile); other
/// exports of the same architecture ship one factor per output channel. Both
/// are the same term here, because `scale_per_block` reads the blocking off
/// the ratio of the two shapes rather than taking a group size on the side.
/// A rank-1 per-channel vector is transmuted to `[rows, 1]` first: that is
/// the same bytes under the rank the pairing needs.
fn bf16_kv_b_proj(b: &mut Builder<'_>) -> Result<(), Error> {
    let f32enc = Encoding::Raw(DType::F32);
    for raw in b.tensors().to_vec() {
        if !raw.name.ends_with(".self_attn.kv_b_proj.weight") {
            continue;
        }
        // A BF16 checkpoint is already what the kernel wants; leaving it to
        // the generic path is what deletes the copy the bind used to make.
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
            // A bare `.scale` shares the base rather than hanging off
            // `.weight`, so ".weight" minus the dot's six characters come
            // off first.
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
        // Both sides shard on the same axis, which for a row-parallel weight
        // is the rows. The loader is what checks the two shards still line
        // up: a rank whose row band is not a whole number of blocks makes
        // the pairing fail with both shapes named, rather than silently
        // reading a factor that describes another rank's rows.
        let (scale_local, scale_shape) = b.shard(factor_expr, factor_shape, axis);
        let scale_name = b.output_name(&factors.name);
        if let Some(declared) = b.define(
            scale_name.clone(),
            scale_local,
            f32enc.clone(),
            Some(scale_shape),
        ) {
            b.mark_internal(declared);
        }

        // Bits/group left at 0 — "the scheme's default" — exactly as the C++
        // `quant_spec` states them, so the two authors' contracts compare
        // equal field for field.
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
