//! What GLM-5.1 binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/glm5/glm5_contract.hpp`. MLA attention
//! plus a DSA indexer plus routed and shared MoE. Nothing about the
//! checkpoint's layout is unusual; what is unusual is that this is the one
//! family whose routed experts the CUDA driver re-quantizes at load time,
//! and the one that ships FP8 experts a runtime FP4 request is allowed to
//! consume.

use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding, QuantScheme, QuantSpec};

use crate::shared::builder::{Builder, is_raw};
use crate::shared::moe::hf_moe_expert_stacks;

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
    // it. `moe/flashinfer_moe.hpp` is what this has to agree with — "the
    // runner reads the gate half from the *second* half of the fc1 output
    // ... the opposite of pie's chunked_swiglu" — and a load that swaps
    // while the matmul does not is silently wrong output, not a load
    // error.
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
        let declared = b.define(
            scale_name.clone(),
            scale_local,
            f32enc.clone(),
            Some(scale_shape),
        );
        b.mark_internal(declared);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;

    const HIDDEN: i64 = 512;
    const LATENT: i64 = 128;

    fn fp8() -> Encoding {
        Encoding::Raw(DType::F8E4M3)
    }

    fn f32e() -> Encoding {
        Encoding::Raw(DType::F32)
    }

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }

    fn tensor(id: u32, name: &str, shape: Vec<i64>, encoding: Encoding) -> RawTensor {
        use model_loader::types::{FileId, TensorId};
        RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 0,
            shape,
            encoding,
        }
    }

    /// One layer's `kv_b_proj` and whatever factors came with it.
    fn checkpoint(weight: Encoding, factors: Option<(&str, Vec<i64>, Encoding)>) -> Vec<RawTensor> {
        let mut ck = vec![tensor(
            1,
            "model.layers.0.self_attn.kv_b_proj.weight",
            vec![HIDDEN, LATENT],
            weight,
        )];
        if let Some((suffix, shape, encoding)) = factors {
            ck.push(tensor(
                2,
                &format!("model.layers.0.self_attn.kv_b_proj.{suffix}"),
                shape,
                encoding,
            ));
        }
        ck
    }

    fn author_over(tensors: Vec<RawTensor>) -> ModelContract {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "glm-5-test",
            LoadShape::mixture(1, 0, 2, true),
            &encoding,
            &target,
            &policy,
        );
        bf16_kv_b_proj(&mut b).expect("the pass does not refuse");
        b.publish_remaining().expect("the tail publishes");
        b.finish().expect("the contract compiles")
    }

    /// What the contract declares, and in what encoding.
    fn declared(contract: &ModelContract) -> Vec<(String, Encoding)> {
        contract
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.encoding.clone()))
            .collect()
    }

    /// Whether the factors are still a tensor the DRIVER binds.
    ///
    /// When the pass claims the pair it declares the scale `Internal` --
    /// the plan still materializes it, because the dequantize term reads
    /// it, but no bind path looks it up. An unclaimed scale is published
    /// `Public` by `publish_remaining` instead.
    ///
    /// This is the only thing that can tell a claimed BF16 weight from an
    /// unclaimed one, because both are DECLARED bf16 -- one by this pass
    /// and one by the generic path.
    fn scale_is_bound_by_the_driver(contract: &ModelContract) -> bool {
        contract.tensors.iter().any(|t| {
            t.name.contains("scale") && t.visibility == model_loader::contract::Visibility::Public
        })
    }

    fn kv_b_proj(contract: &ModelContract) -> Option<Encoding> {
        declared(contract)
            .into_iter()
            .find(|(name, _)| name.ends_with("kv_b_proj.weight"))
            .map(|(_, encoding)| encoding)
    }

    /// The shipped FP8 is consumed and a BF16 tensor is what the kernel binds.
    ///
    /// This is the whole reason the pass exists: the kimi_mla kernels this
    /// family shares read `kv_b_proj` as BF16 and there is no FP8 variant,
    /// so if this pass does nothing the bind gets a tensor its kernel
    /// cannot read.
    #[test]
    fn an_fp8_kv_b_proj_is_published_as_the_bf16_the_kernel_reads() {
        let contract = author_over(checkpoint(
            fp8(),
            Some(("weight_scale_inv", vec![HIDDEN / 128, LATENT / 128], f32e())),
        ));
        assert_eq!(
            kv_b_proj(&contract),
            Some(bf16()),
            "the FP8 weight is dequantized at load, not bound as it shipped"
        );
        assert!(
            !scale_is_bound_by_the_driver(&contract),
            "the factors became the term's own operand, not something the \
             bind path looks up"
        );
    }

    /// A BF16 checkpoint gets no pass at all.
    ///
    /// The generic dense path already publishes it, and running the
    /// dequantize term over a tensor that is already BF16 would be a copy
    /// of the largest attention weight in the model for nothing.
    /// A BF16 checkpoint gets no pass, even when factors are lying beside it.
    ///
    /// The fixture ships the scale on purpose. Without it the encoding
    /// check is unfalsifiable: a BF16 weight run through the dequantize
    /// term would be declared BF16 too, which is what it already was, and
    /// only the CONSUMED scale says which of the two published it.
    #[test]
    fn a_bf16_checkpoint_is_left_to_the_generic_path() {
        let contract = author_over(checkpoint(
            bf16(),
            Some(("weight_scale_inv", vec![HIDDEN / 128, LATENT / 128], f32e())),
        ));
        assert_eq!(
            kv_b_proj(&contract),
            Some(bf16()),
            "it is still published -- by `publish_remaining`, not by this pass"
        );
        assert!(
            scale_is_bound_by_the_driver(&contract),
            "the pass did not claim the pair, so the factors are still a \
             tensor the driver binds"
        );
    }

    /// Both spellings of the blocking are the same term.
    ///
    /// GLM-5.1 ships a DeepSeek-style square block, one factor per 128x128
    /// tile. Other exports of the same architecture ship one factor per
    /// output channel, as a RANK-1 vector. `scale_per_block` reads the
    /// blocking off the ratio of the two shapes, so the rank-1 case only
    /// has to be reshaped to `[rows, 1]` -- the same bytes under the rank
    /// the pairing needs.
    #[test]
    fn a_per_channel_vector_and_a_square_block_both_pair() {
        for (what, shape) in [
            ("a square block", vec![HIDDEN / 128, LATENT / 128]),
            ("a per-channel vector", vec![HIDDEN]),
        ] {
            let contract =
                author_over(checkpoint(fp8(), Some(("weight_scale_inv", shape, f32e()))));
            assert_eq!(
                kv_b_proj(&contract),
                Some(bf16()),
                "{what} names the blocking and the pass pairs with it"
            );
        }
    }

    /// Three spellings of the factor name, and the pass finds each.
    ///
    /// `_scale_inv` is DeepSeek's, `_scale` is the compressed-tensors one,
    /// and a bare `.scale` hangs off the module rather than off `.weight`.
    /// An export whose spelling this pass does not know does not fail --
    /// it falls through to the generic path and binds FP8 to a kernel that
    /// reads BF16, which is why each spelling is stated.
    #[test]
    fn every_spelling_of_the_scale_is_found() {
        for suffix in ["weight_scale_inv", "weight_scale", "scale"] {
            let contract = author_over(checkpoint(
                fp8(),
                Some((suffix, vec![HIDDEN / 128, LATENT / 128], f32e())),
            ));
            assert_eq!(
                kv_b_proj(&contract),
                Some(bf16()),
                "`{suffix}` is a spelling this pass reads"
            );
        }
    }

    /// An FP8 weight with no factors anywhere is not claimed.
    ///
    /// There is nothing to dequantize WITH, so the pass declines and the
    /// generic path publishes what shipped. Stated because the alternative
    /// -- refusing the load -- would reject a checkpoint some other pass
    /// may be able to serve.
    #[test]
    fn an_fp8_weight_with_no_factors_is_left_alone() {
        let contract = author_over(checkpoint(fp8(), None));
        assert_eq!(
            kv_b_proj(&contract),
            Some(fp8()),
            "unpaired FP8 is published as it shipped rather than refused"
        );
    }

    /// The whole author runs the pass, not just this test.
    ///
    /// Every other test here calls `bf16_kv_b_proj` directly, so deleting
    /// the call from [`author_glm5`] left all of them green -- the pass
    /// worked perfectly and nothing ran it. That is the shape the failure
    /// would really take: FP8 `kv_b_proj` bound to a kimi_mla kernel that
    /// reads BF16, which is wrong numbers rather than a load error.
    ///
    /// It does NOT pin the order against `allow_bf16_runtime_quant`,
    /// though the author's comment reads as if something should: swapping
    /// the two lines is invisible here, and correctly so. That call only
    /// sets a flag, which is read by a later pass, so both orderings still
    /// have this pass consume the FP8 pair before anything can act on the
    /// flag. Pinning it would need a fixture that requests a re-quant, and
    /// the ordering that would then matter is this pass against the
    /// CONSUMER of the flag rather than against its setter.
    #[test]
    fn the_author_itself_runs_the_pass() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: checkpoint(
                fp8(),
                Some(("weight_scale_inv", vec![HIDDEN / 128, LATENT / 128], f32e())),
            ),
        };
        let target = StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        };
        let encoding = StoredEncoding::dense();
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "glm-5-test",
            LoadShape::mixture(1, 0, 2, true),
            &encoding,
            &target,
            &policy,
        );
        author_glm5(&mut b).expect("the author does not refuse");
        let contract = b.finish().expect("the contract compiles");
        assert_eq!(
            kv_b_proj(&contract),
            Some(bf16()),
            "the author reaches the pass; without the call the kernel gets \
             the FP8 it cannot read"
        );
        assert!(
            !scale_is_bound_by_the_driver(&contract),
            "and the pair was consumed by THIS pass rather than left for a \
             later one"
        );
    }

    /// The factors have to be F32, and the weight has to be a matrix.
    ///
    /// Both are what `scale_per_block` assumes and neither is checked
    /// downstream: a rank-1 weight has no axis to shard and an integer
    /// factor is not a multiplier.
    #[test]
    fn a_pairing_the_term_cannot_express_is_declined_rather_than_built() {
        let bad_factor_type = author_over(checkpoint(
            fp8(),
            Some(("weight_scale_inv", vec![HIDDEN / 128, LATENT / 128], bf16())),
        ));
        assert_eq!(
            kv_b_proj(&bad_factor_type),
            Some(fp8()),
            "a BF16 factor is not the F32 multiplier the term takes"
        );

        let scalar_factor = author_over(checkpoint(
            fp8(),
            Some(("weight_scale_inv", Vec::new(), f32e())),
        ));
        assert_eq!(
            kv_b_proj(&scalar_factor),
            Some(fp8()),
            "a rank-0 factor states no blocking at all"
        );
    }

    /// And the ranks have to be a pair, on either side.
    ///
    /// `scale_per_block` reads the blocking off the RATIO of a
    /// `[rows, cols]` weight to a `[row_blocks, col_blocks]` factor, so
    /// each side has to be two-dimensional for there to be a ratio. A
    /// rank-1 factor is the one exception and is promoted to `[rows, 1]`
    /// above; nothing else can be recovered, because a third dimension
    /// names a blocking the other side does not have.
    ///
    /// Declining rather than refusing is the same stance
    /// [`an_fp8_weight_with_no_factors_is_left_alone`] takes: the weight
    /// is published as it shipped and a pass that knows what the extra
    /// dimension means may still claim it.
    #[test]
    fn a_rank_the_term_cannot_pair_is_declined_on_either_side() {
        let stacked_weight = author_over(vec![
            tensor(
                1,
                "model.layers.0.self_attn.kv_b_proj.weight",
                vec![2, HIDDEN, LATENT],
                fp8(),
            ),
            tensor(
                2,
                "model.layers.0.self_attn.kv_b_proj.weight_scale_inv",
                vec![HIDDEN / 128, LATENT / 128],
                f32e(),
            ),
        ]);
        assert_eq!(
            kv_b_proj(&stacked_weight),
            Some(fp8()),
            "a rank-3 weight has no single set of rows for the factor to \
             describe"
        );

        let stacked_factor = author_over(checkpoint(
            fp8(),
            Some((
                "weight_scale_inv",
                vec![1, HIDDEN / 128, LATENT / 128],
                f32e(),
            )),
        ));
        assert_eq!(
            kv_b_proj(&stacked_factor),
            Some(fp8()),
            "a rank-3 factor blocks a dimension the weight does not have"
        );
    }
}
