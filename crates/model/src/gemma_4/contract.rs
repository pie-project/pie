//! What Gemma-4 binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/gemma4/gemma4_contract.hpp`. The only
//! family the encode pipeline can scope a load to: an encode-scoped rank
//! declares the vision and audio towers and nothing else, so it never
//! allocates the language model — and because that is a *declaration* rather
//! than a filter applied to a finished contract, the plan it compiles has no
//! trace of the tensors it skipped.

use model_loader::contract::Expr;
use model_loader::error::Error;

use crate::shared::builder::Builder;
use crate::shared::mlx;

/// gemma4, gemma4_text.
pub fn author_gemma4(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_encode_scope()?;
    // The decoder is nested; the vision and audio towers are not, and they
    // have `self_attn.q_proj.weight` of their own.
    b.decoder_layer_prefix("model.language_model.layers.");
    fold_router_scale(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    //
    // The MoE decode runs through flashinfer's CUTLASS grouped GEMM, which
    // reads fc1's output as [linear|gate]; the checkpoint stores [gate|up].
    // `moe/flashinfer_moe.hpp` states the convention this has to agree
    // with -- "fc1 weights must be stacked as [up; gate], not pie's usual
    // [gate; up]" -- and it is a comment rather than an argument, so
    // nothing on the forward side can be asked whether it still holds.
    // Flipping this constant alone swaps gate and up inside every expert,
    // silently and without a shape error.
    const GATE_SECOND: bool = true;
    b.fused_moe_gate_up_tp_slices(GATE_SECOND)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// Fold `1/sqrt(H)` into Gemma-4's per-layer router scale.
///
/// The router pipeline is `rmsnorm_no_scale(x) * scale * (1/sqrt(H))`
/// followed by a linear. Multiplying the two constants together at load time
/// lets the forward collapse the first three steps into one
/// rmsnorm-with-weight call, and `scale` is a `[H]` vector so the fold costs
/// nothing to store.
///
/// `H` is read off the tensor rather than off the config: the fold has to
/// use the same `H` the forward's rmsnorm does, and that one is this
/// vector's length by construction.
///
/// **Rounding.** The kernel rounds to nearest-even; the host loop this
/// replaced truncated, biasing every element down by up to one bf16 ULP. See
/// the C++ header's note about re-recording bit-exact MoE baselines.
///
/// Scoped to `decoder_layer_prefix` for the reason the fused-projection pass
/// is: Gemma-4 nests its decoder under the vision and audio towers, and a
/// suffix match alone would fold a tower's tensor of the same name.
fn fold_router_scale(b: &mut Builder<'_>) -> Result<(), Error> {
    const SUFFIX: &str = ".router.scale";
    // `tensors()` holds source names, so the bound prefix has to be mapped
    // through `source_name` the way the fused-projection pass does.
    let layers = b.source_name(b.decoder_layer_prefix_value());
    for raw in b.tensors().to_vec() {
        if !raw.name.starts_with(&layers) || !raw.name.ends_with(SUFFIX) {
            continue;
        }
        // Refused rather than skipped. A skip publishes the scale
        // UNFOLDED, and the forward has no second place to apply
        // `1/sqrt(H)` -- every router logit would come out scaled by
        // `sqrt(H)` too much, which routes to the wrong experts without
        // ever producing a shape error or a NaN.
        if raw.shape.len() != 1 || raw.shape[0] <= 0 {
            return mlx::fail(format!(
                "gemma_4 router scale: '{}' is {:?}, and the fold needs the \
                 [hidden] vector the forward's rmsnorm reads",
                raw.name, raw.shape
            ));
        }
        // Computed exactly as the forward would have: fp32 `sqrt`, then a
        // reciprocal. Reordering this to `rsqrt` or to fp64 would change the
        // last bit of every router logit.
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

/// The Metal lowering: rename for MLX's binder, bind in place. Ported from
/// `crates/driver-metal/csrc/src/model/gemma4/gemma4_contract.hpp`.
pub fn author_gemma4_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    // KV is shared over the tail of the stack:
    // `layer >= num_hidden_layers - num_kv_shared_layers` attends KV an
    // earlier layer wrote, and the checkpoint still ships its dead k/v/k_norm.
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
    // The towers. Text decode binds none of it.
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
    // Gemma 4 ships tied embeddings; a checkpoint carrying both would declare
    // `shared_embedding` twice and be rejected as a duplicate, which is the
    // truthful outcome.
    if let Some(tail) = raw_name.strip_prefix("lm_head.") {
        return Ok(Some(format!("shared_embedding.{tail}")));
    }
    // Its own output is a valid input: see `mlx::already_lowered`. After the
    // `lm_head.` arm above, which is not an identity for this family at all.
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
    // The PLE table and its projection are layer-less and keep their own
    // names.
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
    // A KV-shared layer attends the KV an earlier layer wrote, so its own
    // k/v projections and k-norm are never bound.
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::contract::{ModelContract, ScaleFactor};
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, DType, Encoding, FileId, TensorId};

    const HIDDEN: i64 = 64;
    const P: &str = "model.language_model.layers.0.";

    fn cuda(tensors: Vec<RawTensor>) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let enc = StoredEncoding::dense();
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 256,
            tile_map_mask: model_loader::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "gemma-4-test",
            LoadShape::dense(1, 32, true),
            &enc,
            &target,
            &policy,
        );
        author_gemma4(&mut b)?;
        b.finish()
    }

    fn scale(shape: &[i64]) -> Vec<RawTensor> {
        vec![RawTensor {
            id: TensorId(0),
            name: format!("{P}router.scale"),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 2,
            shape: shape.to_vec(),
            encoding: Encoding::Raw(DType::BF16),
        }]
    }

    /// `1/sqrt(H)` is folded into the router scale at load, and `H` is
    /// read off the vector rather than off the row.
    ///
    /// The forward collapses `rmsnorm_no_scale(x) * scale * (1/sqrt(H))`
    /// into one rmsnorm-with-weight call, so the constant has to be in
    /// the weight before the model runs. `H` comes from the tensor
    /// because the forward's rmsnorm reads exactly this vector's length;
    /// a config-derived `H` could differ and nothing would say so.
    #[test]
    fn the_router_scale_carries_one_over_sqrt_hidden() {
        let c = cuda(scale(&[HIDDEN])).expect("gemma-4 authors");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("router.scale"))
            .expect("the scale is published");
        let folded = match &t.expr {
            Expr::Scale {
                factor: ScaleFactor::Uniform(bits),
                ..
            } => f32::from_bits(*bits),
            other => panic!("the scale was published unfolded: {other:?}"),
        };
        let want = 1.0f32 / (HIDDEN as f32).sqrt();
        assert!(
            (folded - want).abs() < f32::EPSILON,
            "folded {folded}, wanted {want}"
        );
    }

    /// A router scale that is not the `[hidden]` vector is refused, not
    /// published unfolded.
    ///
    /// Skipping it was the old behaviour and it is the dangerous one:
    /// the forward has no second place to apply `1/sqrt(H)`, so every
    /// router logit comes out scaled by `sqrt(H)` too much and the layer
    /// routes to the wrong experts. No shape error, no NaN, just a
    /// different model.
    #[test]
    fn a_router_scale_that_is_not_the_hidden_vector_is_refused() {
        for (case, shape) in [
            ("rank 2", vec![1, HIDDEN]),
            ("rank 0", Vec::new()),
            ("an empty axis", vec![0]),
        ] {
            let why = cuda(scale(&shape)).expect_err("a malformed scale is refused");
            let Error::Contract(why) = why else {
                panic!("expected a contract refusal, got {why:?}")
            };
            assert!(
                why.contains("router scale") && why.contains("[hidden] vector"),
                "{case}: {why}"
            );
        }
    }

    /// The fold is scoped to the DECODER, so a tower tensor of the same
    /// name keeps its own value.
    ///
    /// Gemma-4 nests its decoder under the vision and audio towers and
    /// those towers have tensors of the same suffix. A suffix match alone
    /// would fold a constant into a tower weight that never wanted one.
    #[test]
    fn a_towers_scale_of_the_same_name_is_left_alone() {
        let mut tensors = scale(&[HIDDEN]);
        tensors.push(RawTensor {
            id: TensorId(1),
            name: "model.vision_tower.layers.0.router.scale".into(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 2,
            shape: vec![HIDDEN],
            encoding: Encoding::Raw(DType::BF16),
        });
        let c = cuda(tensors).expect("gemma-4 authors");
        let tower = c
            .tensors
            .iter()
            .find(|t| t.name.contains("vision_tower"))
            .expect("the tower tensor is still published");
        assert!(
            !matches!(tower.expr, Expr::Scale { .. }),
            "a tower weight was folded: {:?}",
            tower.expr
        );
    }

    /// Every arm of the Metal rename, and what each one prevents.
    ///
    /// A WRONG answer here is silent: `Store::checkpoint_names` returns an
    /// empty candidate list for a name it does not know, so a tensor
    /// renamed to something the binder never asks for is simply absent
    /// from the model, and the layer that wanted it reads zeros.
    #[test]
    fn every_arm_of_the_metal_rename_answers_the_name_the_binder_asks_for() {
        // No KV sharing: -1 is the "no shared tail" sentinel.
        let name = |raw: &str| gemma4_mlx_name(raw, -1).expect("a declared name");

        // The four towers are skipped: text decode binds none of them.
        for tower in ["audio_tower", "vision_tower", "embed_audio", "embed_vision"] {
            assert_eq!(
                name(&format!("model.{tower}.blocks.0.w.weight")),
                None,
                "{tower} was bound into a text decode"
            );
        }

        // `lm_head` is the tied embedding under another name, and this arm
        // sits BEFORE `already_lowered` because it is not an identity.
        assert_eq!(
            name("lm_head.weight"),
            Some("shared_embedding.weight".to_string())
        );

        // A name the lowering already produced is its own answer, which is
        // what makes the pass idempotent over a pie artifact.
        for lowered in [
            "shared_embedding.weight",
            "embed_tokens_per_layer.weight",
            "per_layer_model_projection.weight",
        ] {
            assert_eq!(name(lowered), Some(lowered.to_string()), "{lowered}");
        }

        // The decoder's own spellings, both of them.
        for prefix in ["model.language_model.", "language_model.model."] {
            assert_eq!(
                name(&format!("{prefix}embed_tokens.weight")),
                Some("shared_embedding.weight".to_string()),
                "{prefix}"
            );
            // The PLE table and its projections are layer-less and keep
            // their names once the decoder prefix comes off.
            for direct in [
                "embed_tokens_per_layer.weight",
                "per_layer_model_projection.weight",
                "per_layer_projection_norm.weight",
            ] {
                assert_eq!(
                    name(&format!("{prefix}{direct}")),
                    Some(direct.to_string()),
                    "{prefix}{direct}"
                );
            }
            assert_eq!(
                name(&format!("{prefix}norm.weight")),
                Some("final_norm.weight".to_string())
            );
            assert_eq!(
                name(&format!("{prefix}layers.3.self_attn.q_proj.weight")),
                Some("layers.3.self_attn.q_proj.weight".to_string())
            );
        }
    }

    /// A name with no declared mapping is REFUSED, not passed through.
    ///
    /// Passing it through is the tempting default and the wrong one: an
    /// unrecognised name that survives as itself is a tensor the binder
    /// never asks for, and the model loads a layer short. The refusal
    /// names the tensor so the missing arm can be written.
    #[test]
    fn a_name_with_no_declared_mapping_is_refused_and_named() {
        let why = gemma4_mlx_name("model.mystery_tower.weight", -1)
            .expect_err("an undeclared name is refused");
        let Error::Contract(why) = why else {
            panic!("expected a contract refusal, got {why:?}")
        };
        assert!(
            why.contains("no declared mapping or skip") && why.contains("mystery_tower"),
            "{why}"
        );
    }

    /// The KV-shared tail binds no k/v of its own.
    ///
    /// Those layers attend the KV an earlier layer wrote, and the
    /// checkpoint still ships their dead `k_proj`, `v_proj` and `k_norm`.
    /// Binding them would fill a slot the forward pass reads from
    /// somewhere else -- not a crash, just an attention over the wrong
    /// keys on the last layers of the stack.
    #[test]
    fn the_kv_shared_tail_binds_no_keys_or_values_of_its_own() {
        const FIRST_SHARED: i64 = 30;
        let at = |layer: u32, member: &str| {
            gemma4_mlx_name(
                &format!("model.language_model.layers.{layer}.{member}"),
                FIRST_SHARED,
            )
            .expect("a declared name")
        };
        for member in [
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.k_norm.weight",
        ] {
            assert_eq!(at(FIRST_SHARED as u32, member), None, "shared: {member}");
            assert_eq!(at(33, member), None, "past the boundary: {member}");
            // One layer BEFORE the boundary writes its own KV.
            assert_eq!(
                at(FIRST_SHARED as u32 - 1, member),
                Some(format!("layers.29.{member}")),
                "the last unshared layer: {member}"
            );
        }
        // And a shared layer still binds everything that is not k/v.
        for member in ["self_attn.q_proj.weight", "self_attn.o_proj.weight"] {
            assert_eq!(
                at(FIRST_SHARED as u32, member),
                Some(format!("layers.30.{member}")),
                "a shared layer lost {member}"
            );
        }
    }
}
