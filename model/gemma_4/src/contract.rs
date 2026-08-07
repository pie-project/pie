//! What Gemma-4 binds.
//!
//! Ported from `driver/cuda/src/model/gemma4/gemma4_contract.hpp`. The only
//! family the encode pipeline can scope a load to: an encode-scoped rank
//! declares the vision and audio towers and nothing else, so it never
//! allocates the language model — and because that is a *declaration* rather
//! than a filter applied to a finished contract, the plan it compiles has no
//! trace of the tensors it skipped.

use pie_loader::contract::Expr;
use pie_loader::error::Error;

use pie_model_common::builder::{Builder};
use pie_model_common::mlx;

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
    // `gemma4_moe_gate_up_swapped()` on the forward side is this same
    // constant, and the two have to agree -- flipping one alone swaps gate
    // and up inside every expert, silently and without a shape error.
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
        if raw.shape.len() != 1 || raw.shape[0] <= 0 {
            continue;
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
/// `driver/metal/src/model/gemma4/gemma4_contract.hpp`.
pub fn author_gemma4_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    // KV is shared over the tail of the stack:
    // `layer >= num_hidden_layers - num_kv_shared_layers` attends KV an
    // earlier layer wrote, and the checkpoint still ships its dead k/v/k_norm.
    let first_shared = if b.facts().num_kv_shared_layers > 0 {
        i64::from(b.facts().num_hidden_layers) - i64::from(b.facts().num_kv_shared_layers)
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
