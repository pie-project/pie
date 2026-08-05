//! What the llama-like families bind.
//!
//! Ported from `driver/cuda/src/model/llama_like/llama_like_contract.hpp`.
//! Four binders share this lineage — plain llama/qwen, Mistral-3, Phi-3 and
//! OLMo — and they differ in the contract only where the checkpoint differs.
//! Phi-3 is the one with real work: it ships `self_attn.qkv_proj` and
//! `mlp.gate_up_proj` pre-fused, and the bind path reads the six pieces.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::Expr;
use pie_loader::error::Error;

use crate::common::builder::{Builder, author_dense_contract};
use crate::common::mlx;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// llama, llama3, mistral, qwen2, qwen3: the checkpoint already uses the
/// names the bind path reads. BF16 -> FP8/INT8 runtime quant is wired for
/// these.
pub fn author_llama_like(b: &mut Builder<'_>) -> Result<(), Error> {
    b.allow_bf16_runtime_quant();
    author_dense_contract(b)
}

/// mistral3, ministral3, olmo2, olmo3: dense, nothing beyond the
/// name-pattern rules.
pub fn author_dense(b: &mut Builder<'_>) -> Result<(), Error> {
    author_dense_contract(b)
}

/// Phi-3: undo the two source-side fusions first, so the generic tail never
/// sees the fused tensors and the dense join can re-fuse on CUDA's terms.
pub fn author_phi3(b: &mut Builder<'_>) -> Result<(), Error> {
    phi3_fused_splits(b)?;
    author_dense_contract(b)
}

/// Split Phi-3's fused QKV and gate/up back into the six tensors the
/// llama-like bind path reads.
fn phi3_fused_splits(b: &mut Builder<'_>) -> Result<(), Error> {
    for raw in b.tensors().to_vec() {
        if raw.name.ends_with(".self_attn.qkv_proj.weight") {
            phi3_qkv_split(b, raw)?;
        } else if raw.name.ends_with(".mlp.gate_up_proj.weight") {
            phi3_gate_up_split(b, raw)?;
        }
    }
    Ok(())
}

fn phi3_qkv_split(b: &mut Builder<'_>, raw: &RawTensor) -> Result<(), Error> {
    if raw.shape.len() != 2 {
        return fail(format!("Phi-3 fused QKV '{}' must be 2-D", raw.name));
    }
    let q_rows = raw.shape[1];
    let kv_rows = (raw.shape[0] - q_rows) / 2;
    if q_rows <= 0 || kv_rows <= 0 || q_rows + 2 * kv_rows != raw.shape[0] {
        return fail(format!(
            "Phi-3 fused QKV '{}' has an unsupported shape",
            raw.name
        ));
    }
    let cols = raw.shape[1];
    let base = raw
        .name
        .strip_suffix(".self_attn.qkv_proj.weight")
        .expect("matched above");
    let specs = [
        ("q_proj", 0, q_rows),
        ("k_proj", q_rows, kv_rows),
        ("v_proj", q_rows + kv_rows, kv_rows),
    ];
    for (proj, start, rows) in specs {
        let (expr, local_rows) = b.band(Expr::src(&raw.name), 0, start, rows);
        b.push_expr(
            format!("{base}.self_attn.{proj}.weight"),
            raw,
            vec![local_rows, cols],
            expr,
        );
    }
    Ok(())
}

fn phi3_gate_up_split(b: &mut Builder<'_>, raw: &RawTensor) -> Result<(), Error> {
    if raw.shape.len() != 2 || raw.shape[0] % 2 != 0 {
        return fail(format!(
            "Phi-3 fused gate/up '{}' has an unsupported shape",
            raw.name
        ));
    }
    let half_rows = raw.shape[0] / 2;
    let cols = raw.shape[1];
    let base = raw
        .name
        .strip_suffix(".mlp.gate_up_proj.weight")
        .expect("matched above");
    for (proj, start) in [("gate_proj", 0), ("up_proj", half_rows)] {
        let (expr, local_rows) = b.band(Expr::src(&raw.name), 0, start, half_rows);
        b.push_expr(
            format!("{base}.mlp.{proj}.weight"),
            raw,
            vec![local_rows, cols],
            expr,
        );
    }
    Ok(())
}

/// The Metal lowering of the llama-shaped families: rename for MLX's binder,
/// bind in place. Ported from
/// `driver/metal/src/model/llama/llama_contract.hpp` — what is
/// family-specific is the tensor NAMES, and the mechanics are
/// [`mlx::author_mlx_file`].
pub fn author_llama_mlx(b: &mut Builder<'_>) -> Result<(), Error> {
    // `tie_word_embeddings` is what the config SAYS; a shipped `lm_head` is
    // what the checkpoint DOES. When they disagree the tensors win, because
    // mapping a real `lm_head` onto `shared_embedding` declares that name
    // twice and the load fails with a duplicate rather than with the
    // disagreement.
    let has_lm_head = b
        .tensors()
        .iter()
        .any(|raw| raw.name.starts_with("lm_head."));
    let tied = b.facts().tied_embeddings && !has_lm_head;
    mlx::author_mlx_file(b, "llama", &move |_, raw_name| {
        llama_mlx_name(raw_name, tied)
    })
}

/// The runtime name for a checkpoint tensor, or `None` to skip it. Every
/// name is either mapped or explicitly skipped; an unrecognised one is an
/// error, because a tensor declared under its checkpoint name would never be
/// found by the binder.
fn llama_mlx_name(raw_name: &str, tied: bool) -> Result<Option<String>, Error> {
    // A multimodal release wraps the decoder. Text decode binds none of the
    // towers.
    for skip in [
        "model.visual.",
        "model.vision_tower.",
        "model.audio_tower.",
        "visual.",
    ] {
        if raw_name.starts_with(skip) {
            return Ok(None);
        }
    }
    // Rotary inverse frequencies are recomputed on the GPU from `rope_theta`,
    // so a checkpoint that persists them is shipping a derived tensor.
    if raw_name.contains("rotary_emb.inv_freq") {
        return Ok(None);
    }

    if let Some(tail) = raw_name.strip_prefix("lm_head.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("lm_head.{tail}")
        }));
    }

    // Its own output is a valid input: see `mlx::already_lowered`. After the
    // `lm_head.` arm above, because that one is not an identity when tied.
    if mlx::already_lowered(raw_name) {
        return Ok(Some(raw_name.to_string()));
    }

    // Some releases nest the decoder one level deeper. Accept either.
    let rest = raw_name
        .strip_prefix("model.language_model.")
        .or_else(|| raw_name.strip_prefix("model."));
    let Some(rest) = rest else {
        return mlx::fail(format!(
            "Metal llama schema has no declared mapping or skip for '{raw_name}'"
        ));
    };

    if let Some(tail) = rest.strip_prefix("embed_tokens.") {
        return Ok(Some(if tied {
            format!("shared_embedding.{tail}")
        } else {
            format!("embed_tokens.{tail}")
        }));
    }
    if rest == "norm.weight" {
        return Ok(Some("final_norm.weight".to_string()));
    }

    let (layer, member) = mlx::layer_member(rest, "llama", raw_name)?;
    // The mixture's naming is the same rule in every routed family, so it is
    // asked of one place rather than restated here.
    if let Some(renamed) = mlx::routed_expert_member(raw_name, member, "llama", false)? {
        return Ok(Some(format!("layers.{layer}.{renamed}")));
    }
    Ok(Some(format!("layers.{layer}.{member}")))
}
