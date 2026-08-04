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
