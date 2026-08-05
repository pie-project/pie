//! Phi-3's load contract.
//!
//! Phi-3 stores what every other dense decoder stores, fused differently:
//! `qkv_proj` and `gate_up_proj` arrive already joined. So the contract is the
//! ordinary dense one with two source-side *splits* in front of it — undo the
//! checkpoint's fusion, then let the dense join re-fuse on the device's terms.
//!
//! It sits in its own generation directory rather than inside Llama 3's
//! because the splits are Phi-3's alone; what it shares with Llama 3 is the
//! three-pass dense tail, which is spelled out below like every other
//! generation spells it out.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::Expr;
use pie_loader::error::Error;

use pie_model_common::builder::Builder;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// Phi-3: undo the two source-side fusions first, so the generic tail never
/// sees the fused tensors and the dense join can re-fuse on CUDA's terms.
pub fn author_phi3(b: &mut Builder<'_>) -> Result<(), Error> {
    phi3_fused_splits(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
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
