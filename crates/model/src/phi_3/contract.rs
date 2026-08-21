use model_loader::checkpoint::RawTensor;
use model_loader::contract::Expr;
use model_loader::error::Error;

use crate::shared::builder::Builder;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn author_phi3(b: &mut Builder<'_>) -> Result<(), Error> {
    phi3_fused_splits(b)?;

    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

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
