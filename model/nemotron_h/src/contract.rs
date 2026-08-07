//! What Nemotron-H binds.
//!
//! Ported from `driver/cuda/src/model/nemotron_h/nemotron_h_contract.hpp`.
//! The Mamba2/attention/MoE hybrid keeps its decoder under
//! `language_model.backbone.`, and its MoE GEMM addresses all experts of a
//! layer as one contiguous slab. The contract declares the slab and then
//! declares each expert as a slice of it, so no byte is copied twice.

use pie_loader::checkpoint::RawTensor;
use pie_loader::contract::{Expr, TensorType};
use pie_loader::error::Error;
use pie_loader::types::{DType, Encoding};

use pie_model_common::builder::{Builder, is_raw};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn author_nemotron_h(b: &mut Builder<'_>) -> Result<(), Error> {
    mamba_tp_shards(b)?;
    b.fused_moe_gate_up_tp_slices(false)?;
    packed_expert_views(b)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// A rename, written only when there is something to rename.
///
/// The algebra refuses a node that denotes its operand, so a fold whose
/// target is the shape the operand already has must be left out rather than
/// written and ignored. `to` may carry one `-1`, resolved here the way
/// `infer` resolves it.
fn retype_if_different(node: Expr, from: &[i64], mut to: Vec<i64>, encoding: &Encoding) -> Expr {
    let total: i64 = from.iter().product();
    let known: i64 = to.iter().filter(|&&dim| dim >= 0).product();
    for dim in &mut to {
        if *dim < 0 && known > 0 {
            *dim = total / known;
        }
    }
    if to == from {
        return node;
    }
    node.transmute(TensorType::new(to, encoding.clone()))
}

/// This rank's share of a leading axis that is `units` equal blocks.
///
/// `split` divides an extent, and an extent cannot say what it is a
/// concatenation *of*: `[heads * head_dim, cols]` divides by a `tp_size`
/// that does not divide `heads`, and the split then cuts a head in half.
/// Reshaping the axis to `[units, block]` first puts the divisibility
/// question on the value that has to answer it, so the loader rejects an
/// indivisible world naming this tensor instead of loading a wrong one.
///
/// Both transmutes are pure renames, so the composition still compiles to
/// one contiguous run per band.
fn split_by_unit(
    b: &Builder<'_>,
    rows: Expr,
    rows_shape: &[i64],
    units: i64,
    block: i64,
    out_shape: Vec<i64>,
    encoding: &Encoding,
) -> Expr {
    let folded = retype_if_different(rows, rows_shape, vec![units, block], encoding);
    let local_shape = [b.local_extent(units), block];
    retype_if_different(b.split(folded, 0), &local_shape, out_shape, encoding)
}

/// Elements per row of `shape`, i.e. everything but the leading extent.
fn row_elems(shape: &[i64]) -> i64 {
    shape[1..].iter().product()
}

/// `shape` with its leading extent replaced by `-1`.
fn rows_inferred(shape: &[i64]) -> Vec<i64> {
    let mut out = shape.to_vec();
    out[0] = -1;
    out
}

/// Split every Mamba mixer across the tensor-parallel world.
///
/// The knob (`PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD` in the driver) is a kill
/// switch for bisecting a numerical regression; because it changes the
/// contract, and the cache key is the contract's, flipping it re-plans
/// rather than reusing a cached plan for the other layout.
fn mamba_tp_shards(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.target().tp_size <= 1 || !b.knobs().nemotron_tp_mamba_sharding {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
        let mp = b.source_name(&format!("language_model.backbone.layers.{layer}.mixer."));
        layer_mamba_tp(b, &mp, i64::from(b.facts().mamba_groups))?;
    }
    Ok(())
}

/// Declare this rank's slice of one Mamba mixer.
///
/// The mixer's `in_proj` is five bands stacked on one axis — `z` and `x` of
/// `heads * head_dim` rows, `B` and `C` of `groups * state`, and `dt` of
/// `heads` — and TP splits each band by its own unit. So the local tensor is
/// the concatenation of five independently sharded bands, which is a
/// sentence the algebra can say and a row range cannot.
fn layer_mamba_tp(b: &mut Builder<'_>, mp: &str, groups: i64) -> Result<(), Error> {
    let (
        Some(in_proj),
        Some(conv_w),
        Some(conv_b),
        Some(a_log),
        Some(d),
        Some(dt_bias),
        Some(norm_w),
        Some(out_proj),
    ) = (
        b.find(&format!("{mp}in_proj.weight")),
        b.find(&format!("{mp}conv1d.weight")),
        b.find(&format!("{mp}conv1d.bias")),
        b.find(&format!("{mp}A_log")),
        b.find(&format!("{mp}D")),
        b.find(&format!("{mp}dt_bias")),
        b.find(&format!("{mp}norm.weight")),
        b.find(&format!("{mp}out_proj.weight")),
    )
    else {
        return Ok(()); // an attention or MoE layer; it has no mixer to split.
    };

    // Every extent below is read off a tensor. Only `groups` cannot be: see
    // `ModelFacts::mamba_groups`.
    let in_shape = in_proj.shape.clone();
    let conv_shape = conv_w.shape.clone();
    let out_shape = out_proj.shape.clone();
    if in_shape.len() != 2
        || conv_shape.len() < 2
        || out_shape.len() != 2
        || a_log.shape.len() != 1
        || norm_w.shape.len() != 1
    {
        return fail(format!(
            "nemotron_h: mamba mixer '{mp}' has an unexpected tensor rank"
        ));
    }
    let heads = a_log.shape[0];
    let intermediate = norm_w.shape[0];
    let hidden = in_shape[1];
    let conv_dim = conv_shape[0];
    let group_state = (conv_dim - intermediate) / 2;
    if heads <= 0
        || intermediate % heads != 0
        || conv_dim <= intermediate
        || (conv_dim - intermediate) % 2 != 0
        || groups <= 0
        || group_state % groups != 0
        || in_shape[0] != 2 * intermediate + 2 * group_state + heads
        || out_shape[0] != hidden
        || out_shape[1] != intermediate
    {
        return fail(format!(
            "nemotron_h: mamba mixer '{mp}' does not match heads={heads} groups={groups}"
        ));
    }
    let head_dim = intermediate / heads;
    let state = group_state / groups;

    // The five bands of `in_proj`, in the order the mixer reads them.
    let band_shape = vec![-1, hidden];
    let in_band = |start: i64, rows: i64| Expr::src(&in_proj.name).slice(0, start, rows);
    let z_shape = [intermediate, hidden];
    let bc_shape = [group_state, hidden];
    let enc = &in_proj.encoding;
    let bands = vec![
        split_by_unit(
            b,
            in_band(0, intermediate),
            &z_shape,
            heads,
            head_dim * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(intermediate, intermediate),
            &z_shape,
            heads,
            head_dim * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate, group_state),
            &bc_shape,
            groups,
            state * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate + group_state, group_state),
            &bc_shape,
            groups,
            state * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate + 2 * group_state, heads),
            &[heads, hidden],
            heads,
            hidden,
            band_shape,
            enc,
        ),
    ];
    let local_heads = b.local_extent(heads);
    let local_groups = b.local_extent(groups);
    let local_intermediate = local_heads * head_dim;
    let local_group_state = local_groups * state;
    let encoding = in_proj.encoding.clone();
    let in_proj_id = in_proj.id;
    b.define(
        b.output_name(&in_proj.name),
        Expr::concat(0, bands),
        encoding,
        Some(vec![
            2 * local_intermediate + 2 * local_group_state + local_heads,
            hidden,
        ]),
    );
    b.consume(in_proj_id);

    // `conv1d` carries the same bands minus `z` and `dt`, over whatever
    // trailing extents the checkpoint gave it (`[conv_dim, k]` or
    // `[conv_dim, 1, k]`).
    for raw in [conv_w, conv_b] {
        let shape = raw.shape.clone();
        let cols = row_elems(&shape);
        let published = rows_inferred(&shape);
        let mut z_band = shape.clone();
        z_band[0] = intermediate;
        let mut bc_band = shape.clone();
        bc_band[0] = group_state;
        let conv_band = |start: i64, rows: i64| Expr::src(&raw.name).slice(0, start, rows);
        let enc = &raw.encoding;
        let conv_bands = vec![
            split_by_unit(
                b,
                conv_band(0, intermediate),
                &z_band,
                heads,
                head_dim * cols,
                published.clone(),
                enc,
            ),
            split_by_unit(
                b,
                conv_band(intermediate, group_state),
                &bc_band,
                groups,
                state * cols,
                published.clone(),
                enc,
            ),
            split_by_unit(
                b,
                conv_band(intermediate + group_state, group_state),
                &bc_band,
                groups,
                state * cols,
                published,
                enc,
            ),
        ];
        let mut local_shape = shape;
        local_shape[0] = local_intermediate + 2 * local_group_state;
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(
            b.output_name(&raw.name),
            Expr::concat(0, conv_bands),
            encoding,
            Some(local_shape),
        );
        b.consume(id);
    }

    // One entry per head, so the head axis is already the leading extent.
    for raw in [a_log, d, dt_bias] {
        let expr = b.split(Expr::src(&raw.name), 0);
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(
            b.output_name(&raw.name),
            expr,
            encoding,
            Some(vec![local_heads]),
        );
        b.consume(id);
    }
    let norm_expr = split_by_unit(
        b,
        Expr::src(&norm_w.name),
        &[intermediate],
        heads,
        head_dim,
        vec![-1],
        &norm_w.encoding,
    );
    let encoding = norm_w.encoding.clone();
    let norm_id = norm_w.id;
    b.define(
        b.output_name(&norm_w.name),
        norm_expr,
        encoding,
        Some(vec![local_intermediate]),
    );
    b.consume(norm_id);

    // `out_proj` is `[hidden, heads * head_dim]`: the reduction axis is the
    // columns, so the unit fold goes on axis 1 and the split with it.
    let enc = out_proj.encoding.clone();
    let folded = Expr::src(&out_proj.name)
        .transmute(TensorType::new(vec![hidden, heads, head_dim], enc.clone()));
    let out_expr = b
        .split(folded, 1)
        .transmute(TensorType::new(vec![hidden, -1], enc.clone()));
    let out_proj_id = out_proj.id;
    b.define(
        b.output_name(&out_proj.name),
        out_expr,
        enc,
        Some(vec![hidden, local_intermediate]),
    );
    b.consume(out_proj_id);
    Ok(())
}

/// Publish each layer's experts as one packed slab plus per-expert views
/// into it, which is what the Nemotron-H MoE GEMM addresses.
fn packed_expert_views(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.facts().num_experts == 0 {
        return Ok(());
    }
    for layer in 0..b.facts().num_hidden_layers {
        let base = format!("language_model.backbone.layers.{layer}.mixer.experts");
        if b.find(&format!("{base}.up_proj.packed.weight")).is_some()
            || b.find(&format!("{base}.down_proj.packed.weight")).is_some()
        {
            continue;
        }
        let mut up = Vec::new();
        let mut down = Vec::new();
        let mut complete = true;
        for expert in 0..b.facts().num_experts {
            let tag = format!("{base}.{expert}.");
            let (Some(u), Some(d)) = (
                b.find(&format!("{tag}up_proj.weight")),
                b.find(&format!("{tag}down_proj.weight")),
            ) else {
                complete = false;
                break;
            };
            up.push(u);
            down.push(d);
        }
        if complete {
            layer_packed_experts(b, &base, &up, &down);
        }
    }
    Ok(())
}

fn layer_packed_experts(b: &mut Builder<'_>, base: &str, up: &[&RawTensor], down: &[&RawTensor]) {
    let (Some(first_up), Some(first_down)) = (up.first(), down.first()) else {
        return;
    };
    if first_up.shape.len() != 2
        || first_down.shape.len() != 2
        || !is_raw(&first_up.encoding, DType::BF16)
        || !is_raw(&first_down.encoding, DType::BF16)
    {
        return;
    }
    let full_intermediate = first_up.shape[0];
    let hidden = first_up.shape[1];
    if first_down.shape != [hidden, full_intermediate] {
        return;
    }
    for raw in up {
        if raw.shape != [full_intermediate, hidden] || raw.encoding != first_up.encoding {
            return;
        }
    }
    for raw in down {
        if raw.shape != [hidden, full_intermediate] || raw.encoding != first_down.encoding {
            return;
        }
    }

    let local_intermediate = b.local_extent(full_intermediate);
    let expert_count = up.len() as i64;
    let bf16 = Encoding::Raw(DType::BF16);

    // Each expert contributes its local row band; the pack is their
    // concatenation. The sharding is in the expression, not in a flag.
    let up_name = format!("{base}.up_proj.packed.weight");
    let up_parts: Vec<Expr> = up
        .iter()
        .map(|raw| b.split(Expr::src(&raw.name), 0))
        .collect();
    b.define(
        up_name.clone(),
        Expr::concat(0, up_parts),
        bf16.clone(),
        Some(vec![expert_count * local_intermediate, hidden]),
    );

    let down_parts: Vec<Expr> = down.iter().map(|raw| Expr::src(&raw.name)).collect();
    let (down_expr, down_shape) = b.shard(
        Expr::concat(0, down_parts),
        vec![expert_count * hidden, full_intermediate],
        Some(1),
    );
    let down_name = format!("{base}.down_proj.packed.weight");
    b.define(down_name.clone(), down_expr, bf16.clone(), Some(down_shape));

    for (index, raw) in up.iter().enumerate() {
        let index = index as i64;
        b.define(
            raw.name.clone(),
            Expr::out(&up_name).slice(0, index * local_intermediate, local_intermediate),
            bf16.clone(),
            Some(vec![local_intermediate, hidden]),
        );
        b.consume(raw.id);
    }
    for (index, raw) in down.iter().enumerate() {
        let index = index as i64;
        b.define(
            raw.name.clone(),
            Expr::out(&down_name).slice(0, index * hidden, hidden),
            bf16.clone(),
            Some(vec![hidden, local_intermediate]),
        );
        b.consume(raw.id);
    }
}
