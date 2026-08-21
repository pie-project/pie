use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};

fn router_lanes(n_experts: u32) -> Result<u32, Refusal> {
    if n_experts == 0 {
        return Err(Refusal::Empty { what: "n_experts" });
    }
    Ok(n_experts.min(1024).div_ceil(32) * 32)
}

fn route_rows(width: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let w = width.unsigned_abs();
    Ok(([w, rows.unsigned_abs(), 1], [w.min(256), 1, 1]))
}

fn routed_qmv_grid(rows: i32, out_vec_size: i32, slots: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    let x = rows.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "rows * the simdgroup width",
        at: i64::from(rows) * 32,
    })?;
    Ok([
        x,
        out_vec_size.unsigned_abs().div_ceil(4),
        slots.unsigned_abs(),
    ])
}

fn routed_qmv_widths(
    x_slot_stride: i32,
    y_width: i32,
    slots: i32,
) -> Result<(i32, i32), Refusal> {
    if x_slot_stride <= 0 {
        return Err(Refusal::Empty {
            what: "x_slot_stride",
        });
    }
    if y_width <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    if !y_width.unsigned_abs().is_multiple_of(slots.unsigned_abs()) {
        return Err(Refusal::Narrow {
            what: "an output width the slot count does not divide",
            at: i64::from(y_width),
        });
    }
    Ok((x_slot_stride, y_width / slots))
}

fn routed_qmm_grid(rows: i32, n: i32, tile_m: i32, tile_n: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    let (m, bn) = (tile_m.unsigned_abs(), tile_n.unsigned_abs());
    if m == 0 || !rows.unsigned_abs().is_multiple_of(m) {
        return Err(Refusal::Narrow {
            what: "rows the row tile does not divide",
            at: i64::from(rows),
        });
    }
    if bn == 0 || !n.unsigned_abs().is_multiple_of(bn) {
        return Err(Refusal::Narrow {
            what: "an output width the column tile does not divide",
            at: i64::from(n),
        });
    }
    Ok([
        32 * (n.unsigned_abs() / bn),
        2 * (rows.unsigned_abs() / m),
        2,
    ])
}

fn tile_point(tile_m: i32, tile_n: i32) -> Result<usize, Refusal> {
    let axis = |v: i32, what: &'static str| match v {
        16 => Ok(0),
        32 => Ok(1),
        64 => Ok(2),
        _ => Err(Refusal::Narrow {
            what,
            at: i64::from(v),
        }),
    };
    Ok(axis(tile_m, "the routed qmm's row tile")? * 3
        + axis(tile_n, "the routed qmm's column tile")?)
}

#[routine(canon = topk)]
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    softmax_over_all: Const<u32>,
    logits_pitch: Const<u32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let per_expert_scale = ctx.absent()?;
    let rows = *rows;
    let w = router_lanes(*n_experts)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.fire(
        Fire::at("moe/route.metal", "router_topk_bfloat16").apply(Grid::of([w, rows.unsigned_abs(), 1], [w, 1, 1])),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            per_expert_scale,
            n_experts.arg(),
            experts_per_token.arg(),
            softmax_over_all.arg(),
            logits_pitch.arg(),
        ],
    )
}

#[routine]
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    per_expert_scale: Const<Tensor<bf16>>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    softmax_over_all: Const<u32>,
    logits_pitch: Const<u32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let rows = *rows;
    let w = router_lanes(*n_experts)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.fire(
        Fire::at("moe/route.metal", "router_topk_scaled_bfloat16").apply(Grid::of([w, rows.unsigned_abs(), 1], [w, 1, 1])),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            per_expert_scale.arg(),
            n_experts.arg(),
            experts_per_token.arg(),
            softmax_over_all.arg(),
            logits_pitch.arg(),
        ],
    )
}

#[routine]
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    perm: Out<Tensor<i32>>,
    row_expert: Out<Tensor<i32>>,
    tile_expert: Out<Tensor<i32>>,
    inv: Out<Tensor<i32>>,
    n: Const<u32>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    tile_rows: Const<u32>,
    padded: Const<u32>,
    width: Const<u32>,
    x_pitch: Const<u32>) -> Result<(), Refusal> {
    let w = router_lanes(*n_experts)?;
    ctx.fire(
        Fire::at("moe/route.metal", "route_sort").apply(Grid::of([w, 1, 1], [w, 1, 1])),
        &[
            expert_ids.arg(),
            perm.arg(),
            row_expert.arg(),
            tile_expert.arg(),
            inv.arg(),
            n.arg(),
            n_experts.arg(),
            experts_per_token.arg(),
            tile_rows.arg(),
            padded.arg(),
            width.arg(),
            x_pitch.arg(),
        ],
    )
}

#[routine]
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    perm: In<Tensor<i32>>,
    n: Const<u32>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    tile_rows: Const<u32>,
    padded: Const<u32>,
    width: Const<u32>,
    x_pitch: Const<u32>,
    padded_rows: Const<i32>) -> Result<(), Refusal> {
    let x_width = x.width;
    let padded_rows = *padded_rows;
    let (lanes, group) = route_rows(x_width, padded_rows)?;
    ctx.fire(
        Fire::at("moe/route.metal", "route_gather").apply(Grid::of(lanes, group)),
        &[
            x.arg(),
            out.arg(),
            perm.arg(),
            n.arg(),
            n_experts.arg(),
            experts_per_token.arg(),
            tile_rows.arg(),
            padded.arg(),
            width.arg(),
            x_pitch.arg(),
        ],
    )
}

#[routine(canon = weighted_sum)]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>,
    width: Const<u32>,
    experts_per_token: Const<u32>,
    out_pitch: Const<u32>,
    tokens: Const<i32>) -> Result<(), Refusal> {
    let y_width = y.width;
    let tokens = *tokens;
    let (lanes, group) = route_rows(y_width, tokens)?;
    ctx.fire(
        Fire::at("moe/route.metal", "combine_sorted").apply(Grid::of(lanes, group)),
        &[
            y.arg(),
            expert_weights.arg(),
            out.arg(),
            inv.arg(),
            width.arg(),
            experts_per_token.arg(),
            out_pitch.arg(),
        ],
    )
}

#[routine(canon = sigmoid_gate_add)]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = *rows;
    let (lanes, group) = route_rows((width).try_into().unwrap_or(i32::MAX), rows)?;
    ctx.fire(
        Fire::at("moe/route.metal", "shared_expert_combine").apply(Grid::of(lanes, group)),
        &[routed.arg(), shared.arg(), gate.arg(), out.arg(), width.arg()],
    )
}

#[routine]
pub fn shared_expert_combine_strided(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = routed.width;

    let row_pitch = ctx.param(1)?;
    let rows = *rows;
    let (lanes, group) = route_rows((width).try_into().unwrap_or(i32::MAX), rows)?;
    ctx.fire(
        Fire::at("moe/route.metal", "shared_expert_combine_strided").apply(Grid::of(lanes, group)),
        &[
            routed.arg(),
            shared.arg(),
            gate.arg(),
            out.arg(),
            width.arg(),
            row_pitch.arg(),
        ],
    )
}

#[routine]
pub fn qmv_routed(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let bias = ctx.absent()?;
    let rows = *rows;
    ctx.fire(
        Fire::at("quant/qmv.metal", "affine_qmv_routed_bfloat16_gs_64_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            bias,
            expert_ids.arg(),
            x_slot_stride.arg(),
            x_row_stride.arg(),
            slots_per_row.arg(),
        ],
    )
}

#[routine]
pub fn qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let rows = *rows;
    ctx.fire(
        Fire::at("quant/qmv.metal", "affine_qmv_routed_bias_bfloat16_gs_64_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            bias.arg(),
            expert_ids.arg(),
            x_slot_stride.arg(),
            x_row_stride.arg(),
            slots_per_row.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<u8>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let biases = ctx.absent()?;
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let rows = *rows;
    ctx.fire(
        Fire::at("quant/qmv.metal", "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, [32, 2, 1])),
        &[
            w.arg(),
            scales.arg(),
            biases,
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            bias.arg(),
            expert_ids.arg(),
            x_slot_stride.arg(),
            x_row_stride.arg(),
            slots_per_row.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_routed(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    group: Const<i32>,
    bits: Const<i32>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = *rows;
    let point = crate::quant::qmm_point(
        "_routed",
        "PIE_STAMP_qmm_t_routed",
        *group,
        *bits,
        *tile_m,
        *tile_n,
    )?;
    ctx.fire(
        Fire::at("quant/qmm_t.metal", point.entry)
            .stamp(point.stamp)
            .apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            tile_expert.arg(),
        ],
    )
}

#[routine]
pub fn qmm_t_routed_fp16(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = *rows;
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            [
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32",
                "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64",
            ][point],
        )
        .apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, [32, 2, 2])),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            tile_expert.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    exponents: Const<Tensor<u8>>,
    x: In<Tensor<bf16>>,
    pad: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = *rows;
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.fire(
        Fire::at(
            "quant/qmm_t.metal",
            [
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32",
                "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64",
            ][point],
        )
        .apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, [32, 2, 2])),
        &[
            w.arg(),
            exponents.arg(),
            pad.arg(),
            x.arg(),
            y.arg(),
            k.arg(),
            n.arg(),
            bias.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            pad.arg(),
            tile_expert.arg(),
        ],
    )
}
