use kernels_macros::routine;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;

fn router_grid(rows: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([256, rows.unsigned_abs(), 1])
}

fn rows_by_width(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs(), 1])
}

fn routed_qmv_grid(rows: i32, out_vec_size: i32, slots: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "the output vector",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty { what: "slots" });
    }
    let x = rows
        .unsigned_abs()
        .checked_mul(32)
        .ok_or(Refusal::Grid {
            what: "rows * the matvec's lane count",
            at: i64::from(rows) * 32,
        })?;
    Ok([x, out_vec_size.unsigned_abs(), slots.unsigned_abs()])
}

fn routed_qmm_grid(rows: i32, n: i32, tile_m: i32, tile_n: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "the output width",
        });
    }
    let axis = |extent: i32, tile: i32, what: &'static str| -> Result<u32, Refusal> {
        if tile <= 0 {
            return Err(Refusal::Narrow {
                what,
                at: i64::from(tile),
            });
        }
        let tiles = extent.unsigned_abs().div_ceil(tile.unsigned_abs());
        tiles.checked_mul(16).ok_or(Refusal::Grid {
            what: "a tile count times the workgroup",
            at: i64::from(tiles) * 16,
        })
    };
    Ok([
        axis(n, tile_n, "the routed qmm's column tile")?,
        axis(rows, tile_m, "the routed qmm's row tile")?,
        1,
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

fn affine_qmm_point(group: i32, bits: i32, tile_m: i32, tile_n: i32) -> Result<usize, Refusal> {
    let g = match group {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine group size",
                at: i64::from(group),
            });
        }
    };
    let b = match bits {
        4 => 0,
        8 => 1,
        _ => {
            return Err(Refusal::Narrow {
                what: "affine bit width",
                at: i64::from(bits),
            });
        }
    };
    Ok((g * 2 + b) * 9 + tile_point(tile_m, tile_n)?)
}

#[routine]
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
    ctx.fire(
        Fire::at("moe/route.wgsl", "router_topk_bfloat16").apply(router_grid(rows)?),
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
    ctx.fire(
        Fire::at("moe/route.wgsl", "router_topk_scaled_bfloat16").apply(router_grid(rows)?),
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
    ctx.fire(
        Fire::at("moe/route.wgsl", "route_sort").apply([256, 1, 1]),
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
    ctx.fire(
        Fire::at("moe/route.wgsl", "route_gather").apply(rows_by_width(x_width, padded_rows)?),
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

#[routine]
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
    ctx.fire(
        Fire::at("moe/route.wgsl", "combine_sorted").apply(rows_by_width(y_width, tokens)?),
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

#[routine]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = *rows;
    let w = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "shared_expert_combine").apply(rows_by_width(w, rows)?),
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
    let w = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "shared_expert_combine_strided").apply(rows_by_width(w, rows)?),
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
    let bias = ctx.absent()?;
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("moe/qmv_routed.wgsl", "affine_qmv_routed_bfloat16_gs_64_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("moe/qmv_routed.wgsl", "affine_qmv_routed_bias_bfloat16_gs_64_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("moe/qmv_routed.wgsl", "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
        &[
            w.arg(),
            scales.arg(),
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
    #[allow(unused_variables)]
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
    ctx.fire(
        Fire::at(
            "moe/qmm_t_routed.wgsl",
            [
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_64",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_16",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32",
                "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_64",
            ][affine_qmm_point(*group, *bits, *tile_m, *tile_n)?],
        ).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            tile_expert.arg(),
            k.arg(),
            n.arg(),
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
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "moe/qmm_t_routed.wgsl",
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
            ][tile_point(*tile_m, *tile_n)?],
        ).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            tile_expert.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}

#[routine]
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    exponents: Const<Tensor<u8>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "moe/qmm_t_routed.wgsl",
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
            ][tile_point(*tile_m, *tile_n)?],
        ).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
        &[
            w.arg(),
            exponents.arg(),
            x.arg(),
            y.arg(),
            bias.arg(),
            tile_expert.arg(),
            k.arg(),
            n.arg(),
        ],
    )
}
