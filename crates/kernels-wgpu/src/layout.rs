use kernels_macros::routine;

use crate::routine::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

#[routine(out(out = like(proj)))]
pub fn ple_combine(
    ctx: &Ctx<'_>,
    proj: In<Tensor<bf16>>,
    token: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv_sqrt2: Const<f32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = proj.width;
    let rows = *rows;
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }

    let lanes = width.unsigned_abs() * rows.unsigned_abs();
    ctx.fire(
        Fire::at("layout/ple_combine.wgsl", "ple_combine_bfloat16").apply([lanes, 1, 1]),
        &[proj.arg(), token.arg(), out.arg(), inv_sqrt2.arg()],
    )
}

fn affine_point(group: i32, bits: i32) -> Result<usize, Refusal> {
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
    Ok(g * 2 + b)
}

#[routine]
pub fn embed_gather_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let rows = 1;
    let lanes = elementwise(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_4bit_bfloat16_gs_32_b_4",
                "embed_gather_4bit_bfloat16_gs_32_b_8",
                "embed_gather_4bit_bfloat16_gs_64_b_4",
                "embed_gather_4bit_bfloat16_gs_64_b_8",
                "embed_gather_4bit_bfloat16_gs_128_b_4",
                "embed_gather_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
    )
}

#[routine(canon = "layout.embed")]
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let rows = *rows;
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
        ],
    )
}

#[routine]
pub fn embed_gather_scaled_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let lanes = elementwise(hidden, 1)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

#[routine]
pub fn embed_gather_scaled_mb_4bit(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    embed_scale: Const<f32>,
    group: Const<i32>,
    bits: Const<i32>,
    token_ids: In<Tensor<i32>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let id = token_ids.ptr;
    let hidden = out.width;
    let rows = *rows;
    let lanes = elementwise_rows(hidden, rows)?;
    ctx.fire(
        Fire::at(
            "layout/embed_gather.wgsl",
            [
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_32_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_64_b_8",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_4",
                "embed_gather_scaled_mb_4bit_bfloat16_gs_128_b_8",
            ][affine_point(*group, *bits)?],
        )
        .apply(lanes),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            id.arg(),
            out.arg(),
            hidden.arg(),
            embed_scale.arg(),
        ],
    )
}

#[routine]
pub fn row_gather(
    ctx: &Ctx<'_>,
    input: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    width: Const<u32>,
    sampling_indices: In<Tensor<u32>>,
    count: Const<u32>,
    row_count: Const<i32>,
) -> Result<(), Refusal> {
    let rows = sampling_indices.ptr;

    let count = *count;
    let row_count = *row_count;
    let lanes = elementwise_rows(input.width, row_count)?;
    ctx.fire(
        Fire::at("layout/row_gather.wgsl", "row_gather_bfloat16").apply(lanes),
        &[input.arg(), out.arg(), rows.arg(), width.arg(), count.arg()],
    )
}
