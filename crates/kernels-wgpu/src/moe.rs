use kernels_macros::routine;

use crate::points::{Payload, at_bf16};
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use kernels::routine::Refusal;

/// The `Moe` family, claimed. ONE of seven points lands, and the six that do
/// not fail for THREE different reasons — which is the useful part of this
/// census, because only one of the three is a missing shader.
///
/// # 1. The router writes bf16 weights where the point declares f32
///
/// `moe.topk_softmax` states `weights: Out<Self::Tensor<f32>>`, and the same
/// for `topk_sigmoid` and `topk_sqrt_softplus`: a routing weight is a
/// probability and every plane's router computes it in float.
/// `moe/route.wgsl` declares `expert_weights: array<atomic<u32>>` and stores
/// through `pie_pack_bf16` — the weights come out BF16, two to a word.
///
/// A claim would therefore hand an f32 rectangle to a shader that writes
/// half as many bytes into it and leaves the top half of the plan's rectangle
/// holding whatever the arena last put there. `moe.weighted_sum` next door
/// then reads that f32 rectangle. Nothing refuses; the model is wrong.
///
/// The mismatch is UNCONDITIONAL, so no body can refuse it selectively and
/// the honest row is the family's default. `router_topk`/`router_topk_scaled`
/// keep serving the legacy `moe.topk_softmax` canon meanwhile, against a
/// legacy plan that sized the rectangle bf16.
///
/// **SEAM (P5):** a `PIE_F32_WEIGHTS` store in `route.wgsl` — which also
/// removes the `atomic` and the compare-exchange, since an f32 weight owns
/// its whole word. `topk_sigmoid` and `topk_sqrt_softplus` need shaders as
/// well (this plane's router has a `softmax_over_all` flag and no sigmoid or
/// sqrt-softplus arm at all), so only `topk_softmax` is one edit away.
///
/// # 2. The routed matmuls are `Bank<R: Repr>`, like everything else here
///
/// `moe.matmul_select` and `moe.matmul_select_bias` declare
/// `bank: Const<Self::Tensor<T>>` — one dense expert stack. This plane has
/// `moe/qmv_routed.wgsl` and `moe/qmm_t_routed.wgsl`, and both take an
/// affine or mxfp4 bank as two or three separate weights. Same seam as
/// `layout.embed` and the whole of `Gemm`; see `quant.rs` for the full
/// statement of it. `moe.matmul_select_bias` is already on baker-todo for
/// exactly this reason, from the cuda side.
///
/// # 3. `moe.weighted_sum` needs an operand the point does not declare
///
/// The point is `(routed, weights) -> y`. `combine_sorted` reads a FOURTH
/// buffer, `inv`, the inverse of the permutation `route_sort` built — this
/// backend's MoE is a sort/gather/grouped-matmul/scatter pipeline, so the
/// combine has to undo the sort as it folds. The permutation is not plane
/// staging a body could hide: it is produced by `route_sort` from
/// `expert_ids` and consumed here, so it is a VALUE in the dataflow that the
/// declaration column would have to carry.
///
/// **SEAM (P5/floor):** either (a) the sorted pipeline becomes tier-2 —
/// inherent methods on this plane with the text gating on `inputs.wgpu()` —
/// or (b) an unsorted `combine` shader is written whose `inv` is the identity
/// and which reads `routes` instead, matching the declaration. (b) costs a
/// shader and keeps the text plane-agnostic, which is what the design wants.
#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    /// `y = routed + sigmoid(gate) * shared`: the shared expert folded back
    /// in beside the routed sum.
    ///
    /// The one `Moe` point whose operands are three dense rectangles and one
    /// dense result, which is why it is the one that lands. The width is the
    /// routed row's and the shader is told it, because its output is an
    /// `atomic<u32>` and it derives word ownership from the absolute offset.
    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<Payload<T>>,
        shared: In<Payload<T>>,
        gate: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("moe.sigmoid_gate_add at an element other than bf16")?;
        let width = routed.width;
        self.fire(
            Fire::at("moe/route.wgsl", "shared_expert_combine")
                .apply(rows_by_width(width, routed.rows)?),
            &[routed.arg(), shared.arg(), gate.arg(), y.arg(), width.arg()],
        )
    }
}

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
    let x = rows.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
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

#[routine(canon = "moe.topk_softmax", out(expert_weights = rows(logits) x const(experts_per_token)))]
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    softmax_over_all: Const<u32>,
    logits_pitch: Const<u32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
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

#[routine(out(expert_weights = rows(logits) x const(experts_per_token)))]
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
    rows: Const<i32>,
) -> Result<(), Refusal> {
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
    x_pitch: Const<u32>,
) -> Result<(), Refusal> {
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
    padded_rows: Const<i32>,
) -> Result<(), Refusal> {
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

#[routine(canon = "moe.weighted_sum", out(out = rows(expert_weights) x const(width)))]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>,
    width: Const<u32>,
    experts_per_token: Const<u32>,
    out_pitch: Const<u32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
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

// INLINED into impl Moe; dies with the routine layer.
#[routine(canon = "moe.sigmoid_gate_add", out(out = like(routed)))]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = *rows;
    ctx.fire(
        Fire::at("moe/route.wgsl", "shared_expert_combine").apply(rows_by_width(width, rows)?),
        &[
            routed.arg(),
            shared.arg(),
            gate.arg(),
            out.arg(),
            width.arg(),
        ],
    )
}

#[routine]
pub fn shared_expert_combine_strided(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let width = routed.width;

    let row_pitch = ctx.param(1)?;
    let rows = *rows;
    ctx.fire(
        Fire::at("moe/route.wgsl", "shared_expert_combine_strided")
            .apply(rows_by_width(width, rows)?),
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
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let bias = ctx.absent()?;
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "moe/qmv_routed.wgsl",
            "affine_qmv_routed_bfloat16_gs_64_b_4",
        )
        .apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "moe/qmv_routed.wgsl",
            "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
        )
        .apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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
    rows: Const<i32>,
) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = *rows;
    ctx.fire(
        Fire::at(
            "moe/qmv_routed.wgsl",
            "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
        )
        .apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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
    #[allow(unused_variables)] pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    group: Const<i32>,
    bits: Const<i32>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
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
        )
        .apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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
    #[allow(unused_variables)] pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
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
        )
        .apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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
    #[allow(unused_variables)] pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>,
    rows: Const<i32>,
) -> Result<(), Refusal> {
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
        )
        .apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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
