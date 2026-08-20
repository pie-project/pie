//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `moe/qmm_t_routed.slang` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! `qmv_routed` is compiled for ONE affine format where the dense `qmv_fast`
//! has six, and this paragraph used to say the five missing instantiations
//! had been added. They have not: `moe/qmv_routed.slang` carries three
//! `pie:instantiate` lines and one affine point, and the row beside
//! `qmv_routed` below says why that is the design -- `AffineQ::group_size` is
//! a constant, so a second group point would name an instantiation that
//! dequantises at 64 whatever it claims, and a routed checkpoint at another
//! group is meant to fail BY NAME. The two statements disagreed for as long
//! as both were prose; the shader settles it.

use kernels_macros::routine;
use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, elementwise_rows, keys};
use kernels::routine::Refusal;

/// The workgroup a routed matmul tile walks with, on both axes.
///
/// `qmm_t_routed.slang` states it once as `PIE_TILE_X`/`PIE_TILE_Y` and every
/// walk stride in the body is that constant, so a launch that disagreed would
/// leave the tail of each tile unwritten rather than fail.
const TILE: u32 = 16;

/// Lanes for a routed matvec: one 32-lane row per token, one lane-row per
/// eight output rows, one z per slot.
///
/// The y extent is `out_vec_size` and NOT the output rectangle's width. A
/// routed projection writes a whole token's `k` results end to end, so the
/// buffer is `k` times as wide as one result; giving y the width would launch
/// `k` times the output rows and every copy past the first would run off the
/// end of the expert's weight plane. The rows carry `grid_param = Some(1)`
/// for exactly this reason and this is the same statement in a body.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty token, output or slot count, and
/// [`Refusal::Grid`] when `rows * 32` leaves `u32`.
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

/// Lanes for a routed matmul: one workgroup per `(column tile, row tile)`.
///
/// The tile extents are the SHAPE of the launch and not a hint. `tile_expert`
/// is indexed by `group.y` alone, so the y count is the number of ROW TILES
/// and a launch computed from any other tiling reads a different expert for
/// every tile past the first -- silently, because an expert index is just an
/// offset into the weight plane and every value in range addresses real
/// bytes. `route_sort` fills that table with `p.padded / p.tile_rows` entries,
/// so the two numbers are one number stated twice.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty extent, [`Refusal::Narrow`] for a
/// non-positive tile, and [`Refusal::Grid`] when a tile count leaves `u32`.
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
        tiles.checked_mul(TILE).ok_or(Refusal::Grid {
            what: "a tile count times the workgroup",
            at: i64::from(tiles) * i64::from(TILE),
        })
    };
    Ok([
        axis(n, tile_n, "the routed qmm's column tile")?,
        axis(rows, tile_m, "the routed qmm's row tile")?,
        1,
    ])
}

/// The nine `(BM, BN)` tiles a routed qmm is compiled for, as an index.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a tile the shader tree does not carry.
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

/// The fifty-four affine routed-qmm modules, at `group * 18 + bits * 9 + tile`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry.
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

/// The fifty-four affine routed-qmm modules, in `affine_qmm_point` order.
///
/// Written out rather than built with `format!`. An entrypoint this backend
/// cannot resolve is not an error at the call: `vkCreateComputePipelines`
/// faults on a module that is not there, with the validation layer silent.
/// `tests/routines.rs` sweeps all fifty-four and checks each against the
/// generated entrypoint list, so a typo here is red before it is a fault.
const AFFINE_QMM: &[&str] = &[
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
];

/// The nine pre-cast affine routed-qmm modules, in `tile_point` order.
const FP16_QMM: &[&str] = &[
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64",
];

/// The nine MXFP4 routed-qmm modules, in `tile_point` order.
///
/// The name carries no `gs_`/`b_` segment, unlike both affine tables: the
/// MXFP4 block size is the codec's and not a caller's choice, so there is no
/// axis to spell. Composing these three names from one template would have
/// to know that.
const MXFP4_QMM: &[&str] = &[
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64",
];

/// Which experts a token goes to, and how much of it.
///
/// One workgroup of 1024 per ROW, on y. `route.slang`'s top-k reduces a whole
/// row of logits through groupshared memory and then writes the k picks from
/// lane 0, so a row is a workgroup and not a lane; x is the workgroup itself.
///
/// `per_expert_scale` is declared and not read. The binding sits outside the
/// `PIE_SCALED` guard so the descriptor exists in both modules; slangc deletes
/// it from the unscaled one for being unread, which is why this body may drop
/// the operand rather than bind a slot it never dereferences.
/// [`router_topk_scaled`] is the symbol that reads it.
///
/// FOUR MARKS WHERE `RouterParams` WAS A STORAGE BLOCK. The four are the
/// struct's four fields in the struct's order, which is the statement's order,
/// and `moe/route.slang`'s router arm takes them as the four members of
/// its push block — sixteen bytes, against a guaranteed floor of 128.
///
/// That the signature can NAME them is the point.
/// `driver-metal/tests/packed_params_cover_the_struct.rs` exists because a text
/// stated two words where the shader reads four, and on this plane it would have
/// been quieter still: `robustBufferAccess` is on, so a read past a short block
/// returns ZERO, and a missing `logits_pitch` is not garbage but a plausible
/// number no layer and no assertion would object to. A mark cannot be short —
/// the run is indexed by position and a statement carrying no such word refuses.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row count.
#[routine]
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    // `RouterParams`'s four fields, in its order. `logits_pitch` of zero means
    // the pitch IS `n_experts`, and `softmax_over_all` picks the softmax's
    // DENOMINATOR: zero over the k selected logits, nonzero over every expert.
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    softmax_over_all: Const<u32>,
    logits_pitch: Const<u32>) -> Result<(), Refusal> {
    let _per_expert_scale = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("router_topk_bfloat16", ctx.best()), "router_topk_bfloat16").apply(router_grid(rows)?),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            // THE SCALARS LAST, and the order is the push block's: `words`
            // packs what a body passed in the order it passed it, and
            // `Device::dispatch` refuses a run whose length is not exactly the
            // reflected range — so a member added or dropped is loud, and a
            // member reordered is not.
            n_experts.arg(),
            experts_per_token.arg(),
            softmax_over_all.arg(),
            logits_pitch.arg(),
        ],
    )
}

/// Top-k with a per-expert rescale, indexed by the EXPERT and not by the pick.
///
/// The same four marks as [`router_topk`], in the same order, because it is the
/// same statement shape with one weight added.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row count.
#[routine]
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    per_expert_scale: Const<Tensor<bf16>>,
    // `RouterParams`'s four fields, in its order. A weight `Const` claims the
    // WEIGHT run and a scalar one the params run, so the tensor above takes no
    // slot from these four: `n_experts` is still word 0.
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    softmax_over_all: Const<u32>,
    logits_pitch: Const<u32>) -> Result<(), Refusal> {
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("router_topk_scaled_bfloat16", ctx.best()), "router_topk_scaled_bfloat16").apply(router_grid(rows)?),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            // ONE BINDING LOWER THAN IT WAS: the block sat between the weights
            // and this, and a descriptor set is written densely from the
            // buffers the body passed.
            per_expert_scale.arg(),
            n_experts.arg(),
            experts_per_token.arg(),
            softmax_over_all.arg(),
            logits_pitch.arg(),
        ],
    )
}

/// The router's launch: a whole workgroup per row.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row count.
fn router_grid(rows: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([SORT_LANES, rows.unsigned_abs(), 1])
}

/// The routing sort's workgroup, and therefore every walk stride in it.
///
/// `route.slang` states it as `PIE_SORT_LANES` beside `[numthreads]` because
/// Slang has no `gl_WorkGroupSize` to read it back from.
const SORT_LANES: u32 = 1024;

/// Group the rows by expert: the permutation, the per-row expert, the
/// per-tile expert, and the inverse the combine reads back.
///
/// ONE workgroup, whatever the row count -- which is why this body takes no
/// extent at all. The sort reduces across every `(row, slot)` pair through
/// workgroup-scoped atomics and stripes them over its own 1024 lanes. A
/// launch of one workgroup per row would have each copy clearing and
/// rewriting the permutation the others are reading, and `InterlockedAdd` is
/// scoped to the workgroup so nothing would even serialise it.
///
/// SEVEN MARKS WHERE `MoeRouteParams` WAS A STORAGE BLOCK, and [`route_gather`]
/// takes the same seven in the same order. That sharing is the struct's own
/// point carried forward: `model-dsl` states one seven-word run for both
/// statements, so the padding this kernel writes and the bounds the gather reads
/// cannot disagree. Twenty-eight bytes of push, the widest in `moe/route.slang`.
///
/// # Errors
///
/// Only what the encoder refuses; the grid is a constant.
#[routine]
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    perm: Out<Tensor<i32>>,
    row_expert: Out<Tensor<i32>>,
    tile_expert: Out<Tensor<i32>>,
    inv: Out<Tensor<i32>>,
    // `MoeRouteParams`'s seven fields, in its order. `n` is the number of
    // (row, slot) PAIRS and `padded` is the permutation's length, `n` rounded up
    // so every expert's span is a whole number of `tile_rows` tiles; the two are
    // different numbers and this kernel reads both, so swapping the marks would
    // clear a permutation shorter than it fills.
    n: Const<u32>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    tile_rows: Const<u32>,
    padded: Const<u32>,
    width: Const<u32>,
    x_pitch: Const<u32>) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(crate::routine::module_path("route_sort", ctx.best()), "route_sort").apply([SORT_LANES, 1, 1]),
        &[
            expert_ids.arg(),
            perm.arg(),
            row_expert.arg(),
            tile_expert.arg(),
            // ONE BINDING LOWER THAN IT WAS: the block sat between
            // `tile_expert` and this.
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

/// Gather each sorted slot's token row into the expert-major rectangle.
///
/// The row extent is `padded` and not the token count: the sort rounds the
/// row count up to a whole number of tiles and marks the slack with
/// `perm = -1`, which this kernel writes as a zero row. Launching over the
/// tokens instead would leave the slack holding whatever the arena held, and
/// the routed matmul would then multiply it by real weights.
///
/// [`route_sort`]'s SEVEN MARKS, all of them, and this kernel reads four. `n`,
/// `n_experts` and `tile_rows` are the sort's alone and are carried here anyway
/// — one `MoeRouteParams` layout serves both statements, so `padded` is stated
/// once and read by the kernel that pads and the kernel that is bounded by the
/// padding. An unread member of a push block survives `-O2`
/// (`quant/qmm_t.slang` has carried two since its cast entrypoints were split),
/// so the reflected range is the full twenty-eight bytes and all seven are
/// passed.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    perm: In<Tensor<i32>>,
    // `MoeRouteParams`'s seven fields, in its order — [`route_sort`]'s exactly.
    n: Const<u32>,
    n_experts: Const<u32>,
    experts_per_token: Const<u32>,
    tile_rows: Const<u32>,
    padded: Const<u32>,
    width: Const<u32>,
    x_pitch: Const<u32>) -> Result<(), Refusal> {
    // THE OPERAND'S OWN RECTANGLE, and not the `width` mark beside it. The two
    // are the same number for every text this tree writes; they are not the same
    // FACT. `x.width` is what the arena allocated and is what the grid must
    // cover, while the mark is what the statement said and is what the shader
    // strides by — a disagreement is a fact about the plan, not about this fire.
    let x_width = x.width;
    let padded_rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("route_gather", ctx.best()), "route_gather").apply(elementwise_rows(x_width, padded_rows)?),
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

/// Sum a token's `k` expert results back into one row, weighted.
///
/// The row extent is the TOKEN count here and not `padded`: the inverse map
/// is indexed by token, and a row past the tokens would read `inv` off the
/// end of an allocation that is only as long as the tokens reach.
///
/// THREE MARKS WHERE `ExpertCombineParams` WAS A STORAGE BLOCK, in its order —
/// twelve bytes of push. `out_pitch` of zero means `width`: the mixture's output
/// lands in whatever layout the caller's activations are in, packed for a
/// batched decode and a uniform scratch stride apart for a prefill, and a host
/// with nothing to say writes 0 rather than restating the width.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>,
    // `ExpertCombineParams`'s three fields, in its order.
    width: Const<u32>,
    experts_per_token: Const<u32>,
    out_pitch: Const<u32>) -> Result<(), Refusal> {
    // The OPERAND's rectangle, which is what the grid covers — see
    // [`route_gather`] for why that is not the `width` mark beside it.
    let y_width = y.width;
    let tokens = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("combine_sorted", ctx.best()), "combine_sorted").apply(elementwise_rows(y_width, tokens)?),
        &[
            y.arg(),
            expert_weights.arg(),
            out.arg(),
            // ONE BINDING LOWER THAN IT WAS: the block sat between `out` and
            // this.
            inv.arg(),
            width.arg(),
            experts_per_token.arg(),
            out_pitch.arg(),
        ],
    )
}

/// Blend the shared expert into the routed result, gated per ROW.
///
/// `width` is both the pushed extent and the x axis, and the shader guards on
/// the pushed copy -- so the two are one number and this body passes it once
/// and launches on it.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("shared_expert_combine", ctx.best()), "shared_expert_combine").apply(combine_grid(width.unsigned_abs(), rows)?),
        &[routed.arg(), shared.arg(), gate.arg(), out.arg(), width.arg()],
    )
}

/// The same blend over rows that are not contiguous.
///
/// The x extent is still `width` and not `row_pitch`. The pitch is how far
/// apart the rows are; the width is how much of each row this kernel owns,
/// and the shader returns on `c >= pc.width`. Launching on the pitch would
/// dispatch lanes that do nothing, which is merely wasteful -- but the gate
/// index is what makes the two variants different rather than a copy: here
/// the gate really does stride by the pitch, because `qmv_out_size` answers 1
/// for the shared gate projection and its output is written a full pitch
/// apart like every other projection's.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
#[routine]
pub fn shared_expert_combine_strided(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = routed.width;
    // OUT OF THE STATEMENT'S OWN RUN, at the word HEAD's `Param<1>` named.
    // This body forwards `ctx.params()` as a STRUCT, so the run is the
    // shader's layout and no `Const` mark can name a word inside it; the
    // migration made this `keys::RowPitch`, which no driver answers.
    let row_pitch = ctx.param(1)?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("shared_expert_combine_strided", ctx.best()), "shared_expert_combine_strided").apply(combine_grid(width.unsigned_abs(), rows)?),
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

/// The shared-expert blend's rectangle, with the width arriving unsigned.
///
/// # Errors
///
/// [`Refusal::Grid`] for a width past `i32`, and whatever
/// [`crate::routine::elementwise_rows`] refuses.
fn combine_grid(width: u32, rows: i32) -> Result<[u32; 3], Refusal> {
    let width = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    elementwise_rows(width, rows)
}

/// The MoE expert matvec: one token, the expert its slot names.
///
/// `biases` is the affine codec's zero-point plane; `bias` is the output bias
/// this variant does not add. Two different planes with names one letter
/// apart, bound five bindings apart, and the shader declares both in every
/// affine module -- so a body that swapped them would dequantise against the
/// output bias and still run.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
#[routine]
pub fn qmv_routed(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    // THE SORTED STACK'S THREE STRIDES, WHICH THE STATEMENT CARRIES. All
    // three were `Param<N, i32>` and the migration read them as a fire's
    // facts, so the body asked for keys no driver answers and every routed
    // matvec refused `Unstated`. They are the mixture's own geometry, which
    // `dsl::metal::routed_qmv` computes and states: a row is `k` slots wide
    // and a slot is one, so `x_slot_stride` is the input's width, the row
    // stride is `k` of them, and `slots_per_row` is `k`.
    //
    // `in_vec_size` and `out_vec_size` stood before them and correctly left:
    // they are `x.width` and `y.width`, which the marks carry.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let _bias = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("affine_qmv_routed_bfloat16_gs_64_b_4", ctx.best()), "affine_qmv_routed_bfloat16_gs_64_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            x.arg(),
            y.arg(),
            in_vec_size.arg(),
            out_vec_size.arg(),
            expert_ids.arg(),
            x_slot_stride.arg(),
            x_row_stride.arg(),
            slots_per_row.arg(),
        ],
    )
}

/// The same matvec with the output bias added.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
#[routine]
pub fn qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    // THE SORTED STACK'S THREE STRIDES, WHICH THE STATEMENT CARRIES. All
    // three were `Param<N, i32>` and the migration read them as a fire's
    // facts, so the body asked for keys no driver answers and every routed
    // matvec refused `Unstated`. They are the mixture's own geometry, which
    // `dsl::metal::routed_qmv` computes and states: a row is `k` slots wide
    // and a slot is one, so `x_slot_stride` is the input's width, the row
    // stride is `k` of them, and `slots_per_row` is `k`.
    //
    // `in_vec_size` and `out_vec_size` stood before them and correctly left:
    // they are `x.width` and `y.width`, which the marks carry.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("affine_qmv_routed_bias_bfloat16_gs_64_b_4", ctx.best()), "affine_qmv_routed_bias_bfloat16_gs_64_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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

/// The MXFP4 routed matvec, biased.
///
/// It takes the affine signature unchanged, `biases` included, and the module
/// it names declares bindings 0, 1, 3, 4, 5 and 6 -- a HOLE at 2, exactly
/// where the zero-point plane the MXFP4 codec has no use for would sit. That
/// is not a mismatch to fix by dropping the argument. The descriptor set
/// layout this driver builds is DENSE over `0..max(binding) + 1`, so the hole
/// is a slot that exists and nothing reads, and passing the plane keeps every
/// later buffer at the index its own module declares. Dropping it would slide
/// `x`, `y`, `bias` and `expert_ids` down one and the matvec would read its
/// activations out of the scale plane.
///
/// `model-compiler` picks between this and [`qmv_routed_bias`] on the weight
/// repr alone and makes ONE call for both arms, so the signatures are one
/// signature and this is why.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
#[routine]
pub fn mxfp4_qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<u8>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    // THE SORTED STACK'S THREE STRIDES, WHICH THE STATEMENT CARRIES. All
    // three were `Param<N, i32>` and the migration read them as a fire's
    // facts, so the body asked for keys no driver answers and every routed
    // matvec refused `Unstated`. They are the mixture's own geometry, which
    // `dsl::metal::routed_qmv` computes and states: a row is `k` slots wide
    // and a slot is one, so `x_slot_stride` is the input's width, the row
    // stride is `k` of them, and `slots_per_row` is `k`.
    //
    // `in_vec_size` and `out_vec_size` stood before them and correctly left:
    // they are `x.width` and `y.width`, which the marks carry.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let _biases = ctx.absent()?;
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path("mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", ctx.best()), "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4").apply(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?),
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

/// The routed matmul: a tile of tokens against the expert its tile names.
///
/// `tile_expert` is read at `group.y`, so the tiling is not a tuning knob the
/// launch may round: the y count IS the number of entries `route_sort` wrote,
/// and `tile_m` has to be the `tile_rows` that sort was given.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a point the shader tree does not carry, and
/// whatever [`routed_qmm_grid`] refuses.
#[routine]
pub fn qmm_t_routed(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    // THE STATEMENT'S SECOND INPUT, WHICH THIS GEMM DOES NOT READ.
    // `dsl::metal::routed_qmm` places `[rows, row_expert, tile_expert]`, and
    // the matvec's `row_expert` rides slot 1 so the operand list is the same
    // length either way. The mark is here because the SLOT is a position now:
    // without it `tile_expert` would bind input 1 and read row ids as tile ids.
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    group: Const<i32>,
    bits: Const<i32>,
    tile_m: Const<i32>,
    tile_n: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?], ctx.best()), AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

/// The routed matmul that pre-casts its dequantised weight to `half`.
///
/// One affine point, `gs_64_b_4`, because the pre-cast arm was built for the
/// one checkpoint that wanted it.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a tile the shader tree does not carry, and
/// whatever [`routed_qmm_grid`] refuses.
#[routine]
pub fn qmm_t_routed_fp16(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<bf16>>,
    biases: Const<Tensor<bf16>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    // THE STATEMENT'S SECOND INPUT, WHICH THIS GEMM DOES NOT READ.
    // `dsl::metal::routed_qmm` places `[rows, row_expert, tile_expert]`, and
    // the matvec's `row_expert` rides slot 1 so the operand list is the same
    // length either way. The mark is here because the SLOT is a position now:
    // without it `tile_expert` would bind input 1 and read row ids as tile ids.
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(FP16_QMM[tile_point(*tile_m, *tile_n)?], ctx.best()), FP16_QMM[tile_point(*tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

/// The MXFP4 routed matmul, biased.
///
/// The binding order is NOT the affine one with a hole in it, which is the
/// difference between this family's two codecs and its matvec's. Here the
/// `PIE_MXFP4` arm renumbers: `exponents` at 1, `x` at 2, `y` at 3, `bias` at
/// 4, `tile_expert` at 5 -- six dense bindings against affine's six, with the
/// output bias occupying the slot affine spends on a zero-point plane. So
/// this signature is six buffers and not seven, and taking the matvec's
/// twelve-argument shape here would bind the bias as the activations.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a tile the shader tree does not carry, and
/// whatever [`routed_qmm_grid`] refuses.
#[routine]
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    exponents: Const<Tensor<u8>>,
    x: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    // THE STATEMENT'S SECOND INPUT, WHICH THIS GEMM DOES NOT READ.
    // `dsl::metal::routed_qmm` places `[rows, row_expert, tile_expert]`, and
    // the matvec's `row_expert` rides slot 1 so the operand list is the same
    // length either way. The mark is here because the SLOT is a position now:
    // without it `tile_expert` would bind input 1 and read row ids as tile ids.
    #[allow(unused_variables)]
    pad: In<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(crate::routine::module_path(MXFP4_QMM[tile_point(*tile_m, *tile_n)?], ctx.best()), MXFP4_QMM[tile_point(*tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    type Call = (String, [u32; 3], Vec<ArgValue>);

    /// An `Encode` that remembers what fired, and answers the facts this
    /// family's bodies ask for.
    ///
    /// `rows` backs every `ctx.ask::<i32, keys::Rows>()` in this file, but the
    /// number means different things in different calls: the router wants token
    /// rows, `route_gather` wants the padded row count after sorting, routed
    /// matvecs want token rows while taking their slot count separately, and
    /// routed matmuls want the row-tile extent they launch over. So the probe
    /// keeps one representative nonzero default and the tests that care about a
    /// specific count set it first.
    ///
    /// `row_pitch` is only the shared-expert strided blend's own ask.
    /// `x_slot_stride`, `x_row_stride` and `slots_per_row` are the three facts
    /// both routed matvec bodies ask for. The two `u32` handles are the named
    /// buffer facts this file resolves by source: `ctx.params()` where a test
    /// inspects the exact bound list, and `ctx.absent()` where the generic
    /// buffer catch-all would hide which slot the body deliberately bound as
    /// absent.
    ///
    /// NOTHING IN THIS FILE FORWARDS A PARAMS BLOCK ANY MORE. The four routing
    /// bodies took one -- `RouterParams`, `MoeRouteParams` twice and
    /// `ExpertCombineParams`, each a struct `route.slang` read by field -- and
    /// their fields are `Const<u32>` marks now, so the tests below hand the
    /// numbers over themselves and what a test states is what the shader reads.
    /// The `Params` arm stays in `resolve` for whatever migrates next.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        rows: Cell<i32>,
        row_pitch: Cell<i32>,
        x_slot_stride: Cell<i32>,
        x_row_stride: Cell<i32>,
        slots_per_row: Cell<i32>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
        absent_handle: Cell<u32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                rows: Cell::new(3),
                row_pitch: Cell::new(4096),
                x_slot_stride: Cell::new(1),
                x_row_stride: Cell::new(1),
                slots_per_row: Cell::new(4),
                params_handle: Cell::new(900),
                words: RefCell::default(),
                absent_handle: Cell::new(901),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(
            &self,
            ty: kernels::Ty,
            source: kernels::Source,
        ) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            if source == <keys::XSlotStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.x_slot_stride.get()));
            }
            if source == <keys::XRowStride as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.x_row_stride.get()));
            }
            if source == <keys::SlotsPerRow as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.slots_per_row.get()));
            }
            // THE STATEMENT'S OWN SCALARS, which a body reads by index when its
            // params run is a struct and no `Const` mark can name a word inside
            // it -- see `Asks::param`. The probe answers a number that is
            // plausible for every reader: a stride wide enough for the rows
            // these tests build, and a positive tiling.
            if let kernels::Source::Slot(kernels::Kind::Param, n) = source {
                return Ok(ArgValue::I32(
                    self.words.borrow().get(usize::from(n)).copied().unwrap_or(4096),
                ));
            }
            if source == kernels::Source::Slot(kernels::Kind::Params, 0) {
                return Ok(ArgValue::Buffer {
                    handle: self.params_handle.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if source == kernels::Source::Lit(kernels::Lit::Null) {
                return Ok(ArgValue::Buffer {
                    handle: self.absent_handle.get(),
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            if matches!(ty, kernels::Ty::Buf) {
                return Ok(ArgValue::Buffer {
                    handle: 900,
                    writes: false,
                    rows: 0,
                    width: 0,
                });
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while the
            // same fact went unanswered on a real driver.
            Err(Refusal::Unstated {
                what: "a fact this probe does not answer",
            })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 1, "expected exactly one dispatch");
        calls[0].clone()
    }

    /// The sort is ONE workgroup whatever the routing's size, and the router
    /// is one per ROW.
    ///
    /// These two launches read alike -- both are 1024 lanes on x -- and they
    /// are different kinds of number. `route_sort` reduces across every
    /// `(row, slot)` pair through groupshared counters, so a second workgroup
    /// would not extend the sort, it would run a second whole sort over the
    /// same output: each copy clears `perm` to -1 and refills it while the
    /// other is reading it, and `InterlockedAdd` on groupshared memory does
    /// not serialise across workgroups. The router is the opposite: one
    /// workgroup per row, and giving it one workgroup total would leave every
    /// row but the first unrouted -- `expert_ids` keeps whatever it held,
    /// which `route_sort` then reads as a real expert.
    #[test]
    fn the_sort_is_one_workgroup_and_the_router_is_one_per_row() {
        let seen = Seen::default();
        route_sort(
            &seen,
            In::new(Tensor::<i32>::new(0)),
            Out::new(Tensor::<i32>::new(1)),
            Out::new(Tensor::<i32>::new(2)),
            Out::new(Tensor::<i32>::new(3)),
            Out::new(Tensor::<i32>::new(5)),
            // `MoeRouteParams`: n, n_experts, experts_per_token, tile_rows,
            // padded, width, x_pitch -- seven words, twenty-eight bytes of
            // push, stated here rather than staged into a storage block.
            Const::new(32),
            Const::new(60),
            Const::new(4),
            Const::new(16),
            Const::new(64),
            Const::new(2048),
            Const::new(2048),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [1024, 1, 1]);

        let seen = Seen::default();
        seen.rows.set(7);
        router_topk(
            &seen,
            In::new(Tensor::<bf16>::new(0)),
            Out::new(Tensor::<i32>::new(1)),
            Out::new(Tensor::<bf16>::new(2)),
            // `RouterParams`: n_experts, experts_per_token, softmax_over_all,
            // logits_pitch. The grid is 1024 lanes whatever the count -- unlike
            // metal's, whose threadgroup IS the expert count -- so these four
            // reach the shader and not the launch.
            Const::new(60),
            Const::new(4),
            Const::new(0),
            Const::new(0),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [1024, 7, 1]);
    }

    /// The gather covers the PADDED rows and the combine covers the tokens.
    ///
    /// Two row counts one step apart in the same pipeline, and the sort makes
    /// them differ: it rounds the routed rows up to a whole number of tiles
    /// and marks the slack with `perm = -1`, which the gather writes as zeros.
    /// A gather launched over the tokens leaves that slack holding whatever
    /// the arena held and the routed matmul multiplies it by real weights; a
    /// combine launched over `padded` reads `inv` past the end of an
    /// allocation only as long as the tokens reach.
    #[test]
    fn the_gather_runs_over_padded_rows_and_the_combine_over_tokens() {
        let seen = Seen::default();
        seen.rows.set(96);
        route_gather(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 96, width: 64 },
            Out { ptr: Tensor::<bf16>::new(1), rows: 96, width: 64 },
            In::new(Tensor::<i32>::new(2)),
            // The SORT'S seven words, which this statement carries whole --
            // three of them for the sort's benefit and not this kernel's.
            Const::new(24),
            Const::new(60),
            Const::new(4),
            Const::new(16),
            Const::new(96),
            Const::new(64),
            Const::new(64),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [64, 96, 1]);

        let seen = Seen::default();
        seen.rows.set(24);
        combine_sorted(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 24, width: 64 },
            In::new(Tensor::<bf16>::new(1)),
            Out { ptr: Tensor::<bf16>::new(2), rows: 24, width: 64 },
            In::new(Tensor::<i32>::new(4)),
            // `ExpertCombineParams`: width, experts_per_token, out_pitch.
            Const::new(64),
            Const::new(4),
            Const::new(0),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [64, 24, 1]);
    }

    /// A routed matvec's y axis is `out_vec_size` and its z is the slot count.
    ///
    /// The output buffer is `k` times as wide as one result -- a routed
    /// projection writes a whole token's picks end to end -- so the y extent
    /// is NOT the rectangle's width. Launching on the width would run `k`
    /// times the output rows and every copy past the first would read off the
    /// end of the expert's weight plane. x is `rows * 32` because the
    /// reduction is a warp.
    #[test]
    fn the_routed_matvec_spreads_output_rows_on_y_and_slots_on_z() {
        let seen = Seen::default();
        seen.rows.set(3);
        seen.x_slot_stride.set(1);
        seen.x_row_stride.set(1);
        seen.slots_per_row.set(4);
        qmv_routed(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In { ptr: Tensor::<bf16>::new(3), rows: 3, width: 512 },
            Out { ptr: Tensor::<bf16>::new(4), rows: 3, width: 256 },
            // The stack's three strides, which the STATEMENT carries: the
            // fixture set them on `seen` while they were facts, and states
            // them here now. They precede `expert_ids`, as the signature has
            // them.
            Const::new(1),
            Const::new(1),
            Const::new(4),
            In::new(Tensor::<i32>::new(8)),
        )
        .unwrap();
        let (entrypoint, lanes, args) = one(&seen);
        assert_eq!(entrypoint, "affine_qmv_routed_bfloat16_gs_64_b_4");
        assert_eq!(lanes, [96, 256, 4]);
        assert_eq!(args.len(), 11, "eleven bindings and pushed words");
    }

    /// The MXFP4 matvec drops the plane its codec does not read.
    ///
    /// Its module declares bindings 0, 1, 3, 4, 5, 6 -- nothing at 2, where
    /// the affine zero-point plane sits, because `qmv_routed.slang` compiles
    /// that global out under `PIE_MXFP4` and slangc decorates no binding for
    /// a global it did not compile.
    ///
    /// The descriptor set LAYOUT keeps the hole: `Device::build` is dense
    /// over `0..declared.bindings`, so `x` still lands at descriptor 3. The
    /// CALL does not: `Device::slots` skips every unused index while writing
    /// descriptors and `encode::dispatch` refuses any list whose length is
    /// not `declared.bindings - holes()`. Passing the plane to hold the
    /// numbering is `Refusal::Arity`, not a spare descriptor.
    ///
    /// The SUBJECT changed with the migration: the old test claimed the
    /// signature still accepted the absent plane positionally and the body then
    /// declined to forward it; the current signature is already the forwarded
    /// binding list, because the absent plane moved to `ctx.absent()` inside
    /// the body. This now checks the fact that replaced that positional slot:
    /// the body really does bind a deliberate absence (`ctx.absent()`, given
    /// its own handle here so a regression that forwarded it would show up
    /// in `bound`), and the fired buffer list still skips it.
    #[test]
    fn the_mxfp4_matvec_drops_the_slot_its_codec_does_not_read() {
        let seen = Seen::default();
        seen.rows.set(1);
        seen.x_slot_stride.set(1);
        seen.x_row_stride.set(1);
        seen.slots_per_row.set(1);
        seen.absent_handle.set(777);
        mxfp4_qmv_routed_bias(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<u8>::new(1)),
            In { ptr: Tensor::<bf16>::new(3), rows: 1, width: 512 },
            Out { ptr: Tensor::<bf16>::new(4), rows: 1, width: 256 },
            Const::new(Tensor::<bf16>::new(7)),
            // The stack's three strides, which the STATEMENT carries: the
            // fixture set them on `seen` while they were facts, and states
            // them here now. They precede `expert_ids`, as the signature has
            // them.
            Const::new(1),
            Const::new(1),
            Const::new(4),
            In::new(Tensor::<i32>::new(8)),
        )
        .unwrap();
        let (entrypoint, _, args) = one(&seen);
        assert_eq!(entrypoint, "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4");
        let bound: Vec<u32> = args
            .iter()
            .filter_map(|a| match a {
                ArgValue::Buffer { handle, .. } => Some(*handle),
                _ => None,
            })
            .collect();
        assert_eq!(
            bound,
            vec![0, 1, 3, 4, 7, 8],
            "six buffers, one per binding the module actually decorates"
        );
    }

    /// The MXFP4 matmul does NOT keep it, because its arm renumbers.
    ///
    /// This is the one place the family's two codecs differ in kind rather
    /// than in degree: `qmv_routed.slang` leaves a hole where the zero-point
    /// plane was, and `qmm_t_routed.slang` closes it -- `exponents` at 1, `x`
    /// at 2, `y` at 3, `bias` at 4, `tile_expert` at 5. So the matvec takes
    /// twelve arguments in both codecs and the matmul takes six buffers in
    /// both, but they are not the same six.
    #[test]
    fn the_mxfp4_matmul_renumbers_where_the_matvec_leaves_a_hole() {
        let seen = Seen::default();
        seen.rows.set(32);
        mxfp4_qmm_t_routed_bias(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<u8>::new(1)),
            In { ptr: Tensor::<bf16>::new(2), rows: 32, width: 128 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 32, width: 64 },
            Const::new(Tensor::<bf16>::new(4)),
            In::new(Tensor::<bf16>::new(5)),
            In::new(Tensor::<i32>::new(6)),
            Const::new(32),
            Const::new(16),
        )
        .unwrap();
        let (entrypoint, lanes, args) = one(&seen);
        assert_eq!(entrypoint, "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16");
        assert_eq!(lanes, [64, 16, 1]);
        assert_eq!(args.len(), 8, "six buffers and two pushed words");
        let bound: Vec<u32> = args
            .iter()
            .filter_map(|a| match a {
                ArgValue::Buffer { handle, .. } => Some(*handle),
                _ => None,
            })
            .collect();
        assert_eq!(
            bound,
            vec![0, 1, 2, 3, 4, 6],
            "six DENSE bindings: the activations are third and the output bias is fifth, which is the matvec's order with the hole closed rather than the matvec's order"
        );
    }

    /// The routed matmul's tiling picks the module AND the grid, and the two
    /// may not be told apart.
    ///
    /// `tile_expert` is indexed by `group.y` alone, so the y count is the
    /// number of entries `route_sort` wrote and a launch computed from a
    /// different tiling reads a different expert for every tile past the
    /// first. Silently: an expert index is an offset into the weight plane
    /// and every value in range addresses real bytes.
    #[test]
    fn the_routed_matmuls_tiling_is_both_its_module_and_its_grid() {
        let seen = Seen::default();
        seen.rows.set(65);
        qmm_t_routed(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In { ptr: Tensor::<bf16>::new(3), rows: 65, width: 256 },
            Out { ptr: Tensor::<bf16>::new(4), rows: 65, width: 192 },
            In::new(Tensor::<bf16>::new(5)),
            In::new(Tensor::<i32>::new(6)),
            Const::new(128),
            Const::new(8),
            Const::new(64),
            Const::new(32),
        )
        .unwrap();
        let (entrypoint, lanes, _) = one(&seen);
        assert_eq!(
            entrypoint,
            "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32"
        );
        assert_eq!(lanes, [96, 32, 1]);

        let seen = Seen::default();
        seen.rows.set(65);
        let refused = qmm_t_routed(
            &seen,
            Const::new(Tensor::<u32>::new(0)),
            Const::new(Tensor::<bf16>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            In { ptr: Tensor::<bf16>::new(3), rows: 65, width: 256 },
            Out { ptr: Tensor::<bf16>::new(4), rows: 65, width: 192 },
            In::new(Tensor::<bf16>::new(5)),
            In::new(Tensor::<i32>::new(6)),
            Const::new(128),
            Const::new(8),
            Const::new(48),
            Const::new(32),
        );
        assert!(matches!(refused, Err(Refusal::Narrow { .. })));
    }

    /// A shared-expert blend launches on the WIDTH, not on the pitch.
    ///
    /// The strided form is not a copy of the plain one with a pitch added:
    /// the gate index differs. Here the gate strides by the pitch because
    /// `qmv_out_size` answers 1 for the shared gate projection, so its output
    /// is written a full pitch apart like every other projection's. The grid
    /// is the same rectangle in both, and it is the width because the shader
    /// returns on `c >= pc.width`.
    #[test]
    fn the_shared_expert_blend_launches_on_the_width_in_both_forms() {
        let seen = Seen::default();
        seen.rows.set(9);
        shared_expert_combine(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 9, width: 512 },
            In { ptr: Tensor::<bf16>::new(1), rows: 9, width: 512 },
            In { ptr: Tensor::<bf16>::new(2), rows: 9, width: 512 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 9, width: 512 },
        )
        .unwrap();
        assert_eq!(one(&seen).1, [512, 9, 1]);

        let seen = Seen::default();
        seen.rows.set(9);
        seen.row_pitch.set(4096);
        shared_expert_combine_strided(
            &seen,
            In { ptr: Tensor::<bf16>::new(0), rows: 9, width: 512 },
            In { ptr: Tensor::<bf16>::new(1), rows: 9, width: 512 },
            In { ptr: Tensor::<bf16>::new(2), rows: 9, width: 512 },
            Out { ptr: Tensor::<bf16>::new(3), rows: 9, width: 512 },
        )
        .unwrap();
        let (_, lanes, args) = one(&seen);
        assert_eq!(lanes, [512, 9, 1], "the width, and not the 4096 pitch");
        assert_eq!(args.len(), 6, "four buffers, the width and the pitch");
    }
}
