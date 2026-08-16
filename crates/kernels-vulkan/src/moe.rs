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

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{keys, Ask, Bind, Block, Buf, BufMut, Ctx, Env, Fire, Null, Param, ParamOr, Routine};
use crate::routine::{InSlot, OutSlot, Weight};

/// The entrypoints this family's crossed routines spell, now that their
/// rows are gone. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[&str] = &[
    "combine_sorted",
    "route_gather",
    "route_sort",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32",
    "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64",
    "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
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
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmv_routed_bfloat16_gs_64_b_4",
    "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
    "router_topk_bfloat16",
    "router_topk_scaled_bfloat16",
    "shared_expert_combine",
    "shared_expert_combine_strided",
];

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
/// `per_expert_scale` is bound and not read. The binding is declared outside
/// the `PIE_SCALED` guard, so the descriptor exists in both modules and a set
/// one entry short is a slot holding whatever it last held rather than an
/// error. [`router_topk_scaled`] is the symbol that reads it.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row count.
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: InSlot<0, Buf>,
    expert_ids: OutSlot<0, BufMut>,
    expert_weights: OutSlot<1, BufMut>,
    params: Block<Buf>,
    _per_expert_scale: Null<Buf>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "router_topk_bfloat16",
            lanes: router_grid(*rows)?,
        },
        &[logits.v(), expert_ids.v(), expert_weights.v(), params.v()],
    )
}

/// Top-k with a per-expert rescale, indexed by the EXPERT and not by the pick.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row count.
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: InSlot<0, Buf>,
    expert_ids: OutSlot<0, BufMut>,
    expert_weights: OutSlot<1, BufMut>,
    params: Block<Buf>,
    per_expert_scale: Weight<0, Buf>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "router_topk_scaled_bfloat16",
            lanes: router_grid(*rows)?,
        },
        &[
            logits.v(),
            expert_ids.v(),
            expert_weights.v(),
            params.v(),
            per_expert_scale.v(),
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
/// # Errors
///
/// Only what the encoder refuses; the grid is a constant.
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: InSlot<0, Buf>,
    perm: OutSlot<0, BufMut>,
    row_expert: OutSlot<1, BufMut>,
    tile_expert: OutSlot<2, BufMut>,
    params: Block<Buf>,
    inv: OutSlot<3, BufMut>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "route_sort",
            lanes: [SORT_LANES, 1, 1],
        },
        &[
            expert_ids.v(),
            perm.v(),
            row_expert.v(),
            tile_expert.v(),
            params.v(),
            inv.v(),
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
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: InSlot<0, Buf>,
    out: OutSlot<0, BufMut>,
    perm: InSlot<1, Buf>,
    params: Block<Buf>,
    width: Ask<keys::Width, i32>,
    padded: ParamOr<4, keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "route_gather",
            lanes: crate::routine::elementwise_rows(*width, *padded)?,
        },
        &[x.v(), out.v(), perm.v(), params.v()],
    )
}

/// Sum a token's `k` expert results back into one row, weighted.
///
/// The row extent is the TOKEN count here and not `padded`: the inverse map
/// is indexed by token, and a row past the tokens would read `inv` off the
/// end of an allocation that is only as long as the tokens reach.
///
/// # Errors
///
/// See [`crate::routine::elementwise_rows`].
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: InSlot<0, Buf>,
    expert_weights: InSlot<1, Buf>,
    out: OutSlot<0, BufMut>,
    params: Block<Buf>,
    inv: InSlot<2, Buf>,
    width: Ask<keys::Width, i32>,
    tokens: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "combine_sorted",
            lanes: crate::routine::elementwise_rows(*width, *tokens)?,
        },
        &[y.v(), expert_weights.v(), out.v(), params.v(), inv.v()],
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
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: InSlot<0, Buf>,
    shared: InSlot<1, Buf>,
    gate: InSlot<2, Buf>,
    out: OutSlot<0, BufMut>,
    width: ParamOr<0, keys::Width, u32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "shared_expert_combine",
            lanes: combine_grid(*width, *rows)?,
        },
        &[routed.v(), shared.v(), gate.v(), out.v(), width.v()],
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
pub fn shared_expert_combine_strided(
    ctx: &Ctx<'_>,
    routed: InSlot<0, Buf>,
    shared: InSlot<1, Buf>,
    gate: InSlot<2, Buf>,
    out: OutSlot<0, BufMut>,
    width: ParamOr<0, keys::Width, u32>,
    row_pitch: Param<1, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "shared_expert_combine_strided",
            lanes: combine_grid(*width, *rows)?,
        },
        &[
            routed.v(),
            shared.v(),
            gate.v(),
            out.v(),
            width.v(),
            row_pitch.v(),
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
    crate::routine::elementwise_rows(width, rows)
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
pub fn qmv_routed(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    _bias: Null<Env<Buf>>,
    expert_ids: InSlot<1, Buf>,
    x_slot_stride: Param<2, i32>,
    x_row_stride: Param<3, i32>,
    slots_per_row: Param<4, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmv_routed_bfloat16_gs_64_b_4",
            lanes: routed_qmv_grid(*rows, *out_vec_size, *slots_per_row)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            expert_ids.v(),
            x_slot_stride.v(),
            x_row_stride.v(),
            slots_per_row.v(),
        ],
    )
}

/// The same matvec with the output bias added.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
pub fn qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    bias: Weight<3, Buf>,
    expert_ids: InSlot<1, Buf>,
    x_slot_stride: Param<2, i32>,
    x_row_stride: Param<3, i32>,
    slots_per_row: Param<4, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
            lanes: routed_qmv_grid(*rows, *out_vec_size, *slots_per_row)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            bias.v(),
            expert_ids.v(),
            x_slot_stride.v(),
            x_row_stride.v(),
            slots_per_row.v(),
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
pub fn mxfp4_qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    _biases: Null<Env<Buf>>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    in_vec_size: Param<0, i32>,
    out_vec_size: Param<1, i32>,
    bias: Weight<2, Buf>,
    expert_ids: InSlot<1, Buf>,
    x_slot_stride: Param<2, i32>,
    x_row_stride: Param<3, i32>,
    slots_per_row: Param<4, i32>,
    rows: Ask<keys::Rows, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
            lanes: routed_qmv_grid(*rows, *out_vec_size, *slots_per_row)?,
        },
        &[
            w.v(),
            scales.v(),
            x.v(),
            y.v(),
            in_vec_size.v(),
            out_vec_size.v(),
            bias.v(),
            expert_ids.v(),
            x_slot_stride.v(),
            x_row_stride.v(),
            slots_per_row.v(),
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
pub fn qmm_t_routed(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    tile_expert: InSlot<2, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    rows: Ask<keys::Rows, i32>,
    group: Ask<keys::QuantGroup, i32>,
    bits: Ask<keys::QuantBits, i32>,
    tile_m: Ask<keys::TileM, i32>,
    tile_n: Ask<keys::TileN, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, *n, *tile_m, *tile_n)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            tile_expert.v(),
            k.v(),
            n.v(),
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
pub fn qmm_t_routed_fp16(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    scales: Weight<1, Buf>,
    biases: Weight<2, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    tile_expert: InSlot<2, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    rows: Ask<keys::Rows, i32>,
    tile_m: Ask<keys::TileM, i32>,
    tile_n: Ask<keys::TileN, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: FP16_QMM[tile_point(*tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, *n, *tile_m, *tile_n)?,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            tile_expert.v(),
            k.v(),
            n.v(),
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
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Weight<0, Buf>,
    exponents: Weight<1, Buf>,
    x: InSlot<0, Buf>,
    y: OutSlot<0, BufMut>,
    bias: Weight<2, Buf>,
    tile_expert: InSlot<2, Buf>,
    k: Param<0, i32>,
    n: Param<1, i32>,
    rows: Ask<keys::Rows, i32>,
    tile_m: Ask<keys::TileM, i32>,
    tile_n: Ask<keys::TileN, i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            entrypoint: MXFP4_QMM[tile_point(*tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, *n, *tile_m, *tile_n)?,
        },
        &[
            w.v(),
            exponents.v(),
            x.v(),
            y.v(),
            bias.v(),
            tile_expert.v(),
            k.v(),
            n.v(),
        ],
    )
}

/// The thirteen, in the order the rows above name them.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(combine_sorted),
    crate::routine!(route_gather),
    crate::routine!(route_sort),
    crate::routine!(mxfp4_qmm_t_routed_bias),
    crate::routine!(mxfp4_qmv_routed_bias),
    crate::routine!(qmm_t_routed),
    crate::routine!(qmm_t_routed_fp16),
    crate::routine!(qmv_routed),
    crate::routine!(qmv_routed_bias),
    crate::routine!(router_topk),
    crate::routine!(router_topk_scaled),
    crate::routine!(shared_expert_combine),
    crate::routine!(shared_expert_combine_strided),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Encode};
    use core::cell::RefCell;

    type Call = (String, [u32; 3], Vec<ArgValue>);

    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire<'_>, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0
                .borrow_mut()
                .push((fire.entrypoint.to_string(), fire.lanes, args.to_vec()));
            Ok(())
        }
    }

    fn one(seen: &Seen) -> Call {
        let calls = seen.0.borrow();
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
            InSlot::new(Buf(0)),
            OutSlot::new(BufMut(1)),
            OutSlot::new(BufMut(2)),
            OutSlot::new(BufMut(3)),
            Block::new(Buf(4)),
            OutSlot::new(BufMut(5)),
        )
        .unwrap();
        assert_eq!(one(&seen).1, [1024, 1, 1]);

        let seen = Seen::default();
        router_topk(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), OutSlot::new(BufMut(2)), Block::new(Buf(3)), Null::new(Buf(4)), Ask::new(7)).unwrap();
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
        route_gather(&seen, InSlot::new(Buf(0)), OutSlot::new(BufMut(1)), InSlot::new(Buf(2)), Block::new(Buf(3)), Ask::new(64), ParamOr::new(96)).unwrap();
        assert_eq!(one(&seen).1, [64, 96, 1]);

        let seen = Seen::default();
        combine_sorted(
            &seen,
            InSlot::new(Buf(0)),
            InSlot::new(Buf(1)),
            OutSlot::new(BufMut(2)),
            Block::new(Buf(3)),
            InSlot::new(Buf(4)),
            Ask::new(64),
            Ask::new(24),
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
        qmv_routed(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(512),
            Param::new(256),
            Null::new(Env(Buf(7))),
            InSlot::new(Buf(8)),
            Param::new(1),
            Param::new(1),
            Param::new(4),
            Ask::new(3),
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
    /// The SIGNATURE still takes it, because the row states it and the trace
    /// still has one to hand over -- a row is positional and
    /// `binding::reorder` is what drops its slots on the legacy path. What
    /// the body declines to do is FORWARD it.
    #[test]
    fn the_mxfp4_matvec_drops_the_slot_its_codec_does_not_read() {
        let seen = Seen::default();
        mxfp4_qmv_routed_bias(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Null::new(Env(Buf(2))),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            Param::new(512),
            Param::new(256),
            Weight::new(Buf(7)),
            InSlot::new(Buf(8)),
            Param::new(1),
            Param::new(1),
            Param::new(1),
            Ask::new(1),
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
        mxfp4_qmm_t_routed_bias(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            InSlot::new(Buf(2)),
            OutSlot::new(BufMut(3)),
            Weight::new(Buf(4)),
            InSlot::new(Buf(5)),
            Param::new(128),
            Param::new(64),
            Ask::new(32),
            Ask::new(32),
            Ask::new(16),
        )
        .unwrap();
        let (entrypoint, lanes, args) = one(&seen);
        assert_eq!(entrypoint, "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16");
        // 64 columns over 16-wide tiles is 4 tiles; 32 rows over 32-wide
        // tiles is 1. Times the 16x16 workgroup.
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
            vec![0, 1, 2, 3, 4, 5],
            "six DENSE bindings: the activations are third and the output \
             bias is fifth, which is the matvec's order with the hole closed \
             rather than the matvec's order"
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
        qmm_t_routed(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            InSlot::new(Buf(5)),
            Param::new(256),
            Param::new(192),
            Ask::new(65),
            Ask::new(128),
            Ask::new(8),
            Ask::new(64),
            Ask::new(32),
        )
        .unwrap();
        let (entrypoint, lanes, _) = one(&seen);
        assert_eq!(
            entrypoint,
            "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32"
        );
        // 192 over 32 is 6 tiles; 65 over 64 is 2, and the second covers one
        // row. Both times the 16x16 workgroup.
        assert_eq!(lanes, [96, 32, 1]);

        // A point the tree does not carry is refused by NAME rather than
        // dispatched into a module that is not there.
        let seen = Seen::default();
        let refused = qmm_t_routed(
            &seen,
            Weight::new(Buf(0)),
            Weight::new(Buf(1)),
            Weight::new(Buf(2)),
            InSlot::new(Buf(3)),
            OutSlot::new(BufMut(4)),
            InSlot::new(Buf(5)),
            Param::new(256),
            Param::new(192),
            Ask::new(65),
            Ask::new(128),
            Ask::new(8),
            Ask::new(48),
            Ask::new(32),
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
        shared_expert_combine(&seen, InSlot::new(Buf(0)), InSlot::new(Buf(1)), InSlot::new(Buf(2)), OutSlot::new(BufMut(3)), ParamOr::new(512), Ask::new(9)).unwrap();
        assert_eq!(one(&seen).1, [512, 9, 1]);

        let seen = Seen::default();
        shared_expert_combine_strided(&seen, InSlot::new(Buf(0)), InSlot::new(Buf(1)), InSlot::new(Buf(2)), OutSlot::new(BufMut(3)), ParamOr::new(512), Param::new(4096), Ask::new(9))
            .unwrap();
        let (_, lanes, args) = one(&seen);
        assert_eq!(lanes, [512, 9, 1], "the width, and not the 4096 pitch");
        assert_eq!(args.len(), 6, "four buffers, the width and the pitch");
    }
}
