#![allow(clippy::too_many_arguments)]
//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `quantized_qmm_t.wgsl` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! Declaring the axes is what surfaced the one real coverage gap here, and
//! then closed it: `qmv_routed` was compiled for ONE affine format where the
//! dense `qmv_fast` had six, so a Qwen3-MoE or routed gemma-4 at any other
//! format had no pipeline at all. The five missing instantiations are in
//! `quantized_qmv.wgsl` now, with the evidence for widening rather than
//! refusing. `.wiki/kernel-metal-refactor.md` §9 records it.

use kernels::KernelSig;

pub static KERNELS: &[KernelSig] = &[
    // FIVE outputs, and that is the shape of the thing: a sort states the
    // permutation, the per-row expert, the per-tile expert, and the inverse
    // the combine reads back. A text that named fewer would leave the combine
    // reading whatever was in the buffer.
    // 9 in quantized_qmm_t.wgsl
    // 1 in quantized_qmv.wgsl
    //
    // This row named no operands, which made it the one unstated row in the
    // table that is provably REACHABLE. `model-ir`'s routed-QMV site
    // picks the symbol with a `match` on the weight repr --
    // `WeightRepr::Mxfp4Marlin => "mxfp4_qmv_routed_bias"` against
    // `affine_qmv_routed{_bias}` for everything else -- and then makes ONE
    // `with_params` call for both arms. So a driver does try to bind this, and
    // an operand list it cannot read is a failure at launch rather than dead
    // code.
    //
    // Found from the Vulkan side, by intersecting the operand-less rows with
    // every symbol literal in `model-ir`: of the 57, exactly this one
    // survived. `kernels-vulkan` states it identically.
    //
    // The list below is not invented to fill the hole. `qmv.wgsl` generates
    // this symbol from `instantiate_gptoss_qmv` with `fn = qmv_routed_bias` --
    // the SAME macro and the SAME template function as `qmv_routed_bias`
    // directly above, differing only in the codec and the group/bits point,
    // neither of which appears in the signature. The twelve parameters are
    // therefore identical operand for operand, and this is that row's list
    // copied across rather than reconstructed.
    //
    // `biases` stays in the ABI and stays unread: the MXFP4 codec has no
    // separate bias plane, so the kernel takes the pointer and ignores it. A
    // row is positional, so dropping the slot would shift everything after it.
    // 54 in quantized_qmm_t.wgsl
    // 9 in quantized_qmm_t.wgsl
    // 1 in quantized_qmv.wgsl
    // ONE affine format, and that is the kernel's design rather than a gap:
    // `AffineQ::group_size` is a constant, so a second group point would name
    // an instantiation that dequantises at 64 whatever it claims. A routed
    // checkpoint at another group is meant to fail by name when its pipeline
    // is built -- which `entrypoint()` now does at the call instead of in the
    // shader compiler.
    // 1 in quantized_qmv.wgsl
    // 1 in moe_route.wgsl
    // 1 in moe_route.wgsl
];
/// The entrypoints of this family's routines whose ROWS have been RETIRED.
///
/// `refactor-bigplan.md` §7 Stage 3. Not every kernel here has crossed its
/// arm — this family still states rows for the ones that have not — so this
/// is the retired SUBSET rather than the whole family, and
/// `a_retired_familys_stated_entrypoints_are_what_its_bodies_fire` compares
/// it against the bodies that fire them.
///
/// See [`crate::sample::ENTRYPOINTS`] for why a retired row's entrypoints
/// have to be stated at all.
pub static ENTRYPOINTS: &[&str] = &[
    "affine_qmv_routed_bfloat16_gs_64_b_4",
    "router_topk_bfloat16",
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
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32",
    "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64",
    "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
    "combine_sorted",
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
    "route_gather",
    "route_sort",
    "router_topk_scaled_bfloat16",
    "shared_expert_combine",
    "shared_expert_combine_strided",
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};
use kernels::routine::Refusal;

/// The router's workgroup width, and it is NOT `kernels-vulkan`'s.
///
/// `moe/route.wgsl` declares `const ROUTER_LANES = 256u`;
/// `kernels-vulkan::moe::SORT_LANES` is 1024. WebGPU guarantees only 256
/// invocations per workgroup, so the two trees genuinely differ here — and a
/// [`Fire`] states LANES which the driver divides by the module's own
/// `@workgroup_size`, so copying vulkan's constant would ask for four
/// workgroups where the shader reduces over one and the top-k would read a
/// window that is not there.
const ROUTER_LANES: u32 = 256;

/// The routed matvec's lane count per row, from `@workgroup_size(32, 8)`.
const QMV_LANES: u32 = 32;

/// The routed GEMM's workgroup edge, from `@workgroup_size(16, 16)`.
const TILE: u32 = 16;

/// One workgroup per row, as lanes.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero row count.
fn router_grid(rows: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([ROUTER_LANES, rows.unsigned_abs(), 1])
}

/// A rectangle of `width` columns by `rows` rows, as lanes.
///
/// **Two dimensions, where `kernels-vulkan` states one.** Its route shaders
/// flatten to a single index and take `elementwise_rows`; `moe/route.wgsl`'s
/// gather, combine and both shared-expert forms are `@workgroup_size(16, 16)`
/// and read `gid.x` as the column and `gid.y` as the row. A flat
/// `width * rows` here would give the last rows no lanes at all and the first
/// row far too many — and every one of them would still write, because the
/// bodies guard on `params` rather than on the grid.
///
/// This is what `driver-wgpu::geometry`'s `Rule::RouteRows` already states,
/// `[dims.width, rows, 1]`, and the routine says the same thing.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero width or row count.
fn rows_by_width(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs(), 1])
}

/// The routed matvec's grid: rows on x, the output axis on y, the expert slot
/// on z.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero row count, output width or slot count;
/// [`Refusal::Grid`] if the lane count overflows.
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
        .checked_mul(QMV_LANES)
        .ok_or(Refusal::Grid {
            what: "rows * the matvec's lane count",
            at: i64::from(rows) * i64::from(QMV_LANES),
        })?;
    Ok([x, out_vec_size.unsigned_abs(), slots.unsigned_abs()])
}

/// The routed GEMM's grid: one workgroup per `(column tile, row tile)`.
///
/// # Errors
///
/// [`Refusal::Empty`] for a zero row count or output width;
/// [`Refusal::Narrow`] for a non-positive tile; [`Refusal::Grid`] on overflow.
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

/// Which of the nine `(bm, bn)` points a tile pair names.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a tile the tree has no point for.
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

/// Which of the fifty-four affine points a `(group, bits, bm, bn)` names.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a group size, bit width or tile with no point.
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

/// The whole spelling, from LITERAL tables and never `format!`.
///
/// A composed name that is one point off spells an entrypoint the tree does
/// not carry, and the failure is at first fire rather than at build. These
/// three were generated FROM the shader's own `pie:instantiate` lines and
/// every one of the seventy-two was checked to exist before it was written
/// here.
const _: () = {
    assert!(AFFINE_QMM.len() == 54);
    assert!(FP16_QMM.len() == 9);
    assert!(MXFP4_QMM.len() == 9);
};

/// The router's top-k: which experts each row picks, and how much of each.
///
/// # Errors
///
/// See `router_grid`.
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: Buf,
    expert_ids: BufMut,
    expert_weights: BufMut,
    params: Buf,
    per_expert_scale: Buf,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "router_topk_bfloat16",
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

/// [`router_topk`] with a per-expert gain applied to the weights.
///
/// # Errors
///
/// See `router_grid`.
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: Buf,
    expert_ids: BufMut,
    expert_weights: BufMut,
    params: Buf,
    per_expert_scale: Buf,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
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

/// The counting sort that groups `(row, slot)` pairs by expert.
///
/// ONE workgroup whatever the rows: the histogram is fire-wide and the sort
/// reduces across every pair through workgroup memory. That is why
/// `LaunchRule::RouterSort` had to split from `RouterLane` — the two look
/// alike and one of them must not scale with rows.
///
/// # Errors
///
/// None today; the signature is fallible because every routine's is.
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: Buf,
    perm: BufMut,
    row_expert: BufMut,
    tile_expert: BufMut,
    params: Buf,
    inv: BufMut,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "route_sort",
            lanes: [ROUTER_LANES, 1, 1],
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

/// Gather the rows into sorted order, padding each expert's run to a tile.
///
/// # Errors
///
/// See `rows_by_width`.
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: Buf,
    out: BufMut,
    perm: Buf,
    params: Buf,
    width: Env<i32>,
    padded: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "route_gather",
            lanes: rows_by_width(*width, *padded)?,
        },
        &[x.v(), out.v(), perm.v(), params.v()],
    )
}

/// Blend each row's expert outputs back, through the sort's inverse.
///
/// # Errors
///
/// See `rows_by_width`.
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: Buf,
    expert_weights: Buf,
    out: BufMut,
    params: Buf,
    inv: Buf,
    width: Env<i32>,
    tokens: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "combine_sorted",
            lanes: rows_by_width(*width, *tokens)?,
        },
        &[y.v(), expert_weights.v(), out.v(), params.v(), inv.v()],
    )
}

/// `out = routed + sigmoid(gate) * shared`, the always-on expert folded in.
///
/// # Errors
///
/// See `rows_by_width`; [`Refusal::Grid`] if the width does not fit an `i32`.
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: Buf,
    shared: Buf,
    gate: Buf,
    out: BufMut,
    width: u32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let w = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "shared_expert_combine",
            lanes: rows_by_width(w, *rows)?,
        },
        &[routed.v(), shared.v(), gate.v(), out.v(), width.v()],
    )
}

/// [`shared_expert_combine`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// See [`shared_expert_combine`].
pub fn shared_expert_combine_strided(
    ctx: &Ctx<'_>,
    routed: Buf,
    shared: Buf,
    gate: Buf,
    out: BufMut,
    width: u32,
    row_pitch: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let w = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    ctx.dispatch(
        Fire {
            module: "moe/route.wgsl",
            entrypoint: "shared_expert_combine_strided",
            lanes: rows_by_width(w, *rows)?,
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

/// The routed affine matvec: one expert's weights per `(row, slot)`.
///
/// # Errors
///
/// See `routed_qmv_grid`.
pub fn qmv_routed(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    // The bias plane the UNBIASED form does not read, and still binds: the
    // two symbols share a template and a slot dropped here shifts
    // `expert_ids` into it. `kernels-vulkan` takes it as `_bias` and does not
    // forward it; here the binding is declared and must be filled.
    bias: Buf,
    expert_ids: Buf,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmv_routed.wgsl",
            entrypoint: "affine_qmv_routed_bfloat16_gs_64_b_4",
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
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

/// [`qmv_routed`] with a per-output bias vector.
///
/// # Errors
///
/// See `routed_qmv_grid`.
pub fn qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    bias: Buf,
    expert_ids: Buf,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmv_routed.wgsl",
            entrypoint: "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
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

/// gpt-oss's routed matvec: MXFP4 weights, a shared exponent plane, a bias.
///
/// # Errors
///
/// See `routed_qmv_grid`.
pub fn mxfp4_qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    // ACCEPTED AND NOT FORWARDED, which is `kernels-vulkan`'s answer too and
    // is NOT what §8c would have predicted.
    //
    // The MXFP4 codec has no separate bias plane. The ROW still states the
    // slot, because a row is positional and dropping it would shift every
    // operand after it -- but `moe/qmv_routed.wgsl`'s mxfp4 arm is
    // `//#if`-gated and declares SIX `@group(0)` bindings where the row
    // states seven. So there is no slot to fill, and passing one is
    // `Refusal::Arity` at the first real dispatch.
    //
    // §8c says wgpu binds what vulkan drops, because WGSL declares its
    // bindings in source and `naga` keeps the unread ones. True, and it is
    // about globals the entrypoint never READS. A binding a preprocessor arm
    // never DECLARES is a different thing, and the rule of thumb does not
    // cover it. What settles each case is
    // `every_routine_binds_a_buffer_for_every_binding_its_module_declares`,
    // which measured six here and caught this on its first run.
    _biases: Buf,
    x: Buf,
    y: BufMut,
    in_vec_size: i32,
    out_vec_size: i32,
    bias: Buf,
    expert_ids: Buf,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmv_routed.wgsl",
            entrypoint: "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
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

/// The routed affine GEMM, one expert per row TILE.
///
/// # Errors
///
/// See `affine_qmm_point` and `routed_qmm_grid`.
pub fn qmm_t_routed(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    tile_expert: Buf,
    k: i32,
    n: i32,
    rows: Env<i32>,
    group: Env<i32>,
    bits: Env<i32>,
    tile_m: Env<i32>,
    tile_n: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmm_t_routed.wgsl",
            entrypoint: AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
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

/// [`qmm_t_routed`] with the activation precast to fp16.
///
/// # Errors
///
/// See `tile_point` and `routed_qmm_grid`.
pub fn qmm_t_routed_fp16(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    tile_expert: Buf,
    k: i32,
    n: i32,
    rows: Env<i32>,
    tile_m: Env<i32>,
    tile_n: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmm_t_routed.wgsl",
            entrypoint: FP16_QMM[tile_point(*tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
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

/// gpt-oss's routed GEMM: MXFP4 weights, an exponent plane, a bias.
///
/// # Errors
///
/// See `tile_point` and `routed_qmm_grid`.
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    exponents: Buf,
    x: Buf,
    y: BufMut,
    bias: Buf,
    tile_expert: Buf,
    k: i32,
    n: i32,
    rows: Env<i32>,
    tile_m: Env<i32>,
    tile_n: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "moe/qmm_t_routed.wgsl",
            entrypoint: MXFP4_QMM[tile_point(*tile_m, *tile_n)?],
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
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

pub static ROUTINES: &[Routine] = &[
    crate::routine!(combine_sorted),
    crate::routine!(mxfp4_qmm_t_routed_bias),
    crate::routine!(mxfp4_qmv_routed_bias),
    crate::routine!(qmm_t_routed),
    crate::routine!(qmm_t_routed_fp16),
    crate::routine!(qmv_routed),
    crate::routine!(qmv_routed_bias),
    crate::routine!(route_gather),
    crate::routine!(route_sort),
    crate::routine!(router_topk),
    crate::routine!(router_topk_scaled),
    crate::routine!(shared_expert_combine),
    crate::routine!(shared_expert_combine_strided),
];
