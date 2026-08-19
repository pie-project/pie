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

use kernels_macros::routine;
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

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};
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
#[routine]
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let per_expert_scale = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "router_topk_bfloat16").apply(router_grid(rows)?),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            params,
            per_expert_scale,
        ],
    )
}

/// [`router_topk`] with a per-expert gain applied to the weights.
///
/// # Errors
///
/// See `router_grid`.
#[routine]
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    per_expert_scale: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "router_topk_scaled_bfloat16").apply(router_grid(rows)?),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            params,
            per_expert_scale.arg(),
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
#[routine]
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    perm: Out<Tensor<i32>>,
    row_expert: Out<Tensor<i32>>,
    tile_expert: Out<Tensor<i32>>,
    inv: Out<Tensor<i32>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "route_sort").apply([ROUTER_LANES, 1, 1]),
        &[
            expert_ids.arg(),
            perm.arg(),
            row_expert.arg(),
            tile_expert.arg(),
            params,
            inv.arg(),
        ],
    )
}

/// Gather the rows into sorted order, padding each expert's run to a tile.
///
/// # Errors
///
/// See `rows_by_width`.
#[routine]
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    perm: In<Tensor<i32>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let padded = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "route_gather").apply(rows_by_width(width, padded)?),
        &[x.arg(), out.arg(), perm.arg(), params],
    )
}

/// Blend each row's expert outputs back, through the sort's inverse.
///
/// # Errors
///
/// See `rows_by_width`.
#[routine]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = y.width;
    let tokens = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "combine_sorted").apply(rows_by_width(width, tokens)?),
        &[y.arg(), expert_weights.arg(), out.arg(), params, inv.arg()],
    )
}

/// `out = routed + sigmoid(gate) * shared`, the always-on expert folded in.
///
/// # Errors
///
/// See `rows_by_width`; [`Refusal::Grid`] if the width does not fit an `i32`.
#[routine]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let w = i32::try_from(width).map_err(|_| Refusal::Grid {
        what: "the shared expert's row width",
        at: i64::from(width),
    })?;
    ctx.fire(
        Fire::at("moe/route.wgsl", "shared_expert_combine").apply(rows_by_width(w, rows)?),
        &[routed.arg(), shared.arg(), gate.arg(), out.arg(), width.arg()],
    )
}

/// [`shared_expert_combine`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// See [`shared_expert_combine`].
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

/// The routed affine matvec: one expert's weights per `(row, slot)`.
///
/// # Errors
///
/// See `routed_qmv_grid`.
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
    let bias = ctx.absent()?;
    let in_vec_size = x.width;
    let out_vec_size = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
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

/// [`qmv_routed`] with a per-output bias vector.
///
/// # Errors
///
/// See `routed_qmv_grid`.
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

/// gpt-oss's routed matvec: MXFP4 weights, a shared exponent plane, a bias.
///
/// # Errors
///
/// See `routed_qmv_grid`.
#[routine]
pub fn mxfp4_qmv_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    scales: Const<Tensor<u8>>,
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

/// The routed affine GEMM, one expert per row TILE.
///
/// # Errors
///
/// See `affine_qmm_point` and `routed_qmm_grid`.
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
        Fire::at("moe/qmm_t_routed.wgsl", AFFINE_QMM[affine_qmm_point(*group, *bits, *tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

/// [`qmm_t_routed`] with the activation precast to fp16.
///
/// # Errors
///
/// See `tile_point` and `routed_qmm_grid`.
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
        Fire::at("moe/qmm_t_routed.wgsl", FP16_QMM[tile_point(*tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

/// gpt-oss's routed GEMM: MXFP4 weights, an exponent plane, a bias.
///
/// # Errors
///
/// See `tile_point` and `routed_qmm_grid`.
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
        Fire::at("moe/qmm_t_routed.wgsl", MXFP4_QMM[tile_point(*tile_m, *tile_n)?]).apply(routed_qmm_grid(rows, n, *tile_m, *tile_n)?),
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

