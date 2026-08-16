//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `quant/qmm_t.metal` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! Declaring the axes is what surfaced the one real coverage gap here, and
//! then closed it: `qmv_routed` was compiled for ONE affine format where the
//! dense `qmv_fast` had six, so a Qwen3-MoE or routed gemma-4 at any other
//! format had no pipeline at all. The five missing instantiations are in
//! `quantized_qmv.metal` now, with the evidence for widening rather than
//! refusing. `.wiki/kernel-metal-refactor.md` §9 records it.

#![allow(clippy::too_many_arguments)]

use kernels::routine::Refusal;

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, Routine};

/// The shaders this family's routines reach: `(file, entrypoint)`, one pair
/// per instantiated name.
///
/// A row's `axes` GENERATED these names and its `file` column said where they
/// live. Retiring the row moved who NAMES them, not what exists -- the shader
/// is still compiled and still dispatched -- so the pairs are stated here and
/// [`crate::entrypoints`] reads them back. The FILE rides along because Metal
/// compiles from `(path, entry name)` at run time, and `device_kernels.rs`
/// builds every one of them against a real device; a name without its file
/// would leave that sweep nothing to open. See [`crate::RETIRED`].
pub static ENTRYPOINTS: &[(&str, &str)] = &[
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_128_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_32_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_bfloat16_gs_64_b_8_bm_64_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_64_bn_64",
    ),
    ("quant/qmv.metal", "affine_qmv_routed_bfloat16_gs_64_b_4"),
    (
        "quant/qmv.metal",
        "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
    ),
    ("moe/route.metal", "combine_sorted"),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_16_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_32_bn_64",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_16",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_32",
    ),
    (
        "quant/qmm_t.metal",
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_64_bn_64",
    ),
    (
        "quant/qmv.metal",
        "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
    ),
    ("moe/route.metal", "route_gather"),
    ("moe/route.metal", "route_sort"),
    ("moe/route.metal", "router_topk_bfloat16"),
    ("moe/route.metal", "router_topk_scaled_bfloat16"),
    ("moe/route.metal", "shared_expert_combine"),
    ("moe/route.metal", "shared_expert_combine_strided"),
];

/// A threadgroup lane count for the routing kernels: one lane per expert,
/// rounded up to a whole simdgroup.
///
/// The rounding is load-bearing. `route.metal`'s top-k reduces ACROSS
/// simdgroups through threadgroup memory, so a partial simdgroup leaves a
/// reduction slot holding whatever it held last -- an expert that was never
/// scored competing against the ones that were. Clamped to the kernel's
/// 1024-lane ceiling first, which is the same answer as clamping after.
fn router_lanes(n_experts: i32) -> Result<u32, Refusal> {
    if n_experts <= 0 {
        return Err(Refusal::Empty { what: "n_experts" });
    }
    Ok(n_experts.unsigned_abs().min(1024).div_ceil(32) * 32)
}

/// `LaunchRule::RouteRows`: the row's own width on x, the rows on y.
///
/// The threadgroup is the row clamped to 256 rather than a flat 256, so a row
/// narrower than that does not launch threads past its own end.
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

/// The routed matvec: the dense decomposition -- a threadgroup owns FOUR
/// output rows across two simdgroups -- with two axes the dense shape has not.
///
/// The token row is on x and the expert slot on z, and they are NOT
/// interchangeable: the kernel selects its expert with `sel = row *
/// slots_per_row + slot`, so folding the rows into the slot axis routes every
/// row through row 0's experts.
///
/// Rounded UP on the output axis, and that round-up is load-bearing too: a
/// truncating count drops every output past the last whole four, and at
/// `out_vec_size < 4` it drops the dispatch entirely. A shared expert's gate
/// is `hidden -> ONE logit a token`; its grid was `{32, 0, 1}`, no threads
/// ran, its buffer kept the zeros it was allocated with, and every routed
/// token was combined under `sigmoid(0) = 0.5` instead of its own gate.
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

/// The routed GEMM's grid, at a `(tile_m, tile_n)` tile.
///
/// EXACT division on both axes, refused rather than rounded. The shader has no
/// `M` argument and reads its row count from the grid, so a rounded-up row
/// axis runs tiles over rows that are not there. Both substitutions were
/// measured on the dense twin: handing the GEMM the MATVEC's grid made a
/// prefill entirely NaN, and rounding an axis up made it finite and wrong,
/// which is the worse of the two.
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

/// The nine tiles the shader tree carries, at `tile_m * 3 + tile_n`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a tile the tree does not carry.
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

/// The fifty-four affine routed-qmm instantiations, at `(group, bits) * 9 +
/// tile`.
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

/// Which experts a token goes to, and how much of it.
///
/// One threadgroup per ROW, as wide as the expert count: the top-k reduces a
/// whole row of logits through threadgroup memory and writes the k picks from
/// lane 0, so a row is a threadgroup and not a lane.
///
/// `per_expert_scale` is bound and not read. The slot is positional, so it has
/// to hold an address whether or not this instantiation dereferences it;
/// [`router_topk_scaled`] is the symbol that means it.
///
/// # Errors
///
/// See [`router_lanes`], and [`Refusal::Empty`] for an empty row count.
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: Buf,
    expert_ids: BufMut,
    expert_weights: BufMut,
    params: Buf,
    per_expert_scale: Buf,
    n_experts: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let w = router_lanes(*n_experts)?;
    if *rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.dispatch(
        Fire {
            entrypoint: "router_topk_bfloat16",
            file: ROUTE_FILE,
            lanes: [w, rows.unsigned_abs(), 1],
            group: [w, 1, 1],
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

/// [`router_topk`] with a per-expert rescale, indexed by the EXPERT and not by
/// the pick.
///
/// # Errors
///
/// See [`router_topk`].
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: Buf,
    expert_ids: BufMut,
    expert_weights: BufMut,
    params: Buf,
    per_expert_scale: Buf,
    n_experts: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let w = router_lanes(*n_experts)?;
    if *rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.dispatch(
        Fire {
            entrypoint: "router_topk_scaled_bfloat16",
            file: ROUTE_FILE,
            lanes: [w, rows.unsigned_abs(), 1],
            group: [w, 1, 1],
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

/// The permutation that groups a fire's rows by expert, and the three tables
/// read off it.
///
/// ONE threadgroup whatever the row count, which is the whole reason this is
/// not [`router_topk`]'s grid with a different body: the sort reduces across
/// every `(row, slot)` pair through threadgroup atomics and stripes them over
/// its own lanes. A grid of one copy per row would have each clearing and
/// rewriting the permutation the others are reading.
///
/// FIVE outputs, and that is the shape of the thing: the permutation, the
/// per-row expert, the per-tile expert, and the inverse the combine reads
/// back. A caller that passed fewer would leave the combine reading whatever
/// the arena held.
///
/// # Errors
///
/// See [`router_lanes`].
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: Buf,
    perm: BufMut,
    row_expert: BufMut,
    tile_expert: BufMut,
    params: Buf,
    inv: BufMut,
    n_experts: Env<i32>,
) -> Result<(), Refusal> {
    let w = router_lanes(*n_experts)?;
    ctx.dispatch(
        Fire {
            entrypoint: "route_sort",
            file: ROUTE_FILE,
            lanes: [w, 1, 1],
            group: [w, 1, 1],
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

/// The gather that builds the sorted stack the routed GEMM reads.
///
/// Its rows are the STACK's and not the fire's -- `MoeRouteParams::padded` --
/// which is why the extent is an argument here rather than a fact the rule
/// decides. Handed the fire's token count the gather ran over a quarter of its
/// own output at `top_k = 4` and left the rest whatever the arena held.
///
/// # Errors
///
/// See [`route_rows`].
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: Buf,
    out: BufMut,
    perm: Buf,
    params: Buf,
    width: Env<i32>,
    padded: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = route_rows(*width, *padded)?;
    ctx.dispatch(
        Fire {
            entrypoint: "route_gather",
            file: ROUTE_FILE,
            lanes,
            group,
        },
        &[x.v(), out.v(), perm.v(), params.v()],
    )
}

/// The scatter back: each token's `k` expert results weighted and summed.
///
/// Its rows ARE the fire's, unlike [`route_gather`]'s, and the two share a
/// launch rule -- which is why both state their extent instead of letting the
/// rule pick one for them.
///
/// # Errors
///
/// See [`route_rows`].
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
    let (lanes, group) = route_rows(*width, *tokens)?;
    ctx.dispatch(
        Fire {
            entrypoint: "combine_sorted",
            file: ROUTE_FILE,
            lanes,
            group,
        },
        &[y.v(), expert_weights.v(), out.v(), params.v(), inv.v()],
    )
}

/// `out = routed + sigmoid(gate) * shared`, the always-on expert folded in.
///
/// `out` may alias `routed`, which nothing here needs to know: an alias is two
/// names for one address and a binding is by address.
///
/// The width is an ARGUMENT as well as the grid's x: the kernel is told it
/// because it indexes `row * width + col`, and a grid carries an extent, not a
/// stride.
///
/// # Errors
///
/// See [`route_rows`].
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: Buf,
    shared: Buf,
    gate: Buf,
    out: BufMut,
    width: u32,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    let (lanes, group) = route_rows(width.try_into().unwrap_or(i32::MAX), *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "shared_expert_combine",
            file: ROUTE_FILE,
            lanes,
            group,
        },
        &[routed.v(), shared.v(), gate.v(), out.v(), width.v()],
    )
}

/// [`shared_expert_combine`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// See [`route_rows`].
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
    let (lanes, group) = route_rows(width.try_into().unwrap_or(i32::MAX), *rows)?;
    ctx.dispatch(
        Fire {
            entrypoint: "shared_expert_combine_strided",
            file: ROUTE_FILE,
            lanes,
            group,
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

/// The routed affine matvec, unbiased.
///
/// ONE affine format, and that is the kernel's design rather than a gap:
/// `AffineQ::group_size` is a `constexpr`, so a second group point would name
/// an instantiation that dequantises at 64 whatever it claims. A routed
/// checkpoint at another format is meant to fail BY NAME here, where the
/// caller can still read the refusal, rather than inside the Metal compiler.
///
/// `bias` is bound and not read; [`qmv_routed_bias`] is the symbol that reads
/// it, and the slot is positional so it holds an address either way.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
pub fn qmv_routed(
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
            entrypoint: "affine_qmv_routed_bfloat16_gs_64_b_4",
            file: QMV_FILE,
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
            group: SIMD_PAIR,
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

/// [`qmv_routed`] with a per-output-row bias.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
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
            entrypoint: "affine_qmv_routed_bias_bfloat16_gs_64_b_4",
            file: QMV_FILE,
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
            group: SIMD_PAIR,
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

/// gpt-oss's routed matvec: MXFP4 weights, and a bias that is not a zero
/// point.
///
/// `biases` is bound and not read, and the reason is worth the line: MXFP4
/// carries a shared EXPONENT per block and no zero point, so the slot the
/// affine codec fills with `biases` has nothing to hold here. The checkpoint
/// publishes `weight`, `scales` and `bias` and no `biases` at all -- and the
/// shapes say which is which, `[32, 2880]` for one value per output row where
/// a zero point would be `[32, 2880, 90]`, one per group beside the scales.
///
/// # Errors
///
/// See [`routed_qmv_grid`].
pub fn mxfp4_qmv_routed_bias(
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
            entrypoint: "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
            file: QMV_FILE,
            lanes: routed_qmv_grid(*rows, out_vec_size, slots_per_row)?,
            group: SIMD_PAIR,
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

/// The routed affine GEMM: one tile per expert run of the sorted stack.
///
/// # The five pad slots
///
/// This entrypoint declares buffers at 0..=6 and then `tile_expert` at **12**,
/// with nothing at 7..=11 (`quant/qmm_t.metal:593`). The numbering is the
/// routed MATVEC's, kept deliberately so that one argument-table ordinal
/// serves both pipelines -- the host binds all of an ordinal's slots whichever
/// one the row count selects.
///
/// A Metal argument table is a contiguous run, so the five holes still have to
/// hold an address, and a routine's argument list is positional: the index in
/// the list IS the slot. So `pad` is taken once and bound at each hole.
/// Skipping them would slide `tile_expert` into slot 7 and hand the GEMM the
/// bias pointer where it reads which expert owns the tile. `ssm.rs`'s scan
/// takes a `pad` for the same reason and says so at more length.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`affine_qmm_point`] for a format or tile the
/// shader tree does not carry, and see [`routed_qmm_grid`].
///
/// The instantiation is checked FIRST, as `ssm.rs`'s scan checks its tiling:
/// an entrypoint Metal has no `[[host_name]]` for makes `newFunctionWithName:`
/// return nil at run time, inside a fire, after the plan was accepted.
pub fn qmm_t_routed(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    pad: Buf,
    tile_expert: Buf,
    k: i32,
    n: i32,
    rows: Env<i32>,
    group: Env<i32>,
    bits: Env<i32>,
    tile_m: Env<i32>,
    tile_n: Env<i32>,
) -> Result<(), Refusal> {
    let point = affine_qmm_point(*group, *bits, *tile_m, *tile_n)?;
    ctx.dispatch(
        Fire {
            entrypoint: AFFINE_QMM[point],
            file: QMM_FILE,
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            tile_expert.v(),
        ],
    )
}

/// [`qmm_t_routed`] over activations pre-cast to fp16.
///
/// One affine format here where the plain one has six, and it is the same
/// statement its dense twin makes: the pre-cast path exists for the checkpoint
/// that ships `gs_64_b_4`, and an instantiation for a format nothing publishes
/// would be a name no fire can reach.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`tile_point`], and see [`routed_qmm_grid`].
pub fn qmm_t_routed_fp16(
    ctx: &Ctx<'_>,
    w: Buf,
    scales: Buf,
    biases: Buf,
    x: Buf,
    y: BufMut,
    pad: Buf,
    tile_expert: Buf,
    k: i32,
    n: i32,
    rows: Env<i32>,
    tile_m: Env<i32>,
    tile_n: Env<i32>,
) -> Result<(), Refusal> {
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.dispatch(
        Fire {
            entrypoint: FP16_QMM[point],
            file: QMM_FILE,
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            scales.v(),
            biases.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            tile_expert.v(),
        ],
    )
}

/// gpt-oss's routed GEMM.
///
/// SIX pad slots and not five, and they are not the same five: this entrypoint
/// declares nothing at 2 either, because MXFP4 has no zero point to bind where
/// the affine codec puts `biases`. The bias it does take is at 7, which is a
/// slot the affine form pads.
///
/// # Errors
///
/// [`Refusal::Narrow`] from [`tile_point`], and see [`routed_qmm_grid`].
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Buf,
    exponents: Buf,
    pad: Buf,
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
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.dispatch(
        Fire {
            entrypoint: MXFP4_QMM[point],
            file: QMM_FILE,
            lanes: routed_qmm_grid(*rows, n, *tile_m, *tile_n)?,
            group: QMM_GROUP,
        },
        &[
            w.v(),
            exponents.v(),
            pad.v(),
            x.v(),
            y.v(),
            k.v(),
            n.v(),
            bias.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            pad.v(),
            tile_expert.v(),
        ],
    )
}

/// Two simdgroups of 32, which is what a matvec threadgroup is here: four
/// output rows to a simdgroup, eight to the group.
const SIMD_PAIR: [u32; 3] = [32, 2, 1];

/// The GEMM's threadgroup: four simdgroups, arranged `(32, 2, 2)` because the
/// third axis is the two halves of the K walk and not a third tile.
const QMM_GROUP: [u32; 3] = [32, 2, 2];

const ROUTE_FILE: &str = "moe/route.metal";
const QMV_FILE: &str = "quant/qmv.metal";
const QMM_FILE: &str = "quant/qmm_t.metal";

/// The family, in the order the rows above state it.
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

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do.
    #[derive(Default)]
    struct Seen(RefCell<Vec<Call>>);

    impl Encode for Seen {
        fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.0.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The three routed GEMMs bind a pad at every slot their entrypoint does
    /// not declare, and `tile_expert` lands at twelve.
    ///
    /// Skipping the holes would slide `tile_expert` into slot 7 and hand the
    /// GEMM the bias pointer where it reads which expert owns the tile -- and
    /// nothing would report it, because Metal does not validate a binding and
    /// every one of these is a device pointer.
    #[test]
    fn a_routed_gemm_pads_every_slot_its_entrypoint_leaves_empty() {
        let seen = Seen::default();
        let pad = Buf(99);
        qmm_t_routed(
            &seen,
            Buf(1),
            Buf(2),
            Buf(3),
            Buf(4),
            BufMut(5),
            pad,
            Buf(6),
            2048,
            64,
            Env(32),
            Env(64),
            Env(4),
            Env(32),
            Env(32),
        )
        .expect("a launch");
        mxfp4_qmm_t_routed_bias(
            &seen,
            Buf(1),
            Buf(2),
            pad,
            Buf(4),
            BufMut(5),
            Buf(7),
            Buf(6),
            2048,
            64,
            Env(32),
            Env(32),
            Env(32),
        )
        .expect("a launch");

        let calls = seen.0.borrow();
        for (fire, args) in calls.iter() {
            assert_eq!(
                args.len(),
                13,
                "`{}` binds through slot 12",
                fire.entrypoint
            );
            assert_eq!(
                args[12],
                ArgValue::Buffer(6),
                "`{}` takes tile_expert at TWELVE",
                fire.entrypoint
            );
        }
        assert_eq!(
            calls[0].1[7..12],
            [ArgValue::Buffer(99); 5],
            "the affine form's five holes are 7..=11"
        );
        assert_eq!(
            calls[1].1[2],
            ArgValue::Buffer(99),
            "MXFP4 has no zero point, so slot 2 is a hole the affine form fills"
        );
        assert_eq!(
            calls[1].1[7],
            ArgValue::Buffer(7),
            "and its bias is at 7, which the affine form pads"
        );
    }

    /// A tile that does not divide the extent is refused, not rounded.
    ///
    /// The shader has no `M` argument and reads its row count off the grid, so
    /// a rounded-up axis runs whole tiles over rows that are not there. Both
    /// substitutions were measured on the dense twin: the matvec's grid made a
    /// prefill entirely NaN, and rounding made it finite and WRONG.
    #[test]
    fn a_partial_tile_is_refused_rather_than_rounded() {
        assert!(matches!(
            routed_qmm_grid(48, 64, 32, 32),
            Err(Refusal::Narrow { .. })
        ));
        assert!(matches!(
            routed_qmm_grid(64, 48, 32, 32),
            Err(Refusal::Narrow { .. })
        ));
        assert_eq!(
            routed_qmm_grid(64, 128, 32, 32),
            Ok([32 * 4, 2 * 2, 2]),
            "four column tiles and two row tiles"
        );
        assert!(matches!(
            affine_qmm_point(48, 4, 32, 32),
            Err(Refusal::Narrow {
                what: "affine group size",
                ..
            })
        ));
    }

    /// The routed matvec rounds its output axis UP, and the round-up is what
    /// makes a one-logit projection run at all.
    ///
    /// A truncating count drops every output past the last whole four; at
    /// `out_vec_size < 4` it drops the dispatch entirely. A shared expert's
    /// gate is `hidden -> ONE logit a token`. Its grid was `{32, 0, 1}`, no
    /// threads ran, the buffer kept the zeros it was allocated with, and every
    /// routed token was combined under `sigmoid(0) = 0.5`.
    #[test]
    fn a_one_logit_projection_still_launches_a_thread() {
        assert_eq!(routed_qmv_grid(3, 1, 1), Ok([96, 1, 1]));
        assert_eq!(routed_qmv_grid(1, 5, 4), Ok([32, 2, 4]));
        assert!(matches!(
            routed_qmv_grid(1, 4, 0),
            Err(Refusal::Empty {
                what: "slots_per_row"
            })
        ));
    }

    /// The router's threadgroup is the expert count rounded to a whole
    /// simdgroup, because the top-k reduces ACROSS simdgroups and a partial
    /// one leaves a reduction slot holding whatever it held last.
    #[test]
    fn the_router_rounds_its_threadgroup_to_a_whole_simdgroup() {
        let seen = Seen::default();
        router_topk(
            &seen,
            Buf(1),
            BufMut(2),
            BufMut(3),
            Buf(4),
            Buf(5),
            Env(60),
            Env(7),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(fire.group, [64, 1, 1], "60 experts is two whole simdgroups");
        assert_eq!(fire.lanes, [64, 7, 1], "and a threadgroup per row");
        assert_eq!(
            args.len(),
            5,
            "the unscaled form still binds the scale slot"
        );
        assert_eq!(router_lanes(1024), Ok(1024));
        assert_eq!(router_lanes(2048), Ok(1024), "clamped to the kernel's cap");
    }

    /// The sort is ONE threadgroup whatever the row count.
    ///
    /// `RouterLane`'s grid -- which this shared until the row axis landed on
    /// it -- launches one copy per row, each clearing and rewriting the
    /// permutation the others are reading.
    #[test]
    fn the_sort_is_one_threadgroup_however_many_rows_there_are() {
        let seen = Seen::default();
        route_sort(
            &seen,
            Buf(1),
            BufMut(2),
            BufMut(3),
            BufMut(4),
            Buf(5),
            BufMut(6),
            Env(128),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        let (fire, args) = &calls[0];
        assert_eq!(
            fire.lanes, fire.group,
            "one threadgroup, and it is the grid"
        );
        assert_eq!(args.len(), 6, "five outputs and the params block");
    }

    /// The gather's extent is the SORTED STACK's and the combine's is the
    /// fire's, and they share a launch rule -- which is why both state it.
    ///
    /// Given the fire's count the gather ran over a quarter of its own output
    /// at `top_k = 4` and left the rest whatever the arena held.
    #[test]
    fn the_gather_and_the_combine_state_two_different_row_counts() {
        let seen = Seen::default();
        route_gather(&seen, Buf(1), BufMut(2), Buf(3), Buf(4), Env(2048), Env(64))
            .expect("a launch");
        combine_sorted(
            &seen,
            Buf(1),
            Buf(2),
            BufMut(3),
            Buf(4),
            Buf(5),
            Env(2048),
            Env(16),
        )
        .expect("a launch");
        let calls = seen.0.borrow();
        assert_eq!(calls[0].0.lanes, [2048, 64, 1], "the padded stack");
        assert_eq!(calls[1].0.lanes, [2048, 16, 1], "the fire's own tokens");
        assert_eq!(calls[0].0.group, [256, 1, 1]);
    }
}
