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


use kernels::Grid;
use kernels_macros::routine;
use kernels::routine::Refusal;

use crate::routine::{Asks, Bind, Const, Ctx, Fire, In, Out, Tensor, bf16, keys};

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

/// The routed matvec's two vector widths, which are ONE SLOT's and not one
/// statement's.
///
/// The kernel walks a single expert per thread block: `in_vec_size` strides its
/// weight rows and `out_vec_size` both strides the bank by expert and places
/// the result, so both have to be one expert's numbers. The MARKS carry
/// neither. `dsl::metal::routed_qmv` declares the matvec's value as `[Tokens,
/// width * k]` -- `k` results end to end, which is the shape the elementwise
/// activation between two of them must cover -- and its input, for the down
/// projection, is another such run. So `x.width` and `y.width` are both `k`
/// TIMES what this kernel means by a vector.
///
/// They were read straight off the marks, and the cost was total: at gemma-4's
/// top-8 the bank was addressed `e * 8 * 704 * in_vec_size_w` bytes in, which
/// is past the end of it for every expert but the zeroth, and the result was
/// written eight rows apart into a buffer with one row per route. Both mixture
/// checkpoints on this backend answered `inf` from the first routed matvec of
/// layer 0 and NaN from every layer after it. The statement was already
/// carrying the honest number for the input -- `x_slot_stride` IS one run,
/// which is why `dsl::metal::routed_qmv` states it rather than reading the
/// trailing dim -- so this reads that, and divides the output's by the slot
/// count the same statement carries.
///
/// # Errors
///
/// [`Refusal::Empty`] for a non-positive width or slot count, and
/// [`Refusal::Narrow`] for an output width the slot count does not divide --
/// which is a text whose value shape and whose top-k disagree, not a rounding
/// this could absorb.
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
#[routine]
pub fn router_topk(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let n_experts = ctx.ask::<i32, keys::NumExperts>()?;
    let params = ctx.params()?;
    let per_expert_scale = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let w = router_lanes(n_experts)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.fire(
        Fire::at(ROUTE_FILE, "router_topk_bfloat16").apply(Grid::of([w, rows.unsigned_abs(), 1], [w, 1, 1])),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            params,
            per_expert_scale,
        ],
    )
}

/// [`router_topk`] with a per-expert rescale, indexed by the EXPERT and not by
/// the pick.
///
/// # Errors
///
/// See [`router_topk`].
#[routine]
pub fn router_topk_scaled(
    ctx: &Ctx<'_>,
    logits: In<Tensor<bf16>>,
    expert_ids: Out<Tensor<i32>>,
    expert_weights: Out<Tensor<bf16>>,
    per_expert_scale: Const<Tensor<bf16>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let n_experts = ctx.ask::<i32, keys::NumExperts>()?;
    let params = ctx.params()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let w = router_lanes(n_experts)?;
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    ctx.fire(
        Fire::at(ROUTE_FILE, "router_topk_scaled_bfloat16").apply(Grid::of([w, rows.unsigned_abs(), 1], [w, 1, 1])),
        &[
            logits.arg(),
            expert_ids.arg(),
            expert_weights.arg(),
            params,
            per_expert_scale.arg(),
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
#[routine]
pub fn route_sort(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    perm: Out<Tensor<i32>>,
    row_expert: Out<Tensor<i32>>,
    tile_expert: Out<Tensor<i32>>,
    inv: Out<Tensor<i32>>) -> Result<(), Refusal> {
    // BACK TO AN ASK, BECAUSE THIS ROUTINE'S PARAMS RUN IS A STRUCT.
    // The body forwards `ctx.params()` whole -- the shader reads fields,
    // not a scalar run -- so slot 0 is the struct's first field and a
    // `Const` derived onto it reads that field's bits. HEAD spelled these
    // `Ask<..>` for exactly this reason, and the drivers still answer them.
    let n_experts = ctx.ask::<i32, keys::NumExperts>()?;
    let params = ctx.params()?;
    let w = router_lanes(n_experts)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "route_sort").apply(Grid::of([w, 1, 1], [w, 1, 1])),
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
#[routine]
pub fn route_gather(
    ctx: &Ctx<'_>,
    x: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    perm: In<Tensor<i32>>) -> Result<(), Refusal> {
    let params = ctx.params()?;
    let width = x.width;
    let padded = ctx.ask::<i32, keys::Rows>()?;
    let (lanes, group) = route_rows(width, padded)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "route_gather").apply(Grid::of(lanes, group)),
        &[x.arg(), out.arg(), perm.arg(), params],
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
    let (lanes, group) = route_rows(width, tokens)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "combine_sorted").apply(Grid::of(lanes, group)),
        &[y.arg(), expert_weights.arg(), out.arg(), params, inv.arg()],
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
#[routine]
pub fn shared_expert_combine(
    ctx: &Ctx<'_>,
    routed: In<Tensor<bf16>>,
    shared: In<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let width = routed.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let (lanes, group) = route_rows((width).try_into().unwrap_or(i32::MAX), rows)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "shared_expert_combine").apply(Grid::of(lanes, group)),
        &[routed.arg(), shared.arg(), gate.arg(), out.arg(), width.arg()],
    )
}

/// [`shared_expert_combine`] over rows a `row_pitch` apart.
///
/// # Errors
///
/// See [`route_rows`].
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
    let (lanes, group) = route_rows((width).try_into().unwrap_or(i32::MAX), rows)?;
    ctx.fire(
        Fire::at(ROUTE_FILE, "shared_expert_combine_strided").apply(Grid::of(lanes, group)),
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
/// See [`routed_qmv_grid`] and [`routed_qmv_widths`].
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
    // `in_vec_size` and `out_vec_size` stood before them and left, and the
    // note here said they were `x.width` and `y.width` "which the marks
    // carry". They are NOT: a mark carries a whole statement's run and this
    // kernel walks one slot of it. See `routed_qmv_widths`.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let bias = ctx.absent()?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMV_FILE, "affine_qmv_routed_bfloat16_gs_64_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, SIMD_PAIR)),
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

/// [`qmv_routed`] with a per-output-row bias.
///
/// # Errors
///
/// See [`routed_qmv_grid`] and [`routed_qmv_widths`].
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
    // `in_vec_size` and `out_vec_size` stood before them and left, and the
    // note here said they were `x.width` and `y.width` "which the marks
    // carry". They are NOT: a mark carries a whole statement's run and this
    // kernel walks one slot of it. See `routed_qmv_widths`.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMV_FILE, "affine_qmv_routed_bias_bfloat16_gs_64_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, SIMD_PAIR)),
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
/// See [`routed_qmv_grid`] and [`routed_qmv_widths`].
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
    // `in_vec_size` and `out_vec_size` stood before them and left, and the
    // note here said they were `x.width` and `y.width` "which the marks
    // carry". They are NOT: a mark carries a whole statement's run and this
    // kernel walks one slot of it. See `routed_qmv_widths`.
    x_slot_stride: Const<i32>,
    x_row_stride: Const<i32>,
    slots_per_row: Const<i32>,
    expert_ids: In<Tensor<i32>>) -> Result<(), Refusal> {
    let biases = ctx.absent()?;
    let (in_vec_size, out_vec_size) =
        routed_qmv_widths(*x_slot_stride, y.width, *slots_per_row)?;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    ctx.fire(
        Fire::at(QMV_FILE, "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4").apply(Grid::of(routed_qmv_grid(rows, out_vec_size, *slots_per_row)?, SIMD_PAIR)),
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
    tile_n: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let point = affine_qmm_point(*group, *bits, *tile_m, *tile_n)?;
    ctx.fire(
        Fire::at(QMM_FILE, AFFINE_QMM[point]).apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, QMM_GROUP)),
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
    tile_n: Const<i32>) -> Result<(), Refusal> {
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.fire(
        Fire::at(QMM_FILE, FP16_QMM[point]).apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, QMM_GROUP)),
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
#[routine]
pub fn mxfp4_qmm_t_routed_bias(
    ctx: &Ctx<'_>,
    w: Const<Tensor<u32>>,
    exponents: Const<Tensor<u8>>,
    // THE ORDER IS THE SLOT: `x` is the statement's input 0, `pad` its
    // input 1 and `tile_expert` its input 2. `InSlot<0, _>`/`InSlot<1, _>`
    // used to say that against the declaration order; the marks say it by
    // sitting in it.
    x: In<Tensor<bf16>>,
    pad: In<Tensor<bf16>>,
    y: Out<Tensor<bf16>>,
    bias: Const<Tensor<bf16>>,
    tile_expert: In<Tensor<i32>>,
    tile_m: Const<i32>,
    tile_n: Const<i32>) -> Result<(), Refusal> {
    // THE CONTRACTION IS THE ACTIVATION'S WIDTH, AND IT IS NOT THE PAD'S.
    //
    // This read `pad.width`, and `pad` is the filler bound into the holes this
    // entrypoint leaves at 2 and 8..=11 -- `model-dsl` rides `row_expert`
    // there, an `i32` value ONE element wide, because a real buffer in a hole
    // keeps the operand list the same length as the matvec's. So every fire of
    // gpt-oss's routed GEMM was handed `K = 1` where the bank is 2880 wide,
    // and the affine twins two routines above have always said `x.width`.
    //
    // What that cost is worse than a truncated dot product, because `K` is the
    // ROW STRIDE as well as the loop bound: `qmm_t_cast_loaded_impl` starts
    // its activation loader at `x + y_row * K`, so a tile at sorted row 160
    // read from ELEMENT 160 of row zero instead of from row 160, and the
    // answer for a (token, expert) pair moved whenever the sorted stack put it
    // at a different offset. Adding a token to a prefill moves it. So does
    // adding a SECOND REQUEST to the fire, which is why two conversations
    // batched together did not prefill the way either of them prefills alone.
    //
    // It stayed hidden because the routed MATVEC arm, which is correct, is
    // what a one-token decode selects; only a prefill wide enough to earn the
    // GEMM ever asked this question.
    let k = x.width;
    let n = y.width;
    let rows = ctx.ask::<i32, keys::Rows>()?;
    let point = tile_point(*tile_m, *tile_n)?;
    ctx.fire(
        Fire::at(QMM_FILE, MXFP4_QMM[point]).apply(Grid::of(routed_qmm_grid(rows, n, *tile_m, *tile_n)?, QMM_GROUP)),
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

/// Two simdgroups of 32, which is what a matvec threadgroup is here: four
/// output rows to a simdgroup, eight to the group.
const SIMD_PAIR: [u32; 3] = [32, 2, 1];

/// The GEMM's threadgroup: four simdgroups, arranged `(32, 2, 2)` because the
/// third axis is the two halves of the K walk and not a third tile.
const QMM_GROUP: [u32; 3] = [32, 2, 2];

const ROUTE_FILE: &str = "moe/route.metal";
const QMV_FILE: &str = "quant/qmv.metal";
const QMM_FILE: &str = "quant/qmm_t.metal";


#[cfg(test)]
mod tests {
    use super::*;
    use crate::routine::{ArgValue, Const, Encode, Tensor};
    use core::cell::{Cell, RefCell};

    /// One recorded dispatch: the fire, and the argument list.
    type Call = (Fire, Vec<ArgValue>);

    /// An `Encode` that remembers what it was asked to do, and answers the
    /// three facts this file's TESTED bodies ask for: the staged scalar
    /// block every one of them but `qmm_t_routed`/`mxfp4_qmm_t_routed_bias`
    /// takes with `ctx.params()`, `router_topk`'s unread per-expert scale
    /// with `ctx.absent()`, and `Rows` -- which `router_topk`, the gather,
    /// the combine and both routed GEMMs all ask under that one name whether
    /// the caller thinks of it as a row count, a padded stack, or a fire's
    /// own token count.
    struct Seen {
        calls: RefCell<Vec<Call>>,
        params_handle: Cell<u32>,
        /// THE STATEMENT\'S SCALAR RUN, for a body that reads a word by
        /// index. Empty means "4096 at every slot", which is a plausible
        /// stride for the rows these tests build; a case that means a
        /// particular tiling or split count sets its own.
        words: RefCell<Vec<i32>>,
        absent_handle: Cell<u32>,
        rows: Cell<i32>,
        n_experts: Cell<i32>,
    }

    impl Default for Seen {
        fn default() -> Self {
            Self {
                calls: RefCell::default(),
                params_handle: Cell::new(800),
                words: RefCell::default(),
                absent_handle: Cell::new(900),
                rows: Cell::new(4),
                n_experts: Cell::new(60),
            }
        }
    }

    impl Encode for Seen {
        fn resolve(&self, _ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
            use kernels::keys::Fact;
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
                return Ok(ArgValue::Buffer(self.params_handle.get()));
            }
            if source == kernels::Source::Lit(kernels::Lit::Null) {
                return Ok(ArgValue::Buffer(self.absent_handle.get()));
            }
            // The geometry these bodies read now that their params run is a
            // STRUCT and no slot in it is theirs to take.
            if source == <keys::NumExperts as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.n_experts.get()));
            }
            if source == <keys::Rows as Fact>::SOURCE {
                return Ok(ArgValue::I32(self.rows.get()));
            }
            // Anything else is refused: a probe that invented an answer to a
            // fact it does not know would let a body pass under test while
            // the same fact went unanswered on a real driver.
            Err(Refusal::Unstated { what: "a fact this probe does not answer" })
        }

        fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
            self.calls.borrow_mut().push((fire, args.to_vec()));
            Ok(())
        }
    }

    /// The routed GEMM contracts over the ACTIVATION's width, and not over
    /// the width of the value riding its pad slot.
    ///
    /// `mxfp4_qmm_t_routed_bias` read `pad.width`, and `model-dsl` rides
    /// `row_expert` there -- ONE element wide -- so gpt-oss's mixture ran
    /// every routed projection with `K = 1` against a 2880-wide bank. `K` is
    /// the activation's row stride as well as its loop bound, so a tile at
    /// sorted row 160 read from element 160 of row zero: the answer for one
    /// (token, expert) pair moved when anything moved that pair's place in
    /// the sorted stack, which a longer prefill does and a second request in
    /// the same fire does.
    ///
    /// The two affine forms beside it always said `x.width`, which is why
    /// this asserts all three together.
    #[test]
    fn a_routed_gemm_contracts_over_the_activations_width() {
        let seen = Seen::default();
        seen.rows.set(32);
        let x = || In { ptr: Tensor::<bf16>::new(4), rows: 32, width: 2880 };
        let pad = || In { ptr: Tensor::<bf16>::new(99), rows: 32, width: 1 };
        let y = || Out { ptr: Tensor::<bf16>::new(5), rows: 32, width: 32 };
        qmm_t_routed(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            x(),
            y(),
            pad(),
            In::new(Tensor::<i32>::new(6)),
            Const::new(64),
            Const::new(4),
            Const::new(32),
            Const::new(32))
        .expect("a launch");
        qmm_t_routed_fp16(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            x(),
            y(),
            pad(),
            In::new(Tensor::<i32>::new(6)),
            Const::new(32),
            Const::new(32))
        .expect("a launch");
        mxfp4_qmm_t_routed_bias(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<u8>::new(2)),
            x(),
            pad(),
            y(),
            Const::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<i32>::new(6)),
            Const::new(32),
            Const::new(32))
        .expect("a launch");

        let calls = seen.calls.borrow();
        assert_eq!(calls.len(), 3, "three routed GEMMs fired");
        for (fire, args) in calls.iter() {
            assert_eq!(
                args[5],
                ArgValue::I32(2880),
                "`{}` contracts over the activation's 2880 and not the pad's 1",
                fire.entrypoint
            );
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
        seen.rows.set(32);
        let pad = Tensor::<bf16>::new(99);
        qmm_t_routed(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<bf16>::new(2)),
            Const::new(Tensor::<bf16>::new(3)),
            In::new(Tensor::<bf16>::new(4)),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 32 },
            In::new(pad),
            In::new(Tensor::<i32>::new(6)),
            Const::new(64),
            Const::new(4),
            Const::new(32),
            Const::new(32))
        .expect("a launch");
        mxfp4_qmm_t_routed_bias(
            &seen,
            Const::new(Tensor::<u32>::new(1)),
            Const::new(Tensor::<u8>::new(2)),
            In::new(Tensor::<bf16>::new(4)),
            In::new(pad),
            Out { ptr: Tensor::<bf16>::new(5), rows: 0, width: 32 },
            Const::new(Tensor::<bf16>::new(7)),
            In::new(Tensor::<i32>::new(6)),
            Const::new(32),
            Const::new(32))
        .expect("a launch");

        let calls = seen.calls.borrow();
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
        seen.rows.set(7);
        router_topk(
            &seen,
            In::new(Tensor::<bf16>::new(1)),
            Out::new(Tensor::<i32>::new(2)),
            Out::new(Tensor::<bf16>::new(3)))
        .expect("a launch");
        let calls = seen.calls.borrow();
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
            In::new(Tensor::<i32>::new(1)),
            Out::new(Tensor::<i32>::new(2)),
            Out::new(Tensor::<i32>::new(3)),
            Out::new(Tensor::<i32>::new(4)),
            Out::new(Tensor::<i32>::new(6)))
        .expect("a launch");
        let calls = seen.calls.borrow();
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
        seen.rows.set(64);
        route_gather(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            Out::new(Tensor::<bf16>::new(2)),
            In::new(Tensor::<i32>::new(3)))
        .expect("a launch");
        seen.rows.set(16);
        combine_sorted(
            &seen,
            In { ptr: Tensor::<bf16>::new(1), rows: 0, width: 2048 },
            In::new(Tensor::<bf16>::new(2)),
            Out::new(Tensor::<bf16>::new(3)),
            In::new(Tensor::<i32>::new(5)))
        .expect("a launch");
        let calls = seen.calls.borrow();
        assert_eq!(calls[0].0.lanes, [2048, 64, 1], "the padded stack");
        assert_eq!(calls[1].0.lanes, [2048, 16, 1], "the fire's own tokens");
        assert_eq!(calls[0].0.group, [256, 1, 1]);
    }
}
