//! **NVFP4 WEIGHT-ONLY**: `g16_e2m1_gt_e4m3_f32_n_n`, decoded inside the dot
//! against a bf16/f16 activation. NVIDIA ships official checkpoints in this
//! row and the hardware that decodes it natively is sm120 only, so the
//! decode-in-dot skeleton is what serves it on every other card — the
//! `linear::fp8` twin's bargain, at four bits (wiki alto/next.md §J2).
//!
//! Three factors reach one weight and they are stored three different ways:
//! the e2m1 CODE, an e4m3 scale per SIXTEEN codes, and one f32 for the whole
//! tensor. The first two arrive as planes; the third arrives as an argument,
//! because it is one number and a plane would cost a load per block to say it.
//!
//! **THE PLANE WIDTHS ARE THE FORM'S OWN ALGEBRA.** `Dtype::Nvfp4`
//! states `plane_widths(4096) == [2048, 256, 4]`, and the two widths this
//! entry checks are that statement at any `k`: `k/2` code bytes a row, `k/16`
//! scale bytes a row. A caller whose planes are a different rectangle is
//! refused rather than served a plausible number off the wrong bytes.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/nvfp4.cuh";

const WARP: u32 = 32;

/// Codes under one e4m3 scale — the `g16` in the form's name.
const GROUP: u32 = 16;

/// Weight rows a warp folds at once, and the lanes a block spends on them:
/// `linear::fp8`'s geometry, kept so the decode-in-dot points tile alike.
const ROWS_PER_WARP: u32 = 4;
const BLOCK_LANES: u32 = 128;

/// `linear.matmul` over a weight the store seats as `g16_e2m1_gt_e4m3_f32_n_n`
/// — e2m1 codes at `k/2` bytes a row, an e4m3 scale per sixteen of them at
/// `k/16` bytes a row, and `tensor_scale` over the whole weight.
pub fn matmul(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", act, codes, scales, tensor_scale, y)
}

/// [`matmul`] under the head's own op name, `linear::gemm`'s pairing kept.
pub fn lm_head(
    ctx: &Ctx,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", act, codes, scales, tensor_scale, y)
}

/// The one launch behind both entries. A fire with no rows is the same silent
/// no-op the dense gemm keeps, and for the same capture reason.
fn fire(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: Tensor,
    scales: Tensor,
    tensor_scale: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(codes.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(scales.dtype, Dtype::U8, "a packed plane binds as bytes");
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    if k % GROUP != 0 {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {GROUP}-code nvfp4 groups"),
        ));
    }
    // Two e2m1 codes to the byte, so a row of codes is exactly half the
    // contraction and there is nothing else the width could mean.
    if codes.width != k / 2 {
        return Err(refuse(
            op,
            format!(
                "a {}-byte code row does not store a {k}-wide row of e2m1 nibbles",
                codes.width
            ),
        ));
    }
    if scales.width != k / GROUP {
        return Err(refuse(
            op,
            format!(
                "a {}-byte scale row is not one e4m3 per {GROUP} codes over a {k}-wide row",
                scales.width
            ),
        ));
    }
    if scales.rows != n {
        return Err(refuse(
            op,
            format!("a {}-row scale plane is not {n} rows over a {n}x{k} weight", scales.rows),
        ));
    }
    // **A NON-FINITE TENSOR SCALE IS REFUSED, NOT MULTIPLIED.** It is the one
    // factor that reaches EVERY output of the projection, so a NaN here does
    // not corrupt a row — it flattens the whole logit vector into a sampler
    // that answers nothing, with no fire having failed. Loudly here is the
    // only place it is still attributable.
    if !tensor_scale.is_finite() {
        return Err(refuse(
            op,
            format!("the tensor scale is {tensor_scale}, which every output would carry"),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let tile = (BLOCK_LANES / WARP) * ROWS_PER_WARP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::linear::matmul_nvfp4<{t}, ::pie::i32({ROWS_PER_WARP})>"
            )),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [BLOCK_LANES, 1, 1],
        )),
        &[
            act.arg(),
            codes.arg(),
            scales.arg(),
            y.arg(),
            tensor_scale.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
