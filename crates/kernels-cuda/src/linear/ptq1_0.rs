use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "linear/quant_ptq1_0.cuh";

const WARP: u32 = 32;

/// Simdgroups (warps) per thread block, and the output rows each one computes:
/// the Metal `ptq1_0_qmv` geometry, a thread block of (32, 2, 1) threads.
const SIMDGROUPS: u32 = 2;

const ROWS_PER_SIMDGROUP: u32 = 4;

/// Weights per PTQ1_0 block; the fp16 scale is INLINE per block (single plane).
const PTQ1_0_BLOCK: u32 = 128;

pub fn matmul(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    ptq1_0(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx, act: Tensor, w: Tensor, y: &mut Tensor) -> Result<(), Error> {
    ptq1_0(ctx, "linear.lm_head", act, w, y)
}

fn ptq1_0(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let ti = dtype_dispatch!(op, act.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    // bf16 out is the production (`qwen_3 d27b`) decode; f32 out is the
    // bit-exact oracle read-back path.
    let to = dtype_dispatch!(op, y.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    debug_assert_eq!(
        w.dtype,
        Dtype::Ptq1_0,
        "a stored PTQ1_0 plane binds as ternary blocks"
    );
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    let n = nonzero(op, "N, the columns this projection lands", y.width)?;
    let k = nonzero(op, "K, the contraction this projection walks", act.width)?;
    debug_assert_eq!(w.rows, n, "one weight row per column this projection lands");
    if !k.is_multiple_of(PTQ1_0_BLOCK) {
        return Err(refuse(
            op,
            format!(
                "the contraction is {k}, not a whole number of {PTQ1_0_BLOCK}-weight ternary \
                 blocks: a PTQ1_0 row is `ceil(K/{PTQ1_0_BLOCK}) * 28` bytes and the kernel \
                 reads one 28-byte block per {PTQ1_0_BLOCK} contracted weights"
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let tile = SIMDGROUPS * ROWS_PER_SIMDGROUP;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!("::pie::linear::ptq1_0_qmv<{ti}, {to}>")),
        )
        .apply(Launch::grid(
            [y.rows, n.div_ceil(tile), 1],
            [WARP, SIMDGROUPS, 1],
        )),
        &[
            w.arg(),
            act.arg(),
            y.arg(),
            stated(op, k)?.arg(),
            stated(op, n)?.arg(),
            ctx.stage(),
        ],
    )
}
