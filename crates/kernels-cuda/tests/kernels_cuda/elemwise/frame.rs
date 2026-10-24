#![cfg(feature = "cuda")]

//! The row frame around a hand-written `residual_add_rmsnorm` body lands the
//! hand register kernel's bits: the stream `y` summed in place and the normed
//! row, for a row of one to four vectors a thread, a ragged last vector a
//! thread, and the weight taken as is or as `1 + weight`.

use crate::common::attn_ref::{ten, tuned};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::Bf16;
use kernels_cuda::elemwise::{frame, norm_residual};
use kernels_cuda::jit::Ctx;
use kernels_cuda::tensor::Tensor;

/// The hand kernel's register form (`kernel.elemwise.rows_vec`).
fn hand(gpu: &Gpu) -> Ctx {
    gpu.ctx().with_tuning(tuned(|t| t.rows_vec = true))
}

type Fire = fn(
    &Ctx,
    Tensor,
    &mut Tensor,
    Tensor,
    bool,
    f32,
    &mut Tensor,
) -> Result<(), kernels_cuda::error::Error>;

/// `y` and the normed row after `fire` over one seeded draw.
fn land(fire: Fire, rows: usize, width: usize, plus_one: bool) -> (Vec<u16>, Vec<u16>) {
    let n = rows * width;
    let mut lcg = Lcg::seeded(0xf4a3e ^ ((width as u64) << 8) ^ rows as u64);
    let (x, _) = lcg.row(n);
    let (y, _) = lcg.row(n);
    let (w, _) = lcg.row(width);
    let mut gpu = Gpu::open();
    let (x_at, y_at, w_at, out_at) = (gpu.up(&x), gpu.up(&y), gpu.up(&w), gpu.zeros(n * 2));
    fire(
        &hand(&gpu),
        ten(x_at, rows, width, Bf16),
        &mut ten(y_at, rows, width, Bf16),
        ten(w_at, width, 1, Bf16),
        plus_one,
        1e-6,
        &mut ten(out_at, rows, width, Bf16),
    )
    .expect("the residual norm fires");
    gpu.sync();
    (gpu.down(y_at, n), gpu.down(out_at, n))
}

#[test]
fn the_framed_residual_norm_lands_the_register_kernels_bits() {
    for width in [4096usize, 7168, 8192, 2048, 1544, 6152] {
        for rows in [1usize, 7, 64] {
            for plus_one in [false, true] {
                let case = format!("{rows} rows of {width}, plus one {plus_one}");
                let (y, out) = land(frame::residual_add_rmsnorm, rows, width, plus_one);
                let (y_want, out_want) =
                    land(norm_residual::residual_add_rmsnorm, rows, width, plus_one);
                assert!(y == y_want, "{case}: the summed stream");
                assert!(out == out_want, "{case}: the normed row");
            }
        }
    }
}
