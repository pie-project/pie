#![cfg(feature = "cuda")]

//! A grouped projection whose routes send each row's group g to plane g
//! (DeepSeek-V4's grouped output projection: 8 groups of 4096 into 1024) as
//! one strided batched GEMM: every row's every group against its plane, as
//! an f32 walk over the same bf16 values lands it.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::matmul_grouped_strided;
use kernels_cuda::tensor::Tensor;

fn check(rows: usize, groups: usize, k: usize, n: usize) {
    let mut lcg = Lcg::seeded(0x9a7 + rows as u64);
    let (x_bits, x) = lcg.row(rows * groups * k);
    let (w_bits, w) = lcg.row(groups * n * k);
    let mut gpu = Gpu::open();
    let xt = Tensor::new(
        gpu.up(&x_bits),
        rows as u32,
        (groups * k) as u32,
        Dtype::Bf16,
    );
    let wt = Tensor::new(gpu.up(&w_bits), (groups * n) as u32, k as u32, Dtype::Bf16);
    let y_at = gpu.zeros(rows * groups * n * 2);
    let mut yt = Tensor::new(y_at, rows as u32, (groups * n) as u32, Dtype::Bf16);
    matmul_grouped_strided(&gpu.ctx(), xt, wt, &mut yt, groups as u32)
        .expect("the grouped GEMM fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, rows * groups * n);
    let mut worst = 0f32;
    for r in 0..rows {
        for g in 0..groups {
            for j in 0..n {
                let mut acc = 0f32;
                for i in 0..k {
                    acc += x[(r * groups + g) * k + i] * w[(g * n + j) * k + i];
                }
                let v = from_bf16(got[(r * groups + g) * n + j]);
                worst = worst.max((v - acc).abs() / acc.abs().max(1.0));
            }
        }
    }
    assert!(
        worst <= 2e-2,
        "{rows} rows, {groups} groups of [{n}, {k}]: {worst} relative"
    );
}

#[test]
fn the_strided_grouped_projection_answers_each_group() {
    check(1, 8, 512, 128);
    check(37, 4, 256, 64);
}
