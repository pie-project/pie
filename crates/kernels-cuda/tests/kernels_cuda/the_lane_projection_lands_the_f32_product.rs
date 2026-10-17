#![cfg(feature = "cuda")]

//! The lane projection (`act_x_wt`) of an f32 activation lands the f32
//! product, and of a bf16 one the bf16 product.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::lane_gemm::act_x_wt;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 5;
const K: usize = 96;
const N: usize = 40;

const F32_TOLERANCE: f32 = 2e-5;
const BF16_TOLERANCE: f32 = 2e-2;

fn reference(act: &[f32], w: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; ROWS * N];
    for r in 0..ROWS {
        for c in 0..N {
            let mut acc = 0f32;
            for i in 0..K {
                acc = act[r * K + i].mul_add(w[c * K + i], acc);
            }
            y[r * N + c] = acc;
        }
    }
    y
}

#[test]
fn the_lane_projection_lands_the_f32_product_every_case() {
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let mut rng = Lcg::seeded(if dtype == Dtype::F32 { 11 } else { 23 });
        let (act_bits, act) = rng.row(ROWS * K);
        let (w_bits, w) = rng.row(N * K);
        let want = reference(&act, &w);
        let mut gpu = Gpu::open();
        let (act_at, y_at) = if dtype == Dtype::F32 {
            (gpu.up(&act), gpu.zeros(ROWS * N * 4))
        } else {
            (gpu.up(&act_bits), gpu.zeros(ROWS * N * 2))
        };
        let w_at = gpu.up(&w_bits);
        act_x_wt(
            &gpu.ctx(),
            "linear.matmul",
            Tensor::new(act_at, ROWS as u32, K as u32, dtype),
            Tensor::new(w_at, N as u32, K as u32, Dtype::Bf16),
            &mut Tensor::new(y_at, ROWS as u32, N as u32, dtype),
        )
        .expect("the lane projection fires");
        gpu.sync();
        let (got, tolerance): (Vec<f32>, f32) = if dtype == Dtype::F32 {
            (gpu.down(y_at, ROWS * N), F32_TOLERANCE)
        } else {
            let bits: Vec<u16> = gpu.down(y_at, ROWS * N);
            (bits.into_iter().map(from_bf16).collect(), BF16_TOLERANCE)
        };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let slack = if dtype == Dtype::F32 {
                tolerance
            } else {
                tolerance * w.abs().max(1.0)
            };
            assert!((g - w).abs() <= slack, "{dtype:?} at {i}: {g} against {w}");
        }
    }
}
