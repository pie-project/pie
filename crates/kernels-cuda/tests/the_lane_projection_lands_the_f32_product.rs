#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
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
    the_f32_activation_lands_the_f32_product();
    a_bf16_activation_lands_a_bf16_product();
}

fn the_f32_activation_lands_the_f32_product() {
    let mut rng = Lcg::seeded(11);
    let act: Vec<f32> = (0..ROWS * K).map(|_| rng.unit()).collect();
    let (w_bits, w) = rng.row(N * K);
    let want = reference(&act, &w);

    let mut gpu = Gpu::open();
    let act_at = gpu.up(&act);
    let w_at = gpu.up(&w_bits);
    let y_at = gpu.zeros(ROWS * N * 4);
    let ctx = gpu.ctx();
    act_x_wt(
        &ctx,
        "linear.matmul",
        Tensor::new(act_at, ROWS as u32, K as u32, Dtype::F32),
        Tensor::new(w_at, N as u32, K as u32, Dtype::Bf16),
        &mut Tensor::new(y_at, ROWS as u32, N as u32, Dtype::F32),
    )
    .expect("the lane projection fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(y_at, ROWS * N);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= F32_TOLERANCE, "at {i}: {g} against {w}");
    }
}

fn a_bf16_activation_lands_a_bf16_product() {
    let mut rng = Lcg::seeded(23);
    let (act_bits, act) = rng.row(ROWS * K);
    let (w_bits, w) = rng.row(N * K);
    let want = reference(&act, &w);

    let mut gpu = Gpu::open();
    let act_at = gpu.up(&act_bits);
    let w_at = gpu.up(&w_bits);
    let y_at = gpu.zeros(ROWS * N * 2);
    let ctx = gpu.ctx();
    act_x_wt(
        &ctx,
        "linear.matmul",
        Tensor::new(act_at, ROWS as u32, K as u32, Dtype::Bf16),
        Tensor::new(w_at, N as u32, K as u32, Dtype::Bf16),
        &mut Tensor::new(y_at, ROWS as u32, N as u32, Dtype::Bf16),
    )
    .expect("the lane projection fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, ROWS * N);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let g = from_bf16(*g);
        assert!(
            (g - w).abs() <= BF16_TOLERANCE * w.abs().max(1.0),
            "at {i}: {g} against {w}"
        );
    }
    let _ = to_bf16;
}
