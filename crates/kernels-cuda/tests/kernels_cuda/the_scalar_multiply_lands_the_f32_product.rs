#![cfg(feature = "cuda")]

//! `mul_scalar` scales an f32 plane exactly and a bf16 one to the rounded
//! product.

use crate::common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm::mul_scalar;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 7;
const WIDTH: usize = 5;
const SCALE: f32 = 1000.0 / 3.0;

#[test]
fn the_scalar_multiply_lands_the_f32_product_every_case() {
    let mut rng = Lcg::seeded(41);
    let x: Vec<f32> = (0..ROWS * WIDTH).map(|_| rng.unit()).collect();
    let mut gpu = Gpu::open();
    let at = gpu.up(&x);
    let ctx = gpu.ctx();
    mul_scalar(
        &ctx,
        SCALE,
        &mut Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::F32),
    )
    .expect("the f32 arm fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(at, ROWS * WIDTH);
    for (i, (g, x)) in got.iter().zip(&x).enumerate() {
        assert_eq!(
            g.to_bits(),
            (x * SCALE).to_bits(),
            "at {i}: {g} against {}",
            x * SCALE
        );
    }

    let mut rng = Lcg::seeded(43);
    let (bits, x) = rng.row(ROWS * WIDTH);
    let at = gpu.up(&bits);
    mul_scalar(
        &ctx,
        SCALE,
        &mut Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16),
    )
    .expect("the bf16 arm fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(at, ROWS * WIDTH);
    let scale = from_bf16(common::to_bf16(SCALE));
    for (i, (g, x)) in got.iter().zip(&x).enumerate() {
        let want = from_bf16(common::to_bf16(x * scale));
        assert_eq!(from_bf16(*g), want, "at {i}");
    }
}
