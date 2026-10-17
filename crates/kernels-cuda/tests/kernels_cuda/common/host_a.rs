//! Host references and fixtures several kernel tests share.
#![allow(dead_code)]

use super::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::tensor::Tensor;

/// `tokens` rows of `top` experts each, spread over `experts`.
pub fn routes_of(tokens: usize, experts: usize, top: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top).map(move |j| ((t * 5 + j * 7 + t / 3) % experts) as i32))
        .collect()
}

/// A zeroed bf16 `[rows][width]` tensor.
pub fn bf16_zeros(gpu: &mut Gpu, rows: usize, width: usize) -> Tensor {
    Tensor::new(
        gpu.zeros(rows * width * 2),
        rows as u32,
        width as u32,
        Dtype::Bf16,
    )
}

/// A random bf16 `[rows][width]` tensor, with its values.
pub fn bf16_rows(gpu: &mut Gpu, lcg: &mut Lcg, rows: usize, width: usize) -> (Tensor, Vec<f32>) {
    let (raw, exact) = lcg.row(rows * width);
    (
        Tensor::new(gpu.up(&raw), rows as u32, width as u32, Dtype::Bf16),
        exact,
    )
}
