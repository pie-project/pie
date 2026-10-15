#![cfg(feature = "cuda")]

//! DeepSeek-V4's clamped SwiGLU of routed rows, each times its route's
//! weight before it rounds: over packed rows (`mlp_swiglu_clamp_routed`, a
//! row's gate columns then its up columns) it lands exactly what the split
//! form (`mlp_swiglu_clamp_split_routed`, gate and up planes apart) lands,
//! and both answer the host.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::mlp;
use kernels_cuda::tensor::Tensor;

#[test]
fn the_packed_routed_glu_answers_the_split_one() {
    let (tokens, top, inter, limit) = (5usize, 6usize, 320usize, 10.0f32);
    let rows = tokens * top;
    let mut lcg = Lcg::seeded(0x91f);
    let gate: Vec<u16> = (0..rows * inter)
        .map(|_| to_bf16(lcg.unit() * 14.0))
        .collect();
    let up: Vec<u16> = (0..rows * inter)
        .map(|_| to_bf16(lcg.unit() * 14.0))
        .collect();
    let weights: Vec<f32> = (0..rows).map(|_| lcg.unit().abs() + 0.01).collect();
    let mut packed = Vec::with_capacity(rows * 2 * inter);
    for r in 0..rows {
        packed.extend_from_slice(&gate[r * inter..(r + 1) * inter]);
        packed.extend_from_slice(&up[r * inter..(r + 1) * inter]);
    }
    let mut gpu = Gpu::open();
    let bf = |at: u64, width: usize| Tensor::new(at, rows as u32, width as u32, Dtype::Bf16);
    let gate_t = bf(gpu.up(&gate), inter);
    let up_t = bf(gpu.up(&up), inter);
    let packed_t = bf(gpu.up(&packed), 2 * inter);
    let weights_t = Tensor::new(gpu.up(&weights), tokens as u32, top as u32, Dtype::F32);
    let split_at = gpu.zeros(rows * inter * 2);
    let packed_at = gpu.zeros(rows * inter * 2);
    let ctx = gpu.ctx();
    mlp::swiglu_clamp_split_routed(
        &ctx,
        gate_t,
        up_t,
        weights_t,
        1,
        limit,
        &mut bf(split_at, inter),
    )
    .expect("the split form fires");
    mlp::swiglu_clamp_routed(
        &ctx,
        packed_t,
        inter as u32,
        weights_t,
        1,
        limit,
        &mut bf(packed_at, inter),
    )
    .expect("the packed form fires");
    gpu.sync();
    let split: Vec<u16> = gpu.down(split_at, rows * inter);
    let got: Vec<u16> = gpu.down(packed_at, rows * inter);
    assert_eq!(got, split, "the packed form lands the split form's bits");
    for r in 0..rows {
        for i in 0..inter {
            let g = from_bf16(gate[r * inter + i]).min(limit);
            let u = from_bf16(up[r * inter + i]).clamp(-limit, limit);
            let want = g / (1.0 + (-g).exp()) * u * weights[r];
            let have = from_bf16(got[r * inter + i]);
            assert!(
                (have - want).abs() <= want.abs() / 128.0 + 1e-3,
                "row {r} column {i}: {have} where the host says {want}"
            );
        }
    }
}
