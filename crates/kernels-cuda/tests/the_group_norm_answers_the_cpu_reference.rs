#![cfg(feature = "cuda")]

mod common;

use common::spatial::{Box3, group_norm_ref, near, table};
use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::group_norm;
use kernels_cuda::tensor::Tensor;

fn check(boxes: &[Box3], c: usize, groups: usize, silu: bool) {
    let (grid, rows) = table(boxes);
    let mut lcg = Lcg::seeded(0x9a0 ^ rows as u64);
    let (_, x) = lcg.row(rows * c);
    let x: Vec<f32> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.5 * (i % c) as f32)
        .collect();
    let x_raw: Vec<u16> = x.iter().map(|&v| common::to_bf16(v)).collect();
    let x: Vec<f32> = x_raw.iter().map(|&v| from_bf16(v)).collect();
    let weight: Vec<f32> = (0..c).map(|_| 1.0 + 0.5 * lcg.unit()).collect();
    let bias: Vec<f32> = (0..c).map(|_| lcg.unit()).collect();
    let eps = 1e-5;
    let want = group_norm_ref(&x, boxes, c, groups, &weight, &bias, eps, silu);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let w_at = gpu.up(&weight);
    let b_at = gpu.up(&bias);
    let o_at = gpu.zeros(rows * c * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows as u32, c as u32, Dtype::Bf16);
    group_norm(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32),
        groups as u32,
        Tensor::new(w_at, c as u32, 1, Dtype::F32),
        Tensor::new(b_at, c as u32, 1, Dtype::F32),
        eps,
        silu,
        &mut o,
    )
    .expect("the norm fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows * c);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let g = from_bf16(g);
        assert!(
            near(g, w, 1.0 / 128.0, 1.0 / 128.0),
            "silu {silu}: row {} channel {} is {g} against {w}",
            i / c,
            i % c
        );
    }
}

#[test]
fn the_group_norm_answers_the_cpu_reference_every_case() {
    two_lanes_of_different_boxes_norm_separately();
    the_fused_silu_follows_the_affine();
    a_wide_lane_is_folded_across_its_moment_splits();
}

fn two_lanes_of_different_boxes_norm_separately() {
    check(&[Box3::new(2, 4, 5), Box3::new(3, 3, 4)], 16, 4, false);
}

fn the_fused_silu_follows_the_affine() {
    check(&[Box3::new(2, 4, 5), Box3::new(3, 3, 4)], 24, 8, true);
}

fn a_wide_lane_is_folded_across_its_moment_splits() {
    check(&[Box3::new(5, 40, 48), Box3::new(1, 8, 8)], 32, 8, true);
}
