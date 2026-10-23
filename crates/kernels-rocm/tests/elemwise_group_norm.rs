//! Group norm over a voxel grid against the host's f64 one: lanes norm
//! apart, with and without the SiLU after the affine, and a wide lane folds
//! its moments across splits.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/spatial.rs"]
mod spatial;

use common::{Gpu, bf16s, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::norm::group_norm;
use spatial::{Box3, group_norm_ref, near, table};

fn check(gpu: &Gpu, boxes: &[Box3], c: usize, groups: usize, silu: bool) {
    let (grid, rows) = table(boxes);
    // A per-channel offset gives every group its own mean.
    let x: Vec<f32> = bf16s(rows * c, 0x9a0 ^ rows as u64, 1.0)
        .iter()
        .enumerate()
        .map(|(i, v)| round_bf16(v + 0.5 * (i % c) as f32))
        .collect();
    let weight: Vec<f32> = values(c, 0x9a1, 0.5).iter().map(|v| 1.0 + v).collect();
    let bias = values(c, 0x9a2, 1.0);
    let eps = 1e-5;
    let want = group_norm_ref(&x, boxes, c, groups, &weight, &bias, eps, silu);

    let mut o = gpu.zeros(rows as u32, c as u32, Dtype::Bf16);
    group_norm(
        &gpu.ctx(),
        gpu.tensor(&x, rows as u32, c as u32, Dtype::Bf16),
        gpu.ints_shaped(&grid, boxes.len() as u32, 4),
        groups as u32,
        gpu.tensor(&weight, c as u32, 1, Dtype::F32),
        gpu.tensor(&bias, c as u32, 1, Dtype::F32),
        eps,
        silu,
        &mut o,
    )
    .expect("the norm fires");
    for (i, (&g, &w)) in gpu.read(&o).iter().zip(&want).enumerate() {
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
    let Some(gpu) = Gpu::open() else { return };
    let two = [Box3::new(2, 4, 5), Box3::new(3, 3, 4)];
    check(&gpu, &two, 16, 4, false);
    check(&gpu, &two, 24, 8, true);
    // 9600 voxels in the first lane: 38 splits merged by the finalize's
    // wavefront.
    check(
        &gpu,
        &[Box3::new(5, 40, 48), Box3::new(1, 8, 8)],
        32,
        8,
        true,
    );
}
