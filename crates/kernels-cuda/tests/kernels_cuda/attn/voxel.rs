#![cfg(feature = "cuda")]

//! Spatial attention over whole lanes answers the host reference at each
//! width the kernel instantiates.

use crate::common::host::spatial_attention_ref;
use crate::common::spatial::{Box3, table};
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dense::voxel::{Segment, attention};
use kernels_cuda::tensor::Tensor;

fn check(c: usize) {
    let boxes = [Box3::new(1, 5, 7), Box3::new(2, 3, 4)];
    let (grid, live) = table(&boxes);
    let rows = live + 5;
    let scale = (c as f32).sqrt().recip();
    let mut lcg = Lcg::seeded(0xa77e ^ c as u64);
    let (q_raw, q) = lcg.row(rows * c);
    let (k_raw, k) = lcg.row(rows * c);
    let (v_raw, v) = lcg.row(rows * c);
    let want = spatial_attention_ref([&q, &k, &v], &boxes, c, 0, scale * 4.0);

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let grid_at = gpu.up(&grid);
    let y_at = gpu.zeros(rows * c * 2);
    let t = |at: u64| Tensor::new(at, rows as u32, c as u32, Dtype::Bf16);
    let mut y = t(y_at);
    attention(
        &gpu.ctx(),
        t(q_at),
        t(k_at),
        t(v_at),
        Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32),
        Segment::Lane,
        scale * 4.0,
        &mut y,
        0,
    )
    .unwrap_or_else(|why| panic!("C {c}: {why}"));
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(y_at, rows * c)
        .into_iter()
        .map(from_bf16)
        .collect();

    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 + 1e-2 * w.abs(),
            "C {c}: row {} channel {}: got {g}, want {w}",
            i / c,
            i % c
        );
    }
    assert!(
        got[live * c..].iter().all(|x| *x == 0.0),
        "C {c}: the padded rows past the last lane land zeros"
    );
}

#[test]
fn the_spatial_attention_answers_the_cpu_reference_every_case() {
    for c in [64, 128, 256, 512, 640, 1024] {
        check(c);
    }
}
