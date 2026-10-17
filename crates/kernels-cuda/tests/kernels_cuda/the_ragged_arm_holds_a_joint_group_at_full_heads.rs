#![cfg(feature = "cuda")]

use crate::common::host_b::assert_near;
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn_dense;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

fn against_dense(rows: u32, hd: u32, heads: u32, seed: u64) {
    let table = vec![0i32, rows as i32];
    let rows = rows as usize;
    let width = (heads * hd) as usize;
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, _) = lcg.row(rows * width);
    let (k_raw, _) = lcg.row(rows * width);
    let (v_raw, _) = lcg.row(rows * width);
    let sm_scale = 1.0 / (hd as f32).sqrt();

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let o_dense = gpu.zeros(rows * width * 2);
    let o_ragged = gpu.zeros(rows * width * 2);
    let ctx = gpu.ctx();
    let q = Tensor::new(q_at, rows as u32, width as u32, Dtype::Bf16);
    let k = Tensor::new(k_at, rows as u32, width as u32, Dtype::Bf16);
    let v = Tensor::new(v_at, rows as u32, width as u32, Dtype::Bf16);
    let segments = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);

    let mut dense = Tensor::new(o_dense, rows as u32, width as u32, Dtype::Bf16);
    attn_dense::bidirectional(&ctx, q, k, v, segments, hd, sm_scale, &mut dense)
        .expect("the dense kernel fires");
    let mut ragged = Tensor::new(o_ragged, rows as u32, width as u32, Dtype::Bf16);
    attn_ragged::ragged(
        &ctx,
        q,
        k,
        v,
        segments,
        segments,
        hd,
        sm_scale,
        RaggedMask::None,
        &mut ragged,
    )
    .expect("the ragged arm fires");
    gpu.sync();

    let want: Vec<f32> = gpu
        .down::<u16>(o_dense, rows * width)
        .into_iter()
        .map(from_bf16)
        .collect();
    let got: Vec<u16> = gpu.down(o_ragged, rows * width);
    assert_near(
        &got,
        &want,
        TOLERANCE,
        &format!("{rows} rows, {heads} heads of {hd}"),
    );
}

/// One joint group of 512 + n rows at 24 full heads of 128.
#[test]
fn the_ragged_arm_holds_a_joint_group_at_full_heads_every_case() {
    for (n, seed) in [(4096, 0x4608), (1024, 0x1536), (2500, 0x3012)] {
        against_dense(512 + n, 128, 24, seed);
    }
}
