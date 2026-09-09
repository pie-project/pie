#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
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

    let want: Vec<u16> = gpu.down(o_dense, rows * width);
    let got: Vec<u16> = gpu.down(o_ragged, rows * width);
    let mut worst = 0f32;
    let mut worst_row = 0;
    let mut bad_rows = 0;
    for r in 0..rows {
        let mut row_bad = false;
        for c in 0..width {
            let (g, w) = (from_bf16(got[r * width + c]), from_bf16(want[r * width + c]));
            let err = (g - w).abs();
            if err > TOLERANCE * w.abs().max(1.0) {
                row_bad = true;
            }
            if err > worst {
                worst = err;
                worst_row = r;
            }
        }
        bad_rows += usize::from(row_bad);
    }
    eprintln!("{rows} rows, {heads} heads of {hd}: worst |diff| {worst:.2e} at row {worst_row}, {bad_rows} rows past tolerance");
    assert_eq!(
        bad_rows, 0,
        "{rows} rows, {heads} heads of {hd}: {bad_rows} rows disagree with the dense kernel (worst {worst:.2e} at row {worst_row})"
    );
}

#[test]
fn the_ragged_arm_holds_a_joint_group_at_full_heads_every_case() {
    a_512_plus_4096_row_group_at_24_full_heads();
    a_512_plus_1024_row_group_at_24_full_heads();
    a_512_plus_2500_row_group_at_24_full_heads();
}

fn a_512_plus_4096_row_group_at_24_full_heads() {
    against_dense(512 + 4096, 128, 24, 0x4608);
}

fn a_512_plus_1024_row_group_at_24_full_heads() {
    against_dense(512 + 1024, 128, 24, 0x1536);
}

fn a_512_plus_2500_row_group_at_24_full_heads() {
    against_dense(512 + 2500, 128, 24, 0x3012);
}
