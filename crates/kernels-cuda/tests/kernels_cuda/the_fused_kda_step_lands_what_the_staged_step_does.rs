#![cfg(feature = "cuda")]

//! The KDA decode step that reads its packed rows itself
//! (`kernel.ssm.kda_step_fused`: a block a row, head and slice of the
//! state's value rows) lands what the staged step lands: every output row
//! and every slot's state, a lane whose state does not fold left as it was,
//! at one row and at a batch, under a gate floor and without one.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::ssm;
use kernels_cuda::tensor::{RecurrentPool, Tensor};

const HEAD_DIM: u32 = 128;

fn step(
    gpu: &mut Gpu,
    rows: usize,
    heads: u32,
    mask: Option<Vec<u8>>,
    floor: f32,
    fused: bool,
) -> (Vec<f32>, Vec<f32>) {
    let wide = (heads * HEAD_DIM) as usize;
    let stride = (heads * HEAD_DIM * HEAD_DIM) as usize;
    let mut lcg = Lcg::seeded(0x5e9 ^ rows as u64);
    let (mixed, _) = lcg.row(rows * 3 * wide);
    let (f, _) = lcg.row(rows * wide);
    let (b, _) = lcg.row(rows * heads as usize);
    let dt_bias: Vec<f32> = (0..wide).map(|_| 0.1 * lcg.unit()).collect();
    let a_log: Vec<f32> = (0..heads).map(|_| -0.5 + 0.2 * lcg.unit()).collect();
    let slab: Vec<f32> = (0..rows * stride).map(|_| 0.05 * lcg.unit()).collect();
    let slots: Vec<i32> = (0..rows as i32).rev().collect();
    let slab_at = gpu.up(&slab);
    let y_at = gpu.zeros(rows * wide * 4);
    let mask = match mask {
        Some(v) => Tensor::new(gpu.up(&v), v.len() as u32, 1, Dtype::U8),
        None => Tensor::ABSENT,
    };
    let pool = RecurrentPool {
        slab: Tensor::new(slab_at, rows as u32, stride as u32, Dtype::F32),
        slot_ids: Tensor::new(gpu.up(&slots), rows as u32, 1, Dtype::I32),
        slot_stride_elems: stride as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: mask,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
    };
    let mut y = Tensor::new(y_at, rows as u32, wide as u32, Dtype::F32);
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        ssm_kda_step_fused: fused,
        ..kernels_cuda::tuning::Tuning::DEFAULT
    });
    ssm::kda_step(
        &ctx,
        Tensor::new(gpu.up(&mixed), rows as u32, 3 * wide as u32, Dtype::Bf16),
        Tensor::new(gpu.up(&f), rows as u32, wide as u32, Dtype::Bf16),
        Tensor::new(gpu.up(&b), rows as u32, heads, Dtype::Bf16),
        Tensor::new(gpu.up(&dt_bias), 1, wide as u32, Dtype::F32),
        Tensor::new(gpu.up(&a_log), 1, heads, Dtype::F32),
        &pool,
        heads,
        HEAD_DIM,
        1e-6,
        floor,
        &mut y,
    )
    .expect("the KDA step fires");
    gpu.sync();
    (
        gpu.down(y_at, rows * wide),
        gpu.down(slab_at, rows * stride),
    )
}

fn near(what: &str, got: &[f32], want: &[f32]) {
    let scale = want.iter().map(|w| w.abs()).fold(0f32, f32::max).max(1e-3);
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    assert!(
        worst <= 1e-5 * scale,
        "{what}: the fused step is {worst} from the staged one (scale {scale})"
    );
}

#[test]
fn the_fused_kda_step_lands_what_the_staged_step_does() {
    let mut gpu = Gpu::open();
    for (rows, heads, mask, floor) in [
        (1, 96, None, -5.0),
        (8, 4, Some(vec![1, 0, 1, 1, 0, 1, 1, 1]), -5.0),
        (3, 2, None, 0.0),
    ] {
        let (y0, s0) = step(&mut gpu, rows, heads, mask.clone(), floor, false);
        let (y1, s1) = step(&mut gpu, rows, heads, mask.clone(), floor, true);
        near(&format!("{rows} rows x {heads} heads outputs"), &y1, &y0);
        near(&format!("{rows} rows x {heads} heads states"), &s1, &s0);
    }
}
