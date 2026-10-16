#![cfg(feature = "cuda")]

//! `swiglu_clamp_routed_e4m3`: DeepSeek-V4's routed GLU landing its rows as
//! the e4m3 1x128 read-back of the rows `swiglu_clamp_routed` lands, bit for
//! bit, and those rows read back again are themselves (so the W4A8 select
//! after it may read them as they are).

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{self, Format, Source};
use kernels_cuda::linear::mlp::{swiglu_clamp_routed, swiglu_clamp_routed_e4m3};
use kernels_cuda::tensor::Tensor;

const LIMIT: f32 = 10.0;

fn check(rows: usize, inter: usize, spread: f32) {
    let mut lcg = Lcg::seeded(0x5e1 + (rows * inter) as u64);
    let packed: Vec<u16> = (0..rows * 2 * inter)
        .map(|_| crate::common::to_bf16(lcg.unit() * spread))
        .collect();
    let weights: Vec<f32> = (0..rows).map(|_| lcg.unit().abs() + 0.05).collect();
    let mut gpu = Gpu::open();
    let packed_t = Tensor::new(
        gpu.up(&packed),
        rows as u32,
        (2 * inter) as u32,
        Dtype::Bf16,
    );
    let weights_t = Tensor::new(gpu.up(&weights), rows as u32, 1, Dtype::F32);
    let bytes = rows * inter * 2;
    let (plain_at, back_at, fused_at, again_at) = (
        gpu.zeros(bytes),
        gpu.zeros(bytes),
        gpu.zeros(bytes),
        gpu.zeros(bytes),
    );
    let ctx = gpu.ctx();
    let mut plain = Tensor::new(plain_at, rows as u32, inter as u32, Dtype::Bf16);
    swiglu_clamp_routed(
        &ctx,
        packed_t,
        inter as u32,
        weights_t,
        1,
        LIMIT,
        &mut plain,
    )
    .expect("the plain GLU fires");
    let fmt = Format::E4m3 { gran: 128 };
    act_quant::read_back(&ctx, "test", fmt, Source::rows(plain, None), back_at)
        .expect("the read-back fires");
    let mut fused = Tensor::new(fused_at, rows as u32, inter as u32, Dtype::Bf16);
    assert!(
        swiglu_clamp_routed_e4m3(
            &ctx,
            packed_t,
            inter as u32,
            weights_t,
            1,
            LIMIT,
            &mut fused
        )
        .expect("the read-back GLU fires"),
        "{rows} rows of {inter} are whole 128-column blocks"
    );
    act_quant::read_back(&ctx, "test", fmt, Source::rows(fused, None), again_at)
        .expect("the second read-back fires");
    gpu.sync();
    let back: Vec<u16> = gpu.down(back_at, rows * inter);
    let fused: Vec<u16> = gpu.down(fused_at, rows * inter);
    let again: Vec<u16> = gpu.down(again_at, rows * inter);
    assert_eq!(
        fused, back,
        "{rows} x {inter} (spread {spread}): the GLU's read-back is the plain rows' read-back"
    );
    assert_eq!(
        again, fused,
        "{rows} x {inter} (spread {spread}): read back again, the rows are themselves"
    );
}

#[test]
fn the_routed_glu_lands_its_e4m3_read_back() {
    // DeepSeek-V4-Flash at one and eight requests (six routes each).
    check(6, 2048, 4.0);
    check(48, 2048, 4.0);
    // Odd rows, values past the clamp, and tiny ones near the amax floor.
    check(5, 384, 40.0);
    check(7, 256, 1e-3);
}
