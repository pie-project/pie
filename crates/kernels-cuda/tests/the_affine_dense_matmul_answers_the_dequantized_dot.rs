//! `linear.matmul` over an MLX affine bank (group 64, bf16 scale + zero
//! point per group) at 8 and at 4 bits lands the dot with the DEQUANTIZED
//! rows — at gemma-4-26B-A4B's router shape (`[128, 2816]`, the one U8g64
//! plane in that text) and at a 4-bit projection's.

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant::{OffsetKind, matmul};
use kernels_cuda::tensor::Tensor;

const GROUP: usize = 64;

fn check(bits: u32, n: usize, k: usize, rows: usize, seed: u64) {
    let mut lcg = Lcg::seeded(seed);
    let groups = k / GROUP;
    let levels = ((1u32 << bits) - 1) as f32;
    let per_byte = (8 / bits) as usize;
    let mut codes = vec![0u8; n * k / per_byte];
    let mut scales = vec![0u16; n * groups];
    let mut biases = vec![0u16; n * groups];
    let mut dequant = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..groups {
            let values: Vec<f32> = (0..GROUP).map(|_| lcg.unit()).collect();
            let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = from_bf16(to_bf16(((hi - lo) / levels).max(1e-6)));
            let bias = from_bf16(to_bf16(lo));
            scales[r * groups + g] = to_bf16(scale);
            biases[r * groups + g] = to_bf16(bias);
            for (j, &v) in values.iter().enumerate() {
                let code = (((v - bias) / scale).round().clamp(0.0, levels)) as u32;
                let col = g * GROUP + j;
                let byte = r * (k / per_byte) + col / per_byte;
                let shift = bits as usize * (col % per_byte);
                codes[byte] |= (code << shift) as u8;
                dequant[r * k + col] = code as f32 * scale + bias;
            }
        }
    }
    let (x_raw, x) = lcg.row(rows * k);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let biases_at = gpu.up(&biases);
    let y_at = gpu.zeros(rows * n * 2);
    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, n as u32, (k / per_byte) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, n as u32, (groups * 2) as u32, Dtype::U8);
    let biases_t = Tensor::new(biases_at, n as u32, (groups * 2) as u32, Dtype::U8);
    let mut y_t = Tensor::new(y_at, rows as u32, n as u32, Dtype::Bf16);
    matmul(
        &ctx,
        x_t,
        codes_t,
        scales_t,
        OffsetKind::Post,
        Some(biases_t),
        Dtype::Bf16,
        &mut y_t,
        GroupSeat::RESIDENT,
    )
    .expect("the matmul fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, rows * n);

    let mut bad = 0usize;
    let mut worst = 0f32;
    for t in 0..rows {
        for r in 0..n {
            let mut want = 0f32;
            for c in 0..k {
                want += dequant[r * k + c] * x[t * k + c];
            }
            let g = from_bf16(got[t * n + r]);
            worst = worst.max((g - want).abs() / want.abs().max(1.0));
            if !close(g, want) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("row {t} col {r}: got {g} want {want}");
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bits}-bit: {bad} of {} outputs differ (worst relative error {worst:.4})", rows * n);
}

#[test]
fn the_router_shape_at_eight_bits() {
    check(8, 128, 2816, 26, 0x61);
}

#[test]
fn a_projection_at_four_bits() {
    check(4, 256, 2816, 26, 0x62);
}
