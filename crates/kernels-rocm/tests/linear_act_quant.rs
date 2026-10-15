//! DeepSeek's W8A8 product on the host's terms: activations quantized to
//! e4m3 under a ue8m0 scale a block (the amax floored at 1e-4, the scale
//! 2^ceil(log2(amax / 448)), round to nearest even), landed as their exact
//! bf16 values; and an fp8b128 projection (e4m3 codes under one e8m0
//! exponent a 128x128 block) against them, on the fused-dequant GEMV and on
//! the decoded tile and GEMM, against an f64 dot of the same values.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::linear::act_quant;

/// The value of OCP e4m3fn code `b`.
fn e4m3_value(b: u8) -> f32 {
    let (e, m) = (i32::from((b >> 3) & 0xF), f32::from(b & 7));
    let mag = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 { -mag } else { mag }
}

/// `v` (|v| ≤ 448) rounded to the nearest e4m3fn value, ties to even.
fn e4m3_round(v: f32) -> f32 {
    let a = v.abs();
    let r = if a < 2f32.powi(-6) {
        (a * 512.0).round_ties_even() / 512.0
    } else {
        let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
        let step = 2f32.powi(e - 3);
        (a / step).round_ties_even() * step
    };
    r.min(448.0).copysign(v)
}

/// A row's blocks quantized as the device does: the values `q · s`.
fn host_quantized(x: &[f32], k: usize, gran: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (r, row) in x.chunks(k).enumerate() {
        for b in 0..k / gran {
            let blk = &row[b * gran..(b + 1) * gran];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
            let bits = (amax * (1.0 / 448.0)).to_bits();
            let e = ((bits >> 23) & 0xFF) as i32 - 127 + i32::from(bits & 0x7F_FFFF != 0);
            let s = 2f32.powi(e);
            for (i, v) in blk.iter().enumerate() {
                out[r * k + b * gran + i] = e4m3_round((v / s).clamp(-448.0, 448.0)) * s;
            }
        }
    }
    out
}

/// Bytes on the device as a `[rows, width]` plane of `dtype`.
fn bytes(gpu: &Gpu, b: &[u8], rows: u32, width: u32, dtype: Dtype) -> Tensor {
    let mut padded = b.to_vec();
    padded.resize(b.len().div_ceil(4) * 4, 0);
    let words: Vec<i32> = padded
        .chunks(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let t = gpu.ints_shaped(&words, 1, words.len() as u32);
    Tensor::new(t.ptr, rows, width, dtype)
}

#[test]
fn e4m3_activations_land_their_exact_values() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for gran in [128usize, 32] {
        let (rows, k) = (7usize, 512usize);
        // Blocks of every magnitude, one all zero (the amax floor) and one of
        // values past e4m3's subnormals.
        let mut x: Vec<f32> = values(rows * k, 11, 1.0)
            .into_iter()
            .enumerate()
            .map(|(i, v)| round_bf16(v * 10f32.powi((i / gran) as i32 % 7 - 3)))
            .collect();
        x[..gran].fill(0.0);
        let xt = gpu.tensor(&x, rows as u32, k as u32, Dtype::Bf16);
        let mut y = gpu.zeros(rows as u32, k as u32, Dtype::Bf16);
        act_quant::e4m3_values(&ctx, "test", xt, gran as u32, &mut y).unwrap();
        let want = host_quantized(&x, k, gran);
        assert!(
            want.iter().all(|v| round_bf16(*v) == *v),
            "a quantized value is exact in bf16"
        );
        let have = gpu.read(&y);
        for (i, (h, w)) in have.iter().zip(&want).enumerate() {
            assert_eq!(h.to_bits(), w.to_bits(), "gran {gran} element {i}: {h} != {w} (x {})", x[i]);
        }
    }
}

/// `[m, k] · [n, k]ᵀ` for an fp8b128 weight against quantized activations.
fn run_fp8b128(m: usize, n: usize, k: usize, tile_holds: bool) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let mut seed = 0x9E37_79B9u32 ^ (m * 31 + n) as u32;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    // Codes past the NaNs (0x7f, 0xff), exponents around 2^0.
    let codes: Vec<u8> = (0..n * k)
        .map(|_| loop {
            let b = next() as u8;
            if b & 0x7F != 0x7F {
                break b;
            }
        })
        .collect();
    let (nb, kb) = (n.div_ceil(128), k.div_ceil(128));
    let exps: Vec<u8> = (0..nb * kb).map(|_| 120 + (next() % 10) as u8).collect();
    let w = |r: usize, c: usize| {
        e4m3_value(codes[r * k + c]) * 2f32.powi(i32::from(exps[(r / 128) * kb + c / 128]) - 127)
    };
    let x: Vec<f32> = values(m * k, 21, 2.0).into_iter().map(round_bf16).collect();
    let xq = host_quantized(&x, k, 128);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k).map(|j| f64::from(xq[r * k + j]) * f64::from(w(c, j))).sum::<f64>() as f32
        })
        .collect();
    let mut y = gpu.zeros(m as u32, n as u32, Dtype::Bf16);
    act_quant::matmul_fp8b128(
        &ctx,
        "linear.matmul_act_quant",
        gpu.tensor(&x, m as u32, k as u32, Dtype::Bf16),
        bytes(&gpu, &codes, n as u32, k as u32, Dtype::E4m3),
        bytes(&gpu, &exps, nb as u32, kb as u32, Dtype::E8m0),
        128,
        tile_holds,
        &mut y,
    )
    .unwrap();
    let big = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    close_abs(
        &format!("fp8b128 m{m} n{n} k{k} tile {tile_holds}"),
        &gpu.read(&y),
        &want,
        big * 1e-2,
    );
}

#[test]
fn an_fp8b128_projection_answers_the_quantized_dot() {
    // Decode rows on the GEMV; prefill rows on the decoded tile and GEMM, and
    // past a tile that would not hold the plane, on the GEMV.
    run_fp8b128(1, 384, 1024, true);
    run_fp8b128(13, 256, 512, true);
    run_fp8b128(100, 320, 768, true);
    run_fp8b128(40, 200, 256, false);
}
