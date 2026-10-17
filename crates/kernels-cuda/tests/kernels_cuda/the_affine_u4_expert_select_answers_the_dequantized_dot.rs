#![cfg(feature = "cuda")]

//! The routed affine u4 select (a bf16 scale and bias a group of 64), on
//! the old arms and on tcgen05, answers the dequantized dot.

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_quant};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

const GROUP: usize = 64;

struct Bank {
    codes: Vec<u8>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    dequant: Vec<f32>,
}

fn bank(lcg: &mut Lcg, experts: usize, n: usize, k: usize) -> Bank {
    let groups = k / GROUP;
    let words_per_row = k / 8;
    let mut codes = vec![0u8; experts * n * words_per_row * 4];
    let mut scales = vec![0u16; experts * n * groups];
    let mut biases = vec![0u16; experts * n * groups];
    let mut dequant = vec![0f32; experts * n * k];
    for e in 0..experts {
        for r in 0..n {
            for g in 0..groups {
                let values: Vec<f32> = (0..GROUP).map(|_| lcg.unit()).collect();
                let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let scale = from_bf16(to_bf16(((hi - lo) / 15.0).max(1e-6)));
                let bias = from_bf16(to_bf16(lo));
                let fx = (e * n + r) * groups + g;
                scales[fx] = to_bf16(scale);
                biases[fx] = to_bf16(bias);
                for (j, &v) in values.iter().enumerate() {
                    let code = (((v - bias) / scale).round().clamp(0.0, 15.0)) as u32;
                    let col = g * GROUP + j;
                    let word = (e * n + r) * words_per_row + col / 8;
                    let shift = 4 * (col % 8);
                    let bytes = &mut codes[word * 4..word * 4 + 4];
                    let mut w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    w |= code << shift;
                    bytes.copy_from_slice(&w.to_le_bytes());
                    dequant[(e * n + r) * k + col] = code as f32 * scale + bias;
                }
            }
        }
    }
    Bank {
        codes,
        scales,
        biases,
        dequant,
    }
}

/// A fire of `tokens` routed `top_k` ways; past 16 routes the weights read
/// as bf16 (the grouped kernels' operand).
#[allow(clippy::too_many_arguments)]
fn check(
    tuning: Tuning,
    experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    top_k: usize,
    by_token: bool,
    seed: u64,
) {
    let bf16_weights = tokens * top_k >= 16;
    let mut lcg = Lcg::seeded(seed);
    let bank = bank(&mut lcg, experts, n, k);
    let act_rows = if by_token { tokens } else { tokens * top_k };
    let (x_raw, x) = lcg.row(act_rows * k);
    let mut routes: Vec<i32> = Vec::with_capacity(tokens * top_k);
    for t in 0..tokens {
        for s in 0..top_k {
            routes.push(((t * 37 + s * 17 + 3) % experts) as i32);
        }
    }

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&bank.codes);
    let scales_at = gpu.up(&bank.scales);
    let biases_at = gpu.up(&bank.biases);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(tokens * top_k * n * 2);
    let ctx = gpu.ctx().with_tuning(tuning);
    let x_t = Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, experts as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(
        scales_at,
        experts as u32,
        (n * (k / GROUP) * 2) as u32,
        Dtype::U8,
    );
    let biases_t = Tensor::new(
        biases_at,
        experts as u32,
        (n * (k / GROUP) * 2) as u32,
        Dtype::U8,
    );
    let routes_t = Tensor::new(routes_at, tokens as u32, top_k as u32, Dtype::I32);
    let mut y_t = Tensor::new(y_at, (tokens * top_k) as u32, n as u32, Dtype::Bf16);
    matmul_select_quant(
        &ctx,
        x_t,
        codes_t,
        scales_t,
        Some(biases_t),
        routes_t,
        &mut y_t,
        GroupSeat::RESIDENT,
    )
    .expect("the select fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, tokens * top_k * n);

    let mut bad = Vec::new();
    for route in 0..tokens * top_k {
        let e = routes[route] as usize;
        let xr = if by_token { route / top_k } else { route };
        for r in 0..n {
            let mut want = 0f32;
            for c in 0..k {
                let w = bank.dequant[(e * n + r) * k + c];
                let w = if bf16_weights {
                    from_bf16(to_bf16(w))
                } else {
                    w
                };
                want += w * x[xr * k + c];
            }
            let g = from_bf16(got[route * n + r]);
            if !close(g, want) {
                bad.push((route, r, g, want));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} of {} outputs differ from the dequantized dot; first (route, row, got, want) {:?}",
        bad.len(),
        tokens * top_k * n,
        &bad[..bad.len().min(5)]
    );
}

#[test]
fn the_affine_u4_expert_select_answers_the_dequantized_dot_every_case() {
    // (experts, n, k, tokens, top k, read by token, seed): the GEMV at one
    // token on the up and down legs, the per-route select over eleven
    // groups, every expert of a full bank addressed, and a wide fire on the
    // grouped kernel (the down leg's tiles ragged).
    let cases = [
        (16, 1408, 2816, 1, 8, true, 0x54),
        (16, 2816, 704, 1, 8, false, 0x55),
        (16, 2816, 704, 3, 8, false, 0x52),
        (128, 64, 704, 16, 8, true, 0x53),
        (16, 1408, 2816, 40, 8, true, 0x61),
        (16, 200, 704, 40, 8, false, 0x62),
    ];
    for (experts, n, k, tokens, top_k, by_token, seed) in cases {
        for tuning in [Tuning::DEFAULT, crate::common::TC5] {
            check(tuning, experts, n, k, tokens, top_k, by_token, seed);
        }
    }
}
