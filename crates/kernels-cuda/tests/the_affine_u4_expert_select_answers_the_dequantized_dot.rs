//! `moe_matmul_select_quant` over an MLX affine-U4 bank (group 64, a bf16
//! scale and a bf16 zero point per group) lands, per route, the dot of the
//! activation with the DEQUANTIZED expert rows — at gemma-4-26B-A4B's own
//! shapes: 128 experts, top-8, a `[1408, 2816]` up leg read by token and a
//! `[2816, 704]` down leg read by route (704 is eleven groups, not a power
//! of two).

#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_quant};
use kernels_cuda::tensor::Tensor;

const GROUP: usize = 64;

/// One affine-U4 bank of `experts x n x k`, in the kernel's plane layout,
/// with the dequantized values it stands for.
struct Bank {
    codes: Vec<u8>,
    scales: Vec<u16>,
    biases: Vec<u16>,
    /// `[expert][row][col]`
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
                // A group: 64 uniform values in [-1, 1), quantized min/max.
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

/// `by_token`: the activation holds one row per token (the up leg);
/// otherwise one per route (the down leg).
/// `bf16_weights`: the reference multiplies the dequantized weight ROUNDED
/// TO BF16 — what the tensor-core kernel (and transformers, whose
/// dequantized bank is a bf16 tensor) multiply; the per-route GEMV keeps
/// the weight in f32.
fn check(experts: usize, n: usize, k: usize, tokens: usize, top_k: usize, by_token: bool, seed: u64) {
    check_with(experts, n, k, tokens, top_k, by_token, seed, false);
}

#[allow(clippy::too_many_arguments)]
fn check_with(experts: usize, n: usize, k: usize, tokens: usize, top_k: usize, by_token: bool, seed: u64, bf16_weights: bool) {
    let mut lcg = Lcg::seeded(seed);
    let bank = bank(&mut lcg, experts, n, k);
    let act_rows = if by_token { tokens } else { tokens * top_k };
    let (x_raw, x) = lcg.row(act_rows * k);
    // Routes: distinct experts per token, spread over the whole bank.
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
    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, experts as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, experts as u32, (n * (k / GROUP) * 2) as u32, Dtype::U8);
    let biases_t = Tensor::new(biases_at, experts as u32, (n * (k / GROUP) * 2) as u32, Dtype::U8);
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

    let mut worst = 0f32;
    let mut bad = 0usize;
    for route in 0..tokens * top_k {
        let e = routes[route] as usize;
        let xr = if by_token { route / top_k } else { route };
        for r in 0..n {
            let mut want = 0f32;
            for c in 0..k {
                let w = bank.dequant[(e * n + r) * k + c];
                let w = if bf16_weights { from_bf16(to_bf16(w)) } else { w };
                want += w * x[xr * k + c];
            }
            let g = from_bf16(got[route * n + r]);
            worst = worst.max((g - want).abs() / want.abs().max(1.0));
            if !close(g, want) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("route {route} (expert {e}) row {r}: got {g} want {want}");
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{bad} of {} outputs differ from the dequantized dot (worst relative error {worst:.4})",
        tokens * top_k * n
    );
    eprintln!("ok: worst relative error {worst:.5}");
}

#[test]
fn the_up_leg_read_by_token() {
    // 128 x [1408, 2816] would take a while on the host reference; the
    // layout and the kernel's group walk are the same at 16 experts.
    check(16, 1408, 2816, 3, 8, true, 0x51);
}

#[test]
fn the_down_leg_read_by_route_over_eleven_groups() {
    check(16, 2816, 704, 3, 8, false, 0x52);
}

#[test]
fn every_expert_of_the_full_bank_is_addressed() {
    check(128, 64, 704, 16, 8, true, 0x53);
}

// From 256 routes on the select takes the GROUPED kernel (one block per
// expert × 128 rows, the bank decoded once per expert to bf16 and the dot
// on tensor cores — `moe.rs`, `GROUPED_FROM`); the three above stay on the
// per-route GEMV. Same answer on both legs to bf16-weight arithmetic, with
// rows past a 128-tile and a K that is not a whole number of 128-chunks.

#[test]
fn a_wide_fire_takes_the_grouped_kernel_on_the_up_leg() {
    check_with(16, 1408, 2816, 40, 8, true, 0x61, true);
}

#[test]
fn a_wide_fire_takes_the_grouped_kernel_on_the_down_leg_with_ragged_tiles() {
    check_with(16, 200, 704, 40, 8, false, 0x62, true);
}
