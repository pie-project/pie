#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_quant};
use kernels_cuda::tensor::Tensor;

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

fn check(
    experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    top_k: usize,
    by_token: bool,
    seed: u64,
) {
    check_with(
        experts,
        n,
        k,
        tokens,
        top_k,
        by_token,
        seed,
        tokens * top_k >= 16,
    );
}

#[allow(clippy::too_many_arguments)]
fn check_with(
    experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    top_k: usize,
    by_token: bool,
    seed: u64,
    bf16_weights: bool,
) {
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
    let ctx = gpu.ctx();
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

    let mut worst = 0f32;
    let mut bad = 0usize;
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
        bad,
        0,
        "{bad} of {} outputs differ from the dequantized dot (worst relative error {worst:.4})",
        tokens * top_k * n
    );
    eprintln!("ok: worst relative error {worst:.5}");
}

#[test]
fn the_affine_u4_expert_select_answers_the_dequantized_dot_every_case() {
    the_up_leg_read_by_token();
    the_up_leg_read_by_token_at_one_token_takes_the_gemv();
    the_down_leg_read_by_route_over_eleven_groups();
    the_down_leg_read_by_route_at_one_token_takes_the_gemv();
    every_expert_of_the_full_bank_is_addressed();
    a_wide_fire_takes_the_grouped_kernel_on_the_up_leg();
    a_wide_fire_takes_the_grouped_kernel_on_the_down_leg_with_ragged_tiles();
}

fn the_up_leg_read_by_token() {
    check(16, 1408, 2816, 3, 8, true, 0x51);
}

fn the_up_leg_read_by_token_at_one_token_takes_the_gemv() {
    check(16, 1408, 2816, 1, 8, true, 0x54);
}

fn the_down_leg_read_by_route_over_eleven_groups() {
    check(16, 2816, 704, 3, 8, false, 0x52);
}

fn the_down_leg_read_by_route_at_one_token_takes_the_gemv() {
    check(16, 2816, 704, 1, 8, false, 0x55);
}

fn every_expert_of_the_full_bank_is_addressed() {
    check(128, 64, 704, 16, 8, true, 0x53);
}

fn a_wide_fire_takes_the_grouped_kernel_on_the_up_leg() {
    check_with(16, 1408, 2816, 40, 8, true, 0x61, true);
}

fn a_wide_fire_takes_the_grouped_kernel_on_the_down_leg_with_ragged_tiles() {
    check_with(16, 200, 704, 40, 8, false, 0x62, true);
}

/// A dense affine u4g64 projection on the tcgen05 GEMM against the arm it
/// replaces (dequantize the weight to bf16, then cuBLAS), both agreeing.
/// Ignored: a measurement. Run alone on an idle GPU.
#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn a_dense_affine_projection_is_priced_on_tcgen05() {
    use kernels_cuda::linear::quant::{OffsetKind, matmul_tc5, matmul_via_dense};
    if !kernels_cuda::linear::tc5::supported() {
        return;
    }
    let (n, k) = (4096usize, 4096usize);
    let mut lcg = Lcg::seeded(0xaf);
    let b = bank(&mut lcg, 1, n, k);
    let mut gpu = Gpu::open();
    let codes = Tensor::new(gpu.up(&b.codes), n as u32, (k / 2) as u32, Dtype::U8);
    let factors = (k / GROUP * 2) as u32;
    let scales = Tensor::new(gpu.up(&b.scales), n as u32, factors, Dtype::U8);
    let biases = Tensor::new(gpu.up(&b.biases), n as u32, factors, Dtype::U8);
    for m in [16usize, 128, 1024, 8192] {
        let (x_raw, _) = lcg.row(m * k);
        let x = Tensor::new(gpu.up(&x_raw), m as u32, k as u32, Dtype::Bf16);
        let ya = gpu.zeros(m * n * 2);
        let yb = gpu.zeros(m * n * 2);
        let ctx = gpu.ctx();
        let fire = |tc5: bool, y: u64| {
            let mut out = Tensor::new(y, m as u32, n as u32, Dtype::Bf16);
            let (post, f, seat) = (OffsetKind::Post, Dtype::Bf16, GroupSeat::RESIDENT);
            if tc5 {
                let took = matmul_tc5(
                    &ctx,
                    x,
                    codes,
                    scales,
                    post,
                    Some(biases),
                    f,
                    &mut out,
                    seat,
                )
                .expect("fires");
                assert!(took, "the tcgen05 arm takes a u4g64 projection");
            } else {
                matmul_via_dense(
                    &ctx,
                    x,
                    codes,
                    scales,
                    post,
                    Some(biases),
                    f,
                    &mut out,
                    seat,
                )
                .expect("fires");
            }
        };
        let time = |tc5: bool, y: u64| -> f64 {
            fire(tc5, y);
            gpu.sync();
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                fire(tc5, y);
            }
            gpu.sync();
            t0.elapsed().as_secs_f64() * 1e6 / 20.0
        };
        let dense_us = time(false, ya);
        let tc5_us = time(true, yb);
        let a: Vec<u16> = gpu.down(ya, m * n);
        let c: Vec<u16> = gpu.down(yb, m * n);
        let worst = a
            .iter()
            .zip(&c)
            .map(|(p, q)| (from_bf16(*p) - from_bf16(*q)).abs() / from_bf16(*p).abs().max(1.0))
            .fold(0f32, f32::max);
        eprintln!(
            "m={m} n={n} k={k}: dequantize+cuBLAS {dense_us:.1} us, tcgen05 {tc5_us:.1} us \
             ({:.2}x); worst relative gap {worst:.4}",
            dense_us / tc5_us
        );
        assert!(worst < 0.02, "the two arms disagree by {worst}");
    }
}
