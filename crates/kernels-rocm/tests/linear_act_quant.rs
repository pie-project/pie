//! DeepSeek's W8A8 product on the host's terms: activations quantized to
//! e4m3 under a ue8m0 scale a block (the amax floored at 1e-4, the scale
//! 2^ceil(log2(amax / 448)), round to nearest even), landed as their exact
//! bf16 values; and an fp8b128 projection (e4m3 codes under one e8m0
//! exponent a 128x128 block) against them, on the fused-dequant GEMV and on
//! the decoded tile and GEMM, against an f64 dot of the same values. And
//! transformers' FP8 product (e4m3tile128: e4m3 codes under one f32 factor
//! a 128x128 block, made canonical as a load makes them) against
//! activations quantized under f32 scales a 1x128 block, on the FP8 matrix
//! core, dense and routed, against a host dot that sums each 128-block's
//! exact products and then scales them.

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

/// A row's blocks quantized under f32 scales (`amax / 448`, the amax floored
/// at 1e-10): the codes' values and each block's scale.
fn host_scaled(x: &[f32], k: usize, gran: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0f32; x.len()];
    let mut scales = Vec::new();
    for (r, row) in x.chunks(k).enumerate() {
        for b in 0..k / gran {
            let blk = &row[b * gran..(b + 1) * gran];
            let s = blk.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-10) / 448.0;
            for (i, v) in blk.iter().enumerate() {
                q[r * k + b * gran + i] = e4m3_round((v / s).clamp(-448.0, 448.0));
            }
            scales.push(s);
        }
    }
    (q, scales)
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

/// Codes past the NaNs and f32 factors of no particular form (as a
/// transformers FP8 release stores them), `count` codes and `blocks`
/// factors.
fn tile_planes(count: usize, blocks: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let codes = (0..count)
        .map(|_| loop {
            let b = next() as u8;
            if b & 0x7F != 0x7F {
                break b;
            }
        })
        .collect();
    let factors = (0..blocks).map(|_| 1e-3 + (next() % 10_000) as f32 * 7.31e-7).collect();
    (codes, factors)
}

/// Row `r` of a `[n, k]` e4m3tile128 weight's block-summed dot with `xq`'s
/// row: each 128-block's exact products summed, then scaled by the weight's
/// factor and (`xs`, a scale a block) the activation's.
fn tile_dot(codes: &[u8], factors: &[f32], k: usize, xq: &[f32], xs: Option<&[f32]>, r: usize) -> f64 {
    let kb = k.div_ceil(128);
    (0..kb)
        .map(|b| {
            let part: f64 = (b * 128..((b + 1) * 128).min(k))
                .map(|j| f64::from(xq[j]) * f64::from(e4m3_value(codes[r * k + j])))
                .sum();
            part * f64::from(factors[(r / 128) * kb + b]) * xs.map_or(1.0, |xs| f64::from(xs[b]))
        })
        .sum()
}

/// `codes` made canonical for the FP8 matrix core, as a load does; the NaN
/// codes it counted.
fn canonical(gpu: &Gpu, codes: Tensor, bytes: usize) -> u32 {
    let nans = gpu.zeros(1, 1, Dtype::F32);
    act_quant::e4m3fn_canonical(&gpu.ctx(), "test", codes.ptr, bytes as u64, nans.ptr).unwrap();
    gpu.read(&nans)[0].to_bits()
}

#[test]
fn a_nan_code_is_counted_and_negative_zero_reads_as_zero() {
    let Some(gpu) = Gpu::open() else { return };
    let codes: Vec<u8> = vec![0x80, 0x00, 0x7F, 0x3C, 0xFF, 0x80, 0x01, 0x7E];
    let t = bytes(&gpu, &codes, 1, 8, Dtype::U8);
    assert_eq!(canonical(&gpu, t, 8), 2, "0x7f and 0xff are NaN codes");
}

fn run_fp8_mfma(m: usize, n: usize, k: usize, edge_codes: bool) {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (nb, kb) = (n.div_ceil(128), k / 128);
    let (mut codes, factors) = tile_planes(n * k, nb * kb, (m * 37 + n) as u32);
    if edge_codes {
        // Negative zeros (fnuz's NaN) and the largest codes.
        for (i, c) in codes.iter_mut().enumerate().step_by(7) {
            *c = [0x80, 0x00, 0x7E, 0xFE, 0x01, 0x81][i % 6];
        }
    }
    let x: Vec<f32> = values(m * k, 41, 2.0).into_iter().map(round_bf16).collect();
    let (xq, xs) = host_scaled(&x, k, 128);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let r = i / n;
            tile_dot(&codes, &factors, k, &xq[r * k..(r + 1) * k], Some(&xs[r * kb..(r + 1) * kb]), i % n) as f32
        })
        .collect();
    let mut y = gpu.zeros(m as u32, n as u32, Dtype::Bf16);
    let wt = bytes(&gpu, &codes, n as u32, k as u32, Dtype::E4m3);
    assert_eq!(canonical(&gpu, wt, n * k), 0);
    act_quant::matmul_fp8(
        &ctx,
        "linear.matmul_act_quant",
        gpu.tensor(&x, m as u32, k as u32, Dtype::Bf16),
        wt,
        gpu.tensor(&factors, nb as u32, kb as u32, Dtype::F32),
        &mut y,
    )
    .unwrap();
    let got = gpu.read(&y);
    let big = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= w.abs() * 2f32.powi(-8) + 1e-5 * big,
            "fp8 mfma m{m} n{n} k{k}: [{}, {}] is {g}, the block-summed dot {w}",
            i / n,
            i % n
        );
    }
}

#[test]
fn an_fp8_projection_on_the_matrix_core_folds_each_block() {
    // Decode rows (one slice and several), rows past one item, a ragged last
    // row block, and the codes fnuz reads differently.
    run_fp8_mfma(1, 576, 1024, false);
    run_fp8_mfma(7, 256, 2048, true);
    run_fp8_mfma(32, 384, 768, false);
    run_fp8_mfma(150, 320, 512, true);
    run_fp8_mfma(64, 6144, 1024, false);
    // The prefill tile: rows past one 128-row item, a ragged last weight
    // tile.
    run_fp8_mfma(300, 320, 768, true);
    run_fp8_mfma(128, 4096, 2048, false);
}

#[test]
fn an_fp8_bank_on_the_matrix_core_folds_each_route() {
    use kernels_rocm::linear::moe;
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let experts = 6usize;
    for (tokens, top_k, n, k) in [(1usize, 4usize, 256usize, 512usize), (24, 6, 384, 1024), (96, 4, 256, 768), (40, 6, 192, 256), (200, 4, 320, 512)] {
        let (nb, kb) = (n.div_ceil(128), k / 128);
        let (codes, factors) = tile_planes(experts * n * k, experts * nb * kb, (tokens * 11 + n) as u32);
        let x: Vec<f32> = values(tokens * k, 43, 2.0).into_iter().map(round_bf16).collect();
        let routes: Vec<i32> = (0..tokens * top_k)
            .map(|i| ((i / top_k * 5 + (i % top_k) * 7 + i / 3) % experts) as i32)
            .collect();
        let op = "linear.moe_matmul_select_act_quant";
        let xt = gpu.tensor(&x, tokens as u32, k as u32, Dtype::Bf16);
        let _ = op;
        let (host, host_s) = host_scaled(&x, k, 128);
        let mut y = gpu.zeros((tokens * top_k) as u32, n as u32, Dtype::Bf16);
        let ct = bytes(&gpu, &codes, experts as u32, (n * k) as u32, Dtype::U8);
        assert_eq!(canonical(&gpu, ct, experts * n * k), 0);
        moe::matmul_select_fp8_mfma(
            &ctx,
            xt,
            ct,
            bytes(&gpu, &factors.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(), experts as u32, (4 * nb * kb) as u32, Dtype::U8),
            gpu.ints_shaped(&routes, tokens as u32, top_k as u32),
            &mut y,
            moe::GroupSeat::RESIDENT,
        )
        .unwrap();
        let got = gpu.read(&y);
        let mut big = 0f32;
        let want: Vec<f32> = (0..tokens * top_k * n)
            .map(|i| {
                let (route, c) = (i / n, i % n);
                let e = routes[route] as usize;
                let t = route / top_k;
                let v = tile_dot(
                    &codes[e * n * k..(e + 1) * n * k],
                    &factors[e * nb * kb..(e + 1) * nb * kb],
                    k,
                    &host[t * k..(t + 1) * k],
                    Some(&host_s[t * kb..(t + 1) * kb]),
                    c,
                ) as f32;
                big = big.max(v.abs());
                v
            })
            .collect();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= w.abs() * 2f32.powi(-8) + 1e-5 * big,
                "fp8 mfma bank t{tokens} k{top_k} {n}x{k}: route {} column {} is {g}, the block-summed dot {w}",
                i / n,
                i % n
            );
        }
    }
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_act_quant -- --ignored --nocapture"]
fn bench_fp8_projections() {
    // GLM-5.3's fp8 projections on the FP8 matrix core at decode rows, cold
    // over copies 1 GB together; µs of the product alone (its quantizer
    // apart) and the weight's GB/s.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let rows: Vec<usize> = std::env::var("FP8_M").map_or(vec![1, 8, 32, 128], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    for (name, n, k) in [
        ("q_a", 2048usize, 6144usize),
        ("q_b", 16384, 2048),
        ("kv_a", 576, 6144),
        ("o", 6144, 16384),
        ("dense gate_up", 24576, 6144),
        ("dense down", 6144, 12288),
        ("shared gate_up", 4096, 6144),
        ("shared down", 6144, 2048),
    ] {
        let (nb, kb) = (n.div_ceil(128), k / 128);
        let copies = (1usize << 30).div_ceil(n * k).clamp(2, 16);
        let planes: Vec<(Tensor, Tensor)> = (0..copies)
            .map(|c| {
                let (codes, factors) = tile_planes(n * k, nb * kb, 77 + c as u32);
                let ct = bytes(&gpu, &codes, n as u32, k as u32, Dtype::E4m3);
                assert_eq!(canonical(&gpu, ct, n * k), 0);
                (ct, gpu.tensor(&factors, nb as u32, kb as u32, Dtype::F32))
            })
            .collect();
        for &m in &rows {
            let x: Vec<f32> = values(m * k, 5, 1.0);
            let xt = gpu.tensor(&x, m as u32, k as u32, Dtype::Bf16);
            let y = gpu.zeros(m as u32, n as u32, Dtype::Bf16);
            let mut i = 0usize;
            let us = gpu.time_graph(24, || {
                let (c, f) = planes[i % copies];
                i += 1;
                let mut yt = y;
                act_quant::matmul_fp8(&ctx, "bench", xt, c, f, &mut yt).unwrap();
            });
            eprintln!("{name:15} m={m:4} n={n:6} k={k:6}: {us:8.1} us {:6.0} GB/s", (n * k) as f64 / us / 1e3);
        }
    }
}
