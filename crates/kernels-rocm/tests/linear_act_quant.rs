//! FP8 products against host references: e4m3 activation quantization
//! under ue8m0 and f32 block scales, and fp8b128 / fp8b32 / e4m3tile128
//! projections, dense and routed, on the GEMV, the decoded GEMM and the FP8
//! matrix core.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, e4m3_value, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::linear::act_quant;

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
            assert_eq!(
                h.to_bits(),
                w.to_bits(),
                "gran {gran} element {i}: {h} != {w} (x {})",
                x[i]
            );
        }
    }
}

/// `[m, k] · [n, k]ᵀ` for an fp8b128 weight against quantized activations.
fn run_fp8b128(m: usize, n: usize, k: usize, tile_holds: bool) {
    run_fp8b(m, n, k, tile_holds, 128);
}

/// The same at block side `side` (128, or V4.1's 32), the activations in
/// 1x`side` blocks.
fn run_fp8b(m: usize, n: usize, k: usize, tile_holds: bool, side: usize) {
    run_fp8b_on(m, n, k, tile_holds, side, false);
}

/// The same, `lanes`: an fp8b32 plane put in the FP8 matrix core's order at
/// load, on the matrix core.
fn run_fp8b_on(m: usize, n: usize, k: usize, tile_holds: bool, side: usize, lanes: bool) {
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
        .map(|_| {
            loop {
                let b = next() as u8;
                if b & 0x7F != 0x7F {
                    break b;
                }
            }
        })
        .collect();
    let (nb, kb) = (n.div_ceil(side), k.div_ceil(side));
    let exps: Vec<u8> = (0..nb * kb).map(|_| 120 + (next() % 10) as u8).collect();
    let w = |r: usize, c: usize| {
        e4m3_value(codes[r * k + c]) * 2f32.powi(i32::from(exps[(r / side) * kb + c / side]) - 127)
    };
    let x: Vec<f32> = values(m * k, 21, 2.0).into_iter().map(round_bf16).collect();
    let xq = host_quantized(&x, k, side);
    let want: Vec<f32> = (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k)
                .map(|j| f64::from(xq[r * k + j]) * f64::from(w(c, j)))
                .sum::<f64>() as f32
        })
        .collect();
    let mut y = gpu.zeros(m as u32, n as u32, Dtype::Bf16);
    let big = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    if lanes {
        let (wt, nans) = canonical(
            &gpu,
            gpu.upload(&codes, n as u32, k as u32, Dtype::E4m3),
            n,
            k,
        );
        assert_eq!(nans, 0);
        act_quant::matmul_fp8b32_lanes(
            &ctx,
            "linear.matmul_act_quant",
            gpu.tensor(&x, m as u32, k as u32, Dtype::Bf16),
            wt,
            gpu.upload(&exps, nb as u32, kb as u32, Dtype::E8m0),
            &mut y,
        )
        .unwrap();
        for (i, (g, w)) in gpu.read(&y).iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= w.abs() * 2f32.powi(-8) + 1e-5 * big,
                "fp8b32 lanes m{m} n{n} k{k}: [{}, {}] is {g}, the quantized dot {w}",
                i / n,
                i % n
            );
        }
        return;
    }
    act_quant::matmul_fp8b128(
        &ctx,
        "linear.matmul_act_quant",
        gpu.tensor(&x, m as u32, k as u32, Dtype::Bf16),
        gpu.upload(&codes, n as u32, k as u32, Dtype::E4m3),
        gpu.upload(&exps, nb as u32, kb as u32, Dtype::E8m0),
        side as u32,
        tile_holds,
        &mut y,
    )
    .unwrap();
    close_abs(
        &format!("fp8b{side} m{m} n{n} k{k} tile {tile_holds}"),
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

#[test]
fn an_fp8b32_projection_answers_the_quantized_dot() {
    run_fp8b(1, 384, 1024, true, 32);
    run_fp8b(13, 256, 512, true, 32);
    run_fp8b(100, 320, 768, true, 32);
    run_fp8b(40, 200, 256, false, 32);
}

#[test]
fn an_fp8b32_projection_on_the_matrix_core_folds_each_step() {
    // Decode rows (one slice and several), rows past one item, and the
    // prefill tile past one 128-row item over a ragged last weight tile.
    run_fp8b_on(1, 384, 1024, true, 32, true);
    run_fp8b_on(13, 256, 512, true, 32, true);
    run_fp8b_on(64, 2048, 1024, true, 32, true);
    run_fp8b_on(100, 320, 768, true, 32, true);
    run_fp8b_on(300, 336, 768, true, 32, true);
    // The prefill tiles: whole tiles and a last round's K slices summed by
    // the last to finish, a ragged last weight tile.
    run_fp8b_on(600, 336, 768, true, 32, true);
    run_fp8b_on(1100, 1040, 1536, true, 32, true);
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
        .map(|_| {
            loop {
                let b = next() as u8;
                if b & 0x7F != 0x7F {
                    break b;
                }
            }
        })
        .collect();
    let factors = (0..blocks)
        .map(|_| 1e-3 + (next() % 10_000) as f32 * 7.31e-7)
        .collect();
    (codes, factors)
}

/// Row `r` of a `[n, k]` e4m3tile128 weight's block-summed dot with `xq`'s
/// row: each 128-block's exact products summed, then scaled by the weight's
/// factor (a `tall` x 128 block) and (`xs`, a scale a block) the activation's.
fn tile_dot(
    codes: &[u8],
    factors: &[f32],
    k: usize,
    xq: &[f32],
    xs: Option<&[f32]>,
    r: usize,
    tall: usize,
) -> f64 {
    let kb = k.div_ceil(128);
    (0..kb)
        .map(|b| {
            let part: f64 = (b * 128..((b + 1) * 128).min(k))
                .map(|j| f64::from(xq[j]) * f64::from(e4m3_value(codes[r * k + j])))
                .sum();
            part * f64::from(factors[(r / tall) * kb + b]) * xs.map_or(1.0, |xs| f64::from(xs[b]))
        })
        .sum()
}

/// `codes` (`rows` x `k`) put in the FP8 matrix core's order and made
/// canonical, as a load does, into a plane of their own; the NaN codes it
/// counted.
fn canonical(gpu: &Gpu, codes: Tensor, rows: usize, k: usize) -> (Tensor, u32) {
    let nans = gpu.zeros(1, 1, Dtype::F32);
    let out = gpu.zeros(codes.rows, codes.width.div_ceil(2), Dtype::Bf16);
    act_quant::e4m3fn_shuffle(
        &gpu.ctx(),
        "test",
        codes.ptr,
        out.ptr,
        rows as u64,
        k as u64,
        nans.ptr,
    )
    .unwrap();
    (
        Tensor::new(out.ptr, codes.rows, codes.width, codes.dtype),
        gpu.read(&nans)[0].to_bits(),
    )
}

#[test]
fn a_nan_code_is_counted_and_negative_zero_reads_as_zero() {
    let Some(gpu) = Gpu::open() else { return };
    let mut codes = vec![0x3Cu8; 16 * 64];
    for (i, c) in [0x80u8, 0x7F, 0xFF, 0x80, 0x01, 0x7E].iter().enumerate() {
        codes[i * 97] = *c;
    }
    let t = gpu.upload(&codes, 16, 64, Dtype::U8);
    assert_eq!(
        canonical(&gpu, t, 16, 64).1,
        2,
        "0x7f and 0xff are NaN codes"
    );
}

fn run_fp8_mfma(m: usize, n: usize, k: usize, edge_codes: bool) {
    run_fp8_mfma_rows(m, n, k, edge_codes, 128);
}

/// `run_fp8_mfma` under factors a `tall` x 128 block (at 64: `e4m3g128` per-row factors folded at load).
fn run_fp8_mfma_rows(m: usize, n: usize, k: usize, edge_codes: bool, tall: usize) {
    run_fp8_mfma_tuned(m, n, k, edge_codes, tall, false);
}

/// `run_fp8_mfma_rows`, `narrow`: past 16 rows on the 64 x 128 prefill tile.
fn run_fp8_mfma_tuned(m: usize, n: usize, k: usize, edge_codes: bool, tall: usize, narrow: bool) {
    let Some(gpu) = Gpu::open() else { return };
    let mut ctx = gpu.ctx();
    if narrow {
        let mut tuning = ctx.tuning();
        tuning.fp8_decode_until = 16;
        tuning.fp8_narrow_until = u32::MAX;
        ctx.set_tuning(tuning);
    }
    let (nb, kb) = (n.div_ceil(tall), k / 128);
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
            tile_dot(
                &codes,
                &factors,
                k,
                &xq[r * k..(r + 1) * k],
                Some(&xs[r * kb..(r + 1) * kb]),
                i % n,
                tall,
            ) as f32
        })
        .collect();
    let mut y = gpu.zeros(m as u32, n as u32, Dtype::Bf16);
    let wt = gpu.upload(&codes, n as u32, k as u32, Dtype::E4m3);
    let (wt, nans) = canonical(&gpu, wt, n, k);
    assert_eq!(nans, 0);
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
    // A decode row, rows past one item with a ragged last row block, the
    // prefill tile past one 128-row item, and sliced prefill tiles; half of
    // them over the codes fnuz reads differently.
    run_fp8_mfma(1, 576, 1024, false);
    run_fp8_mfma(150, 320, 512, true);
    run_fp8_mfma(300, 320, 768, true);
    run_fp8_mfma(1100, 1040, 1536, false);
    // The narrow tile: one tile of rows, then rows past several, whole
    // tiles and sliced.
    run_fp8_mfma_tuned(40, 384, 1024, true, 128, true);
    run_fp8_mfma_tuned(300, 1040, 1536, false, 128, true);
    run_fp8_mfma_tuned(150, 320, 768, true, 64, true);
}

#[test]
fn a_projection_under_64_row_factors_folds_each_block() {
    // `e4m3g128` folded to 64-row groups (MiMo's rows restart their blocks
    // on 64-row bounds): the decode tile, the prefill tile whole and sliced.
    run_fp8_mfma_rows(1, 576, 1024, false, 64);
    run_fp8_mfma_rows(33, 320, 768, true, 64);
    run_fp8_mfma_rows(300, 320, 768, true, 64);
    run_fp8_mfma_rows(1100, 1040, 1536, false, 64);
}

#[test]
fn an_fp8_bank_on_the_matrix_core_folds_each_route() {
    use kernels_rocm::linear::moe;
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let experts = 6usize;
    for (tokens, top_k, n, k) in [
        (1usize, 4usize, 256usize, 512usize),
        (24, 6, 384, 1024),
        (96, 4, 256, 768),
        (40, 6, 192, 256),
        (200, 4, 320, 512),
        (300, 6, 336, 384),
        (600, 8, 512, 1024),
    ] {
        let (nb, kb) = (n.div_ceil(128), k / 128);
        let (codes, factors) =
            tile_planes(experts * n * k, experts * nb * kb, (tokens * 11 + n) as u32);
        let x: Vec<f32> = values(tokens * k, 43, 2.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let routes: Vec<i32> = (0..tokens * top_k)
            .map(|i| ((i / top_k * 5 + (i % top_k) * 7 + i / 3) % experts) as i32)
            .collect();
        let op = "linear.moe_matmul_select_act_quant";
        let xt = gpu.tensor(&x, tokens as u32, k as u32, Dtype::Bf16);
        let _ = op;
        let (host, host_s) = host_scaled(&x, k, 128);
        let mut y = gpu.zeros((tokens * top_k) as u32, n as u32, Dtype::Bf16);
        let ct = gpu.upload(&codes, experts as u32, (n * k) as u32, Dtype::U8);
        let (ct, nans) = canonical(&gpu, ct, experts * n, k);
        assert_eq!(nans, 0);
        moe::matmul_select_fp8_mfma(
            &ctx,
            xt,
            ct,
            gpu.upload(
                &factors
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
                experts as u32,
                (4 * nb * kb) as u32,
                Dtype::U8,
            ),
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
                    128,
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
