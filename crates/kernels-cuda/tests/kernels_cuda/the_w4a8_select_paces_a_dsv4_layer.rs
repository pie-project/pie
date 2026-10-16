#![cfg(feature = "cuda")]

//! A DeepSeek-V4-Flash MoE layer's W4A8 selects (256 experts, top 6, MXFP4
//! codes under atom-laid scales, e4m3 activations in 128s) at decode widths:
//! the grouped tcgen05 GEMM against the activations read back to bf16 on
//! the e2m1-convert register-dequant kernel. Each fire takes the next of 16
//! routings. Ignored: a measurement. Run alone on an idle GPU:
//!
//!     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//!         the_w4a8_select_paces -- --ignored --nocapture
//!
//! `PIE_W4_TOKENS` (a comma list) keeps those token counts.

use crate::common::Gpu;
use dtype::Dtype;
use kernels_cuda::linear::act_quant::{Bank, Format, Source, read_back, select};
use kernels_cuda::linear::moe::select_mxfp4_atoms;
use kernels_cuda::linear::tc5::{Elem, atom_bytes, supported};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;
use std::time::Instant;

const EXPERTS: usize = 256;
const TOP: usize = 6;
const HIDDEN: usize = 4096;
const INTER: usize = 2048;
const ROUTINGS: usize = 16;
const FMT: Format = Format::E4m3 { gran: 128 };

fn routing(tokens: usize, seed: u64) -> Vec<i32> {
    let mut bits = seed | 1;
    let mut out = Vec::with_capacity(tokens * TOP);
    for _ in 0..tokens {
        let mut picked: Vec<i32> = Vec::with_capacity(TOP);
        while picked.len() < TOP {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            let e = (bits % EXPERTS as u64) as i32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        out.extend(picked);
    }
    out
}

#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_w4a8_select_paces_a_dsv4_layer() {
    if !supported() {
        return;
    }
    let mut gpu = Gpu::open();
    // The gate-up leg (2 x INTER over HIDDEN) and the down leg.
    let legs = [("gate-up", 2 * INTER, HIDDEN), ("down", HIDDEN, INTER)];
    let banks: Vec<(u64, u64)> = legs
        .iter()
        .map(|&(_, n, k)| {
            let codes = gpu.zeros(EXPERTS * n * k / 2);
            let atoms = gpu.up(&vec![127u8; atom_bytes(EXPERTS * n, k, 32)]);
            (codes, atoms)
        })
        .collect();
    let cvt = gpu.ctx().with_tuning(Tuning {
        mxfp4_mma_from: 1,
        mxfp4_mma_until: 65_536,
        mxfp4_cvt: true,
        mxfp4_a8_until: 65_536,
        ..Tuning::DEFAULT
    });
    println!("| tokens | leg | tcgen05 W4A8 us | read back + e2m1 convert us |");
    println!("|---|---|---|---|");
    let counts: Vec<usize> = std::env::var("PIE_W4_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16, 24, 32, 48, 64]);
    for tokens in counts {
        let routes_at: Vec<u64> = (0..ROUTINGS)
            .map(|r| gpu.up(&routing(tokens, 0x51d + 7919 * r as u64 + tokens as u64)))
            .collect();
        for (&(leg, n, k), &(codes, atoms)) in legs.iter().zip(&banks) {
            let x = Tensor::new(
                gpu.zeros(tokens * k * 2),
                tokens as u32,
                k as u32,
                Dtype::Bf16,
            );
            let back_at = gpu.zeros(tokens * k * 2);
            let y_at = gpu.zeros(tokens * TOP * n * 2);
            let routes_t = |i: usize| {
                Tensor::new(
                    routes_at[i % ROUTINGS],
                    tokens as u32,
                    TOP as u32,
                    Dtype::I32,
                )
            };
            let tc5 = |i: usize| {
                let mut y = Tensor::new(y_at, (tokens * TOP) as u32, n as u32, Dtype::Bf16);
                select(
                    &cvt,
                    "bench",
                    FMT,
                    x,
                    routes_t(i),
                    None,
                    Bank {
                        codes,
                        atoms,
                        experts: EXPERTS as u32,
                        n: n as u32,
                        k: k as u32,
                        elem: Elem::E2m1,
                        gran: 32,
                        alpha_e: 0,
                        alpha_from: None,
                    },
                    &mut y,
                )
                .expect("the W4A8 select fires");
            };
            let conv = |i: usize| {
                let mut y = Tensor::new(y_at, (tokens * TOP) as u32, n as u32, Dtype::Bf16);
                let back = read_back(&cvt, "bench", FMT, Source::rows(x, None), back_at)
                    .expect("the read-back fires");
                let codes_t = Tensor::new(codes, EXPERTS as u32, (n * k / 2) as u32, Dtype::U8);
                assert!(
                    select_mxfp4_atoms(&cvt, back, codes_t, atoms, routes_t(i), &mut y)
                        .expect("the convert select fires")
                );
            };
            let time = |f: &dyn Fn(usize)| {
                let reps: usize = std::env::var("PIE_W4_REPS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(64);
                (0..reps.max(ROUTINGS)).for_each(&f);
                gpu.sync();
                let t0 = Instant::now();
                (0..reps).for_each(&f);
                gpu.sync();
                t0.elapsed().as_secs_f64() * 1e6 / reps as f64
            };
            let (a, b) = (time(&tc5), time(&conv));
            println!("| {tokens} | {leg} | {a:.1} | {b:.1} |");
        }
    }
}

/// `count` bytes, each `base + (r & mask)` of a xorshift draw `r`.
fn noise(count: usize, mask: u8, base: u8) -> Vec<u8> {
    let mut bits = 0x9e37_79b9_7f4a_7c15u64;
    (0..count)
        .map(|_| {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            base.wrapping_add((bits >> 24) as u8 & mask)
        })
        .collect()
}

/// The whole W4A8 routed MLP kept by expert (DeepSeek's: quantize, the
/// gate-up with its GLU epilogue, the down leg, the weighted sum), each
/// fire the next of 16 routings. `PIE_W4_TOKENS` as above.
#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_w4a8_routed_mlp_paces_a_dsv4_layer() {
    use kernels_cuda::linear::moe::{GluBank, experts_weighted_first};
    use kernels_cuda::linear::tc5::Glu;
    if !supported() {
        return;
    }
    let mut gpu = Gpu::open();
    let bank = |gpu: &mut Gpu, n: usize, k: usize| {
        let e = EXPERTS as u32;
        GluBank::Mxfp4A8 {
            // Random codes and scales near 1: zeros would draw less
            // power than a checkpoint's planes do.
            codes: Tensor::new(
                gpu.up(&noise(EXPERTS * n * k / 2, 0xff, 0)),
                e,
                (n * k / 2) as u32,
                Dtype::U8,
            ),
            atoms: Tensor::new(
                gpu.up(&noise(atom_bytes(EXPERTS * n, k, 32), 3, 125)),
                e,
                (n * k / 32) as u32,
                Dtype::U8,
            ),
        }
    };
    let up = bank(&mut gpu, 2 * INTER, HIDDEN);
    let down = bank(&mut gpu, HIDDEN, INTER);
    let ctx = gpu.ctx().with_tuning(Tuning {
        moe_w4a8_fused_from: 1,
        ..crate::common::SM100
    });
    println!("| tokens | routed MLP us |");
    println!("|---|---|");
    let counts: Vec<usize> = std::env::var("PIE_W4_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![16, 32, 64, 128]);
    for tokens in counts {
        let routes_at: Vec<u64> = (0..ROUTINGS)
            .map(|r| gpu.up(&routing(tokens, 0x51d + 7919 * r as u64 + tokens as u64)))
            .collect();
        let t = tokens as u32;
        let x_raw: Vec<u16> = noise(tokens * HIDDEN * 2, 0xff, 0)
            .chunks(2)
            .map(|b| 0x3c00 | (u16::from(b[0]) & 0x7f) | (u16::from(b[1] & 0x80) << 8))
            .collect();
        let x = Tensor::new(gpu.up(&x_raw), t, HIDDEN as u32, Dtype::Bf16);
        let shared = Tensor::new(
            gpu.zeros(tokens * HIDDEN * 2),
            t,
            HIDDEN as u32,
            Dtype::Bf16,
        );
        let weights = Tensor::new(gpu.zeros(tokens * TOP * 4), t, TOP as u32, Dtype::F32);
        let y_at = gpu.zeros(tokens * HIDDEN * 2);
        let fire = |i: usize| {
            let mut y = Tensor::new(y_at, t, HIDDEN as u32, Dtype::Bf16);
            let routes = Tensor::new(routes_at[i % ROUTINGS], t, TOP as u32, Dtype::I32);
            assert!(
                experts_weighted_first(
                    &ctx,
                    x,
                    up,
                    Glu::SwigluClamp { limit: 10.0 },
                    down,
                    routes,
                    weights,
                    INTER as u32,
                    shared,
                    &mut y,
                )
                .expect("the routed MLP fires"),
                "the grouped GEMM takes the fire"
            );
        };
        // `PIE_W4_REPS` fires a point (64): thousands hold the GPU at its
        // power cap, as a model's decode does.
        let reps: usize = std::env::var("PIE_W4_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        // As many again first, to warm (and heat) the GPU.
        (0..reps.max(ROUTINGS)).for_each(&fire);
        gpu.sync();
        let t0 = Instant::now();
        (0..reps).for_each(&fire);
        gpu.sync();
        let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("| {tokens} | {us:.1} |");
    }
}
