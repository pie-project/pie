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
                const REPS: usize = 64;
                (0..ROUTINGS).for_each(&f);
                gpu.sync();
                let t0 = Instant::now();
                (0..REPS).for_each(&f);
                gpu.sync();
                t0.elapsed().as_secs_f64() * 1e6 / REPS as f64
            };
            let (a, b) = (time(&tc5), time(&conv));
            println!("| {tokens} | {leg} | {a:.1} | {b:.1} |");
        }
    }
}
