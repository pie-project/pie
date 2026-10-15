#![cfg(feature = "cuda")]

//! A Qwen3.6-35B-A3B MoE layer's bf16 selects (256 experts, top 8, hidden
//! 2048, 512 a gate or up) at decode widths: the gate-up leg read by token,
//! the down leg by route, each fire on the next of 16 routings. Ignored: a
//! measurement. Run alone on an idle GPU:
//!
//!     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//!         the_bf16_select_paces -- --ignored --nocapture
//!
//! `PIE_SELECT_TOKENS` (a comma list) keeps those token counts.

use crate::common::Gpu;
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select};
use kernels_cuda::tensor::Tensor;
use std::time::Instant;

const EXPERTS: usize = 256;
const TOP: usize = 8;
const HIDDEN: usize = 2048;
const INTER: usize = 512;
const ROUTINGS: usize = 16;

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
fn the_bf16_select_paces_a_qwen36_layer() {
    let mut gpu = Gpu::open();
    let (up_n, down_n) = (2 * INTER, HIDDEN);
    let up = gpu.zeros(EXPERTS * up_n * HIDDEN * 2);
    let down = gpu.zeros(EXPERTS * down_n * INTER * 2);
    let ctx = gpu.ctx();
    let bytes_of = |routes: &[i32], n: usize, k: usize| {
        let mut hit = [false; EXPERTS];
        routes.iter().for_each(|&e| hit[e as usize] = true);
        hit.iter().filter(|&&h| h).count() as f64 * (n * k * 2) as f64
    };
    println!("| tokens | leg | us | GB/s (distinct experts' bytes) |");
    println!("|---|---|---|---|");
    let counts: Vec<usize> = std::env::var("PIE_SELECT_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 16, 32, 64]);
    for tokens in counts {
        let routings: Vec<Vec<i32>> = (0..ROUTINGS)
            .map(|r| routing(tokens, 0x51ed + 7919 * r as u64 + tokens as u64))
            .collect();
        let routes_at: Vec<u64> = routings.iter().map(|r| gpu.up(r)).collect();
        let routes_t = |i: usize| {
            Tensor::new(
                routes_at[i % ROUTINGS],
                tokens as u32,
                TOP as u32,
                Dtype::I32,
            )
        };
        let x = Tensor::new(
            gpu.zeros(tokens * HIDDEN * 2),
            tokens as u32,
            HIDDEN as u32,
            Dtype::Bf16,
        );
        let h = Tensor::new(
            gpu.zeros(tokens * TOP * INTER * 2),
            (tokens * TOP) as u32,
            INTER as u32,
            Dtype::Bf16,
        );
        let up_out = gpu.zeros(tokens * TOP * up_n * 2);
        let down_out = gpu.zeros(tokens * TOP * down_n * 2);
        let gate_up = |i: usize| {
            let mut y = Tensor::new(up_out, (tokens * TOP) as u32, up_n as u32, Dtype::Bf16);
            matmul_select(
                &ctx,
                x,
                Tensor::new(up, EXPERTS as u32, (up_n * HIDDEN) as u32, Dtype::Bf16),
                routes_t(i),
                &mut y,
                ExpertTable::RESIDENT,
            )
            .expect("the gate-up select fires");
        };
        let down_leg = |i: usize| {
            let mut y = Tensor::new(down_out, (tokens * TOP) as u32, down_n as u32, Dtype::Bf16);
            matmul_select(
                &ctx,
                h,
                Tensor::new(down, EXPERTS as u32, (down_n * INTER) as u32, Dtype::Bf16),
                routes_t(i),
                &mut y,
                ExpertTable::RESIDENT,
            )
            .expect("the down select fires");
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
        let mean = |n: usize, k: usize| {
            routings.iter().map(|r| bytes_of(r, n, k)).sum::<f64>() / ROUTINGS as f64
        };
        let a = time(&gate_up);
        println!(
            "| {tokens} | gate-up | {a:.1} | {:.0} |",
            mean(up_n, HIDDEN) / a * 1e-3
        );
        let b = time(&down_leg);
        println!(
            "| {tokens} | down | {b:.1} | {:.0} |",
            mean(down_n, INTER) / b * 1e-3
        );
    }
}
