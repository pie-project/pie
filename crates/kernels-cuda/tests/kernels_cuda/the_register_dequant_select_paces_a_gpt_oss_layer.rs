#![cfg(feature = "cuda")]

//! A gpt-oss-120b MoE layer's selects (128 experts, top 4, mxfp4 W4A16) at
//! decode widths: the register-dequant MMA kernel against the tcgen05 GEMM,
//! the gate-up leg with its GLU landed and the down leg bare. Each fire
//! takes the next of 16 routings, so it reads other experts' weights than
//! the last. Ignored: a measurement. Run alone on an idle GPU:
//!
//!     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//!         the_register_dequant_select_paces -- --ignored --nocapture
//!
//! `PIE_W4_TOKENS` (a comma list) keeps those token counts.

use crate::common::Gpu;
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, GluBank, GroupSeat, matmul_select_bias, select_glu};
use kernels_cuda::linear::tc5::Glu;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;
use std::time::Instant;

const EXPERTS: usize = 128;
const TOP: usize = 4;
const HIDDEN: usize = 2880;
const INTER: usize = 2880;
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
fn the_register_dequant_select_paces_a_gpt_oss_layer() {
    let mut gpu = Gpu::open();
    let (up_n, down_n) = (2 * INTER, HIDDEN);
    let up_codes = gpu.zeros(EXPERTS * up_n * HIDDEN / 2);
    let up_scales = gpu.up(&vec![127u8; EXPERTS * up_n * HIDDEN / 32]);
    let up_bias = gpu.zeros(EXPERTS * up_n * 2);
    let down_codes = gpu.zeros(EXPERTS * down_n * INTER / 2);
    let down_scales = gpu.up(&vec![127u8; EXPERTS * down_n * INTER / 32]);
    let down_bias = gpu.zeros(EXPERTS * down_n * 2);
    let tc5 = gpu.ctx().with_tuning(Tuning {
        moe_tc5: true,
        moe_tc5_from: 1,
        ..Tuning::DEFAULT
    });
    let old = gpu.ctx();
    let mma = gpu.ctx().with_tuning(Tuning {
        moe_tc5: true,
        mxfp4_mma_until: 65_536,
        mxfp4_mma_from: 1,
        ..Tuning::DEFAULT
    });
    // Bytes each fire reads: the routed experts' codes and scales.
    let bytes_of = |routes: &[i32], n: usize, k: usize| {
        let mut hit = [false; EXPERTS];
        routes.iter().for_each(|&e| hit[e as usize] = true);
        hit.iter().filter(|&&h| h).count() as f64 * (n * k / 2 + n * k / 32) as f64
    };
    println!(
        "| tokens | leg | registry default us | tcgen05 us | register-dequant us | GB/s (default / tcgen05 / register) |"
    );
    println!("|---|---|---|---|---|---|");
    let counts: Vec<usize> = std::env::var("PIE_W4_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 16, 32, 64, 128, 256, 512, 1024]);
    for tokens in counts {
        let routings: Vec<Vec<i32>> = (0..ROUTINGS)
            .map(|r| routing(tokens, 0x9e37 + 7919 * r as u64 + tokens as u64))
            .collect();
        let routes_at: Vec<u64> = routings.iter().map(|r| gpu.up(r)).collect();
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
        let y_at = gpu.zeros(tokens * TOP * down_n * 2);
        let full_at = gpu.zeros(tokens * TOP * up_n * 2);
        let routes_t = |i: usize| {
            Tensor::new(
                routes_at[i % ROUTINGS],
                tokens as u32,
                TOP as u32,
                Dtype::I32,
            )
        };
        let up = |ctx: &kernels_cuda::Ctx, i: usize| {
            let mut out = h;
            let fused = select_glu(
                ctx,
                x,
                GluBank::Mxfp4 {
                    codes: Tensor::new(
                        up_codes,
                        EXPERTS as u32,
                        (up_n * HIDDEN / 2) as u32,
                        Dtype::U8,
                    ),
                    scales: Tensor::new(
                        up_scales,
                        EXPERTS as u32,
                        (up_n * HIDDEN / 32) as u32,
                        Dtype::U8,
                    ),
                    bias: Some(Tensor::new(
                        up_bias,
                        EXPERTS as u32,
                        up_n as u32,
                        Dtype::Bf16,
                    )),
                },
                routes_t(i),
                &mut out,
                Glu::SwigluClampAlpha {
                    limit: 7.0,
                    alpha: 1.702,
                },
                ExpertTable::RESIDENT,
            )
            .expect("the gate-up select fires");
            if !fused {
                // The registry's arm: the select alone (its GLU apart).
                let mut full =
                    Tensor::new(full_at, (tokens * TOP) as u32, up_n as u32, Dtype::Bf16);
                matmul_select_bias(
                    ctx,
                    x,
                    Tensor::new(
                        up_codes,
                        EXPERTS as u32,
                        (up_n * HIDDEN / 2) as u32,
                        Dtype::U8,
                    ),
                    Tensor::new(
                        up_scales,
                        EXPERTS as u32,
                        (up_n * HIDDEN / 32) as u32,
                        Dtype::U8,
                    ),
                    Tensor::new(up_bias, EXPERTS as u32, up_n as u32, Dtype::Bf16),
                    routes_t(i),
                    &mut full,
                    GroupSeat::RESIDENT,
                )
                .expect("the gate-up select fires");
            }
        };
        let down = |ctx: &kernels_cuda::Ctx, i: usize| {
            let mut y = Tensor::new(y_at, (tokens * TOP) as u32, down_n as u32, Dtype::Bf16);
            matmul_select_bias(
                ctx,
                h,
                Tensor::new(
                    down_codes,
                    EXPERTS as u32,
                    (down_n * INTER / 2) as u32,
                    Dtype::U8,
                ),
                Tensor::new(
                    down_scales,
                    EXPERTS as u32,
                    (down_n * INTER / 32) as u32,
                    Dtype::U8,
                ),
                Tensor::new(down_bias, EXPERTS as u32, down_n as u32, Dtype::Bf16),
                routes_t(i),
                &mut y,
                GroupSeat::RESIDENT,
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
        let up_bytes = routings
            .iter()
            .map(|r| bytes_of(r, up_n, HIDDEN))
            .sum::<f64>()
            / ROUTINGS as f64;
        let down_bytes = routings
            .iter()
            .map(|r| bytes_of(r, down_n, INTER))
            .sum::<f64>()
            / ROUTINGS as f64;
        let (o, a, b) = (
            time(&|i| up(&old, i)),
            time(&|i| up(&tc5, i)),
            time(&|i| up(&mma, i)),
        );
        println!(
            "| {tokens} | gate-up + GLU | {o:.1} | {a:.1} | {b:.1} | {:.0} / {:.0} / {:.0} |",
            up_bytes / o * 1e-3,
            up_bytes / a * 1e-3,
            up_bytes / b * 1e-3
        );
        let (o, a, b) = (
            time(&|i| down(&old, i)),
            time(&|i| down(&tc5, i)),
            time(&|i| down(&mma, i)),
        );
        println!(
            "| {tokens} | down | {o:.1} | {a:.1} | {b:.1} | {:.0} / {:.0} / {:.0} |",
            down_bytes / o * 1e-3,
            down_bytes / a * 1e-3,
            down_bytes / b * 1e-3
        );
    }
}
