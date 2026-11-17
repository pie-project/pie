//! Probe (by hand): does a row's matmul land the same bits whatever rows
//! fire beside it? At Qwen3.5-0.8B's shapes, row 0 of an m-row fire against
//! the one-row fire.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::linear::gemm;

#[test]
#[ignore = "a probe run by hand"]
fn rows_agree_probe() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (name, n, k) in [
        ("attn q+gate", 4096u32, 1024u32),
        ("attn k|v", 512, 1024),
        ("attn qkv fused", 5120, 1024),
        ("attn o", 1024, 2048),
        ("lin qkv", 6144, 1024),
        ("lin z", 2048, 1024),
        ("lin b|a", 16, 1024),
        ("lin qkvz fused", 8192, 1024),
        ("lin out", 1024, 2048),
        ("mlp gate_up", 7168, 1024),
        ("mlp down", 1024, 3584),
        ("mtp fc", 1024, 2048),
        ("lm_head", 248320, 1024),
    ] {
        let a: Vec<f32> = values((8 * k) as usize, 1, 1.0).into_iter().map(round_bf16).collect();
        let w: Vec<f32> = values((n * k) as usize, 2, 0.05).into_iter().map(round_bf16).collect();
        let wt = gpu.tensor(&w, n, k, Dtype::Bf16);
        let mut base: Vec<Vec<f32>> = Vec::new();
        let mut line = format!("{name:<16} n {n:>6} k {k:>5}:");
        for m in 1..=5u32 {
            let at = gpu.tensor(&a[..(m * k) as usize], m, k, Dtype::Bf16);
            let mut y = gpu.zeros(m, n, Dtype::Bf16);
            if name == "lm_head" {
                gpu.laid(&ctx, |ctx| gemm::lm_head(ctx, at, wt, &mut y)).unwrap();
            } else {
                gpu.laid(&ctx, |ctx| gemm::matmul(ctx, at, wt, &mut y)).unwrap();
            }
            let got = gpu.read(&y);
            let rows: Vec<Vec<f32>> = (0..m as usize).map(|r| got[r * n as usize..(r + 1) * n as usize].to_vec()).collect();
            if m == 1 {
                base = vec![rows[0].clone()];
            }
            // row 0 against the one-row fire; row 1 against the two-row fire's
            if base.len() < 2 && m == 2 {
                base.push(rows[1].clone());
            }
            let d0 = rows[0].iter().zip(&base[0]).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            let d1 = if m >= 2 { rows[1].iter().zip(&base[1]).filter(|(x, y)| x.to_bits() != y.to_bits()).count() } else { 0 };
            line += &format!("  m{m}: row0 {d0} row1 {d1}");
        }
        eprintln!("[rows] {line}");
    }
}
