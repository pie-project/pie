//! Device time of `linear::gemm::matmul` at Qwen 3.5 0.8B's projection
//! shapes, rows from decode to prefill.
//! `cargo test --release --features rocm --test bench_linear -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::linear::gemm;

/// (name, n, k)
pub const SHAPES: [(&str, u32, u32); 7] = [
    ("gdn_in_qkvz", 8192, 1024),
    ("gdn_out", 1024, 2048),
    ("attn_q_gate", 4096, 1024),
    ("attn_kv", 1024, 1024),
    ("mlp_gate_up", 7168, 1024),
    ("mlp_down", 1024, 3584),
    ("lm_head", 248_320, 1024),
];

pub const ROWS: [u32; 10] = [1, 4, 6, 8, 16, 64, 256, 1024, 4096, 8192];

#[test]
#[ignore = "benchmark"]
fn bench_linear() {
    let Some(gpu) = Gpu::open() else { return };
    // BENCH_LINEAR_SHAPES="n,k;n,k" and BENCH_LINEAR_ROWS="m,m" replace the
    // defaults.
    let shapes: Vec<(&str, u32, u32)> = std::env::var("BENCH_LINEAR_SHAPES").map_or_else(
        |_| SHAPES.to_vec(),
        |v| {
            v.split(';')
                .map(|s| {
                    let mut it = s.split(',').map(|x| x.parse().unwrap());
                    ("custom", it.next().unwrap(), it.next().unwrap())
                })
                .collect()
        },
    );
    let rows: Vec<u32> = std::env::var("BENCH_LINEAR_ROWS")
        .map_or_else(|_| ROWS.to_vec(), |v| v.split(',').map(|x| x.parse().unwrap()).collect());
    for (name, n, k) in shapes {
        let w = gpu.tensor(&values((n * k) as usize, 2, 0.05), n, k, Dtype::Bf16);
        for &m in &rows {
            if name == "lm_head" && m > 1024 {
                continue;
            }
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let mut y = gpu.zeros(m, n, Dtype::Bf16);
            let us = gpu.time_graph(20, || gemm::matmul(&gpu.ctx(), a, w, &mut y).unwrap());
            let tflops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k) / us / 1e6;
            let gbs = (f64::from(n) * f64::from(k) * 2.0 + f64::from(m) * f64::from(k + n) * 2.0)
                / us
                / 1e3;
            eprintln!(
                "pie {name:12} m={m:5} n={n:6} k={k:5}: {us:9.1} us {tflops:7.1} TFLOP/s {gbs:7.0} GB/s"
            );
        }
    }
}
