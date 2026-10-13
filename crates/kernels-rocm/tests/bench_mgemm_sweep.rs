//! The decode-batch GEMM's tilings swept at Qwen3.8-27B's projection shapes,
//! each against hipBLASLt, weights cycled past the last-level cache.
//! `cargo test --release --features rocm --test bench_mgemm_sweep -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::linear::{force_mgemm, gemm};

#[test]
#[ignore = "benchmark"]
fn bench_mgemm_sweep() {
    let Some(gpu) = Gpu::open() else { return };
    let ms: Vec<u32> = std::env::var("SWEEP_M").ok().map_or(vec![128], |v| {
        v.split(',').map(|x| x.parse().unwrap()).collect()
    });
    for (name, n, k) in [
        ("qkvz", 16384u32, 5120u32),
        ("out", 5120, 6144),
        ("gate_up", 34816, 5120),
        ("down", 5120, 17408),
    ] {
        let copies = (512u64 << 20)
            .div_ceil(u64::from(n) * u64::from(k) * 2)
            .max(2) as usize;
        let host = values((n * k) as usize, 2, 0.05);
        let ws: Vec<_> = (0..copies)
            .map(|_| gpu.tensor(&host, n, k, Dtype::Bf16))
            .collect();
        for &m in &ms {
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let mut y = gpu.zeros(m, n, Dtype::Bf16);
            let mut time = |label: String| {
                let mut at = 0usize;
                let us = gpu.time_graph(8 * copies as u32, || {
                    at = (at + 1) % copies;
                    gemm::matmul(&gpu.ctx(), a, ws[at], &mut y).unwrap()
                });
                (us, label)
            };
            let mut rows = Vec::new();
            if let Ok(one) = std::env::var("SWEEP_ONE") {
                let f: Vec<u32> = one.split(',').map(|x| x.parse().unwrap()).collect();
                force_mgemm(Some([f[0], f[1], f[2], f[3]]));
                let (us, label) = time(one.clone());
                force_mgemm(None);
                eprintln!("{name:8} m={m:3}: {us:7.1} us  {label}");
                continue;
            }
            for nbw in [1u32, 2] {
                for kc in [32u32] {
                    for depth in [2u32, 3] {
                        for splits in [1u32, 2, 4, 8] {
                            force_mgemm(Some([nbw, kc, depth, splits]));
                            rows.push(time(format!(
                                "nbw={nbw} kc={kc} depth={depth} splits={splits}"
                            )));
                        }
                    }
                }
            }
            force_mgemm(None);
            rows.sort_by(|a, b| a.0.total_cmp(&b.0));
            let bytes = f64::from(n) * f64::from(k) * 2.0;
            for (us, label) in rows.iter().take(4) {
                eprintln!(
                    "{name:8} m={m:3}: {us:7.1} us {:5.0} GB/s  {label}",
                    bytes / us / 1e3
                );
            }
        }
    }
}
