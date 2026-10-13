//! Host cost of `gemm::matmul` launches at shapes a prefill meets: the first
//! call of a shape (plan) and the repeats.
//! `cargo test --release --features rocm --test bench_lt_host -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::linear::gemm;

#[test]
#[ignore = "benchmark"]
fn bench_lt_host() {
    let Some(gpu) = Gpu::open() else { return };
    let (n, k) = (16384u32, 5376u32);
    let w = gpu.tensor(&values((n * k) as usize, 2, 0.05), n, k, Dtype::Bf16);
    for m in [256u32, 256, 200, 201, 8192, 8192, 1000] {
        let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
        let mut y = gpu.zeros(m, n, Dtype::Bf16);
        gpu.sync();
        let t = std::time::Instant::now();
        gemm::matmul(&gpu.ctx(), a, w, &mut y).unwrap();
        let first = t.elapsed().as_micros();
        gpu.sync();
        let t = std::time::Instant::now();
        for _ in 0..10 {
            gemm::matmul(&gpu.ctx(), a, w, &mut y).unwrap();
        }
        let host = t.elapsed().as_micros() / 10;
        gpu.sync();
        let dev = gpu.time(10, || gemm::matmul(&gpu.ctx(), a, w, &mut y).unwrap());
        eprintln!("m={m:5}: first call {first} us host, repeat {host} us host, device {dev:.1} us");
    }
}
