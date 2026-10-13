//! The batch engine's bf16 projection against hipBLASLt's heuristic pick at
//! the dense models' decode shapes (Gemma-4 31B, Qwen3.8-27B), with a cold
//! weight: the launches cycle through copies that together outgrow the
//! 256 MiB last-level cache, captured in one HIP graph. The gate/up shapes
//! also run fused with the SwiGLU, against hipBLASLt then the swiglu.
//! `cargo test --release --features rocm --test bench_dense_batch -- --ignored --nocapture`
//! (`BENCH_NK="n,k;n,k"`, `BENCH_M="m,m"` narrow it).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::rt;
use kernels_rocm::linear::{gemm, mlp};
use kernels_rocm::tuning::Tuning;

/// The copies of one weight, past the last-level cache together.
const COLD_BYTES: usize = 768 << 20;

#[test]
#[ignore = "benchmark"]
fn bench_dense_batch() {
    let Some(gpu) = Gpu::open() else { return };
    // (n, k, gate/up): the gate/up banks fuse with the SwiGLU at n / 2.
    let shapes: Vec<(u32, u32, bool)> = std::env::var("BENCH_NK").ok().map_or(
        vec![
            (43008, 5376, true),
            (5376, 21504, false),
            (16384, 5376, false),
            (5376, 8192, false),
            (34816, 5120, true),
            (5120, 17408, false),
            (16384, 5120, false),
            (5120, 6144, false),
        ],
        |v| {
            v.split(';')
                .map(|p| {
                    let mut it = p.split(',').map(|x| x.parse().unwrap());
                    let (n, k) = (it.next().unwrap(), it.next().unwrap());
                    // A third field marks a gate/up bank.
                    (n, k, it.next().is_some())
                })
                .collect()
        },
    );
    let ms: Vec<u32> = std::env::var("BENCH_M")
        .ok()
        .map_or(vec![8, 16, 32, 64, 128, 256], |v| {
            v.split(',').map(|x| x.parse().unwrap()).collect()
        });
    let mut batch = Tuning::DEFAULT;
    batch.dense_batch_from = 1;
    batch.dense_batch_until = 4096;
    batch.dense_batch_weight_mi = 0;
    batch.dense_batch_weight_max_mi = 65_536;
    let mut lt = Tuning::DEFAULT;
    lt.dense_batch_until = 0;
    let (on, off) = (gpu.ctx().with_tuning(batch), gpu.ctx().with_tuning(lt));
    for (n, k, gate_up) in shapes {
        let block = gpu.tensor(&values((64 * k) as usize, 2, 0.05), 64, k, Dtype::Bf16);
        let bytes = n as usize * k as usize * 2;
        let copies: Vec<Tensor> = (0..(COLD_BYTES / bytes + 1).max(2))
            .map(|_| {
                let w = gpu.zeros(n, k, Dtype::Bf16);
                let chunk = 64 * k as usize * 2;
                for at in (0..bytes).step_by(chunk) {
                    // SAFETY: both ranges are live device allocations of at
                    // least these sizes.
                    unsafe {
                        rt::hipMemcpy(
                            (w.ptr as usize + at) as *mut _,
                            block.ptr as usize as *const _,
                            chunk.min(bytes - at),
                            rt::hipMemcpyDeviceToDevice,
                        );
                    }
                }
                w
            })
            .collect();
        let reps = (2 * copies.len()).max(16) as u32;
        let tb = |us: f64| bytes as f64 / us / 1e6;
        for m in ms.iter().copied() {
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let mut y = gpu.zeros(m, n, Dtype::Bf16);
            let mut time = |ctx: &kernels_rocm::Ctx| {
                let mut i = 0;
                gpu.time_graph(reps, || {
                    gemm::matmul(ctx, a, copies[i % copies.len()], &mut y).unwrap();
                    i += 1;
                })
            };
            let (b, l) = (time(&on), time(&off));
            println!(
                "m={m:4} n={n:5} k={k:5}: batch {b:7.1} us {:4.2} TB/s  hipblaslt {l:7.1} us {:4.2} TB/s  x{:.2}",
                tb(b),
                tb(l),
                l / b
            );
            if gate_up {
                let i = n / 2;
                let mut h = gpu.zeros(m, i, Dtype::Bf16);
                let mut packed = gpu.zeros(m, n, Dtype::Bf16);
                let mut j = 0;
                let fused = gpu.time_graph(reps, || {
                    assert!(gemm::matmul_swiglu(&on, a, copies[j % copies.len()], &mut h).unwrap());
                    j += 1;
                });
                let mut j = 0;
                let apart = gpu.time_graph(reps, || {
                    let w = copies[j % copies.len()];
                    if !gemm::matmul_swiglu(&off, a, w, &mut h).unwrap() {
                        gemm::matmul(&off, a, w, &mut packed).unwrap();
                        mlp::swiglu(&off, packed, i, 1, &mut h).unwrap();
                    }
                    j += 1;
                });
                println!(
                    "m={m:4} n={n:5} k={k:5}: swiglu fused {fused:7.1} us {:4.2} TB/s  hipblaslt+swiglu {apart:7.1} us  x{:.2}",
                    tb(fused),
                    apart / fused
                );
            }
        }
    }
}
