//! Every hipBLASLt algorithm the heuristic offers at the dense models' decode
//! and mixed-batch projection shapes (Gemma-4 31B, Qwen3.8-27B), timed with a
//! cold weight: the launches cycle through copies that together outgrow the
//! 256 MiB last-level cache, as a decode step streams its projections.
//! `cargo test --release --features rocm --test bench_lt_survey -- --ignored --nocapture`
//! (`SURVEY_NK="n,k;n,k"`, `SURVEY_M="m,m"` narrow it).
//!
//! On MI300X these shapes run at the 750 W socket limit (the gfx clock sits
//! near 1.1 GHz, pure weight streaming alone already draws ~740 W), so the
//! heuristic's pick is within ~15% of the library's best everywhere.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::hip::{lt, rt};

/// The copies of one weight, past the last-level cache together.
const COLD_BYTES: usize = 768 << 20;

#[test]
#[ignore = "benchmark"]
fn bench_lt_survey() {
    let Some(gpu) = Gpu::open() else { return };
    let ws = gpu.zeros(1, (lt::WORKSPACE / 2) as u32, Dtype::Bf16);
    let shapes: Vec<(u32, u32)> = std::env::var("SURVEY_NK").ok().map_or(
        vec![
            (43008, 5376),
            (5376, 21504),
            (16384, 5376),
            (5376, 8192),
            (34816, 5120),
            (5120, 17408),
            (16384, 5120),
            (5120, 6144),
        ],
        |v| {
            v.split(';')
                .map(|p| {
                    let mut it = p.split(',').map(|x| x.parse().unwrap());
                    (it.next().unwrap(), it.next().unwrap())
                })
                .collect()
        },
    );
    let ms: Vec<u32> = std::env::var("SURVEY_M")
        .ok()
        .map_or(vec![64, 128, 192, 256, 363, 512, 768, 1024], |v| {
            v.split(',').map(|x| x.parse().unwrap()).collect()
        });
    for (n, k) in shapes {
        // One 64-row block of values, tiled down every copy.
        let block = gpu.tensor(&values((64 * k) as usize, 2, 0.05), 64, k, Dtype::Bf16);
        let bytes = n as usize * k as usize * 2;
        let copies: Vec<_> = (0..(COLD_BYTES / bytes + 1).max(2))
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
        let ptrs: Vec<u64> = copies.iter().map(|w| w.ptr).collect();
        for m in ms.iter().copied() {
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let y = gpu.zeros(m, n, Dtype::Bf16);
            let s = lt::Shape {
                m,
                n,
                k,
                bias: false,
            };
            let reps = (2 * ptrs.len()).max(20) as u32;
            // SAFETY: the operands are live device tensors of the shape's
            // extents; the stream is idle.
            let mut got = unsafe {
                lt::survey(
                    s,
                    a.ptr,
                    &ptrs,
                    y.ptr,
                    ws.ptr,
                    gpu.stream.raw().cast(),
                    64,
                    reps,
                )
            }
            .unwrap();
            let first = got
                .iter()
                .find(|(i, _, _)| *i == 0)
                .map_or(f64::NAN, |x| x.1);
            got.sort_by(|a, b| a.1.total_cmp(&b.1));
            let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
            if let Some((i, us, wsb)) = got.first().copied() {
                eprintln!(
                    "m={m:4} n={n:5} k={k:5}: {} algos, heuristic#0 {first:7.1} us \
                     ({:4.0} TF/s, {:4.2} TB/s), best #{i} {us:7.1} us (x{:.2}, ws {wsb})",
                    got.len(),
                    flops / first / 1e6,
                    bytes as f64 / first / 1e6,
                    first / us,
                );
            }
        }
    }
}
