//! Every hipBLASLt algorithm the heuristic offers at Qwen3.8-27B's decode
//! projection shapes, timed.
//! `cargo test --release --features rocm --test bench_lt_survey -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::hip::lt;

#[test]
#[ignore = "benchmark"]
fn bench_lt_survey() {
    let Some(gpu) = Gpu::open() else { return };
    let ws = gpu.zeros(1, (lt::WORKSPACE / 2) as u32, Dtype::Bf16);
    let shapes: Vec<(u32, u32)> = std::env::var("SURVEY_NK").ok().map_or(
        vec![(34816, 5120), (5120, 17408), (16384, 5120), (5120, 6144)],
        |v| {
            v.split(';')
                .map(|p| {
                    let mut it = p.split(',').map(|x| x.parse().unwrap());
                    (it.next().unwrap(), it.next().unwrap())
                })
                .collect()
        },
    );
    for (n, k) in shapes {
        let w = gpu.tensor(&values((n * k) as usize, 2, 0.05), n, k, Dtype::Bf16);
        let ms: Vec<u32> = std::env::var("SURVEY_M")
            .ok()
            .map_or(vec![32, 64, 128, 256], |v| {
                v.split(',').map(|x| x.parse().unwrap()).collect()
            });
        for m in ms.iter().copied() {
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let y = gpu.zeros(m, n, Dtype::Bf16);
            let s = lt::Shape {
                m,
                n,
                k,
                bias: false,
            };
            let mut got = unsafe {
                lt::survey(
                    s,
                    a.ptr,
                    w.ptr,
                    y.ptr,
                    ws.ptr,
                    gpu.stream.raw().cast(),
                    64,
                    10,
                )
            }
            .unwrap();
            let first = got
                .iter()
                .find(|(i, _, _)| *i == 0)
                .map(|x| x.1)
                .unwrap_or(f64::NAN);
            got.sort_by(|a, b| a.1.total_cmp(&b.1));
            let best = got.first().copied();
            let bytes = f64::from(n) * f64::from(k) * 2.0;
            if let Some((i, us, wsb)) = best {
                eprintln!(
                    "m={m:3} n={n:5} k={k:5}: {} algos, heuristic#0 {first:7.1} us, best #{i} {us:7.1} us ({:5.0} GB/s, ws {wsb})",
                    got.len(),
                    bytes / us / 1e3
                );
            }
        }
    }
}
