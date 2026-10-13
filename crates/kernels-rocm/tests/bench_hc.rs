//! Device time of the hyper-connection kernels at the decode shapes of
//! GLM-5.3-Flash (H 4096), DeepSeek-V4.1-Flash (H 5120) and Qwen3.8 Flash
//! Next (H 2560), four streams each. Each call reads planes of its own, as
//! many as it takes to step past the 256 MiB last-level cache, so a call
//! reads cold planes as a decode step's does.
//! `cargo test --release --features rocm --test bench_hc -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::elemwise::hc;

const M: u32 = 4;
const ROWS: [u32; 5] = [1, 7, 32, 128, 512];

/// The bytes of planes a timing walks before it repeats one.
const COLD_BYTES: f64 = 400.0 * 1024.0 * 1024.0;

fn report(op: &str, h: u32, n: u32, us: f64, bytes: f64) {
    eprintln!(
        "{op:16} H={h:5} rows={n:4}: {us:8.2} us {:7.0} GB/s",
        bytes / us / 1e3
    );
}

/// Copies of one host plane on the device.
fn copies(gpu: &Gpu, host: &[f32], rows: u32, width: u32, dtype: Dtype, n: usize) -> Vec<Tensor> {
    (0..n)
        .map(|_| gpu.tensor(host, rows, width, dtype))
        .collect()
}

/// Mean device microseconds of `f(i)` over `sets` calls, call i on set i.
fn time_sets(gpu: &Gpu, sets: usize, mut f: impl FnMut(usize)) -> f64 {
    let mut i = 0;
    gpu.time_graph(sets as u32, || {
        f(i % sets);
        i += 1;
    })
}

#[test]
#[ignore = "benchmark"]
fn bench_hc() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let mix_hc = 2 * M + M * M;
    for h in [4096u32, 5120, 2560] {
        let wide = M * h;
        let fn_w = gpu.tensor(
            &values((mix_hc * wide) as usize, 2, 0.05),
            mix_hc,
            wide,
            Dtype::F32,
        );
        let scale = gpu.tensor(&[0.9, 1.1, 1.3], 1, 3, Dtype::F32);
        let base = gpu.tensor(&values(mix_hc as usize, 4, 0.5), 1, mix_hc, Dtype::F32);
        for n in ROWS {
            let (nf, wf, hf) = (f64::from(n), f64::from(wide), f64::from(h));
            // A set: the streams, their f32 norm and a narrow row or two.
            let sets =
                ((COLD_BYTES / (nf * wf * 8.0 + nf * hf * 4.0)).ceil() as usize).clamp(4, 256);
            let streams_h = values((n * wide) as usize, 1, 2.0);
            let streams = copies(&gpu, &streams_h, n, wide, Dtype::Bf16, sets);
            if h != 2560 {
                let mut normed: Vec<Tensor> =
                    (0..sets).map(|_| gpu.zeros(n, wide, Dtype::F32)).collect();
                let us = time_sets(&gpu, sets, |i| {
                    hc::rmsnorm_f32(&ctx, streams[i], 1e-6, &mut normed[i]).unwrap();
                });
                report("hc_rmsnorm_f32", h, n, us, nf * wf * 6.0);
                let mut mixes: Vec<Tensor> = (0..sets)
                    .map(|_| gpu.zeros(n, mix_hc, Dtype::F32))
                    .collect();
                let us = time_sets(&gpu, sets, |i| {
                    hc::project(&ctx, normed[i], fn_w, M, &mut mixes[i]).unwrap();
                });
                report(
                    "hc_project",
                    h,
                    n,
                    us,
                    nf * wf * 4.0 + f64::from(mix_hc) * wf * 4.0,
                );
                let us = time_sets(&gpu, sets, |i| {
                    hc::norm_project(&ctx, streams[i], fn_w, 1e-6, M, &mut mixes[i]).unwrap();
                });
                report(
                    "hc_norm_project",
                    h,
                    n,
                    us,
                    nf * wf * 2.0 + f64::from(mix_hc) * wf * 4.0,
                );
                let mut x: Vec<Tensor> = (0..sets).map(|_| gpu.zeros(n, h, Dtype::Bf16)).collect();
                let mut post = gpu.zeros(n, M, Dtype::F32);
                let mut comb = gpu.zeros(n, M * M, Dtype::F32);
                let us = time_sets(&gpu, sets, |i| {
                    hc::gates(
                        &ctx, mixes[i], streams[i], scale, base, M, 1e-6, 2.0, 20, &mut x[i],
                        &mut post, &mut comb,
                    )
                    .unwrap();
                });
                report("hc_gates", h, n, us, nf * (wf + hf) * 2.0);
                let us = time_sets(&gpu, sets, |i| {
                    let mut y = streams[i];
                    hc::fold(&ctx, x[i], streams[i], post, comb, &mut y).unwrap();
                });
                report("hc_fold", h, n, us, nf * (2.0 * wf + hf) * 2.0);
            } else {
                let gates = copies(
                    &gpu,
                    &values((n * wide) as usize, 3, 2.0),
                    n,
                    wide,
                    Dtype::Bf16,
                    sets,
                );
                let mut y: Vec<Tensor> = (0..sets).map(|_| gpu.zeros(n, h, Dtype::Bf16)).collect();
                let us = time_sets(&gpu, sets, |i| {
                    hc::mix(&ctx, gates[i], streams[i], M, &mut y[i]).unwrap();
                });
                report("hc_mix", h, n, us, nf * (2.0 * wf + hf) * 2.0);
                let g = gpu.tensor(&values((n * M) as usize, 5, 2.0), n, M, Dtype::Bf16);
                let us = time_sets(&gpu, sets, |i| {
                    let mut hyper = streams[i];
                    hc::inject(&ctx, y[i], g, M, &mut hyper).unwrap();
                });
                report("hc_inject", h, n, us, nf * (2.0 * wf + hf) * 2.0);
            }
        }
    }
}
