//! The K-quant matmuls' speed on the first Vulkan device: the hand-written
//! kernel beside kernels-slang's tiers, the same weights and rows, the Slang
//! results checked against the hand-written ones. Ignored: it asks for a real
//! GPU (`cargo test kquant_speed -- --ignored --nocapture`). Each kernel fires
//! `REPS` times a frame; the best of `TRIALS` frames is reported.

use std::time::Instant;

use kernels_slang::linear::kquant::{Scheme, Tactic};
use kernels_vulkan::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const REPS: usize = 50;
const TRIALS: usize = 7;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

/// The best trial's microseconds a dispatch.
fn time(gpu: &Gpu, run: &dyn Fn(&Gpu)) -> f64 {
    run(gpu);
    (0..TRIALS)
        .map(|_| {
            let t0 = Instant::now();
            run(gpu);
            t0.elapsed().as_secs_f64() * 1e6 / REPS as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// Random words with every super-block's scales a small f16.
fn weights(scheme: Scheme, blocks: usize) -> Vec<u32> {
    let width = scheme.block_bytes() as usize;
    let mut bytes: Vec<u8> = noise(9, blocks * width)
        .iter()
        .map(|v| (v.to_bits() >> 7) as u8)
        .collect();
    let scales: &[usize] = match scheme {
        Scheme::Q2K => &[80, 82],
        Scheme::Q3K => &[108],
        Scheme::Q4K | Scheme::Q5K => &[0, 2],
        Scheme::Q6K => &[208],
    };
    for block in bytes.chunks_mut(width) {
        for at in scales {
            block[*at..*at + 2].copy_from_slice(&0x1400u16.to_le_bytes());
        }
    }
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

#[test]
#[ignore = "a benchmark: a real GPU"]
fn kquant_speed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let chip = gpu.device.chip();
    eprintln!("device: {} (subgroup {})", chip.name, chip.subgroup);
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (n, k) = (4096usize, 4096usize);
    for scheme in [Scheme::Q4K, Scheme::Q6K, Scheme::Q3K] {
        let row_bytes = k / 256 * scheme.block_bytes() as usize;
        let w = gpu.up_u32(&weights(scheme, n * k / 256));
        let wt = t(w, n, row_bytes, Dtype::U8);
        for m in [1usize, 16] {
            let xt = t(gpu.up(&noise(3, m * k)).0, m, k, Dtype::Bf16);
            let ht = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
            let hand = time(&gpu, &|g| {
                g.run(|c| {
                    for _ in 0..REPS {
                        kernels_vulkan::linear::kquant::matmul(c, xt, wt, ht)?;
                    }
                    Ok(())
                })
            });
            let want: Vec<f64> = gpu
                .down(ht.buf, m * n)
                .iter()
                .map(|v| f64::from(*v))
                .collect();
            let mut line = format!("{scheme:?} m {m}: hand {hand:.1} us");
            for (subgroup, slices) in [(false, 2), (false, 4), (true, 2), (true, 4)] {
                if subgroup && chip.subgroup < 32 {
                    continue;
                }
                let tier = Tactic {
                    tier: if subgroup {
                        kernels_slang::Tier::Subgroup
                    } else {
                        kernels_slang::Tier::Baseline
                    },
                    slices,
                };
                let yt = t(gpu.zeros(m * n), m, n, Dtype::Bf16);
                let slang = time(&gpu, &|g| {
                    g.run_slang(|c| {
                        for _ in 0..REPS {
                            kernels_slang::linear::kquant::matmul(c, tier, st(xt), st(wt), st(yt))?;
                        }
                        Ok(())
                    })
                });
                crate::gpu::close(
                    &gpu.down(yt.buf, m * n),
                    &want,
                    &format!("{scheme:?} m {m} subgroup {subgroup} slices {slices}"),
                );
                line += &format!(
                    ", sg {subgroup} s{slices} {slang:.1} us ({:+.1}%)",
                    (slang / hand - 1.0) * 100.0
                );
            }
            eprintln!("{line}");
        }
    }
}
