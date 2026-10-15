#![cfg(feature = "cuda")]

//! The hyper-connection projection (`elementwise.hc_project`) against a host
//! dot: DeepSeek-V4's four 4096-wide streams projected to a layer's 24 mix
//! columns, a sublayer's 8 without its combine, and a head's 4, at decode
//! and prefill row counts.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::elemwise::hc;
use kernels_cuda::tensor::Tensor;

fn check(rows: usize, streams: usize, hidden: usize, mix_hc: usize) {
    let fan = streams * hidden;
    let mut lcg = Lcg::seeded(0x4c0 + rows as u64 + mix_hc as u64);
    let normed: Vec<f32> = (0..rows * fan).map(|_| lcg.unit()).collect();
    let w: Vec<f32> = (0..mix_hc * fan).map(|_| lcg.unit() * 0.01).collect();
    let mut gpu = Gpu::open();
    let n_at = gpu.up(&normed);
    let w_at = gpu.up(&w);
    let m_at = gpu.zeros(rows * mix_hc * 4);
    let mut mixes = Tensor::new(m_at, rows as u32, mix_hc as u32, Dtype::F32);
    hc::project(
        &gpu.ctx(),
        Tensor::new(n_at, rows as u32, fan as u32, Dtype::F32),
        Tensor::new(w_at, mix_hc as u32, fan as u32, Dtype::F32),
        streams as u32,
        &mut mixes,
    )
    .expect("the projection fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(m_at, rows * mix_hc);
    for r in 0..rows {
        for o in 0..mix_hc {
            let want: f64 = (0..fan)
                .map(|d| f64::from(normed[r * fan + d]) * f64::from(w[o * fan + d]))
                .sum();
            let mag: f64 = (0..fan)
                .map(|d| (f64::from(normed[r * fan + d]) * f64::from(w[o * fan + d])).abs())
                .sum();
            let g = f64::from(got[r * mix_hc + o]);
            assert!(
                (g - want).abs() <= 1e-5 * mag.max(1e-3),
                "{rows} rows, {mix_hc} columns: row {r} column {o} is {g}, the dot {want}"
            );
        }
    }
}

#[test]
fn the_hyper_connection_projection_answers_the_dot() {
    check(8, 4, 4096, 24);
    check(70, 4, 4096, 24);
    check(3, 4, 4096, 8);
    check(5, 4, 4096, 4);
}

/// The stream rows' f32 norm against a host one, 16384 wide.
#[test]
fn the_hyper_connection_norm_answers_the_host() {
    use crate::common::{from_bf16, to_bf16};
    let (rows, width) = (5usize, 16384usize);
    let mut lcg = Lcg::seeded(0x40f);
    let x: Vec<u16> = (0..rows * width).map(|_| to_bf16(lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let y_at = gpu.zeros(rows * width * 4);
    let mut y = Tensor::new(y_at, rows as u32, width as u32, Dtype::F32);
    hc::rmsnorm_f32(
        &gpu.ctx(),
        Tensor::new(x_at, rows as u32, width as u32, Dtype::Bf16),
        1e-6,
        &mut y,
    )
    .expect("the norm fires");
    gpu.sync();
    let got: Vec<f32> = gpu.down(y_at, rows * width);
    for r in 0..rows {
        let row: Vec<f64> = (0..width)
            .map(|d| f64::from(from_bf16(x[r * width + d])))
            .collect();
        let s = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + 1e-6).sqrt();
        for d in 0..width {
            let want = row[d] * s;
            let g = f64::from(got[r * width + d]);
            assert!(
                (g - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "row {r} column {d}: {g} vs {want}"
            );
        }
    }
}

/// The combined gates (pre, post, the Sinkhorn combine and the pre-mixed
/// input) against the walk written out on the host, DeepSeek-V4's 4 streams.
#[test]
fn the_hyper_connection_gates_answer_the_host_walk() {
    use crate::common::{from_bf16, to_bf16};
    let (rows, m, h, iters) = (7usize, 4usize, 4096usize, 20usize);
    let (eps, alpha) = (1e-6f32, 2.0f32);
    let mix_hc = 2 * m + m * m;
    let mut lcg = Lcg::seeded(0x9a7e);
    let mixes: Vec<f32> = (0..rows * mix_hc).map(|_| lcg.unit() * 3.0).collect();
    let scale: Vec<f32> = vec![0.7, 1.3, 0.9];
    let base: Vec<f32> = (0..mix_hc).map(|_| lcg.unit()).collect();
    let streams: Vec<u16> = (0..rows * m * h).map(|_| to_bf16(lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let (x_at, post_at, comb_at) = (
        gpu.zeros(rows * h * 2),
        gpu.zeros(rows * m * 4),
        gpu.zeros(rows * m * m * 4),
    );
    let mut x = Tensor::new(x_at, rows as u32, h as u32, Dtype::Bf16);
    let mut post = Tensor::new(post_at, rows as u32, m as u32, Dtype::F32);
    let mut comb = Tensor::new(comb_at, rows as u32, (m * m) as u32, Dtype::F32);
    let mixes_t = Tensor::new(gpu.up(&mixes), rows as u32, mix_hc as u32, Dtype::F32);
    let streams_t = Tensor::new(gpu.up(&streams), rows as u32, (m * h) as u32, Dtype::Bf16);
    let scale_t = Tensor::new(gpu.up(&scale), 1, 3, Dtype::F32);
    let base_t = Tensor::new(gpu.up(&base), 1, mix_hc as u32, Dtype::F32);
    hc::gates(
        &gpu.ctx(),
        mixes_t,
        streams_t,
        scale_t,
        base_t,
        m as u32,
        eps,
        alpha,
        iters as u32,
        true,
        &mut x,
        &mut post,
        &mut comb,
    )
    .expect("the gates fire");
    gpu.sync();
    let gx: Vec<u16> = gpu.down(x_at, rows * h);
    let gp: Vec<f32> = gpu.down(post_at, rows * m);
    let gc: Vec<f32> = gpu.down(comb_at, rows * m * m);
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    for r in 0..rows {
        let row = &mixes[r * mix_hc..][..mix_hc];
        let pre: Vec<f32> = (0..m)
            .map(|i| sig(row[i] * scale[0] + base[i]) + eps)
            .collect();
        for i in 0..m {
            let want = sig(row[m + i] * scale[1] + base[m + i]) * alpha;
            assert!((gp[r * m + i] - want).abs() <= 1e-5, "row {r} post {i}");
        }
        let mut c: Vec<f32> = (0..m * m)
            .map(|t| row[2 * m + t] * scale[2] + base[2 * m + t])
            .collect();
        for i in 0..m {
            let mx = (0..m).map(|j| c[i * m + j]).fold(f32::MIN, f32::max);
            for j in 0..m {
                c[i * m + j] = (c[i * m + j] - mx).exp();
            }
            let s: f32 = (0..m).map(|j| c[i * m + j]).sum();
            for j in 0..m {
                c[i * m + j] = c[i * m + j] / s + eps;
            }
        }
        let col = |c: &mut Vec<f32>| {
            for j in 0..m {
                let s: f32 = (0..m).map(|i| c[i * m + j]).sum::<f32>() + eps;
                for i in 0..m {
                    c[i * m + j] /= s;
                }
            }
        };
        col(&mut c);
        for _ in 0..iters - 1 {
            for i in 0..m {
                let s: f32 = (0..m).map(|j| c[i * m + j]).sum::<f32>() + eps;
                for j in 0..m {
                    c[i * m + j] /= s;
                }
            }
            col(&mut c);
        }
        for t in 0..m * m {
            assert!(
                (gc[r * m * m + t] - c[t]).abs() <= 1e-4,
                "row {r} combine {t}: {} vs {}",
                gc[r * m * m + t],
                c[t]
            );
        }
        for d in 0..h {
            let want: f32 = (0..m)
                .map(|i| pre[i] * from_bf16(streams[(r * m + i) * h + d]))
                .sum();
            let g = from_bf16(gx[r * h + d]);
            assert!(
                (g - want).abs() <= 1e-2 * want.abs().max(0.05),
                "row {r} column {d}: {g} vs {want}"
            );
        }
    }
}
