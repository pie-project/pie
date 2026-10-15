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
