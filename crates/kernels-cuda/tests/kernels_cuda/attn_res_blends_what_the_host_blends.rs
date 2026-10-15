#![cfg(feature = "cuda")]

//! Kimi-K3's AttnRes blend (`elementwise.res_blend`): each token's closed
//! blocks and running prefix scored by the RMS-normed dot with a projection,
//! softmaxed, and blended under it; the blocks read where they lie, apart.
//! Read against the host, at Kimi's width (16-byte runs) and an odd one.

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm::res_blend;
use kernels_cuda::tensor::Tensor;

#[test]
fn attn_res_blends_what_the_host_blends() {
    for (tokens, hidden, blocks) in [
        (1usize, 7168usize, 3usize),
        (37, 7168, 8),
        (5, 1001, 2),
        (3, 64, 0),
    ] {
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded(0xa7 ^ (tokens * hidden + blocks) as u64);
        let (prefix_raw, _) = lcg.row(tokens * hidden);
        let planes: Vec<Vec<u16>> = (0..blocks).map(|_| lcg.row(tokens * hidden).0).collect();
        let (norm_raw, _) = lcg.row(hidden);
        let (proj_raw, _) = lcg.row(hidden);
        let eps = 1e-5f32;
        let prefix = Tensor::new(
            gpu.up(&prefix_raw),
            tokens as u32,
            hidden as u32,
            Dtype::Bf16,
        );
        // Each block its own allocation: the blend reads them apart.
        let block_t: Vec<Tensor> = planes
            .iter()
            .map(|p| Tensor::new(gpu.up(p), tokens as u32, hidden as u32, Dtype::Bf16))
            .collect();
        let norm = Tensor::new(gpu.up(&norm_raw), 1, hidden as u32, Dtype::Bf16);
        let proj = Tensor::new(gpu.up(&proj_raw), 1, hidden as u32, Dtype::Bf16);
        let mut y = Tensor::new(
            gpu.zeros(tokens * hidden * 2),
            tokens as u32,
            hidden as u32,
            Dtype::Bf16,
        );
        res_blend(&gpu.ctx(), prefix, &block_t, norm, eps, proj, &mut y).expect("the blend fires");
        gpu.sync();
        let got: Vec<u16> = gpu.down(y.ptr, tokens * hidden);
        let f = |v: &[u16], i: usize| f64::from(from_bf16(v[i]));
        for t in 0..tokens {
            let rows: Vec<&Vec<u16>> = planes.iter().chain(std::iter::once(&prefix_raw)).collect();
            let logits: Vec<f64> = rows
                .iter()
                .map(|r| {
                    let ss: f64 = (0..hidden).map(|h| f(r, t * hidden + h).powi(2)).sum();
                    let scale = 1.0 / (ss / hidden as f64 + f64::from(eps)).sqrt();
                    (0..hidden)
                        .map(|h| f(r, t * hidden + h) * scale * f(&norm_raw, h) * f(&proj_raw, h))
                        .sum()
                })
                .collect();
            let m = logits.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
            let sum: f64 = e.iter().sum();
            for h in 0..hidden {
                let want: f64 = rows
                    .iter()
                    .zip(&e)
                    .map(|(r, p)| p / sum * f(r, t * hidden + h))
                    .sum();
                let g = from_bf16(got[t * hidden + h]);
                assert!(
                    close(g, want as f32),
                    "{tokens} x {hidden}, {blocks} blocks: token {t} column {h}: {g} vs {want}"
                );
            }
        }
    }
}
