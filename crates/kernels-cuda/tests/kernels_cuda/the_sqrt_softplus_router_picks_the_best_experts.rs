#![cfg(feature = "cuda")]

//! DeepSeek-V4's router (`linear.moe_topk_sqrt_softplus`): each row's `K`
//! best experts by `sqrt(softplus(logit)) + bias`, the lowest expert on a
//! tie, weighted by their unbiased scores renormalized and scaled. Read
//! against the serial walk on the host; the logits are spaced well apart
//! save the exact ties planted to test the tie rule.

use crate::common::{Gpu, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe;
use kernels_cuda::tensor::Tensor;

#[test]
fn the_sqrt_softplus_router_picks_the_best_experts() {
    let (rows, experts, k) = (9usize, 256usize, 6usize);
    let (renorm, scaling) = (true, 1.5f32);
    let mut logits: Vec<u16> = Vec::with_capacity(rows * experts);
    for r in 0..rows {
        for e in 0..experts {
            // A permutation of evenly spaced values, a few planted ties.
            let slot = (e * 97 + r * 31) % experts;
            let v = if r % 3 == 0 && e % 50 == 7 { 2.5 } else { -4.0 + 0.03 * slot as f32 };
            logits.push(to_bf16(v));
        }
    }
    let bias: Vec<f32> = (0..experts).map(|e| if e % 5 == 0 { 0.0 } else { -0.001 }).collect();
    let mut gpu = Gpu::open();
    let l_at = gpu.up(&logits);
    let b_at = gpu.up(&bias);
    let (i_at, w_at) = (gpu.zeros(rows * k * 4), gpu.zeros(rows * k * 4));
    let mut routes = Tensor::new(i_at, rows as u32, k as u32, Dtype::I32);
    let mut weights = Tensor::new(w_at, rows as u32, k as u32, Dtype::F32);
    moe::topk_sqrt_softplus(
        &gpu.ctx(),
        Tensor::new(l_at, rows as u32, experts as u32, Dtype::Bf16),
        Tensor::new(b_at, 1, experts as u32, Dtype::F32),
        experts as u32,
        k as u32,
        renorm,
        scaling,
        &mut routes,
        &mut weights,
    )
    .expect("the router fires");
    gpu.sync();
    let got_i: Vec<i32> = gpu.down(i_at, rows * k);
    let got_w: Vec<f32> = gpu.down(w_at, rows * k);
    let sp = |x: f32| {
        let s = if x > 20.0 { x } else { x.exp().ln_1p() };
        s.max(0.0).sqrt()
    };
    for r in 0..rows {
        let orig: Vec<f32> = (0..experts).map(|e| sp(from_bf16(logits[r * experts + e]))).collect();
        let mut scores: Vec<f32> = (0..experts).map(|e| orig[e] + bias[e]).collect();
        let mut want_i = Vec::new();
        let mut want_w = Vec::new();
        for _ in 0..k {
            let mut best = (-1i32, f32::MIN);
            for (e, &v) in scores.iter().enumerate() {
                if v > best.1 {
                    best = (e as i32, v);
                }
            }
            want_i.push(best.0);
            want_w.push(orig[best.0 as usize]);
            scores[best.0 as usize] = f32::MIN;
        }
        let sum: f32 = want_w.iter().sum();
        let scale = if renorm && sum > 0.0 { scaling / sum } else { scaling };
        assert_eq!(&got_i[r * k..(r + 1) * k], &want_i[..], "row {r}: the picked experts");
        for j in 0..k {
            let want = want_w[j] * scale;
            assert!(
                (got_w[r * k + j] - want).abs() <= 1e-5 * want.abs().max(1.0),
                "row {r} pick {j}: weight {} vs {want}",
                got_w[r * k + j]
            );
        }
    }
}
