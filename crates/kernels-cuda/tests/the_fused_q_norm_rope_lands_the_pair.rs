#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope;
use kernels_cuda::tensor::Tensor;

fn check(head_dim: usize, heads: usize, rotary_dim: usize, window: Option<(u32, u32, u32)>) {
    let (rows, live, base) = window.unwrap_or((6, 6, 0));
    let planes = (base + rows) as usize;
    let width = heads * head_dim;
    let mut lcg = Lcg::seeded(0x9e ^ (head_dim as u64) ^ ((rotary_dim as u64) << 8));
    let (x_raw, x) = lcg.row(planes * width);
    let (w_raw, w) = lcg.row(head_dim);
    let positions: Vec<i32> = (0..planes as i32).map(|r| 3 + 7 * r).collect();
    let (y_raw, _) = lcg.row(planes * width);
    let (theta, eps) = (1.0e6f32, 1e-6f32);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let p_at = gpu.up(&positions);
    let y_at = gpu.up(&y_raw);
    let ctx = gpu.ctx();
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let mut y = Tensor::new(y_at, rows, width as u32, Dtype::Bf16);
    rope::rmsnorm_rope_partial_q(
        &ctx,
        Tensor::new(x_at, rows, width as u32, Dtype::Bf16),
        Tensor::new(w_at, head_dim as u32, 1, Dtype::Bf16),
        head_dim as u32,
        eps,
        Tensor::new(p_at, rows, 1, Dtype::I32),
        rotary_dim as u32,
        theta,
        &mut y,
    )
    .expect("the fused q path fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, planes * width);

    let half = head_dim / 2;
    for r in 0..planes {
        let touched = r >= base as usize && r < (base + live) as usize;
        if !touched {
            assert_eq!(
                &got[r * width..(r + 1) * width],
                &y_raw[r * width..(r + 1) * width],
                "row {r} moved"
            );
            continue;
        }
        for h in 0..heads {
            let at = r * width + h * head_dim;
            let xr = &x[at..at + head_dim];
            let ms: f32 = xr.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for dp in 0..half {
                let a = xr[dp] * inv * w[dp];
                let b = xr[dp + half] * inv * w[dp + half];
                let (wa, wb) = if dp < rotary_dim / 2 {
                    let freq = theta.powf(-2.0 * dp as f32 / head_dim as f32);
                    let ang = positions[r] as f32 * freq;
                    let (s, c) = ang.sin_cos();
                    (a * c - b * s, b * c + a * s)
                } else {
                    (a, b)
                };
                for (i, want) in [(dp, wa), (dp + half, wb)] {
                    let g = from_bf16(got[at + i]);
                    assert!(
                        (g - want).abs() <= want.abs() * (1.0 / 64.0) + 1.5e-2,
                        "head_dim {head_dim} rotary {rotary_dim} window {window:?}: y[{r}][{h}][{i}] = {g}, want {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_fused_q_norm_rope_lands_the_pair_every_case() {
    a_full_rotary_head_lands_the_pair();
    a_partial_rotary_head_lands_the_pair_and_leaves_the_rest_normed();
}

fn a_full_rotary_head_lands_the_pair() {
    check(256, 4, 256, None);
}

fn a_partial_rotary_head_lands_the_pair_and_leaves_the_rest_normed() {
    check(256, 8, 128, None);
    check(128, 2, 64, Some((6, 4, 1)));
}
