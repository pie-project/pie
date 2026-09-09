#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm::{self, PostNorm};
use kernels_cuda::tensor::Tensor;

fn round(v: f32) -> f32 {
    from_bf16(to_bf16(v))
}

fn check(hidden: usize, window: Option<(u32, u32, u32)>) {
    let (rows, live, base) = window.unwrap_or((8, 8, 0));
    let planes = (base + rows) as usize;
    let mut lcg = Lcg::seeded(0x1234 ^ hidden as u64);
    let (x_raw, x) = lcg.row(planes * hidden);
    let (y_raw, y0) = lcg.row(planes * hidden);
    let (w0_raw, w0) = lcg.row(hidden);
    let (w1_raw, w1) = lcg.row(hidden);
    let s_raw = [to_bf16(0.75)];
    let (eps0, eps1) = (1e-6f32, 1e-6f32);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let y_at = gpu.up(&y_raw);
    let w0_at = gpu.up(&w0_raw);
    let w1_at = gpu.up(&w1_raw);
    let s_at = gpu.up(&s_raw);
    let t_at = gpu.zeros(planes * hidden * 2);
    let scaled_at = gpu.zeros(planes * hidden * 2);
    let out_at = gpu.zeros(planes * hidden * 2);
    let ctx = gpu.ctx();
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let plane = |at: u64| Tensor::new(at, rows, hidden as u32, Dtype::Bf16);
    let mut t = plane(t_at);
    let mut y = plane(y_at);
    let mut scaled = plane(scaled_at);
    let mut out = plane(out_at);
    norm::rmsnorm_residual_add(
        &ctx,
        plane(x_at),
        Tensor::new(w0_at, hidden as u32, 1, Dtype::Bf16),
        eps0,
        &mut t,
        &mut y,
        Some((Tensor::new(s_at, 1, 1, Dtype::Bf16), &mut scaled)),
        Some(PostNorm {
            weight: Tensor::new(w1_at, hidden as u32, 1, Dtype::Bf16),
            plus_one: true,
            eps: eps1,
            out: &mut out,
        }),
    )
    .expect("the chain fires");
    gpu.sync();
    let got_t: Vec<u16> = gpu.down(t_at, planes * hidden);
    let got_y: Vec<u16> = gpu.down(y_at, planes * hidden);
    let got_s: Vec<u16> = gpu.down(scaled_at, planes * hidden);
    let got_o: Vec<u16> = gpu.down(out_at, planes * hidden);

    let close = |got: u16, want: f32, what: &str, r: usize, i: usize| {
        let g = from_bf16(got);
        assert!(
            (g - want).abs() <= want.abs() * (1.0 / 64.0) + 1.5e-2,
            "hidden {hidden} window {window:?}: {what}[{r}][{i}] = {g}, want {want}"
        );
    };
    for r in 0..planes {
        let span = r * hidden..(r + 1) * hidden;
        let touched = r >= base as usize && r < (base + live) as usize;
        if !touched {
            assert_eq!(
                &got_y[span.clone()],
                &y_raw[span.clone()],
                "row {r}: a padded row moved"
            );
            continue;
        }
        let xr = &x[span.clone()];
        let ms0: f32 = xr.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv0 = 1.0 / (ms0 + eps0).sqrt();
        let mut fin = vec![0f32; hidden];
        for i in 0..hidden {
            let tv = round(xr[i] * inv0 * w0[i]);
            close(got_t[r * hidden + i], tv, "t", r, i);
            let folded = round(y0[r * hidden + i] + tv);
            close(got_y[r * hidden + i], folded, "y", r, i);
            let sv = round(folded * 0.75);
            close(got_s[r * hidden + i], sv, "scaled", r, i);
            fin[i] = sv;
        }
        let ms1: f32 = fin.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let inv1 = 1.0 / (ms1 + eps1).sqrt();
        for i in 0..hidden {
            close(
                got_o[r * hidden + i],
                fin[i] * inv1 * (w1[i] + 1.0),
                "out",
                r,
                i,
            );
        }
    }
}

#[test]
fn the_fused_chain_lands_the_four_launches_every_case() {
    the_eight_wide_chain_lands_the_four_launches();
    the_scalar_chain_lands_the_four_launches();
}

fn the_eight_wide_chain_lands_the_four_launches() {
    check(2560, None);
    check(2560, Some((8, 5, 2)));
}

fn the_scalar_chain_lands_the_four_launches() {
    check(2564, None);
    check(1028, Some((8, 3, 1)));
}
