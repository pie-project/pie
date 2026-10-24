//! The row norms against a host reference: f32 accumulation, one bf16
//! rounding on the store.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::{norm, norm_residual, pointwise};

fn rms(x: &[f32], w: Option<&[f32]>, plus_one: bool, eps: f32) -> Vec<f32> {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let s = w.map_or(1.0, |w| w[i]) + if plus_one { 1.0 } else { 0.0 };
            v * inv * s
        })
        .collect()
}

fn rows(x: &[f32], width: usize, f: impl Fn(&[f32]) -> Vec<f32>) -> Vec<f32> {
    x.chunks(width).flat_map(f).collect()
}

#[test]
fn elemwise_norm() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // An aligned width (the packed path) and an odd one (the element path).
    for (rows_n, width) in [(5u32, 1024u32), (3, 1000)] {
        let n = (rows_n * width) as usize;
        let xs: Vec<f32> = values(n, 1, 2.0).into_iter().map(round_bf16).collect();
        let ws: Vec<f32> = values(width as usize, 2, 0.5)
            .into_iter()
            .map(round_bf16)
            .collect();
        let x = gpu.tensor(&xs, rows_n, width, Dtype::Bf16);
        let w = gpu.tensor(&ws, 1, width, Dtype::Bf16);

        let mut y = gpu.zeros(rows_n, width, Dtype::Bf16);
        norm::rmsnorm_plus_one(&ctx, x, w, 1e-6, &mut y).unwrap();
        let want = rows(&xs, width as usize, |r| rms(r, Some(&ws), true, 1e-6));
        close_bf16("rmsnorm_plus_one", &gpu.read(&y), &want, 1.0);

        let mut y = gpu.zeros(rows_n, width, Dtype::Bf16);
        norm::rmsnorm(&ctx, x, w, 1e-6, &mut y).unwrap();
        let want = rows(&xs, width as usize, |r| rms(r, Some(&ws), false, 1e-6));
        close_bf16("rmsnorm", &gpu.read(&y), &want, 1.0);

        // y += x, out = rmsnorm(y)·(w+1)
        let ys: Vec<f32> = values(n, 3, 2.0).into_iter().map(round_bf16).collect();
        let mut stream = gpu.tensor(&ys, rows_n, width, Dtype::Bf16);
        let mut out = gpu.zeros(rows_n, width, Dtype::Bf16);
        norm_residual::residual_add_rmsnorm(&ctx, x, &mut stream, w, true, 1e-6, &mut out).unwrap();
        let summed: Vec<f32> = ys.iter().zip(&xs).map(|(a, b)| round_bf16(a + b)).collect();
        close_bf16(
            "residual_add_rmsnorm stream",
            &gpu.read(&stream),
            &summed,
            0.0,
        );
        let want = rows(&summed, width as usize, |r| rms(r, Some(&ws), true, 1e-6));
        close_bf16("residual_add_rmsnorm out", &gpu.read(&out), &want, 1.0);

        let mut stream = gpu.tensor(&ys, rows_n, width, Dtype::Bf16);
        pointwise::residual_add(&ctx, x, &mut stream).unwrap();
        close_bf16("residual_add", &gpu.read(&stream), &summed, 0.0);
    }

    // Per head: 4 rows of 8 heads x 256 (Qwen 3.5's q/k norm geometry).
    let (r, heads, hd) = (4u32, 8u32, 256u32);
    let xs: Vec<f32> = values((r * heads * hd) as usize, 4, 3.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let ws: Vec<f32> = values(hd as usize, 5, 0.3)
        .into_iter()
        .map(round_bf16)
        .collect();
    let x = gpu.tensor(&xs, r, heads * hd, Dtype::Bf16);
    let w = gpu.tensor(&ws, 1, hd, Dtype::Bf16);
    let mut y = gpu.zeros(r, heads * hd, Dtype::Bf16);
    norm::rmsnorm_per_head_plus_one(&ctx, x, w, hd, 1e-6, &mut y).unwrap();
    let want = rows(&xs, hd as usize, |h| rms(h, Some(&ws), true, 1e-6));
    close_bf16("rmsnorm_per_head_plus_one", &gpu.read(&y), &want, 1.0);
    let mut y = gpu.zeros(r, heads * hd, Dtype::Bf16);
    norm::rmsnorm_no_scale(&ctx, x, 512, 1e-6, &mut y).unwrap();
    let want = rows(&xs, 512, |h| rms(h, None, false, 1e-6));
    close_bf16("rmsnorm_no_scale per head", &gpu.read(&y), &want, 1.0);

    // Gated (GDN output): f32 x and w, bf16 gate, silu and sigmoid.
    let (r, heads, hd) = (3u32, 4u32, 128u32);
    let n = (r * heads * hd) as usize;
    let xs = values(n, 6, 1.5);
    let gs: Vec<f32> = values(n, 7, 3.0).into_iter().map(round_bf16).collect();
    let ws = values(hd as usize, 8, 1.0);
    let x = gpu.tensor(&xs, r, heads * hd, Dtype::F32);
    let g = gpu.tensor(&gs, r, heads * hd, Dtype::Bf16);
    let w = gpu.tensor(&ws, 1, hd, Dtype::F32);
    for sigmoid in [false, true] {
        let mut y = gpu.zeros(r, heads * hd, Dtype::Bf16);
        norm::rmsnorm_gated(&ctx, x, g, Some(w), hd, 1e-6, sigmoid, &mut y).unwrap();
        let normed = rows(&xs, hd as usize, |h| rms(h, Some(&ws), false, 1e-6));
        let want: Vec<f32> = normed
            .iter()
            .zip(&gs)
            .map(|(v, g)| {
                let s = 1.0 / (1.0 + (-g).exp());
                v * if sigmoid { s } else { g * s }
            })
            .collect();
        close_bf16("rmsnorm_gated", &gpu.read(&y), &want, 1.0);
    }

    // A window: the launch covers 6 rows, the staged extent is rows 2..5.
    let (r, width) = (6u32, 512u32);
    let xs: Vec<f32> = values((r * width) as usize, 9, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let ws: Vec<f32> = values(width as usize, 10, 0.5)
        .into_iter()
        .map(round_bf16)
        .collect();
    let x = gpu.tensor(&xs, r, width, Dtype::Bf16);
    let w = gpu.tensor(&ws, 1, width, Dtype::Bf16);
    let win = gpu.ints(&[3, 2, 0, 0]);
    let mut y = gpu.zeros(r, width, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    norm::rmsnorm_plus_one(&ctx, x, w, 1e-6, &mut y).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&y);
    let w_ = width as usize;
    for row in 0..r as usize {
        let have = &got[row * w_..(row + 1) * w_];
        if (2..5).contains(&row) {
            close_bf16(
                "windowed row",
                have,
                &rms(&xs[row * w_..(row + 1) * w_], Some(&ws), true, 1e-6),
                1.0,
            );
        } else {
            assert!(
                have.iter().all(|v| *v == 0.0),
                "row {row} is outside the window and stays untouched"
            );
        }
    }
}
