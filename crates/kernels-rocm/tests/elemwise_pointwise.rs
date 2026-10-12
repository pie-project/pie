//! Rotary, the sigmoid gate and swiglu against a host reference.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::{gate, rope};
use kernels_rocm::linear::mlp;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn elemwise_pointwise() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // Qwen 3.5's full-attention geometry: head 256, rotary 64, span 32.
    let (rows, qh, kh, hd, rot, theta) = (5u32, 8u32, 2u32, 256u32, 64u32, 1e7f32);
    let span = rot / 2;
    let positions = [0i32, 1, 17, 4095, 131_071];
    let qs: Vec<f32> = values((rows * qh * hd) as usize, 1, 1.0).into_iter().map(round_bf16).collect();
    let ks: Vec<f32> = values((rows * kh * hd) as usize, 2, 1.0).into_iter().map(round_bf16).collect();
    let mut q = gpu.tensor(&qs, rows, qh * hd, Dtype::Bf16);
    let mut k = gpu.tensor(&ks, rows, kh * hd, Dtype::Bf16);
    let pos = gpu.column(&positions);
    rope::partial(&ctx, &mut q, &mut k, pos, rot, hd, theta, span).unwrap();
    let rotate = |x: &[f32], heads: u32| -> Vec<f32> {
        let mut out = x.to_vec();
        for (r, &at) in positions.iter().enumerate() {
            let p = f64::from(at);
            for h in 0..heads as usize {
                let base = (r * heads as usize + h) * hd as usize;
                for i in 0..span as usize {
                    let ang = p * f64::from(theta).powf(-(i as f64) / f64::from(span));
                    let (s, c) = ang.sin_cos();
                    let (a, b) = (f64::from(x[base + i]), f64::from(x[base + i + span as usize]));
                    out[base + i] = (a * c - b * s) as f32;
                    out[base + i + span as usize] = (b * c + a * s) as f32;
                }
            }
        }
        out
    };
    // An f32 angle at position 131071 is good to ~8e-3 rad (its ulp), and
    // the store rounds to bf16 (~4e-3 at |x| ≤ 1): inputs are in [-1, 1).
    close_abs("rope_partial q", &gpu.read(&q), &rotate(&qs, qh), 2e-2);
    close_abs("rope_partial k", &gpu.read(&k), &rotate(&ks, kh), 2e-2);

    let (rows, width) = (3u32, 1000u32);
    let n = (rows * width) as usize;
    let xs: Vec<f32> = values(n, 3, 2.0).into_iter().map(round_bf16).collect();
    let gs: Vec<f32> = values(n, 4, 4.0).into_iter().map(round_bf16).collect();
    let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let g = gpu.tensor(&gs, rows, width, Dtype::Bf16);
    gate::sigmoid_mul(&ctx, g, 1, &mut x).unwrap();
    let want: Vec<f32> = xs.iter().zip(&gs).map(|(x, g)| x * sigmoid(*g)).collect();
    close_bf16("gate_sigmoid_mul", &gpu.read(&x), &want, 1.0);

    let (rows, inter) = (4u32, 1536u32);
    let packed: Vec<f32> = values((rows * 2 * inter) as usize, 5, 3.0).into_iter().map(round_bf16).collect();
    let p = gpu.tensor(&packed, rows, 2 * inter, Dtype::Bf16);
    let mut y = gpu.zeros(rows, inter, Dtype::Bf16);
    mlp::swiglu(&ctx, p, inter, 1, &mut y).unwrap();
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            let row = &packed[r * 2 * inter as usize..(r + 1) * 2 * inter as usize];
            (0..inter as usize).map(move |i| {
                let (g, u) = (row[i], row[inter as usize + i]);
                g * sigmoid(g) * u
            })
        })
        .collect();
    close_bf16("mlp_swiglu", &gpu.read(&y), &want, 1.0);
}
