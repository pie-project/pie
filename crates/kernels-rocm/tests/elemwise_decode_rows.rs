//! The residual add, the sigmoid gate and the partial rotary at Qwen3.8-
//! Flash-Next's decode shapes (hidden 2560, 24 query heads over 2 kv heads
//! of 256, a quarter of each rotated), at 1, 7, 128 and 512 rows, and under
//! a staged window that pads the launch far past its live rows (rows past
//! the window untouched). The benchmark times each at those rows and at a
//! window of 128 live rows in an 8192-row bucket.
//! `cargo test --release --features rocm --test elemwise_decode_rows -- --ignored --nocapture`

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::{gate, norm, rope};

const HIDDEN: u32 = 2560;
const QH: u32 = 24;
const KH: u32 = 2;
const HD: u32 = 256;
const ROT: u32 = 64;
const THETA: f32 = 1e7;

fn bf(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The four window words {rows, row_offset, lanes, lane_offset}.
fn window(gpu: &Gpu, rows: u32, offset: u32) -> kernels_rocm::Tensor {
    gpu.ints(&[rows as i32, offset as i32, rows as i32, offset as i32])
}

#[test]
fn residual_add_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for rows in [1u32, 7, 128, 512] {
        let n = (rows * HIDDEN) as usize;
        let (xs, ys) = (bf(n, 1, 2.0), bf(n, 2, 2.0));
        let x = gpu.tensor(&xs, rows, HIDDEN, Dtype::Bf16);
        let mut y = gpu.tensor(&ys, rows, HIDDEN, Dtype::Bf16);
        norm::residual_add(&ctx, x, &mut y).unwrap();
        let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| round_bf16(a + b)).collect();
        assert_eq!(gpu.read(&y), want, "residual_add rows{rows}");
    }
    // A width the packs do not divide.
    let (rows, width) = (5u32, 1001u32);
    let n = (rows * width) as usize;
    let (xs, ys) = (bf(n, 3, 2.0), bf(n, 4, 2.0));
    let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let mut y = gpu.tensor(&ys, rows, width, Dtype::Bf16);
    norm::residual_add(&ctx, x, &mut y).unwrap();
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| round_bf16(a + b)).collect();
    assert_eq!(gpu.read(&y), want, "residual_add width {width}");
    // A 512-row launch whose window holds rows 7..107.
    for width in [HIDDEN, 1001] {
        let rows = 512u32;
        let n = (rows * width) as usize;
        let (xs, ys) = (bf(n, 5, 2.0), bf(n, 6, 2.0));
        let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let mut y = gpu.tensor(&ys, rows, width, Dtype::Bf16);
        ctx.arm_stage(window(&gpu, 100, 7).ptr);
        norm::residual_add(&ctx, x, &mut y).unwrap();
        ctx.disarm_stage();
        let w = width as usize;
        let want: Vec<f32> = (0..n)
            .map(|i| {
                if (7 * w..107 * w).contains(&i) {
                    round_bf16(xs[i] + ys[i])
                } else {
                    ys[i]
                }
            })
            .collect();
        assert_eq!(gpu.read(&y), want, "residual_add windowed width {width}");
    }
}

#[test]
fn gate_sigmoid_mul_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let width = QH * HD;
    for rows in [1u32, 7, 128, 512] {
        let n = (rows * width) as usize;
        let (xs, gs) = (bf(n, 7, 2.0), bf(n, 8, 4.0));
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let g = gpu.tensor(&gs, rows, width, Dtype::Bf16);
        gate::sigmoid_mul(&ctx, g, 1, &mut x).unwrap();
        let want: Vec<f32> = xs.iter().zip(&gs).map(|(x, g)| x * sigmoid(*g)).collect();
        close_bf16(
            &format!("gate_sigmoid_mul rows{rows}"),
            &gpu.read(&x),
            &want,
            1.0,
        );
    }
    // Windowed, two rows a stated row: live rows 2·3.. 2·(3 + 40).
    for width in [width, 1001] {
        let rows = 256u32;
        let n = (rows * width) as usize;
        let (xs, gs) = (bf(n, 9, 2.0), bf(n, 10, 4.0));
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let g = gpu.tensor(&gs, rows, width, Dtype::Bf16);
        ctx.arm_stage(window(&gpu, 40, 3).ptr);
        gate::sigmoid_mul(&ctx, g, 2, &mut x).unwrap();
        ctx.disarm_stage();
        let got = gpu.read(&x);
        let w = width as usize;
        let live = 6 * w..86 * w;
        let want: Vec<f32> = (0..n)
            .map(|i| {
                if live.contains(&i) {
                    xs[i] * sigmoid(gs[i])
                } else {
                    xs[i]
                }
            })
            .collect();
        close_bf16(
            &format!("gate_sigmoid_mul windowed width {width}"),
            &got,
            &want,
            1.0,
        );
        for i in (0..n).filter(|i| !live.contains(i)) {
            assert_eq!(
                got[i], xs[i],
                "gate_sigmoid_mul wrote past its window at {i}"
            );
        }
    }
}

/// softmax-free host rotary: pair i with i + span, f64 angles.
fn rotate(x: &[f32], positions: &[i32], heads: u32) -> Vec<f32> {
    let span = (ROT / 2) as usize;
    let mut out = x.to_vec();
    for (r, &at) in positions.iter().enumerate() {
        let p = f64::from(at);
        for h in 0..heads as usize {
            let base = (r * heads as usize + h) * HD as usize;
            for i in 0..span {
                let ang = p * f64::from(THETA).powf(-(i as f64) / span as f64);
                let (s, c) = ang.sin_cos();
                let (a, b) = (f64::from(x[base + i]), f64::from(x[base + i + span]));
                out[base + i] = (a * c - b * s) as f32;
                out[base + i + span] = (b * c + a * s) as f32;
            }
        }
    }
    out
}

#[test]
fn rope_partial_at_decode_rows() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for rows in [1u32, 7, 128, 512] {
        let positions: Vec<i32> = (0..rows as i32).map(|r| (r * 977 + 3) % 70_000).collect();
        let qs = bf((rows * QH * HD) as usize, 11, 1.0);
        let ks = bf((rows * KH * HD) as usize, 12, 1.0);
        let mut q = gpu.tensor(&qs, rows, QH * HD, Dtype::Bf16);
        let mut k = gpu.tensor(&ks, rows, KH * HD, Dtype::Bf16);
        rope::partial(
            &ctx,
            &mut q,
            &mut k,
            gpu.column(&positions),
            ROT,
            HD,
            THETA,
            ROT / 2,
        )
        .unwrap();
        close_abs(
            &format!("rope_partial q rows{rows}"),
            &gpu.read(&q),
            &rotate(&qs, &positions, QH),
            2e-2,
        );
        close_abs(
            &format!("rope_partial k rows{rows}"),
            &gpu.read(&k),
            &rotate(&ks, &positions, KH),
            2e-2,
        );
    }
}
