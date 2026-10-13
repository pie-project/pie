//! The flat binary ops, activations, clamps and the per-head sigmoid gate
//! against a host reference: f32 arithmetic, one rounding on the store.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::{activation, binary, clip, gate};

fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `have` holds `want` on flat elements `live` and `before` elsewhere.
fn windowed(what: &str, have: &[f32], want: &[f32], before: &[f32], live: std::ops::Range<usize>) {
    for (i, ((h, w), b)) in have.iter().zip(want).zip(before).enumerate() {
        if live.contains(&i) {
            close_bf16(what, &[*h], &[*w], 1.0);
        } else {
            assert_eq!(h, b, "{what}[{i}] is outside the window and stays");
        }
    }
}

#[test]
fn binary_and_activation() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, width) in [(3u32, 1000u32), (1, 7)] {
        let n = (rows * width) as usize;
        for dtype in [Dtype::Bf16, Dtype::F32] {
            let round = |v: f32| {
                if dtype == Dtype::Bf16 {
                    round_bf16(v)
                } else {
                    v
                }
            };
            let ulps = if dtype == Dtype::Bf16 { 0.5 } else { 0.0 };
            let xs: Vec<f32> = values(n, 1, 3.0).into_iter().map(round).collect();
            let ys: Vec<f32> = values(n, 2, 3.0).into_iter().map(round).collect();
            let x = gpu.tensor(&xs, rows, width, dtype);
            let y = gpu.tensor(&ys, rows, width, dtype);

            let mut o = gpu.zeros(rows, width, dtype);
            binary::add(&ctx, x, y, &mut o).unwrap();
            let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a + b).collect();
            // A sum that cancels is exact in f32 and rounded once to bf16.
            close_abs(
                "add",
                &gpu.read(&o),
                &want.iter().map(|v| round(*v)).collect::<Vec<_>>(),
                0.0,
            );

            let mut o = gpu.zeros(rows, width, dtype);
            binary::mul(&ctx, x, y, &mut o).unwrap();
            let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a * b).collect();
            close_bf16("mul", &gpu.read(&o), &want, ulps);

            // The fast exp and tanh are within a few f32 ulps; the store
            // rounds them once.
            let tol = if dtype == Dtype::Bf16 { 1.0 } else { 1e-3 };
            let mut o = gpu.zeros(rows, width, dtype);
            activation::silu(&ctx, x, &mut o).unwrap();
            let want: Vec<f32> = xs.iter().map(|v| v * sigmoid(*v)).collect();
            close_bf16("silu", &gpu.read(&o), &want, tol);

            let mut o = gpu.zeros(rows, width, dtype);
            activation::tanh(&ctx, x, &mut o).unwrap();
            let want: Vec<f32> = xs.iter().map(|v| v.tanh()).collect();
            close_bf16("tanh", &gpu.read(&o), &want, tol);
        }
    }

    // A window over rows 1..3 of 4; the left operand has spare rows.
    let (rows, width) = (4u32, 300u32);
    let (n, wd) = ((rows * width) as usize, width as usize);
    let xs = bf16s(n + wd, 3, 2.0);
    let ys = bf16s(n, 4, 2.0);
    let x = gpu.tensor(&xs, rows + 1, width, Dtype::Bf16);
    let y = gpu.tensor(&ys, rows, width, Dtype::Bf16);
    let zeros = vec![0.0; n];
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    let mut o = gpu.zeros(rows, width, Dtype::Bf16);
    binary::mul(&ctx, x, y, &mut o).unwrap();
    let mut a = gpu.zeros(rows, width, Dtype::Bf16);
    activation::silu(&ctx, x, &mut a).unwrap();
    ctx.disarm_stage();
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| a * b).collect();
    windowed("windowed mul", &gpu.read(&o), &want, &zeros, wd..3 * wd);
    let want: Vec<f32> = xs[..n].iter().map(|v| v * sigmoid(*v)).collect();
    windowed("windowed silu", &gpu.read(&a), &want, &zeros, wd..3 * wd);
}

#[test]
fn clamps() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, width) in [(3u32, 1024u32), (2, 33)] {
        let n = (rows * width) as usize;
        let xs = bf16s(n, 5, 4.0);

        // Bounds between bf16 steps clamp to the rounded bound.
        let (lo, hi) = (-1.2345f32, 2.6789f32);
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        clip::clamp(&ctx, lo, hi, &mut x).unwrap();
        let want: Vec<f32> = xs
            .iter()
            .map(|v| v.clamp(round_bf16(lo), round_bf16(hi)))
            .collect();
        close_bf16("clamp", &gpu.read(&x), &want, 0.0);

        let (lo, hi) = (round_bf16(-0.75), round_bf16(1.5));
        let lt = gpu.tensor(&[lo], 1, 1, Dtype::Bf16);
        let ht = gpu.tensor(&[hi], 1, 1, Dtype::Bf16);
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        clip::clamp_learned(&ctx, lt, ht, &mut x).unwrap();
        let want: Vec<f32> = xs.iter().map(|v| v.clamp(lo, hi)).collect();
        close_bf16("clamp_learned", &gpu.read(&x), &want, 0.0);
    }

    // A window over row 2 of 3.
    let (rows, width) = (3u32, 100u32);
    let wd = width as usize;
    let xs = bf16s((rows * width) as usize, 6, 4.0);
    let (lo, hi) = (round_bf16(-1.0), round_bf16(1.0));
    let lt = gpu.tensor(&[lo], 1, 1, Dtype::Bf16);
    let ht = gpu.tensor(&[hi], 1, 1, Dtype::Bf16);
    let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let mut c = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let win = gpu.ints(&[1, 2, 0, 0]);
    ctx.arm_stage(win.ptr);
    clip::clamp_learned(&ctx, lt, ht, &mut x).unwrap();
    clip::clamp(&ctx, -1.0, 1.0, &mut c).unwrap();
    ctx.disarm_stage();
    let want: Vec<f32> = xs.iter().map(|v| v.clamp(lo, hi)).collect();
    windowed(
        "windowed clamp_learned",
        &gpu.read(&x),
        &want,
        &xs,
        2 * wd..3 * wd,
    );
    windowed("windowed clamp", &gpu.read(&c), &want, &xs, 2 * wd..3 * wd);
}

#[test]
fn gate_sigmoid_mul_heads() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Qwen3.8-Next's per-head output gate (16 x 256), and small ragged heads.
    for (rows, heads, hd, scale) in [(3u32, 16u32, 256u32, 1.0f32), (2, 5, 36, 2.0)] {
        let width = heads * hd;
        let xs = bf16s((rows * width) as usize, 7, 2.0);
        let gs = bf16s((rows * heads) as usize, 8, 4.0);
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let g = gpu.tensor(&gs, rows, heads, Dtype::Bf16);
        gate::sigmoid_mul_heads(&ctx, g, hd, scale, &mut x).unwrap();
        let want: Vec<f32> = xs
            .iter()
            .enumerate()
            .map(|(i, v)| v * scale * sigmoid(gs[i / hd as usize]))
            .collect();
        close_bf16("gate_sigmoid_mul_heads", &gpu.read(&x), &want, 1.0);
    }

    // A window over row 1 of 3.
    let (rows, heads, hd) = (3u32, 4u32, 64u32);
    let wd = (heads * hd) as usize;
    let xs = bf16s((rows as usize) * wd, 9, 2.0);
    let gs = bf16s((rows * heads) as usize, 10, 4.0);
    let mut x = gpu.tensor(&xs, rows, heads * hd, Dtype::Bf16);
    let g = gpu.tensor(&gs, rows, heads, Dtype::Bf16);
    let win = gpu.ints(&[1, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    gate::sigmoid_mul_heads(&ctx, g, hd, 1.0, &mut x).unwrap();
    ctx.disarm_stage();
    let want: Vec<f32> = xs
        .iter()
        .enumerate()
        .map(|(i, v)| v * sigmoid(gs[i / hd as usize]))
        .collect();
    windowed("windowed gate heads", &gpu.read(&x), &want, &xs, wd..2 * wd);
}
