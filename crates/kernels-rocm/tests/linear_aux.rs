//! The small projections: the lane GEMM (an f32 activation), the LoRA
//! correction, and the relative-position bias, against host references.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::elemwise::posenc;
use kernels_rocm::linear::lane_gemm;
use kernels_rocm::linear::lora::{self, Segments};

fn bf16s(n: u32, seed: u64, scale: f32) -> Vec<f32> {
    values(n as usize, seed, scale)
        .into_iter()
        .map(round_bf16)
        .collect()
}

#[test]
fn linear_lane_gemm() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, n, k) in [(3u32, 37u32, 1000u32), (1, 130, 64)] {
        let a = values((rows * k) as usize, 1, 1.0);
        let w = bf16s(n * k, 2, 0.1);
        let at = gpu.tensor(&a, rows, k, Dtype::F32);
        let wt = gpu.tensor(&w, n, k, Dtype::Bf16);
        let want: Vec<f32> = (0..(rows * n) as usize)
            .map(|i| {
                let (r, c) = (i / n as usize, i % n as usize);
                (0..k as usize)
                    .map(|j| f64::from(a[r * k as usize + j]) * f64::from(w[c * k as usize + j]))
                    .sum::<f64>() as f32
            })
            .collect();
        let mut y = gpu.zeros(rows, n, Dtype::F32);
        lane_gemm::act_x_wt(&ctx, "linear.matmul", at, wt, &mut y).unwrap();
        close_abs(
            &format!("lane_gemm f32 {rows}x{n}x{k}"),
            &gpu.read(&y),
            &want,
            1e-4,
        );
        let mut y = gpu.zeros(rows, n, Dtype::Bf16);
        lane_gemm::act_x_wt(&ctx, "linear.matmul", at, wt, &mut y).unwrap();
        close_bf16(
            &format!("lane_gemm bf16 {rows}x{n}x{k}"),
            &gpu.read(&y),
            &want,
            1.0,
        );
    }
}

#[test]
fn linear_lora_correct() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, input, out, rank, adapters) = (6u32, 256u32, 300u32, 8u32, 3u32);
    let x = bf16s(rows * input, 3, 1.0);
    let a = bf16s(adapters * rank * input, 4, 0.1);
    let b = bf16s(adapters * out * rank, 5, 0.1);
    let y0 = bf16s(rows * out, 6, 1.0);
    let routes = [2i32, 0, 1, 1, 2, 0];
    // The host: t = bf16(x · Aᵀ), then y += B · t, rounded once.
    let (i_, o_, r_) = (input as usize, out as usize, rank as usize);
    let t: Vec<f32> = (0..rows as usize * r_)
        .map(|i| {
            let (row, j) = (i / r_, i % r_);
            let ad = routes[row] as usize;
            round_bf16(
                (0..i_)
                    .map(|c| f64::from(x[row * i_ + c]) * f64::from(a[(ad * r_ + j) * i_ + c]))
                    .sum::<f64>() as f32,
            )
        })
        .collect();
    let want = |row: usize, c: usize| -> f32 {
        let ad = routes[row] as usize;
        let acc: f32 = (0..r_)
            .map(|j| b[(ad * o_ + c) * r_ + j] * t[row * r_ + j])
            .sum();
        y0[row * o_ + c] + acc
    };

    let xt = gpu.tensor(&x, rows, input, Dtype::Bf16);
    let at = gpu.tensor(&a, adapters, rank * input, Dtype::Bf16);
    let bt = gpu.tensor(&b, adapters, out * rank, Dtype::Bf16);
    let rt = gpu.column(&routes);
    let mut y = gpu.tensor(&y0, rows, out, Dtype::Bf16);
    lora::correct(&ctx, xt, at, bt, rt, &mut y, None).unwrap();
    let all: Vec<f32> = (0..rows as usize * o_)
        .map(|i| want(i / o_, i % o_))
        .collect();
    close_bf16("lora", &gpu.read(&y), &all, 2.0);

    // Segments: rows 1..3 and 4..6 only; the others keep their values.
    let mut y = gpu.tensor(&y0, rows, out, Dtype::Bf16);
    let list = gpu.ints(&[1, 2, 4, 2]);
    let segments = Segments {
        list,
        count: 2,
        cap: 2,
        max_rows: 2,
    };
    lora::correct(&ctx, xt, at, bt, rt, &mut y, Some(segments)).unwrap();
    let got = gpu.read(&y);
    for row in 0..rows as usize {
        let have = &got[row * o_..(row + 1) * o_];
        if [1, 2, 4, 5].contains(&row) {
            close_bf16(
                &format!("lora segment row {row}"),
                have,
                &all[row * o_..(row + 1) * o_],
                2.0,
            );
        } else {
            assert_eq!(
                have,
                &y0[row * o_..(row + 1) * o_],
                "row {row} is in no segment"
            );
        }
    }
}

#[test]
fn linear_rel_bias() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, heads, d_rel, extent) = (5u32, 3u32, 24u32, 300u32);
    let x = bf16s(rows * heads * d_rel, 7, 1.0);
    let w = bf16s(d_rel * extent, 8, 0.5);
    let xt = gpu.tensor(&x, rows, heads * d_rel, Dtype::Bf16);
    let wt = gpu.tensor(&w, d_rel, extent, Dtype::Bf16);
    let (h_, d_, e_) = (heads as usize, d_rel as usize, extent as usize);
    let want: Vec<f32> = (0..rows as usize * h_ * e_)
        .map(|i| {
            let (rh, d) = (i / e_, i % e_);
            (0..d_).map(|j| x[rh * d_ + j] * w[j * e_ + d]).sum()
        })
        .collect();
    let mut bias = gpu.zeros(rows, heads * extent, Dtype::F32);
    posenc::rel_bias(&ctx, xt, wt, heads, d_rel, extent, &mut bias).unwrap();
    close_abs("rel_bias", &gpu.read(&bias), &want, 1e-4);

    // A window over rows 2..4.
    let win = gpu.ints(&[2, 2, 0, 0]);
    let mut bias = gpu.zeros(rows, heads * extent, Dtype::F32);
    ctx.arm_stage(win.ptr);
    posenc::rel_bias(&ctx, xt, wt, heads, d_rel, extent, &mut bias).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&bias);
    let width = h_ * e_;
    for row in 0..rows as usize {
        let (have, w) = (
            &got[row * width..(row + 1) * width],
            &want[row * width..(row + 1) * width],
        );
        if (2..4).contains(&row) {
            close_abs(&format!("rel_bias windowed row {row}"), have, w, 1e-4);
        } else {
            assert!(
                have.iter().all(|v| *v == 0.0),
                "row {row} is outside the window"
            );
        }
    }
}
