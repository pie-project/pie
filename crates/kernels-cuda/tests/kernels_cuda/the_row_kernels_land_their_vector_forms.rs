#![cfg(feature = "cuda")]

//! `kernel.elemwise.rows_vec` swaps the RMS norms at 128 and 256 wide, the
//! scale-free per-head norm of whole 256-element runs, the q/gate cut, the
//! sigmoid gate and the rotation of a head's tail for their
//! bandwidth-shaped forms: each lands what the old form lands (the cut, the
//! gate and the rotation bit for bit, the norms within a bf16 ulp), on the
//! staged window's rows and nowhere else.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::{gate, norm, rope};
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

const HEADS: u32 = 6;

/// Every output of one pass, `vec` choosing the forms; `window` is
/// `(rows, live, base)`.
fn pass(vec: bool, head_dim: u32, window: Option<(u32, u32, u32)>) -> Vec<Vec<u16>> {
    let (rows, live, base) = window.unwrap_or((40, 40, 0));
    let planes = (base + rows) as usize;
    let width = (HEADS * head_dim) as usize;
    let mut lcg = Lcg::seeded(0x5eed + u64::from(head_dim));
    let (packed_raw, _) = lcg.row(planes * width * 2);
    let (x_raw, _) = lcg.row(planes * width);
    let (gate_raw, _) = lcg.row(planes * width);
    let (w_raw, _) = lcg.row(head_dim as usize);

    let mut gpu = Gpu::open();
    let packed = gpu.up(&packed_raw);
    let x = gpu.up(&x_raw);
    let gate_at = gpu.up(&gate_raw);
    let gated = gpu.up(&x_raw);
    let w = gpu.up(&w_raw);
    let outs: Vec<u64> = (0..6).map(|_| gpu.zeros(planes * width * 2)).collect();
    let turned: Vec<u64> = (0..2).map(|_| gpu.up(&x_raw)).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 37 + 3).collect();
    let positions = gpu.up(&positions);
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: vec,
        ..Tuning::DEFAULT
    });
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let t = |at: u64, w: u32| Tensor::new(at, rows, w, Dtype::Bf16);
    let weight = Tensor::new(w, head_dim, 1, Dtype::Bf16);
    let width = width as u32;

    norm::rmsnorm_per_head_plus_one(
        &ctx,
        t(x, width),
        weight,
        head_dim,
        1e-6,
        &mut t(outs[0], width),
    )
    .expect("the per-head norm fires");
    norm::rmsnorm_per_head(
        &ctx,
        t(x, width),
        weight,
        head_dim,
        1e-6,
        &mut t(outs[1], width),
    )
    .expect("the plain per-head norm fires");
    // The whole row as one head-wide norm, rows `HEADS` times as many.
    let tall = |at: u64| Tensor::new(at, rows * HEADS, head_dim, Dtype::Bf16);
    if window.is_none() {
        norm::rmsnorm_plus_one(&ctx, tall(x), weight, 1e-6, &mut tall(outs[2]))
            .expect("the whole-row norm fires");
    }
    layout::split_q_gate(
        &ctx,
        t(packed, 2 * width),
        head_dim,
        &mut t(outs[3], width),
        &mut t(outs[4], width),
    )
    .expect("the cut fires");
    gate::sigmoid_mul(&ctx, t(gate_at, width), 1, &mut t(gated, width)).expect("the gate fires");
    norm::rmsnorm_no_scale(&ctx, t(x, width), head_dim, 1e-6, &mut t(outs[5], width))
        .expect("the scale-free norm fires");
    // DeepSeek-V4's 64-wide tail, forward interleaved and inverse split.
    for (k, (interleaved, inverse)) in [(true, false), (false, true)].into_iter().enumerate() {
        rope::partial_last(
            &ctx,
            &mut t(turned[k], width),
            Tensor::new(positions, rows, 1, Dtype::I32),
            64,
            head_dim,
            10000.0,
            interleaved,
            inverse,
            None,
        )
        .expect("the rotation fires");
    }
    gpu.sync();

    let n = planes * width as usize;
    let mut got: Vec<Vec<u16>> = outs.iter().map(|&at| gpu.down(at, n)).collect();
    got.push(gpu.down(gated, n));
    for &at in &turned {
        got.push(gpu.down(at, n));
    }
    got
}

fn check(head_dim: u32, window: Option<(u32, u32, u32)>) {
    let old = pass(false, head_dim, window);
    let new = pass(true, head_dim, window);
    let names = [
        "per-head plus-one norm",
        "per-head norm",
        "whole-row norm",
        "q",
        "gate",
        "scale-free norm",
        "gated",
        "interleaved rotation",
        "split inverse rotation",
    ];
    for (k, name) in names.iter().enumerate() {
        if k < 3 || k == 5 {
            for (i, (&g, &w)) in new[k].iter().zip(&old[k]).enumerate() {
                let (g, w) = (from_bf16(g), from_bf16(w));
                assert!(
                    (g - w).abs() <= w.abs() * (1.0 / 128.0) + 1e-6,
                    "d={head_dim} {window:?}: {name} element {i} is {g} against {w}"
                );
            }
        } else {
            assert!(
                new[k] == old[k],
                "d={head_dim} {window:?}: the {name} plane differs"
            );
        }
    }
}

#[test]
fn the_row_kernels_land_their_vector_forms() {
    for head_dim in [128, 256, 512] {
        check(head_dim, None);
        check(head_dim, Some((24, 17, 5)));
    }
}

/// DeepSeek-V4's q at a prefill's rows: the scale-free norm of 64 heads of
/// 512 and the rotation of their 64-wide tails, both forms.
#[test]
#[ignore = "pacing; run with --ignored --nocapture"]
fn the_row_kernels_pace_deepseeks_q() {
    let rows = 8192u32;
    let width = 64 * 512u32;
    let mut gpu = Gpu::open();
    let n = rows as usize * width as usize;
    let x = gpu.zeros(n * 2);
    let y = gpu.zeros(n * 2);
    let positions: Vec<i32> = (0..rows as i32).collect();
    let positions = gpu.up(&positions);
    for vec in [false, true] {
        let ctx = gpu.ctx().with_tuning(Tuning {
            rows_vec: vec,
            ..Tuning::DEFAULT
        });
        let t = |at: u64| Tensor::new(at, rows, width, Dtype::Bf16);
        let time = |what: &str, f: &dyn Fn()| {
            f();
            gpu.sync();
            let iters = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                f();
            }
            gpu.sync();
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
            eprintln!("| vec {vec:5} | {what:9} | {us:7.1} us |");
        };
        time("qnorm", &|| {
            norm::rmsnorm_no_scale(&ctx, t(x), 512, 1e-6, &mut t(y)).unwrap();
        });
        time("rope", &|| {
            rope::partial_last(
                &ctx,
                &mut t(y),
                Tensor::new(positions, rows, 1, Dtype::I32),
                64,
                512,
                10000.0,
                true,
                false,
                None,
            )
            .unwrap();
        });
    }
}
