#![cfg(feature = "cuda")]

//! `kernel.elemwise.rows_vec` swaps the RMS norms at 128 and 256 wide, the
//! q/gate cut and the sigmoid gate for their bandwidth-shaped forms: each
//! lands what the old form lands (the cut and the gate bit for bit, the norm
//! within a bf16 ulp), on the staged window's rows and nowhere else.

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::{gate, norm};
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
    let outs: Vec<u64> = (0..5).map(|_| gpu.zeros(planes * width * 2)).collect();
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
    gpu.sync();

    let n = planes * width as usize;
    let mut got: Vec<Vec<u16>> = outs.iter().map(|&at| gpu.down(at, n)).collect();
    got.push(gpu.down(gated, n));
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
        "gated",
    ];
    for (k, name) in names.iter().enumerate() {
        if k < 3 {
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
    for head_dim in [128, 256] {
        check(head_dim, None);
        check(head_dim, Some((24, 17, 5)));
    }
}
