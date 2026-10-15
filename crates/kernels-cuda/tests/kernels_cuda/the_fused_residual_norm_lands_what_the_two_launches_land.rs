#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

fn check(window: Option<(u32, u32, u32)>, hidden: usize, one_pass: bool) {
    let (rows, live, base) = window.unwrap_or((64, 64, 0));
    let planes = (base + rows) as usize;
    let mut lcg = Lcg::seeded(0x7e5);
    let (x_raw, _) = lcg.row(planes * hidden);
    let (y_raw, _) = lcg.row(planes * hidden);
    let (w_raw, _) = lcg.row(hidden);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let y_ref = gpu.up(&y_raw);
    let y_fused = gpu.up(&y_raw);
    let out_ref = gpu.zeros(planes * hidden * 2);
    let out_fused = gpu.zeros(planes * hidden * 2);
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: one_pass,
        ..Tuning::DEFAULT
    });
    let x = Tensor::new(x_at, planes as u32, hidden as u32, Dtype::Bf16);
    let w = Tensor::new(w_at, hidden as u32, 1, Dtype::Bf16);
    let tensor = |at: u64| Tensor::new(at, planes as u32, hidden as u32, Dtype::Bf16);

    norm::residual_add(&ctx, x, &mut tensor(y_ref)).expect("the add fires");
    norm::rmsnorm_plus_one(&ctx, tensor(y_ref), w, 1e-6, &mut tensor(out_ref))
        .expect("the norm fires");

    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    let staged = |at: u64| Tensor::new(at, rows, hidden as u32, Dtype::Bf16);
    let mut y = staged(y_fused);
    let mut out = staged(out_fused);
    norm::residual_add_rmsnorm(&ctx, staged(x_at), &mut y, w, true, 1e-6, &mut out)
        .expect("the pair fires");
    gpu.sync();

    let got_y: Vec<u16> = gpu.down(y_fused, planes * hidden);
    let got_out: Vec<u16> = gpu.down(out_fused, planes * hidden);
    let want_y: Vec<u16> = gpu.down(y_ref, planes * hidden);
    let want_out: Vec<u16> = gpu.down(out_ref, planes * hidden);
    for r in 0..planes {
        let span = r * hidden..(r + 1) * hidden;
        let touched = r >= base as usize && r < (base + live) as usize;
        if touched {
            assert_eq!(
                got_y[span.clone()],
                want_y[span.clone()],
                "row {r}: the stream differs"
            );
            for (i, (&got, &want)) in got_out[span.clone()]
                .iter()
                .zip(&want_out[span.clone()])
                .enumerate()
            {
                let (g, w) = (from_bf16(got), from_bf16(want));
                assert!(
                    (g - w).abs() <= w.abs() * (1.0 / 128.0) + 1e-6,
                    "row {r} column {i}: the normed row is {g} against {w}"
                );
            }
        } else {
            assert_eq!(
                got_y[span.clone()],
                y_raw[span.clone()],
                "row {r}: a padded row moved"
            );
            assert!(
                got_out[span].iter().all(|&v| v == 0),
                "row {r}: a padded row was normed"
            );
        }
    }
}

#[test]
fn the_fused_residual_norm_lands_what_the_two_launches_land_every_case() {
    the_pair_lands_the_two_launches_bits();
    the_pair_retires_a_buckets_padded_rows_off_the_staged_window();
}

/// Both forms: the two-pass block a row, and the one-pass form holding its
/// vectors in registers (one to three a thread at these widths).
fn the_pair_lands_the_two_launches_bits() {
    for one_pass in [false, true] {
        for hidden in [1024, 2048, 5120] {
            check(None, hidden, one_pass);
        }
    }
}

fn the_pair_retires_a_buckets_padded_rows_off_the_staged_window() {
    for one_pass in [false, true] {
        check(Some((4, 2, 1)), 5120, one_pass);
    }
}
