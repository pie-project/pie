#![cfg(feature = "cuda")]

//! A fused norm-and-fold told a row goes unread (`kernel.elemwise.skip_unread`)
//! leaves that row as it found it and lands every other one bit for bit as
//! the chain that keeps them all: the normed row `t`, the stream `y` folded in
//! place, the rescaled copy and the post-norm's output, at Gemma 4 E4B's width
//! (320 runs over a 512-thread block) and a 1536-wide row. A wide fire taken
//! 128 threads a row (`kernel.elemwise.norm_fold_block`) lands what a run a
//! thread does, to a bf16 rounding.

use crate::common::attn_ref::tuned;
use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::norm_residual::{self, Keep, PostNorm};
use kernels_cuda::tensor::Tensor;

struct Landed {
    t: Vec<u16>,
    y: Vec<u16>,
    scaled: Vec<u16>,
    out: Vec<u16>,
}

fn land(
    rows: usize,
    width: usize,
    scaled: bool,
    post: bool,
    keep: Keep,
    block: u32,
    seed: u64,
) -> (Landed, Landed) {
    let n = rows * width;
    let mut lcg = Lcg::seeded(seed);
    let (x, _) = lcg.row(n);
    let (y, _) = lcg.row(n);
    let (w0, _) = lcg.row(width);
    let (w1, _) = lcg.row(width);
    let (s, _) = lcg.row(1);
    // what the outputs held before the chain ran
    let (t0, _) = lcg.row(n);
    let (scaled0, _) = lcg.row(n);
    let (out0, _) = lcg.row(n);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let y_at = gpu.up(&y);
    let w0_at = gpu.up(&w0);
    let w1_at = gpu.up(&w1);
    let s_at = gpu.up(&s);
    let t_at = gpu.up(&t0);
    let scaled_at = gpu.up(&scaled0);
    let out_at = gpu.up(&out0);
    let ctx = gpu.ctx().with_tuning(tuned(|t| t.norm_fold_block = block));
    let plane = |at| Tensor::new(at, rows as u32, width as u32, Dtype::Bf16);
    let mut t = plane(t_at);
    let mut yt = plane(y_at);
    let mut sc = plane(scaled_at);
    let mut out = plane(out_at);
    norm_residual::rmsnorm_residual_add_keeping(
        &ctx,
        plane(x_at),
        Tensor::new(w0_at, width as u32, 1, Dtype::Bf16),
        1e-6,
        &mut t,
        &mut yt,
        scaled.then(|| (Tensor::new(s_at, 1, 1, Dtype::Bf16), &mut sc)),
        post.then(|| PostNorm {
            weight: Tensor::new(w1_at, width as u32, 1, Dtype::Bf16),
            plus_one: false,
            eps: 1e-6,
            out: &mut out,
        }),
        keep,
    )
    .expect("the fused norm-and-fold fires");
    gpu.sync();
    let landed = Landed {
        t: gpu.down(t_at, n),
        y: gpu.down(y_at, n),
        scaled: gpu.down(scaled_at, n),
        out: gpu.down(out_at, n),
    };
    let before = Landed {
        t: t0,
        y,
        scaled: scaled0,
        out: out0,
    };
    (landed, before)
}

#[test]
fn the_norm_fold_lands_only_what_it_keeps() {
    for width in [2560usize, 1536] {
        for (scaled, post) in [(false, false), (false, true), (true, false), (true, true)] {
            let seed = 0x5eed ^ width as u64 ^ (u64::from(scaled) << 20) ^ (u64::from(post) << 21);
            let (all, _) = land(5, width, scaled, post, Keep::ALL, 0, seed);
            for keep in [
                Keep { t: false, y: true },
                Keep { t: true, y: false },
                Keep { t: false, y: false },
            ] {
                let case = format!("width {width} scaled {scaled} post {post} {keep:?}");
                let (got, before) = land(5, width, scaled, post, keep, 0, seed);
                let t_want = if keep.t { &all.t } else { &before.t };
                let y_want = if keep.y { &all.y } else { &before.y };
                assert!(&got.t == t_want, "{case}: t");
                assert!(&got.y == y_want, "{case}: y");
                if scaled {
                    assert!(got.scaled == all.scaled, "{case}: scaled");
                }
                if post {
                    assert!(got.out == all.out, "{case}: out");
                }
            }
        }
    }
}

#[test]
fn a_wide_fire_folds_at_a_smaller_block_as_at_a_run_a_thread() {
    let close = |what: &str, got: &[u16], want: &[u16]| {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let (g, w) = (from_bf16(*g), from_bf16(*w));
            // a sum of two O(1) rows can cancel: half a ulp of either
            // operand, not of the result
            assert!(
                (g - w).abs() <= w.abs() * (1.0 / 64.0) + 1.5e-2,
                "{what}[{i}] = {g}, want {w}"
            );
        }
    };
    for (scaled, post) in [(false, false), (false, true), (true, true)] {
        let seed = 0xb10c ^ (u64::from(scaled) << 20) ^ (u64::from(post) << 21);
        let (want, _) = land(1100, 2560, scaled, post, Keep::ALL, 0, seed);
        let (got, _) = land(1100, 2560, scaled, post, Keep::ALL, 128, seed);
        let case = format!("scaled {scaled} post {post}");
        close(&format!("{case} t"), &got.t, &want.t);
        close(&format!("{case} y"), &got.y, &want.y);
        if scaled {
            close(&format!("{case} scaled"), &got.scaled, &want.scaled);
        }
        if post {
            close(&format!("{case} out"), &got.out, &want.out);
        }
    }
}
