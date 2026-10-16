#![cfg(feature = "cuda")]

//! DeepSeek-V4's norm-then-rotate pairs (`norm_rope_last`): q's scale-free
//! norm of 64 heads of 512, the kv plane's and compressed latent's scaled
//! norm of one 512-wide head, the index keys' of one 128-wide head, each
//! with its 64-wide tail turned forward or back, plain or YaRN-scaled. Under
//! `kernel.elemwise.rows_vec` the one pass lands what the two launches land
//! within a bf16 ulp of the norm, on the staged window's rows and nowhere
//! else; off, it is the two launches.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope::{self, Yarn};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

struct Case {
    heads: u32,
    head_dim: u32,
    weighted: bool,
    inverse: bool,
    yarn: bool,
}

fn pass(vec: bool, c: &Case, window: Option<(u32, u32, u32)>) -> Vec<u16> {
    let (rows, live, base) = window.unwrap_or((37, 37, 0));
    let planes = (base + rows) as usize;
    let width = (c.heads * c.head_dim) as usize;
    let mut lcg = Lcg::seeded(0x40e + u64::from(c.head_dim) + u64::from(c.heads));
    let x: Vec<u16> = (0..planes * width)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<u16> = (0..c.head_dim).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 997 + 5).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let pos_at = gpu.up(&positions);
    let y_at = gpu.up(&vec![0x3f80u16; planes * width]);
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: vec,
        ..Tuning::DEFAULT
    });
    if window.is_some() {
        let win_at = gpu.up(&[live, base, 0u32, 0u32]);
        ctx.arm_stage(win_at);
    }
    rope::norm_rope_last(
        &ctx,
        Tensor::new(x_at, rows, width as u32, Dtype::Bf16),
        c.weighted
            .then(|| Tensor::new(w_at, c.head_dim, 1, Dtype::Bf16)),
        c.head_dim,
        1e-6,
        Tensor::new(pos_at, rows, 1, Dtype::I32),
        64,
        10000.0,
        true,
        c.inverse,
        c.yarn.then_some(Yarn {
            factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position: 65536,
        }),
        &mut Tensor::new(y_at, rows, width as u32, Dtype::Bf16),
    )
    .expect("the norm and rotation fire");
    gpu.sync();
    gpu.down(y_at, planes * width)
}

#[test]
fn the_norm_and_tail_rotation_land_in_one_pass() {
    let cases = [
        Case {
            heads: 64,
            head_dim: 512,
            weighted: false,
            inverse: false,
            yarn: true,
        },
        Case {
            heads: 64,
            head_dim: 512,
            weighted: false,
            inverse: true,
            yarn: false,
        },
        Case {
            heads: 1,
            head_dim: 512,
            weighted: true,
            inverse: false,
            yarn: true,
        },
        Case {
            heads: 1,
            head_dim: 128,
            weighted: true,
            inverse: false,
            yarn: false,
        },
        Case {
            heads: 4,
            head_dim: 256,
            weighted: false,
            inverse: false,
            yarn: true,
        },
    ];
    for c in &cases {
        for window in [None, Some((24, 17, 5))] {
            let old = pass(false, c, window);
            let new = pass(true, c, window);
            for (i, (&g, &w)) in new.iter().zip(&old).enumerate() {
                let (gf, wf) = (from_bf16(g), from_bf16(w));
                assert!(
                    (gf - wf).abs() <= wf.abs() / 64.0 + 2e-2,
                    "heads {} d {} weighted {} {window:?}: element {i} is {gf} against {wf}",
                    c.heads,
                    c.head_dim,
                    c.weighted
                );
            }
        }
    }
}

/// DeepSeek-V4's q norm and rotation at a prefill's and a decode's rows,
/// the one pass against the two launches.
#[test]
#[ignore = "pacing; run with --ignored --nocapture"]
fn the_norm_and_tail_rotation_pace_deepseeks_q() {
    let width = 64 * 512u32;
    let mut gpu = Gpu::open();
    let most = 8192usize;
    let x = gpu.zeros(most * width as usize * 2);
    let y = gpu.zeros(most * width as usize * 2);
    let positions: Vec<i32> = (0..most as i32).collect();
    let positions = gpu.up(&positions);
    for rows in [8u32, 64, 8192] {
        for vec in [false, true] {
            let ctx = gpu.ctx().with_tuning(Tuning {
                rows_vec: vec,
                ..Tuning::DEFAULT
            });
            let fire = || {
                rope::norm_rope_last(
                    &ctx,
                    Tensor::new(x, rows, width, Dtype::Bf16),
                    None,
                    512,
                    1e-6,
                    Tensor::new(positions, rows, 1, Dtype::I32),
                    64,
                    10000.0,
                    true,
                    false,
                    None,
                    &mut Tensor::new(y, rows, width, Dtype::Bf16),
                )
                .unwrap();
            };
            fire();
            gpu.sync();
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                fire();
            }
            gpu.sync();
            let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
            eprintln!("| rows {rows:5} | one pass {vec:5} | {us:7.1} us |");
        }
    }
}
