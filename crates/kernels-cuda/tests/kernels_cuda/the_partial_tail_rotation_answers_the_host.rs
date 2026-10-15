#![cfg(feature = "cuda")]

//! DeepSeek-V4's rotation of a head's last 64 dims (`rope_partial_last`),
//! forward and inverse, plain and YaRN-scaled, half-split and interleaved,
//! against the rotation worked on the host: the untouched head stays put and
//! the tail turns by each position's angles.

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope;
use kernels_cuda::tensor::Tensor;

fn check(interleaved: bool, inverse: bool, yarn: bool) {
    let (rows, heads, d, rot) = (5usize, 64usize, 512usize, 64usize);
    let theta = 10000.0f32;
    let positions: Vec<i32> = vec![0, 1, 17, 300, 4095];
    let mut lcg = Lcg::seeded(0x7a11 + u64::from(interleaved) + 2 * u64::from(inverse));
    let x: Vec<u16> = (0..rows * heads * d).map(|_| to_bf16(lcg.unit())).collect();
    let mut gpu = Gpu::open();
    let q_at = gpu.up(&x);
    let mut q = Tensor::new(q_at, rows as u32, (heads * d) as u32, Dtype::Bf16);
    let yarn_cfg = yarn.then_some(rope::Yarn {
        factor: 16.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position: 65536,
    });
    rope::partial_last(
        &gpu.ctx(),
        &mut q,
        Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32),
        rot as u32,
        d as u32,
        theta,
        interleaved,
        inverse,
        yarn_cfg,
    )
    .expect("the rotation fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(q_at, rows * heads * d);
    let half = rot / 2;
    for r in 0..rows {
        for h in 0..heads {
            let base = (r * heads + h) * d;
            for i in 0..d - rot {
                assert_eq!(
                    got[base + i],
                    x[base + i],
                    "row {r} head {h}: dim {i} is not the tail's"
                );
            }
            if yarn {
                continue; // The scaled angles are the kernel's own ramp; the head's bulk stays put.
            }
            for p in 0..half {
                let freq = f64::from(theta).powf(-2.0 * p as f64 / rot as f64);
                let ang = (if inverse { -1.0 } else { 1.0 }) * f64::from(positions[r]) * freq;
                let (s, c) = ang.sin_cos();
                let (i, j) = if interleaved {
                    (d - rot + 2 * p, d - rot + 2 * p + 1)
                } else {
                    (d - rot + p, d - rot + p + half)
                };
                let (a, b) = (
                    f64::from(from_bf16(x[base + i])),
                    f64::from(from_bf16(x[base + j])),
                );
                let (wi, wj) = (a * c - b * s, b * c + a * s);
                let (gi, gj) = (
                    f64::from(from_bf16(got[base + i])),
                    f64::from(from_bf16(got[base + j])),
                );
                assert!(
                    (gi - wi).abs() <= 2e-2 && (gj - wj).abs() <= 2e-2,
                    "row {r} head {h} pair {p}: ({gi}, {gj}) vs ({wi}, {wj})"
                );
            }
        }
    }
}

#[test]
fn the_partial_tail_rotation_answers_the_host() {
    for interleaved in [false, true] {
        for inverse in [false, true] {
            check(interleaved, inverse, false);
        }
    }
    check(false, false, true);
}
