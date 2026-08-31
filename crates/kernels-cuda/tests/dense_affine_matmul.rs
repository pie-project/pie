//! **THE DENSE AFFINE PROJECTION, ON A REAL DEVICE** (qwen4 stored-form
//! wave): `linear::quant::matmul` / `lm_head` over an MLX affine triplet,
//! held against a host fold of the same planes, at both code widths.
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test dense_affine_matmul -- --nocapture
//! ```
//!
//! The gates:
//!
//! ```text
//! (a) eight-bit codes: y = act · (s·c + b)^T matches the host fold, on a
//!     rectangle whose N is not a whole row tile (the clamp path runs)
//! (b) four-bit codes: the same, through the nibble decode
//! (c) `lm_head` answers what `matmul` answers (one launch, two names)
//! (d) a factor plane grouped by anything but sixty-four is refused, and a
//!     missing zero-point plane is refused by name
//! ```
//!
//! The reference folds in the kernel's own order — codes to a group in
//! storage order, `s·Σ c·x + b·Σ x` per group — so what the comparison
//! allows is float reassociation across the warp reduce, nothing more.

#![cfg(feature = "_cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::quant;
use kernels_cuda::tensor::Tensor;

const ROWS: u32 = 5;
const N: u32 = 100;
const K: u32 = 128;
const GROUP: usize = 64;

struct Planes {
    codes: Vec<u8>,
    scales: Vec<u16>,
    biases: Vec<u16>,
}

/// A seeded triplet and the bf16 numbers the device will read out of it.
fn triplet(bits: u32, rng: &mut Lcg) -> Planes {
    let groups = K as usize / GROUP;
    let mut codes = Vec::new();
    let mut scales = Vec::with_capacity(N as usize * groups);
    let mut biases = Vec::with_capacity(N as usize * groups);
    let mut byte = 0u8;
    for at in 0..(N as usize * K as usize) {
        let code = (rng.unit().abs() * 255.0) as u32;
        match bits {
            8 => codes.push(code as u8),
            4 => {
                let nibble = (code & 0xF) as u8;
                if at % 2 == 0 {
                    byte = nibble;
                } else {
                    codes.push(byte | (nibble << 4));
                }
            }
            _ => unreachable!(),
        }
    }
    for _ in 0..(N as usize * groups) {
        scales.push(to_bf16(rng.unit() * 0.05));
        biases.push(to_bf16(rng.unit()));
    }
    Planes {
        codes,
        scales,
        biases,
    }
}

/// The host fold, in the kernel's own per-group order.
fn fold(bits: u32, planes: &Planes, x: &[f32]) -> Vec<f32> {
    let groups = K as usize / GROUP;
    let mut y = vec![0.0f32; ROWS as usize * N as usize];
    for t in 0..ROWS as usize {
        let xt = &x[t * K as usize..][..K as usize];
        for r in 0..N as usize {
            let mut acc = 0.0f32;
            for g in 0..groups {
                let mut part = 0.0f32;
                let mut xsum = 0.0f32;
                for j in 0..GROUP {
                    let at = r * K as usize + g * GROUP + j;
                    let code = match bits {
                        8 => f32::from(planes.codes[at]),
                        4 => {
                            let byte = planes.codes[at / 2];
                            f32::from(if at % 2 == 0 { byte & 0xF } else { byte >> 4 })
                        }
                        _ => unreachable!(),
                    };
                    let xv = xt[g * GROUP + j];
                    part += code * xv;
                    xsum += xv;
                }
                let fx = r * groups + g;
                acc += part * from_bf16(planes.scales[fx]);
                acc += xsum * from_bf16(planes.biases[fx]);
            }
            y[t * N as usize + r] = acc;
        }
    }
    y
}

fn fired(bits: u32, head: bool) -> (Vec<f32>, Vec<f32>) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded(0x5eed + u64::from(bits));
    let planes = triplet(bits, &mut rng);
    let (x_raw, x) = rng.row(ROWS as usize * K as usize);

    let act = Tensor::new(gpu.up(&x_raw), ROWS, K, Dtype::Bf16);
    let codes_width = if bits == 8 { K } else { K / 2 };
    let codes = Tensor::new(gpu.up(&planes.codes), N, codes_width, Dtype::U8);
    let scales = Tensor::new(gpu.up(&planes.scales), N, (K / 64) * 2, Dtype::U8);
    let biases = Tensor::new(gpu.up(&planes.biases), N, (K / 64) * 2, Dtype::U8);
    let y_at = gpu.zeros(ROWS as usize * N as usize * 2);
    let mut y = Tensor::new(y_at, ROWS, N, Dtype::Bf16);

    let entry = if head { quant::lm_head } else { quant::matmul };
    entry(
        &gpu.ctx(),
        act,
        codes,
        scales,
        Some(biases),
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the affine projection fires");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, ROWS as usize * N as usize);
    (got.into_iter().map(from_bf16).collect(), fold(bits, &planes, &x))
}

fn held(bits: u32, head: bool) {
    let (got, want) = fired(bits, head);
    for (at, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            close(*g, *w),
            "y[{at}] answered {g} where the host fold says {w} (bits {bits})"
        );
    }
}

#[test]
fn the_eight_bit_projection_matches_the_host_fold() {
    held(8, false);
}

#[test]
fn the_four_bit_projection_matches_the_host_fold() {
    held(4, false);
}

#[test]
fn the_head_answers_what_the_matmul_answers() {
    held(8, true);
}

#[test]
fn a_foreign_grouping_and_a_missing_plane_are_refused() {
    let mut gpu = Gpu::open();
    let act = Tensor::new(gpu.zeros(2 * K as usize), 1, K, Dtype::Bf16);
    let codes = Tensor::new(gpu.zeros(K as usize), 1, K, Dtype::U8);
    let mut y = Tensor::new(gpu.zeros(2), 1, 1, Dtype::Bf16);

    // A 32-wide grouping: the factor plane carries twice the factors.
    let wide = Tensor::new(gpu.zeros((K as usize / 32) * 2), 1, (K / 32) * 2, Dtype::U8);
    let refused = quant::matmul(
        &gpu.ctx(),
        act,
        codes,
        wide,
        Some(wide),
        &mut y,
        GroupSeat::RESIDENT,
    );
    assert!(refused.is_err(), "a 32-wide grouping fired");

    let scales = Tensor::new(gpu.zeros((K as usize / 64) * 2), 1, (K / 64) * 2, Dtype::U8);
    let refused = quant::matmul(&gpu.ctx(), act, codes, scales, None, &mut y, GroupSeat::RESIDENT);
    assert!(refused.is_err(), "a two-plane dense projection fired");
}
