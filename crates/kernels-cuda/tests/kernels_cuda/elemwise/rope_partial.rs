#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope;
use kernels_cuda::tensor::Tensor;

/// Rotates the first `rotary_dim/2` pairs of a head, each `span` apart, by
/// `pos * theta^(-i/span)`: `span = rotary_dim/2` is HF transformers', mlx-lm's
/// and llama.cpp's partial rotary, `span = head_dim/2` their proportional one
/// (Gemma 4's global layers). Everything else is untouched.
fn turn(head: &mut [f32], pos: i32, rotary_dim: usize, span: usize, theta: f32) {
    for dp in 0..rotary_dim / 2 {
        let (a, b) = (head[dp], head[dp + span]);
        let freq = theta.powf(-(dp as f32) / span as f32);
        let (s, c) = (pos as f32 * freq).sin_cos();
        head[dp] = a * c - b * s;
        head[dp + span] = b * c + a * s;
    }
}

fn check(head_dim: usize, q_heads: usize, kv_heads: usize, rotary_dim: usize, span: usize) {
    let rows = 6usize;
    let (q_width, k_width) = (q_heads * head_dim, kv_heads * head_dim);
    let mut lcg = Lcg::seeded(0x5e ^ (head_dim as u64) ^ ((rotary_dim as u64) << 8));
    let (q_raw, q_in) = lcg.row(rows * q_width);
    let (k_raw, k_in) = lcg.row(rows * k_width);
    // positions far from zero: the two conventions coincide near 0 and part ways
    // with distance, so a drifting convention only shows up out here.
    let positions: Vec<i32> = (0..rows as i32).map(|r| 700 + 331 * r).collect();
    let theta = 1.0e7f32;

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let p_at = gpu.up(&positions);
    let ctx = gpu.ctx();
    let mut q = Tensor::new(q_at, rows as u32, q_width as u32, Dtype::Bf16);
    let mut k = Tensor::new(k_at, rows as u32, k_width as u32, Dtype::Bf16);
    rope::partial(
        &ctx,
        &mut q,
        &mut k,
        Tensor::new(p_at, rows as u32, 1, Dtype::I32),
        rotary_dim as u32,
        head_dim as u32,
        theta,
        span as u32,
    )
    .expect("the partial rope fires");
    gpu.sync();
    let got_q: Vec<u16> = gpu.down(q_at, rows * q_width);
    let got_k: Vec<u16> = gpu.down(k_at, rows * k_width);

    for (label, src, got, heads, width) in [
        ("q", &q_in, &got_q, q_heads, q_width),
        ("k", &k_in, &got_k, kv_heads, k_width),
    ] {
        for (r, &position) in positions.iter().enumerate().take(rows) {
            for h in 0..heads {
                let at = r * width + h * head_dim;
                let mut want = src[at..at + head_dim].to_vec();
                turn(&mut want, position, rotary_dim, span, theta);
                for (i, want) in want.into_iter().enumerate() {
                    let g = from_bf16(got[at + i]);
                    assert!(
                        (g - want).abs() <= want.abs() * (1.0 / 64.0) + 1.5e-2,
                        "{label} head_dim {head_dim} rotary {rotary_dim}: \
                         [{r}][{h}][{i}] = {g}, want {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_partial_rope_turns_only_the_rotary_pairs_every_case() {
    // (head_dim, q_heads, kv_heads, rotary_dim, span): a full rotary head;
    // partial ones (Qwen3 A3B's 64 of 256) leaving the tail where it was;
    // Gemma 4's global layers, a quarter of 512 turning, each pair half a
    // head apart.
    for (head_dim, q_heads, kv_heads, rotary_dim, span) in [
        (128, 4, 2, 128, 64),
        (256, 16, 2, 64, 32),
        (128, 8, 1, 64, 32),
        (512, 4, 1, 128, 256),
    ] {
        check(head_dim, q_heads, kv_heads, rotary_dim, span);
    }
}
