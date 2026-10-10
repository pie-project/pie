#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope_mrope;
use kernels_cuda::tensor::Tensor;

/// Upstream's interleaved mrope (Qwen3-VL, Qwen3.5): pair `i` of the first
/// `rotary_dim/2` reads axis `h` when `i % 3 == 1 && i < 3·s1`, `w` when
/// `i % 3 == 2 && i < 3·s2`, else `t`, and turns `(i, i + rotary_dim/2)` by
/// `pos · theta^(-2i/rotary_dim)`; past `rotary_dim` nothing moves.
fn turn(head: &mut [f32], pos: [i32; 3], sections: [u32; 3], rotary_dim: usize, theta: f32) {
    let half = rotary_dim / 2;
    for i in 0..half {
        let axis = match i % 3 {
            1 if i < 3 * sections[1] as usize => 1,
            2 if i < 3 * sections[2] as usize => 2,
            _ => 0,
        };
        let freq = theta.powf(-2.0 * i as f32 / rotary_dim as f32);
        let (s, c) = (pos[axis] as f32 * freq).sin_cos();
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * c - b * s;
        head[i + half] = b * c + a * s;
    }
}

fn check(head_dim: usize, q_heads: usize, kv_heads: usize, rotary_dim: usize, sections: [u32; 3]) {
    let rows = 6usize;
    let (q_width, k_width) = (q_heads * head_dim, kv_heads * head_dim);
    let mut lcg = Lcg::seeded(0x3d ^ (head_dim as u64) ^ ((rotary_dim as u64) << 8));
    let (q_raw, q_in) = lcg.row(rows * q_width);
    let (k_raw, k_in) = lcg.row(rows * k_width);
    // distinct axes, far from zero: a pair reading the wrong axis or the wrong
    // ladder only parts from upstream once the angles grow.
    let positions: Vec<[i32; 3]> = (0..rows as i32)
        .map(|r| [700 + 331 * r, 40 + 7 * r, 90 + 13 * r])
        .collect();
    let flat: Vec<i32> = positions.iter().flatten().copied().collect();
    let theta = 1.0e7f32;

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let p_at = gpu.up(&flat);
    let ctx = gpu.ctx();
    let mut q = Tensor::new(q_at, rows as u32, q_width as u32, Dtype::Bf16);
    let mut k = Tensor::new(k_at, rows as u32, k_width as u32, Dtype::Bf16);
    rope_mrope::interleaved(
        &ctx,
        &mut q,
        &mut k,
        Tensor::new(p_at, rows as u32, rope_mrope::AXES, Dtype::I32),
        sections,
        rotary_dim as u32,
        head_dim as u32,
        theta,
    )
    .expect("the interleaved mrope fires");
    gpu.sync();
    let got_q: Vec<u16> = gpu.down(q_at, rows * q_width);
    let got_k: Vec<u16> = gpu.down(k_at, rows * k_width);

    for (label, src, got, heads, width) in [
        ("q", &q_in, &got_q, q_heads, q_width),
        ("k", &k_in, &got_k, kv_heads, k_width),
    ] {
        for r in 0..rows {
            for h in 0..heads {
                let at = r * width + h * head_dim;
                let mut want = src[at..at + head_dim].to_vec();
                turn(&mut want, positions[r], sections, rotary_dim, theta);
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
fn the_interleaved_mrope_turns_the_rotary_prefix_as_its_own_head_every_case() {
    a_full_rotary_head_turns_end_to_end();
    a_partial_rotary_head_turns_its_prefix_and_leaves_the_tail();
}

fn a_full_rotary_head_turns_end_to_end() {
    // Qwen3-VL: head_dim 128, every pair rotated, sections [24, 20, 20].
    check(128, 4, 2, 128, [24, 20, 20]);
}

fn a_partial_rotary_head_turns_its_prefix_and_leaves_the_tail() {
    // Qwen3.5: head_dim 256, rotary_dim 64, sections [11, 11, 10].
    check(256, 8, 2, 64, [11, 11, 10]);
}
