#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::rope::{Freqs, Pairs, rope};
use kernels_cuda::tensor::Tensor;

/// The host's turn of one head: pair `i` is `(i, i + span)`, or `(2i, 2i+1)`
/// interleaved, at `pos · inv_freq[i]` (an M-RoPE pair reading its axis as
/// Qwen's interleaved form does), cos and sin scaled by `mscale`, in f64.
fn turn(head: &mut [f32], pos: [i32; 3], pairs: Pairs, inv_freq: &[f32], mscale: f32) {
    for (i, &f) in inv_freq.iter().enumerate() {
        let axis = match (pairs.sections, i % 3) {
            (Some(s), 1) if i < 3 * s[1] as usize => 1,
            (Some(s), 2) if i < 3 * s[2] as usize => 2,
            _ => 0,
        };
        let angle = f64::from(pos[axis] as f32 * f);
        let (s, c) = angle.sin_cos();
        let (s, c) = (s * f64::from(mscale), c * f64::from(mscale));
        let (j0, j1) = if pairs.interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + pairs.span as usize)
        };
        let (a, b) = (f64::from(head[j0]), f64::from(head[j1]));
        head[j0] = (a * c - b * s) as f32;
        head[j1] = (b * c + a * s) as f32;
    }
}

fn check(pairs: Pairs, turned: usize, q_heads: usize, kv_heads: usize, mscale: f32) {
    let rows = 5usize;
    let head_dim = pairs.head_dim as usize;
    let (q_width, k_width) = (q_heads * head_dim, kv_heads * head_dim);
    let mut lcg = Lcg::seeded(0x7ab1e ^ (head_dim as u64) ^ ((turned as u64) << 9));
    let (q_raw, q_in) = lcg.row(rows * q_width);
    let (k_raw, k_in) = lcg.row(rows * k_width.max(1));
    // A ladder no theta spells (each pair its own scale), and positions out
    // past 128K, where a fast sine drifts off the reference's cos/sin cache.
    let inv_freq: Vec<f32> = (0..turned)
        .map(|i| 10_000f32.powf(-(i as f32) / turned as f32) / (1.0 + (i % 5) as f32))
        .collect();
    let positions: Vec<[i32; 3]> = (0..rows as i32)
        .map(|r| [131_000 - 977 * r, 40 + 7 * r, 90 + 13 * r])
        .collect();
    let axes = if pairs.sections.is_some() { 3 } else { 1 };
    let flat: Vec<i32> = positions.iter().flat_map(|p| p[..axes].to_vec()).collect();

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let p_at = gpu.up(&flat);
    let ctx = gpu.ctx();
    let mut q = Tensor::new(q_at, rows as u32, q_width as u32, Dtype::Bf16);
    let mut k = Tensor::new(k_at, rows as u32, k_width as u32, Dtype::Bf16);
    rope(
        &ctx,
        "elemwise.rope",
        &mut q,
        &mut k,
        Tensor::new(p_at, rows as u32, axes as u32, Dtype::I32),
        pairs,
        Freqs {
            inv_freq: &inv_freq,
            mscale,
        },
    )
    .expect("the stated table fires");
    gpu.sync();
    let got_q: Vec<u16> = gpu.down(q_at, rows * q_width);
    let got_k: Vec<u16> = gpu.down(k_at, rows * k_width.max(1));

    for (label, src, got, heads, width) in [
        ("q", &q_in, &got_q, q_heads, q_width),
        ("k", &k_in, &got_k, kv_heads, k_width),
    ] {
        for (r, &position) in positions.iter().enumerate() {
            for h in 0..heads {
                let at = r * width + h * head_dim;
                let mut want = src[at..at + head_dim].to_vec();
                turn(&mut want, position, pairs, &inv_freq, mscale);
                for (i, want) in want.into_iter().enumerate() {
                    let g = from_bf16(got[at + i]);
                    assert!(
                        (g - want).abs() <= want.abs() / 128.0 + 1e-2,
                        "{label} {pairs:?}: [{r}][{h}][{i}] = {g}, want {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_stated_rope_table_turns_each_pair_at_its_frequency() {
    let pairs = |head_dim, span, interleaved, sections| Pairs {
        head_dim,
        span,
        interleaved,
        sections,
    };
    // Llama's whole head, neox halves.
    check(pairs(128, 64, false, None), 64, 4, 2, 1.0);
    // An interleaved whole head, YaRN's attention factor on cos and sin.
    check(pairs(64, 32, true, None), 32, 2, 1, 1.1386);
    // Qwen3.5's partial rotary: 64 of a 256-wide head, k's tail untouched.
    check(pairs(256, 32, false, None), 32, 4, 1, 1.0);
    // q alone (k zero wide).
    check(pairs(128, 64, false, None), 64, 3, 0, 1.0);
    // Qwen3.5's interleaved M-RoPE over its rotary prefix.
    check(pairs(256, 32, false, Some([11, 11, 10])), 32, 4, 2, 1.0);
}
