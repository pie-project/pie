//! The norm family against host references.

use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::elemwise::norm;

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

fn rms_ref(x: &[f32], w: &[f32], axis: usize, eps: f32, plus_one: bool) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for (run, chunk) in x.chunks(axis).enumerate() {
        let ms: f32 = chunk.iter().map(|v| v * v).sum::<f32>() / axis as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for (i, v) in chunk.iter().enumerate() {
            let g = if plus_one { 1.0 + w[i] } else { w[i] };
            out[run * axis + i] = g * (v * inv);
        }
    }
    out
}

#[test]
fn rmsnorm_and_its_per_head_plus_one_form_answer_the_host() {
    let (rows, width, head) = (5u32, 256u32, 64u32);
    let xs = data((rows * width) as usize, 1);
    let ws = data(width as usize, 2);
    let hs = data(head as usize, 3);
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let w = b.bf16(1, width, &ws);
    let h = b.bf16(1, head, &hs);
    let y = b.zeros(dtype::Dtype::Bf16, rows, width);
    let z = b.zeros(dtype::Dtype::Bf16, rows, width);
    if !b
        .run(|ctx| {
            norm::rmsnorm(ctx, x, w, 1e-6, y)?;
            norm::rmsnorm_per_head_plus_one(ctx, x, h, head, 1e-6, z)
        })
        .unwrap()
    {
        return;
    }
    assert_close(
        &b.read_f32(y),
        &rms_ref(&xs, &ws, width as usize, 1e-6, false),
        1e-2,
        1e-2,
    );
    assert_close(
        &b.read_f32(z),
        &rms_ref(&xs, &hs, head as usize, 1e-6, true),
        1e-2,
        1e-2,
    );
}

#[test]
fn a_residual_add_lands_in_its_second_operand() {
    let xs = data(64, 4);
    let ys = data(64, 5);
    let mut b = Bench::new();
    let x = b.bf16(4, 16, &xs);
    let y = b.bf16(4, 16, &ys);
    if !b.run(|ctx| norm::residual_add(ctx, x, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(a, b)| round_bf16(a + b)).collect();
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
}
