mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::elemwise::rope;

fn rope_ref(x: &[f32], positions: &[i32], rotary: usize, head: usize, theta: f32) -> Vec<f32> {
    let width = x.len() / positions.len();
    let mut out = x.to_vec();
    for (r, pos) in positions.iter().enumerate() {
        for h in 0..width / head {
            let base = r * width + h * head;
            for i in 0..rotary / 2 {
                let inv = theta.powf(-(2.0 * i as f32) / rotary as f32);
                let ang = *pos as f32 * inv;
                let (s, c) = ang.sin_cos();
                let (x1, x2) = (x[base + i], x[base + i + rotary / 2]);
                out[base + i] = round_bf16(x1 * c - x2 * s);
                out[base + i + rotary / 2] = round_bf16(x1 * s + x2 * c);
            }
        }
    }
    out
}

#[test]
fn a_partial_rotation_turns_q_and_k_in_place() {
    let (rows, head, rotary) = (4u32, 8u32, 6u32);
    let (qw, kw) = (3 * head, 2 * head);
    let qs = data((rows * qw) as usize, 9);
    let ks = data((rows * kw) as usize, 10);
    let positions = [0i32, 1, 7, 30];
    let theta = 10_000.0;
    let mut b = Bench::new();
    let q = b.bf16(rows, qw, &qs);
    let k = b.bf16(rows, kw, &ks);
    let p = b.i32(rows, 1, &positions);
    if !b
        .run(|ctx| rope::partial(ctx, q, k, p, rotary, head, theta))
        .unwrap()
    {
        return;
    }
    assert_close(
        &b.read_f32(q),
        &rope_ref(&qs, &positions, rotary as usize, head as usize, theta),
        2e-2,
        2e-2,
    );
    assert_close(
        &b.read_f32(k),
        &rope_ref(&ks, &positions, rotary as usize, head as usize, theta),
        2e-2,
        2e-2,
    );
}

/// The multimodal forms: each head's pairs turn by the row's `(t, h, w)`
/// position on the pair's axis, as the form's table says.
fn mrope_ref(
    x: &[f32],
    positions: &[i32],
    head: usize,
    table: &[kernels_cerebras::elemwise::rope_mrope::Pair],
) -> Vec<f32> {
    let mut y = x.to_vec();
    let rows = x.len() / (y.len() / x.len()).max(1);
    let _ = rows;
    let width = x.len() / (positions.len() / 3);
    for (r, row) in y.chunks_mut(width).enumerate() {
        for h in 0..width / head {
            for p in table {
                let pos = positions[r * 3 + p.axis as usize] as f32;
                let (c, s) = ((pos * p.freq).cos(), (pos * p.freq).sin());
                let (a, b) = (h * head + p.lo as usize, h * head + p.hi as usize);
                let (x1, x2) = (row[a], row[b]);
                row[a] = x1 * c - x2 * s;
                row[b] = x1 * s + x2 * c;
            }
        }
    }
    y
}

fn mrope_case(form: &str, sections: [u32; 3], rotary: u32, head: u32) {
    use kernels_cerebras::elemwise::rope_mrope as m;
    let rows = 4u32;
    let (qw, kw) = (2 * head, head);
    let qs = data((rows * qw) as usize, 11);
    let ks = data((rows * kw) as usize, 12);
    let positions = [0i32, 0, 0, 3, 1, 2, 7, 7, 0, 30, 2, 9];
    let theta = 10_000.0;
    let table = match form {
        "interleaved" => m::interleaved_pairs(sections, rotary, theta),
        "blocked" => m::blocked_table(sections, rotary, head, theta).unwrap(),
        _ => m::split_table(sections, rotary, theta).unwrap(),
    };
    let mut b = Bench::new();
    let q = b.bf16(rows, qw, &qs);
    let k = b.bf16(rows, kw, &ks);
    let p = b.i32(rows, 3, &positions);
    let ran = b
        .run(|ctx| match form {
            "interleaved" => m::interleaved(ctx, q, k, p, sections, rotary, head, theta),
            "blocked" => m::blocked(ctx, q, k, p, sections, rotary, head, theta),
            _ => m::split(ctx, q, k, p, sections, rotary, head, theta),
        })
        .unwrap();
    if !ran {
        return;
    }
    assert_close(
        &b.read_f32(q),
        &mrope_ref(&qs, &positions, head as usize, &table),
        2e-2,
        2e-2,
    );
    assert_close(
        &b.read_f32(k),
        &mrope_ref(&ks, &positions, head as usize, &table),
        2e-2,
        2e-2,
    );
}

#[test]
fn multimodal_rotations_follow_their_pair_tables() {
    mrope_case("interleaved", [1, 1, 1], 8, 16);
    mrope_case("blocked", [2, 1, 1], 8, 16);
    mrope_case("split", [2, 1, 1], 8, 16);
}
