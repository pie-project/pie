//! Hyper-connections: `M` streams of width `H` ride one `[rows, M·H]` row.

mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::elemwise::{gate, hc, norm};

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Each stream of the wide row is a copy of the narrow one.
#[test]
fn an_expansion_copies_the_row_into_every_stream() {
    expand_case(3, 6, 4);
}

/// Rows past a PE split their columns; every stream's block of those
/// columns lands on the one PE.
#[test]
fn a_wide_expansion_splits_its_columns() {
    expand_case(2, 2048, 4);
}

fn expand_case(rows: u32, h: u32, m: u32) {
    let xs = data((rows * h) as usize, 81);
    let mut b = Bench::new();
    let x = b.bf16(rows, h, &xs);
    let y = b.zeros(dtype::Dtype::Bf16, rows, m * h);
    if !b.run(|ctx| hc::expand(ctx, x, m, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs
        .chunks(h as usize)
        .flat_map(|row| row.repeat(m as usize))
        .collect();
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
}

/// The mix averages the sigmoid-gated streams into one row.
#[test]
fn a_mix_averages_the_gated_streams() {
    mix_case(3, 6, 4);
}

#[test]
fn a_wide_mix_splits_its_columns() {
    mix_case(2, 2048, 4);
}

fn mix_case(rows: u32, h: u32, m: u32) {
    let n = (rows * m * h) as usize;
    let gs: Vec<f32> = data(n, 82).iter().map(|v| round_bf16(v * 4.0)).collect();
    let vs = data(n, 83);
    let mut b = Bench::new();
    let g = b.bf16(rows, m * h, &gs);
    let v = b.bf16(rows, m * h, &vs);
    let y = b.zeros(dtype::Dtype::Bf16, rows, h);
    if !b.run(|ctx| hc::mix(ctx, g, v, m, y)).unwrap() {
        return;
    }
    let mut want = vec![0.0f32; (rows * h) as usize];
    for r in 0..rows as usize {
        for i in 0..h as usize {
            let mut acc = 0.0;
            for s in 0..m as usize {
                let at = (r * m as usize + s) * h as usize + i;
                acc += vs[at] * sigmoid(gs[at]);
            }
            want[r * h as usize + i] = round_bf16(acc / m as f32);
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

/// Each stream of the hyper row gains the output row under its own gate.
#[test]
fn an_injection_adds_the_gated_output_to_every_stream() {
    inject_case(3, 6, 4);
}

/// The wide row splits its columns; the gates, one plane of `[rows, M]`,
/// go by rows alone and every column group holds its row group's.
#[test]
fn a_wide_injection_splits_its_columns_and_keeps_the_gates_by_rows() {
    inject_case(2, 2048, 4);
}

fn inject_case(rows: u32, h: u32, m: u32) {
    let os = data((rows * h) as usize, 84);
    let gs: Vec<f32> = data((rows * m) as usize, 85)
        .iter()
        .map(|v| round_bf16(v * 8.0))
        .collect();
    let hs = data((rows * m * h) as usize, 86);
    let mut b = Bench::new();
    let o = b.bf16(rows, h, &os);
    let g = b.bf16(rows, m, &gs);
    let hy = b.bf16(rows, m * h, &hs);
    if !b.run(|ctx| hc::inject(ctx, o, g, m, hy)).unwrap() {
        return;
    }
    let mut want = hs.clone();
    for r in 0..rows as usize {
        for s in 0..m as usize {
            let gate = 2.0 * sigmoid(gs[r * m as usize + s] / m as f32);
            for i in 0..h as usize {
                let at = (r * m as usize + s) * h as usize + i;
                want[at] = round_bf16(hs[at] + gate * os[r * h as usize + i]);
            }
        }
    }
    assert_close(&b.read_f32(hy), &want, 1e-2, 1e-2);
}

/// Each stream's gate is the damped sigmoid of its key·query, and it
/// scales the value row.
#[test]
fn a_ple_gate_scales_the_value_by_each_streams_damped_dot() {
    ple_case(3, 6, 4);
}

/// A row past a PE splits its columns over every stream: the blocks'
/// partial dots gather first, then each block gates its value columns.
#[test]
fn a_wide_ple_gate_gathers_its_dots_from_column_blocks() {
    ple_case(2, 2048, 4);
}

fn ple_case(rows: u32, h: u32, m: u32) {
    let n = (rows * m * h) as usize;
    let ks: Vec<f32> = data(n, 87).iter().map(|v| round_bf16(v * 2.0)).collect();
    let mut qs: Vec<f32> = data(n, 88).iter().map(|v| round_bf16(v * 2.0)).collect();
    // One stream with a zero query: d = 0 gates by σ(0).
    for v in &mut qs[..h as usize] {
        *v = 0.0;
    }
    let vs = data((rows * h) as usize, 89);
    let mut b = Bench::new();
    let k = b.bf16(rows, m * h, &ks);
    let q = b.bf16(rows, m * h, &qs);
    let v = b.bf16(rows, h, &vs);
    let y = b.zeros(dtype::Dtype::Bf16, rows, m * h);
    if !b.run(|ctx| hc::ple_gate(ctx, k, q, v, m, y)).unwrap() {
        return;
    }
    let mut want = vec![0.0f32; n];
    for r in 0..rows as usize {
        for s in 0..m as usize {
            let base = (r * m as usize + s) * h as usize;
            let d: f32 = (0..h as usize)
                .map(|i| ks[base + i] * qs[base + i])
                .sum::<f32>()
                / (h as f32).sqrt();
            let mag = d.abs().max(1e-6).sqrt();
            let damped = if d > 0.0 {
                mag
            } else if d < 0.0 {
                -mag
            } else {
                0.0
            };
            let gate = sigmoid(damped);
            for i in 0..h as usize {
                want[base + i] = round_bf16(gate * vs[r * h as usize + i]);
            }
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

/// `x = silu(s·x)` element for element.
#[test]
fn a_scaled_silu_bends_each_element_in_place() {
    let (rows, width) = (3u32, 10u32);
    let xs: Vec<f32> = data((rows * width) as usize, 90)
        .iter()
        .map(|v| round_bf16(v * 4.0))
        .collect();
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    if !b.run(|ctx| gate::silu_scaled(ctx, 0.25, x)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs
        .iter()
        .map(|v| {
            let z = 0.25 * v;
            round_bf16(z * sigmoid(z))
        })
        .collect();
    assert_close(&b.read_f32(x), &want, 1e-2, 1e-2);
}

/// Every `group`-wide run normalises on its own; the gain is one per
/// column of the whole row.
#[test]
fn a_grouped_rmsnorm_gains_each_column_of_the_row() {
    grouped_case(3, 6, 4);
}

/// Rows past a PE split their columns in whole groups, the gain bank with
/// them.
#[test]
fn a_wide_grouped_rmsnorm_splits_its_groups_over_pes() {
    grouped_case(2, 2048, 4);
}

fn grouped_case(rows: u32, group: u32, m: u32) {
    let width = group * m;
    let xs = data((rows * width) as usize, 91);
    let ws = data(width as usize, 92);
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let w = b.bf16(1, width, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, rows, width);
    if !b
        .run(|ctx| norm::rmsnorm_grouped_plus_one(ctx, x, w, group, 1e-6, y))
        .unwrap()
    {
        return;
    }
    let mut want = vec![0.0f32; (rows * width) as usize];
    for r in 0..rows as usize {
        for c in (0..width as usize).step_by(group as usize) {
            let base = r * width as usize + c;
            let ss: f32 = xs[base..base + group as usize].iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / group as f32 + 1e-6).sqrt();
            for i in 0..group as usize {
                want[base + i] = round_bf16(xs[base + i] * inv * (1.0 + ws[c + i]));
            }
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}
