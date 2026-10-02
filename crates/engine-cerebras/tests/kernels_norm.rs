mod common;

use common::data;
use engine_cerebras::bench::{Bench, assert_close, round_bf16};
use kernels_cerebras::elemwise::norm;

fn rms_ref(x: &[f32], w: &[f32], width: usize, eps: f32, plus_one: bool) -> Vec<f32> {
    x.chunks(width)
        .flat_map(|row| {
            let ss: f32 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / width as f32 + eps).sqrt();
            row.iter()
                .zip(w)
                .map(|(v, g)| round_bf16(v * inv * if plus_one { 1.0 + g } else { *g }))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn rmsnorm_and_its_plus_one_form_answer_the_host() {
    rmsnorm_case(5, 48, 1);
}

/// Rows past a PE (2 x 8192, twice over with y) split their columns: the
/// sums of squares of the blocks gather first, then each block scales.
#[test]
fn a_wide_rmsnorm_splits_its_columns_in_two_phases() {
    rmsnorm_case(2, 8192, 3);
}

fn rmsnorm_case(rows: u32, width: u32, seed: u32) {
    let xs = data((rows * width) as usize, seed);
    let ws = data(width as usize, seed + 1);
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let w = b.bf16(1, width, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, rows, width);
    let z = b.zeros(dtype::Dtype::Bf16, rows, width);
    let ran = b
        .run(|ctx| {
            norm::rmsnorm(ctx, x, w, 1e-6, y)?;
            norm::rmsnorm_plus_one(ctx, x, w, 1e-6, z)
        })
        .unwrap();
    if !ran {
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
        &rms_ref(&xs, &ws, width as usize, 1e-6, true),
        1e-2,
        1e-2,
    );
}

#[test]
fn a_residual_add_lands_in_its_second_operand() {
    let (rows, width) = (3u32, 20u32);
    let xs = data((rows * width) as usize, 3);
    let ys = data((rows * width) as usize, 4);
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let y = b.bf16(rows, width, &ys);
    if !b.run(|ctx| norm::residual_add(ctx, x, y)).unwrap() {
        return;
    }
    let want: Vec<f32> = xs.iter().zip(&ys).map(|(p, q)| round_bf16(p + q)).collect();
    assert_close(&b.read_f32(y), &want, 0.0, 0.0);
    assert_close(&b.read_f32(x), &xs, 0.0, 0.0);
}

#[test]
fn per_head_plus_one_normalises_each_head_run() {
    let (rows, head, heads) = (3u32, 16u32, 3u32);
    let xs = data((rows * head * heads) as usize, 81);
    let ws = data(head as usize, 82);
    let mut b = Bench::new();
    let x = b.bf16(rows, head * heads, &xs);
    let w = b.bf16(1, head, &ws);
    let y = b.zeros(dtype::Dtype::Bf16, rows, head * heads);
    if !b
        .run(|ctx| norm::rmsnorm_per_head_plus_one(ctx, x, w, head, 1e-6, y))
        .unwrap()
    {
        return;
    }
    assert_close(
        &b.read_f32(y),
        &rms_ref(&xs, &ws, head as usize, 1e-6, true),
        1e-2,
        1e-2,
    );
}

#[test]
fn a_gated_rmsnorm_applies_silu_or_sigmoid_of_z() {
    gated_rmsnorm_case(3, 12, 2);
}

/// Rows past a PE (8 x 2048 x three planes) spread over PEs by row.
#[test]
fn a_wide_gated_rmsnorm_spreads_its_rows_over_pes() {
    gated_rmsnorm_case(8, 128, 16);
}

fn gated_rmsnorm_case(rows: u32, vd: u32, heads: u32) {
    let acc: Vec<f32> = data((rows * vd * heads) as usize, 91)
        .iter()
        .map(|v| v * 2.5)
        .collect();
    let zs: Vec<f32> = data((rows * vd * heads) as usize, 92)
        .iter()
        .map(|v| round_bf16(v * 3.0))
        .collect();
    let gw = data(vd as usize, 93);
    let mut b = Bench::new();
    let x = b.f32(rows, vd * heads, &acc);
    let z = b.bf16(rows, vd * heads, &zs);
    let w = b.f32(1, vd, &gw);
    let y_silu = b.zeros(dtype::Dtype::Bf16, rows, vd * heads);
    let y_sig = b.zeros(dtype::Dtype::Bf16, rows, vd * heads);
    let ran = b
        .run(|ctx| {
            norm::rmsnorm_gated(ctx, x, z, w, vd, 1e-6, false, y_silu)?;
            norm::rmsnorm_gated(ctx, x, z, w, vd, 1e-6, true, y_sig)
        })
        .unwrap();
    if !ran {
        return;
    }
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let gated = |sig: bool| -> Vec<f32> {
        let mut out = vec![0f32; acc.len()];
        for (run, c) in acc.chunks(vd as usize).enumerate() {
            let ms: f32 = c.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for (i, v) in c.iter().enumerate() {
                let zv = zs[run * vd as usize + i];
                let g = if sig { sigmoid(zv) } else { zv * sigmoid(zv) };
                out[run * vd as usize + i] = round_bf16(v * inv * gw[i] * g);
            }
        }
        out
    };
    assert_close(&b.read_f32(y_silu), &gated(false), 1e-2, 1e-2);
    assert_close(&b.read_f32(y_sig), &gated(true), 1e-2, 1e-2);
}

/// `out += bias` per column, over a tile when the rows exceed a PE.
#[test]
fn a_bias_adds_to_every_row() {
    for (rows, width) in [(3u32, 20u32), (4, 4096)] {
        let xs = data((rows * width) as usize, 61);
        let bs = data(width as usize, 62);
        let mut b = Bench::new();
        let out = b.f32(rows, width, &xs);
        let bias = b.f32(1, width, &bs);
        if !b.run(|ctx| norm::add_bias(ctx, bias, out)).unwrap() {
            return;
        }
        let want: Vec<f32> = xs
            .iter()
            .enumerate()
            .map(|(i, v)| v + bs[i % width as usize])
            .collect();
        assert_close(&b.read_f32(out), &want, 1e-5, 1e-5);
    }
}

fn layernorm_ref(x: &[f32], w: &[f32], bias: &[f32], width: usize, eps: f32) -> Vec<f32> {
    x.chunks(width)
        .flat_map(|row| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + eps).sqrt();
            row.iter()
                .enumerate()
                .map(|(i, v)| (v - mean) * inv * w[i] + bias[i])
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Layernorm on one PE, and over column blocks with the row's sums
/// gathered first when a row exceeds a PE.
#[test]
fn a_layernorm_centres_and_scales_each_row() {
    for (rows, width, seed) in [(3u32, 24u32, 63u32), (2, 6144, 66)] {
        let xs = data((rows * width) as usize, seed);
        let ws = data(width as usize, seed + 1);
        let bs = data(width as usize, seed + 2);
        let mut b = Bench::new();
        let x = b.f32(rows, width, &xs);
        let w = b.f32(1, width, &ws);
        let bias = b.f32(1, width, &bs);
        let y = b.zeros(dtype::Dtype::F32, rows, width);
        if !b
            .run(|ctx| norm::layernorm(ctx, x, w, bias, 1e-5, y))
            .unwrap()
        {
            return;
        }
        assert_close(
            &b.read_f32(y),
            &layernorm_ref(&xs, &ws, &bs, width as usize, 1e-5),
            2e-2,
            2e-2,
        );
    }
}
