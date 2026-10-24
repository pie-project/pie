//! The timestep sinusoid against the diffusers formula, and T5's relative
//! bucket bias table against the bucket torch lands at every distance.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::elemwise::posenc::{relative_bucket_bias, sinusoid};

fn sinusoid_reference(
    t: f32,
    dim: usize,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
) -> Vec<f32> {
    let half = dim / 2;
    let angles: Vec<f32> = (0..half)
        .map(|i| scale * (t * (-max_period.ln() * i as f32 / half as f32).exp()))
        .collect();
    let mut row: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
    row.extend(angles.iter().map(|a| a.cos()));
    if flip_sin_cos {
        row.rotate_left(half);
    }
    if dim % 2 == 1 {
        row.push(0.0);
    }
    row
}

fn embed(gpu: &Gpu, t: &[f32], dim: u32, max_period: f32, flip: bool, scale: f32) -> Vec<f32> {
    let rows = t.len() as u32;
    let ts = gpu.tensor(t, rows, 1, Dtype::F32);
    let mut o = gpu.zeros(rows, dim, Dtype::F32);
    sinusoid(&gpu.ctx(), ts, dim, max_period, flip, scale, &mut o).expect("the embedding fires");
    gpu.read(&o)
}

fn within(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tolerance, "{what} at {i}: {g} against {w}");
    }
}

#[test]
fn the_sinusoid_embedding_matches_the_diffusers_formula() {
    let Some(gpu) = Gpu::open() else { return };

    let got = embed(&gpu, &[1.0, 500.0], 8, 10_000.0, false, 1.0);
    within(
        &got[..8],
        &[
            0.841_470_98,
            0.099_833_414,
            0.009_999_833,
            0.001,
            0.540_302_3,
            0.995_004_2,
            0.999_95,
            0.999_999_5,
        ],
        2e-6,
        "t = 1",
    );
    within(
        &got[8..],
        &[
            -0.467_771_8,
            -0.262_374_85,
            -0.958_924_3,
            0.479_425_54,
            -0.883_849_26,
            0.964_966_03,
            0.283_662_2,
            0.877_582_55,
        ],
        // Angles up to 500 rad: one f32 ulp of the argument is 3e-5 there,
        // so the device's sincos answers within the bound the angle allows.
        1e-5,
        "t = 500",
    );

    let odd = embed(&gpu, &[0.5], 5, 10_000.0, false, 1.0);
    within(
        &odd,
        &[0.479_425_54, 0.004_999_979, 0.877_582_55, 0.999_987_5, 0.0],
        2e-6,
        "an odd width",
    );

    let flipped = embed(&gpu, &[3.0], 6, 10_000.0, true, 2.0);
    within(
        &flipped,
        &[
            0.960_170_3,
            0.961_470_2,
            0.999_916_45,
            -0.279_415_5,
            0.274_909_27,
            0.012_926_248,
        ],
        2e-6,
        "flipped and scaled",
    );

    const ROWS: usize = 19;
    const DIM: usize = 64;
    let t: Vec<f32> = values(ROWS, 7, 1.0)
        .into_iter()
        .map(|v| (v + 1.0) * 4.0)
        .collect();
    let got = embed(&gpu, &t, DIM as u32, 256.0, true, 1.25);
    for (row, &tv) in t.iter().enumerate() {
        within(
            &got[row * DIM..(row + 1) * DIM],
            &sinusoid_reference(tv, DIM, 256.0, true, 1.25),
            1e-5,
            &format!("row {row}"),
        );
    }
}

fn bucket(d: i64, bidirectional: bool, mut num_buckets: i64, max_distance: f32) -> i64 {
    let mut out = 0;
    let n = if bidirectional {
        num_buckets /= 2;
        if d > 0 {
            out += num_buckets;
        }
        d.abs()
    } else {
        -d.min(0)
    };
    let max_exact = num_buckets / 2;
    if n < max_exact {
        return out + n;
    }
    let x = (n as f32 / max_exact as f32).ln();
    let ratio = (f64::from(max_distance) / max_exact as f64).ln() as f32;
    let large = max_exact + (x / ratio * (num_buckets - max_exact) as f32).trunc() as i64;
    out + large.min(num_buckets - 1)
}

/// A bucket's embedding is `64 · head + bucket`: exact in bf16 and f32, so the
/// table names which bucket each cell read.
const STEP: usize = 64;

fn table(
    gpu: &Gpu,
    heads: u32,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    dtype: Dtype,
) {
    let span = 2 * max_len - 1;
    let emb: Vec<f32> = (0..(num_buckets * heads) as usize)
        .map(|i| (STEP * (i % heads as usize) + i / heads as usize) as f32)
        .collect();
    let e = gpu.tensor(&emb, num_buckets, heads, dtype);
    let mut y = gpu.zeros(heads, span, Dtype::F32);
    relative_bucket_bias(
        &gpu.ctx(),
        e,
        max_len,
        num_buckets,
        max_distance,
        bidirectional,
        &mut y,
    )
    .expect("the table computes");
    let got = gpu.read(&y);
    for h in 0..heads as usize {
        for c in 0..span as usize {
            let d = c as i64 - (i64::from(max_len) - 1);
            let want =
                (STEP * h) as i64 + bucket(d, bidirectional, i64::from(num_buckets), max_distance);
            assert_eq!(
                got[h * span as usize + c] as i64,
                want,
                "{num_buckets}/{max_distance} bidirectional={bidirectional} {dtype:?}: head {h} distance {d}"
            );
        }
    }
}

#[test]
fn the_relative_bucket_bias_table_reads_the_bucket_torch_lands() {
    let Some(gpu) = Gpu::open() else { return };
    table(&gpu, 4, 512, 32, 128.0, true, Dtype::Bf16);
    table(&gpu, 3, 300, 16, 64.0, true, Dtype::F32);
    table(&gpu, 3, 300, 64, 256.0, true, Dtype::F32);
    table(&gpu, 2, 512, 32, 1024.0, true, Dtype::F32);
    table(&gpu, 2, 200, 32, 128.0, false, Dtype::F32);

    let e = gpu.zeros(32, 4, Dtype::F32);
    let mut y = gpu.zeros(4, 15, Dtype::F32);
    assert!(
        relative_bucket_bias(&gpu.ctx(), e, 8, 32, 8.0, true, &mut y).is_err(),
        "max_distance 8 at max_exact 8 leaves the logarithm's ratio at zero"
    );
    let mut narrow = gpu.zeros(4, 14, Dtype::F32);
    assert!(
        relative_bucket_bias(&gpu.ctx(), e, 8, 32, 128.0, true, &mut narrow).is_err(),
        "a table 14 wide is not 2 · 8 − 1"
    );
}
