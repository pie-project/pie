#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::elemwise::sinusoid;
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 2e-6;

const TOLERANCE_AT_ANGLE: f32 = 1e-5;

fn reference(t: f32, dim: usize, max_period: f32, flip_sin_cos: bool, scale: f32) -> Vec<f32> {
    let half = dim / 2;
    let angles: Vec<f32> = (0..half)
        .map(|i| {
            let freq = (-max_period.ln() * i as f32 / half as f32).exp();
            scale * (t * freq)
        })
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

fn run(t: &[f32], dim: usize, max_period: f32, flip_sin_cos: bool, scale: f32) -> Vec<f32> {
    let mut gpu = Gpu::open();
    let t_at = gpu.up(t);
    let o_at = gpu.zeros(t.len() * dim * 4);
    let ctx = gpu.ctx();
    sinusoid(
        &ctx,
        Tensor::new(t_at, t.len() as u32, 1, Dtype::F32),
        dim as u32,
        max_period,
        flip_sin_cos,
        scale,
        &mut Tensor::new(o_at, t.len() as u32, dim as u32, Dtype::F32),
    )
    .expect("the embedding fires");
    gpu.sync();
    gpu.down(o_at, t.len() * dim)
}

fn agrees(got: &[f32], want: &[f32], what: &str) {
    within(got, want, TOLERANCE, what);
}

fn within(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tolerance, "{what} at {i}: {g} against {w}");
    }
}

#[test]
fn the_sinusoid_embedding_matches_the_diffusers_formula_every_case() {
    the_rows_are_the_ones_the_reference_prints();
    an_odd_width_pads_its_last_column_with_zero();
    flipping_puts_the_cosines_first_and_the_scale_multiplies_the_angle();
    a_whole_rectangle_answers_the_host_reference();
}

fn the_rows_are_the_ones_the_reference_prints() {
    let got = run(&[1.0, 500.0], 8, 10_000.0, false, 1.0);
    agrees(
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
        "t = 1",
    );
    agrees(
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
        "t = 500",
    );
}

fn an_odd_width_pads_its_last_column_with_zero() {
    let got = run(&[0.5], 5, 10_000.0, false, 1.0);
    agrees(
        &got,
        &[0.479_425_54, 0.004_999_979, 0.877_582_55, 0.999_987_5, 0.0],
        "an odd width",
    );
}

fn flipping_puts_the_cosines_first_and_the_scale_multiplies_the_angle() {
    let got = run(&[3.0], 6, 10_000.0, true, 2.0);
    agrees(
        &got,
        &[
            0.960_170_3,
            0.961_470_2,
            0.999_916_45,
            -0.279_415_5,
            0.274_909_27,
            0.012_926_248,
        ],
        "flipped and scaled",
    );
}

fn a_whole_rectangle_answers_the_host_reference() {
    const ROWS: usize = 19;
    const DIM: usize = 64;

    let mut lcg = Lcg::seeded(0x511e);
    let t: Vec<f32> = (0..ROWS).map(|_| lcg.unit() * 8.0).collect();
    let got = run(&t, DIM, 256.0, true, 1.25);
    for (row, &tv) in t.iter().enumerate() {
        within(
            &got[row * DIM..(row + 1) * DIM],
            &reference(tv, DIM, 256.0, true, 1.25),
            TOLERANCE_AT_ANGLE,
            &format!("row {row}"),
        );
    }
}
