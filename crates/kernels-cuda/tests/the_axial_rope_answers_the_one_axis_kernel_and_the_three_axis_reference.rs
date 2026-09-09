#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::elemwise::{RopeForm, rope, rope_axes};
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 12;
const HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const WIDTH: usize = HEADS * HEAD_DIM;

fn reference(
    x: &[f32],
    positions: &[f32],
    dims: &[u32],
    thetas: &[f32],
    form: RopeForm,
    rotary_dim: usize,
) -> Vec<f32> {
    let mut o = x.to_vec();
    let axes = dims.len();
    let angles = rotary_dim / 2;
    for row in 0..ROWS {
        for head in 0..HEADS {
            let base = row * WIDTH + head * HEAD_DIM;
            let (mut first_angle, mut first_channel) = (0usize, 0usize);
            for axis in 0..axes {
                let block = dims[axis] as usize;
                for within in 0..block / 2 {
                    let freq = thetas[axis].powf(-2.0 * within as f32 / block as f32);
                    let angle = positions[row * axes + axis] * freq;
                    let (sin, cos) = (angle.sin(), angle.cos());
                    let (lo, hi) = match form {
                        RopeForm::Interleaved => {
                            (first_channel + 2 * within, first_channel + 2 * within + 1)
                        }
                        RopeForm::Neox => {
                            let p = first_angle + within;
                            (p, p + angles)
                        }
                        RopeForm::Split => {
                            (first_channel + within, first_channel + block / 2 + within)
                        }
                        RopeForm::SplitLadder => unreachable!(
                            "this reference walks per-head axis blocks at \
                             negative exponents; SplitLadder is neither"
                        ),
                    };
                    let (a, b) = (x[base + lo], x[base + hi]);
                    o[base + lo] = a.mul_add(cos, -(b * sin));
                    o[base + hi] = b.mul_add(cos, a * sin);
                }
                first_angle += block / 2;
                first_channel += block;
            }
        }
    }
    o
}

#[test]
fn the_axial_rope_answers_the_one_axis_kernel_and_the_three_axis_reference_every_case() {
    one_axis_is_the_scalar_kernels_own_rotation();
    three_axes_answer_the_reference_in_every_form();
}

fn one_axis_is_the_scalar_kernels_own_rotation() {
    let mut lcg = Lcg::seeded(0xa1e5);
    let (x_raw, _) = lcg.row(ROWS * WIDTH);
    let pos: Vec<i32> = (0..ROWS).map(|r| (r * 7 % 41) as i32).collect();
    #[allow(clippy::cast_precision_loss)]
    let pos_f: Vec<f32> = pos.iter().map(|&p| p as f32).collect();

    for (form, interleaved) in [(RopeForm::Neox, false), (RopeForm::Interleaved, true)] {
        let mut gpu = Gpu::open();
        let scalar_at = gpu.up(&x_raw);
        let empty = gpu.zeros(2);
        let axial_in = gpu.up(&x_raw);
        let axial_out = gpu.zeros(ROWS * WIDTH * 2);
        let pos_at = gpu.up(&pos);
        let pos_f_at = gpu.up(&pos_f);
        let ctx = gpu.ctx();
        let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);

        rope::full(
            &ctx,
            &mut rect(scalar_at),
            &mut Tensor::new(empty, ROWS as u32, 0, Dtype::Bf16),
            Tensor::new(pos_at, ROWS as u32, 1, Dtype::I32),
            HEAD_DIM as u32,
            10_000.0,
            interleaved,
        )
        .expect("the scalar rotation fires");
        rope_axes(
            &ctx,
            rect(axial_in),
            Tensor::new(pos_f_at, ROWS as u32, 1, Dtype::F32),
            [HEAD_DIM as u32, 0, 0, 0],
            [10_000.0, 0.0, 0.0, 0.0],
            form,
            HEAD_DIM as u32,
            HEAD_DIM as u32,
            &mut rect(axial_out),
        )
        .expect("the axial rotation fires");
        gpu.sync();

        let want: Vec<u16> = gpu.down(scalar_at, ROWS * WIDTH);
        let got: Vec<u16> = gpu.down(axial_out, ROWS * WIDTH);
        assert_eq!(
            got, want,
            "{form:?}: the one-axis rotation is not the scalar kernel's"
        );
    }
}

fn three_axes_answer_the_reference_in_every_form() {
    const ROTARY: usize = 48;

    let dims = [16u32, 20, 12];
    let thetas = [256.0f32, 10_000.0, 2000.0];
    let mut lcg = Lcg::seeded(0x3a1e5);
    let (x_raw, x) = lcg.row(ROWS * WIDTH);
    let positions: Vec<f32> = (0..ROWS * 3)
        .map(|i| (i as f32) * 0.5 + if i % 3 == 0 { 0.25 } else { 0.0 })
        .collect();

    for form in [RopeForm::Interleaved, RopeForm::Neox, RopeForm::Split] {
        let mut gpu = Gpu::open();
        let x_at = gpu.up(&x_raw);
        let pos_at = gpu.up(&positions);
        let o_at = gpu.zeros(ROWS * WIDTH * 2);
        let ctx = gpu.ctx();
        let rect = |at: u64| Tensor::new(at, ROWS as u32, WIDTH as u32, Dtype::Bf16);

        rope_axes(
            &ctx,
            rect(x_at),
            Tensor::new(pos_at, ROWS as u32, 3, Dtype::F32),
            [dims[0], dims[1], dims[2], 0],
            [thetas[0], thetas[1], thetas[2], 0.0],
            form,
            ROTARY as u32,
            HEAD_DIM as u32,
            &mut rect(o_at),
        )
        .expect("the rotation fires");
        gpu.sync();

        let got: Vec<u16> = gpu.down(o_at, ROWS * WIDTH);
        let want = reference(&x, &positions, &dims, &thetas, form, ROTARY);
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let (g, w) = (from_bf16(g), w);
            assert!(
                (g - w).abs() <= 1e-3 + w.abs() / 128.0,
                "{form:?} at {i}: {g} against {w}"
            );
        }
        for row in 0..ROWS {
            for head in 0..HEADS {
                for col in ROTARY..HEAD_DIM {
                    let at = row * WIDTH + head * HEAD_DIM + col;
                    assert_eq!(got[at], to_bf16(x[at]), "{form:?}: the tail moved at {at}");
                }
            }
        }
    }
}
