//! Argmax on the first Vulkan device, against a host reference: the first
//! index of the largest value, NaNs skipped and a row of NaNs landing 0,
//! bf16 and f32, rows narrower and wider than a workgroup, each landed at
//! its column of a deeper plane.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

fn first_max(row: &[f32]) -> i32 {
    let mut best: Option<(f32, usize)> = None;
    for (i, v) in row.iter().enumerate() {
        if !v.is_nan() && best.is_none_or(|(b, _)| *v > b) {
            best = Some((*v, i));
        }
    }
    best.map_or(0, |(_, i)| i as i32)
}

#[test]
fn argmax_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (rows, depth, column) = (4usize, 3usize, 1usize);
    for width in [8usize, 1000, 5000] {
        let mut v = noise(width as u64, rows * width);
        // Row 0 ties its maximum at two places, row 1 hides it behind NaNs,
        // row 3 is all NaN.
        v[3] = 9.0;
        v[width - 1] = 9.0;
        v[width] = f32::NAN;
        v[width + width / 2] = 7.5;
        v[width + width / 2 + 1] = f32::NAN;
        for x in &mut v[3 * width..] {
            *x = f32::NAN;
        }
        for dtype in [Dtype::Bf16, Dtype::F32] {
            let (xb, held) = if dtype == Dtype::F32 {
                (gpu.up_f32(&v), v.clone())
            } else {
                gpu.up(&v)
            };
            let want: Vec<i32> = held.chunks(width).map(first_max).collect();
            let xt = Tensor::new(xb, rows as u32, width as u32, dtype);
            let y = gpu.up_i32(&vec![-5; rows * depth]);
            let yt = Tensor::new(y, rows as u32, depth as u32, Dtype::I32);
            gpu.run(|c| kernels_slang::sampling::argmax::argmax(c, xt, column as u32, yt));

            let got: Vec<i32> = gpu
                .down_f32(y, rows * depth)
                .iter()
                .map(|v| v.to_bits() as i32)
                .collect();
            for r in 0..rows {
                assert_eq!(
                    got[r * depth + column],
                    want[r],
                    "argmax {dtype:?} {width}: row {r}"
                );
                assert_eq!(
                    got[r * depth],
                    -5,
                    "argmax {dtype:?} {width}: row {r} wrote another column"
                );
            }
            eprintln!("argmax {dtype:?} {width}: exact");
        }
    }
}
