#![cfg(feature = "cuda")]

use crate::common::spatial::{Box3, pixel_shuffle_ref, table};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{GridRule, derive_grid, pixel_shuffle};
use kernels_cuda::tensor::Tensor;

const BOXES: [Box3; 2] = [Box3::new(3, 2, 3), Box3::new(2, 3, 2)];

const R: [u32; 3] = [2, 2, 2];

const TRIM: u32 = 1;

#[test]
fn a_trimmed_pixel_shuffle_drops_its_anchor_frames() {
    let c_out = 3usize;
    let vol = 8usize;
    let (grid, rows) = table(&BOXES);

    let mut lcg = Lcg::seeded(0x7a17);
    let (x_raw, x) = lcg.row(rows * c_out * vol);
    let (plain, plain_boxes) = pixel_shuffle_ref(&x, &BOXES, c_out, [2, 2, 2]);
    let trimmed_boxes: Vec<Box3> = plain_boxes
        .iter()
        .map(|b| Box3::new(b.t - TRIM as usize, b.h, b.w))
        .collect();
    let (want_grid, rows_out) = table(&trimmed_boxes);
    let mut want = vec![0f32; rows_out * c_out];
    let (mut src, mut dst) = (0usize, 0usize);
    for (l, b) in plain_boxes.iter().enumerate() {
        let keep = trimmed_boxes[l];
        let skip = TRIM as usize * b.plane();
        for row in 0..keep.voxels() {
            for c in 0..c_out {
                want[(dst + row) * c_out + c] = plain[(src + skip + row) * c_out + c];
            }
        }
        src += b.voxels();
        dst += keep.voxels();
    }

    let allocated: usize = plain_boxes.iter().map(|b| b.voxels()).sum();
    assert!(rows_out < allocated, "the trim shrinks the live rows");

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.zeros(BOXES.len() * 4 * 4);
    let o_at = gpu.zeros(allocated * c_out * 2);
    let ctx = gpu.ctx();

    let g = Tensor::new(grid_at, BOXES.len() as u32, 4, Dtype::I32);
    let mut og = Tensor::new(o_grid_at, BOXES.len() as u32, 4, Dtype::I32);
    derive_grid(&ctx, g, GridRule::Shuffle { r: R, trim_t: TRIM }, &mut og)
        .expect("the rule fires");

    let mut o = Tensor::new(o_at, allocated as u32, c_out as u32, Dtype::Bf16);
    pixel_shuffle(
        &ctx,
        Tensor::new(x_at, rows as u32, (c_out * vol) as u32, Dtype::Bf16),
        g,
        R,
        TRIM,
        &mut o,
        og,
    )
    .expect("the shuffle fires");
    gpu.sync();

    let got_grid: Vec<i32> = gpu.down(o_grid_at, BOXES.len() * 4);
    assert_eq!(
        got_grid, want_grid,
        "the device rule's boxes and offsets are the host's"
    );
    assert_eq!(got_grid, vec![5, 4, 6, 0, 3, 6, 4, 120]);

    let got: Vec<u16> = gpu.down(o_at, allocated * c_out);
    let want_bits: Vec<u16> = want.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(
        &got[..rows_out * c_out],
        &want_bits[..],
        "every live row is the shuffled voxel {TRIM} frame(s) on"
    );
    assert!(
        got[rows_out * c_out..].iter().all(|&x| x == 0),
        "the rows past the trimmed claim stay zero"
    );
}
