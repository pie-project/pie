#![cfg(feature = "cuda")]

mod common;

use common::spatial::{
    Box3, avg_down_ref, pixel_shuffle_ref, pixel_unshuffle_ref, table, upsample_ref,
};
use common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{
    avg_down, patchify, pixel_shuffle, pixel_unshuffle, unpatchify, upsample_nearest,
};
use kernels_cuda::tensor::Tensor;

#[test]
fn the_voxel_reshapes_land_the_reference_ordering_every_case() {
    pixel_shuffle_lands_torchs_ordering_on_the_hand_computed_example();
    pixel_unshuffle_inverts_pixel_shuffle_and_both_match_the_reference();
    patchify_and_unpatchify_round_trip_over_the_token_box();
    upsample_nearest_repeats_every_frame();
    upsample_nearest_keeps_the_first_frame_single();
    avg_down_answers_the_avgdown3d_reference();
    avg_down_pads_a_short_chunks_time_axis_in_front();
}

fn pixel_shuffle_lands_torchs_ordering_on_the_hand_computed_example() {
    let x: Vec<u16> = [0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0]
        .map(to_bf16)
        .to_vec();
    let (grid, rows) = table(&[Box3::new(1, 1, 2)]);
    let (o_grid, rows_out) = table(&[Box3::new(1, 2, 4)]);
    assert_eq!((rows, rows_out), (2, 8));
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows_out as u32, 1, Dtype::Bf16);
    pixel_shuffle(
        &ctx,
        Tensor::new(x_at, rows as u32, 4, Dtype::Bf16),
        Tensor::new(grid_at, 1, 4, Dtype::I32),
        [1, 2, 2],
        0,
        &mut o,
        Tensor::new(o_grid_at, 1, 4, Dtype::I32),
    )
    .expect("the shuffle fires");
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, rows_out)
        .into_iter()
        .map(from_bf16)
        .collect();
    assert_eq!(got, [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
}

const BOXES: [Box3; 2] = [Box3::new(2, 4, 6), Box3::new(4, 2, 4)];

const R: [u32; 3] = [2, 2, 2];

fn pixel_unshuffle_inverts_pixel_shuffle_and_both_match_the_reference() {
    let c = 3usize;
    let vol = 8usize;
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x5a);
    let (x_raw, x) = lcg.row(rows * c * vol);
    let (want_up, up_boxes) = pixel_shuffle_ref(&x, &BOXES, c, [2, 2, 2]);
    let (o_grid, rows_up) = table(&up_boxes);
    let (want_back, back_boxes) = pixel_unshuffle_ref(&want_up, &up_boxes, c, [2, 2, 2]);
    assert_eq!(back_boxes, BOXES.to_vec());
    assert_eq!(want_back, x, "the host references invert each other");

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let up_at = gpu.zeros(rows_up * c * 2);
    let back_at = gpu.zeros(rows * c * vol * 2);
    let ctx = gpu.ctx();
    let x_t = Tensor::new(x_at, rows as u32, (c * vol) as u32, Dtype::Bf16);
    let g = Tensor::new(grid_at, 2, 4, Dtype::I32);
    let og = Tensor::new(o_grid_at, 2, 4, Dtype::I32);
    let mut up = Tensor::new(up_at, rows_up as u32, c as u32, Dtype::Bf16);
    pixel_shuffle(&ctx, x_t, g, R, 0, &mut up, og).expect("the shuffle fires");
    let mut back = Tensor::new(back_at, rows as u32, (c * vol) as u32, Dtype::Bf16);
    pixel_unshuffle(&ctx, up, og, R, &mut back, g).expect("the unshuffle fires");
    gpu.sync();
    let got_up: Vec<u16> = gpu.down(up_at, rows_up * c);
    let got_back: Vec<u16> = gpu.down(back_at, rows * c * vol);
    let want_up: Vec<u16> = want_up.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got_up, want_up, "the shuffle's ordering");
    assert_eq!(got_back, x_raw, "the round trip");
}

fn patchify_and_unpatchify_round_trip_over_the_token_box() {
    let c = 4usize;
    let p = [1u32, 2, 2];
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x9a7);
    let (x_raw, x) = lcg.row(rows * c);
    let (want_tokens, token_boxes) = pixel_unshuffle_ref(&x, &BOXES, c, [1, 2, 2]);
    let (t_grid, tokens) = table(&token_boxes);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let t_grid_at = gpu.up(&t_grid);
    let tok_at = gpu.zeros(tokens * c * 4 * 2);
    let back_at = gpu.zeros(rows * c * 2);
    let ctx = gpu.ctx();
    let g = Tensor::new(grid_at, 2, 4, Dtype::I32);
    let tg = Tensor::new(t_grid_at, 2, 4, Dtype::I32);
    let mut tok = Tensor::new(tok_at, tokens as u32, (c * 4) as u32, Dtype::Bf16);
    patchify(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        g,
        p,
        &mut tok,
        tg,
    )
    .expect("patchify fires");
    let mut back = Tensor::new(back_at, rows as u32, c as u32, Dtype::Bf16);
    unpatchify(&ctx, tok, tg, p, &mut back, g).expect("unpatchify fires");
    gpu.sync();
    let got_tok: Vec<u16> = gpu.down(tok_at, tokens * c * 4);
    let got_back: Vec<u16> = gpu.down(back_at, rows * c);
    let want_tok: Vec<u16> = want_tokens.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got_tok, want_tok, "`(c pt ph pw)` token channels");
    assert_eq!(got_back, x_raw, "the round trip");
}

fn upsample(keep_first: bool) {
    let c = 8usize;
    let f = [2usize, 2, 3];
    let (grid, rows) = table(&BOXES);
    let mut lcg = Lcg::seeded(0x0b);
    let (x_raw, x) = lcg.row(rows * c);
    let (want, out_boxes) = upsample_ref(&x, &BOXES, c, f, keep_first);
    let (o_grid, rows_out) = table(&out_boxes);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * c * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows_out as u32, c as u32, Dtype::Bf16);
    upsample_nearest(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        Tensor::new(grid_at, 2, 4, Dtype::I32),
        [2, 2, 3],
        keep_first,
        &mut o,
        Tensor::new(o_grid_at, 2, 4, Dtype::I32),
    )
    .expect("the upsample fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows_out * c);
    let want: Vec<u16> = want.iter().map(|&v| to_bf16(v)).collect();
    assert_eq!(got, want, "keep_first {keep_first}");
}

fn upsample_nearest_repeats_every_frame() {
    upsample(false);
}

fn upsample_nearest_keeps_the_first_frame_single() {
    upsample(true);
}

fn avg_down_case(boxes: &[Box3], c: usize, r: [u32; 3], group: u32) {
    let (grid, rows) = table(boxes);
    let ru = r.map(|v| v as usize);
    let mut lcg = Lcg::seeded(0xa7 ^ u64::from(r[0] * 31 + r[1] * 7 + r[2]) ^ (group as u64) << 8);
    let (x_raw, x) = lcg.row(rows * c);
    let (want, out_boxes) = avg_down_ref(&x, boxes, c, ru, group as usize);
    let (o_grid, rows_out) = table(&out_boxes);
    let c_out = c * ru[0] * ru[1] * ru[2] / group as usize;

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(rows_out * c_out * 2);
    let ctx = gpu.ctx();
    let lanes = boxes.len() as u32;
    let mut o = Tensor::new(o_at, rows_out as u32, c_out as u32, Dtype::Bf16);
    avg_down(
        &ctx,
        Tensor::new(x_at, rows as u32, c as u32, Dtype::Bf16),
        Tensor::new(grid_at, lanes, 4, Dtype::I32),
        r,
        group,
        &mut o,
        Tensor::new(o_grid_at, lanes, 4, Dtype::I32),
    )
    .unwrap_or_else(|why| panic!("{r:?} group {group}: {why}"));
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, rows_out * c_out)
        .into_iter()
        .map(from_bf16)
        .collect();
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        worst = worst.max((g - w).abs());
        assert!(
            (g - w).abs() <= 4e-3 + 8e-3 * w.abs(),
            "{r:?} group {group}: row {} channel {}: got {g}, want {w}",
            i / c_out,
            i % c_out
        );
    }
    eprintln!("avg_down {r:?} group {group}: {rows} -> {rows_out} rows, max |err| {worst:.5}");
}

fn avg_down_answers_the_avgdown3d_reference() {
    const EVEN: [Box3; 2] = [Box3::new(4, 4, 6), Box3::new(2, 2, 4)];
    const ODD: [Box3; 2] = [Box3::new(1, 4, 6), Box3::new(3, 2, 4)];
    avg_down_case(&EVEN, 5, [1, 2, 2], 4);
    avg_down_case(&EVEN, 5, [2, 2, 2], 4);
    avg_down_case(&ODD, 5, [2, 2, 2], 4);
    avg_down_case(&ODD, 3, [2, 2, 2], 8);
    avg_down_case(&ODD, 3, [1, 1, 1], 1);
}

fn avg_down_pads_a_short_chunks_time_axis_in_front() {
    let boxes = [Box3::new(1, 2, 2)];
    let (grid, rows) = table(&boxes);
    let x: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0].map(to_bf16).to_vec();
    let (o_grid, rows_out) = table(&[Box3::new(1, 1, 1)]);
    assert_eq!((rows, rows_out), (4, 1));
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let o_at = gpu.zeros(2 * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, 1, 2, Dtype::Bf16);
    avg_down(
        &ctx,
        Tensor::new(x_at, 4, 1, Dtype::Bf16),
        Tensor::new(grid_at, 1, 4, Dtype::I32),
        [2, 2, 2],
        4,
        &mut o,
        Tensor::new(o_grid_at, 1, 4, Dtype::I32),
    )
    .expect("the avg-down fires");
    gpu.sync();
    let got: Vec<f32> = gpu
        .down::<u16>(o_at, 2)
        .into_iter()
        .map(from_bf16)
        .collect();
    assert_eq!(got, [0.0, 2.5]);
}
