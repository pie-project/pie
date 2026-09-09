#![cfg(feature = "cuda")]

mod common;

use common::spatial::{Box3, near, out_boxes, table};
use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{Conv3d, ConvPath, TimePad, conv3d, conv3d_on};
use kernels_cuda::tensor::Tensor;

fn run(c_in: usize, c_out: usize, boxes: &[Box3], conv: Conv3d) {
    let taps = conv.taps() as usize;
    let (grid, rows) = table(boxes);
    let outs = out_boxes(&conv, boxes);
    let (o_grid, rows_out) = table(&outs);
    let cache_rows: usize = boxes.iter().map(|b| conv.pad[0] as usize * b.plane()).sum();
    let mut lcg = Lcg::seeded(0xb0a);
    let (x_raw, _) = lcg.row(rows * c_in);
    let (w_raw, _) = lcg.row(c_out * c_in * taps);
    let (cache_raw, _) = lcg.row(cache_rows * c_in);
    let bias: Vec<f32> = (0..c_out).map(|_| lcg.unit()).collect();

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let cache_at = gpu.up(&cache_raw);
    let bias_at = gpu.up(&bias);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let direct = gpu.zeros(rows_out * c_out * 2);
    let mma = gpu.zeros(rows_out * c_out * 2);
    let ctx = gpu.ctx();
    for (path, at) in [(ConvPath::Direct, direct), (ConvPath::TensorCore, mma)] {
        let mut o = Tensor::new(at, rows_out as u32, c_out as u32, Dtype::Bf16);
        conv3d_on(
            &ctx,
            path,
            Tensor::new(x_at, rows as u32, c_in as u32, Dtype::Bf16),
            Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32),
            Tensor::new(w_at, c_out as u32, (c_in * taps) as u32, Dtype::Bf16),
            Some(Tensor::new(bias_at, c_out as u32, 1, Dtype::F32)),
            conv,
            (conv.causal_t && cache_rows > 0).then_some(Tensor::new(
                cache_at,
                cache_rows as u32,
                c_in as u32,
                Dtype::Bf16,
            )),
            &mut o,
            Tensor::new(o_grid_at, boxes.len() as u32, 4, Dtype::I32),
        )
        .unwrap_or_else(|e| panic!("{path:?} fires: {e}"));
    }
    gpu.sync();
    let a: Vec<u16> = gpu.down(direct, rows_out * c_out);
    let b: Vec<u16> = gpu.down(mma, rows_out * c_out);
    let scale = (taps * c_in) as f32;
    let mut differing = 0usize;
    for (i, (&da, &db)) in a.iter().zip(&b).enumerate() {
        let (fa, fb) = (from_bf16(da), from_bf16(db));
        assert!(
            near(fb, fa, 1.0 / 128.0, scale * 1e-4),
            "row {} channel {}: tensor cores {fb} against fma {fa}",
            i / c_out,
            i % c_out
        );
        differing += usize::from(da != db);
    }
    assert!(
        differing * 20 < a.len(),
        "{differing} of {} outputs differ by a rounding step — more than the odd tie",
        a.len()
    );
}

#[test]
fn the_tensor_core_conv_lands_what_the_direct_kernel_lands_every_case() {
    the_two_kernels_agree_over_a_causal_cached_block();
    the_two_kernels_agree_off_the_tile_boundaries();
    the_unnamed_entry_takes_the_tensor_cores_on_a_vectorisable_shape();
}

fn the_two_kernels_agree_over_a_causal_cached_block() {
    run(
        64,
        64,
        &[Box3::new(3, 9, 17), Box3::new(2, 7, 13)],
        Conv3d {
            k: [3, 3, 3],
            stride: [1, 1, 1],
            pad: [2, 1, 1],
            pad_back: [2, 1, 1],
            causal_t: true,
            time_pad: TimePad::Zero,
        },
    );
}

fn the_two_kernels_agree_off_the_tile_boundaries() {
    run(
        40,
        72,
        &[Box3::new(2, 11, 19), Box3::new(3, 5, 7)],
        Conv3d {
            k: [3, 3, 3],
            stride: [2, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
            time_pad: TimePad::Zero,
        },
    );
}

fn the_unnamed_entry_takes_the_tensor_cores_on_a_vectorisable_shape() {
    let (c_in, c_out) = (32usize, 40usize);
    let boxes = [Box3::new(2, 5, 9)];
    let conv = Conv3d::conv2d([3, 3], [1, 1], [1, 1]);
    let taps = conv.taps() as usize;
    let (grid, rows) = table(&boxes);
    let outs = out_boxes(&conv, &boxes);
    let (o_grid, rows_out) = table(&outs);
    let mut lcg = Lcg::seeded(0xa07);
    let (x_raw, _) = lcg.row(rows * c_in);
    let (w_raw, _) = lcg.row(c_out * c_in * taps);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let w_at = gpu.up(&w_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let auto_at = gpu.zeros(rows_out * c_out * 2);
    let mma_at = gpu.zeros(rows_out * c_out * 2);
    let ctx = gpu.ctx();
    let x = Tensor::new(x_at, rows as u32, c_in as u32, Dtype::Bf16);
    let w = Tensor::new(w_at, c_out as u32, (c_in * taps) as u32, Dtype::Bf16);
    let g = Tensor::new(grid_at, 1, 4, Dtype::I32);
    let og = Tensor::new(o_grid_at, 1, 4, Dtype::I32);
    let mut auto = Tensor::new(auto_at, rows_out as u32, c_out as u32, Dtype::Bf16);
    conv3d(&ctx, x, g, w, None, conv, None, &mut auto, og).expect("the entry fires");
    let mut mma = Tensor::new(mma_at, rows_out as u32, c_out as u32, Dtype::Bf16);
    conv3d_on(
        &ctx,
        ConvPath::TensorCore,
        x,
        g,
        w,
        None,
        conv,
        None,
        &mut mma,
        og,
    )
    .expect("the tensor-core kernel fires");
    gpu.sync();
    let a: Vec<u16> = gpu.down(auto_at, rows_out * c_out);
    let b: Vec<u16> = gpu.down(mma_at, rows_out * c_out);
    assert_eq!(a, b, "the same kernel, bit for bit");
}
