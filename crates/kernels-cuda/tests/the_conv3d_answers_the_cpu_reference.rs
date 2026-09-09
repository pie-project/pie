#![cfg(feature = "cuda")]

mod common;

use common::spatial::{Box3, conv3d_ref, near, out_boxes, table};
use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::spatial::{Conv3d, ConvPath, TimePad, conv_weight_taps_major, conv3d_on};
use kernels_cuda::tensor::Tensor;

struct Case {
    name: &'static str,
    boxes: Vec<Box3>,
    c_in: usize,
    c_out: usize,
    conv: Conv3d,
    cache: bool,
    bias: bool,
}

fn check(case: &Case, path: ConvPath) {
    let Case {
        name,
        boxes,
        c_in,
        c_out,
        conv,
        cache,
        bias,
    } = case;
    let (c_in, c_out) = (*c_in, *c_out);
    let taps = conv.taps() as usize;
    let (grid, rows) = table(boxes);
    let outs = out_boxes(conv, boxes);
    let (o_grid, rows_out) = table(&outs);

    let mut lcg = Lcg::seeded(0x5ea7 ^ rows as u64 ^ (c_out as u64) << 8);
    let (x_raw, x) = lcg.row(rows * c_in);
    let (w_raw, w) = lcg.row(c_out * c_in * taps);
    let bias_vals: Vec<f32> = (0..c_out).map(|_| lcg.unit()).collect();
    let cache_rows: usize = boxes.iter().map(|b| conv.pad[0] as usize * b.plane()).sum();
    let (cache_raw, cache_vals) = lcg.row(cache_rows * c_in);

    let want = conv3d_ref(
        &x,
        boxes,
        c_in,
        &w,
        c_out,
        bias.then_some(bias_vals.as_slice()),
        conv,
        cache.then_some(cache_vals.as_slice()),
    );

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let w_nat = gpu.up(&w_raw);
    let w_at = gpu.zeros(w_raw.len() * 2);
    let bias_at = gpu.up(&bias_vals);
    let cache_at = gpu.up(&cache_raw);
    let slack = 5usize;
    let o_at = gpu.up(&vec![0x7fc0u16; (rows_out + slack) * c_out]);
    let ctx = gpu.ctx();

    let mut w_t = Tensor::new(w_at, c_out as u32, (c_in * taps) as u32, Dtype::Bf16);
    conv_weight_taps_major(
        &ctx,
        Tensor::new(w_nat, c_out as u32, (c_in * taps) as u32, Dtype::Bf16),
        c_in as u32,
        taps as u32,
        &mut w_t,
    )
    .expect("the relabelling fires");
    let mut o = Tensor::new(o_at, (rows_out + slack) as u32, c_out as u32, Dtype::Bf16);
    conv3d_on(
        &ctx,
        path,
        Tensor::new(x_at, rows as u32, c_in as u32, Dtype::Bf16),
        Tensor::new(grid_at, boxes.len() as u32, 4, Dtype::I32),
        w_t,
        bias.then_some(Tensor::new(bias_at, c_out as u32, 1, Dtype::F32)),
        *conv,
        cache.then_some(Tensor::new(
            cache_at,
            cache_rows as u32,
            c_in as u32,
            Dtype::Bf16,
        )),
        &mut o,
        Tensor::new(o_grid_at, outs.len() as u32, 4, Dtype::I32),
    )
    .unwrap_or_else(|e| panic!("{name} on {path:?} fires: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, (rows_out + slack) * c_out);
    assert!(
        got[rows_out * c_out..].iter().all(|&v| v == 0),
        "{name} on {path:?}: a padded row past the lanes did not land zeros"
    );

    let scale = (taps * c_in) as f32;
    let worst = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| (from_bf16(g) - w).abs())
        .fold(0f32, f32::max);
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "{name} on {path:?}: worst |diff| {worst} at peak |want| {peak} over {} outputs",
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let g = from_bf16(g);
        assert!(
            near(g, w, 1.0 / 128.0, scale * 1e-4),
            "{name} on {path:?}: row {} channel {} is {g} against {w}",
            i / c_out,
            i % c_out
        );
    }
}

fn k3(stride: u32, pad: [u32; 3], causal_t: bool, time_pad: TimePad) -> Conv3d {
    Conv3d {
        k: [3, 3, 3],
        stride: [stride; 3],
        pad,        pad_back: pad,
        causal_t,
        time_pad,
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "symmetric stride 1, two lanes, C_in 16 -> C_out 24, bias",
            boxes: vec![Box3::new(3, 5, 6), Box3::new(2, 4, 7)],
            c_in: 16,
            c_out: 24,
            conv: k3(1, [1, 1, 1], false, TimePad::Zero),
            cache: false,
            bias: true,
        },
        Case {
            name: "symmetric stride 2, two lanes, C_in 8 -> C_out 16, no bias",
            boxes: vec![Box3::new(5, 7, 9), Box3::new(4, 6, 8)],
            c_in: 8,
            c_out: 16,
            conv: k3(2, [1, 1, 1], false, TimePad::Zero),
            cache: false,
            bias: false,
        },
        Case {
            name: "symmetric replicating the first and last frames",
            boxes: vec![Box3::new(3, 5, 5), Box3::new(1, 4, 6)],
            c_in: 16,
            c_out: 16,
            conv: k3(1, [1, 1, 1], false, TimePad::Replicate),
            cache: false,
            bias: true,
        },
        Case {
            name: "causal zero-padded, no cache",
            boxes: vec![Box3::new(4, 5, 5), Box3::new(3, 4, 6)],
            c_in: 16,
            c_out: 16,
            conv: k3(1, [2, 1, 1], true, TimePad::Zero),
            cache: false,
            bias: true,
        },
        Case {
            name: "causal with the previous tile's two frames",
            boxes: vec![Box3::new(4, 5, 5), Box3::new(3, 4, 6)],
            c_in: 16,
            c_out: 16,
            conv: k3(1, [2, 1, 1], true, TimePad::Zero),
            cache: true,
            bias: true,
        },
        Case {
            name: "causal replicating the first frame",
            boxes: vec![Box3::new(4, 5, 5), Box3::new(3, 4, 6)],
            c_in: 16,
            c_out: 8,
            conv: k3(1, [2, 1, 1], true, TimePad::Replicate),
            cache: false,
            bias: true,
        },
        Case {
            name: "causal time downsample: k (3,1,1), stride (2,1,1), one cached frame",
            boxes: vec![Box3::new(4, 3, 4), Box3::new(8, 2, 3)],
            c_in: 16,
            c_out: 16,
            conv: Conv3d {
                k: [3, 1, 1],
                stride: [2, 1, 1],
                pad: [1, 0, 0],
                pad_back: [1, 0, 0],
                causal_t: true,
                time_pad: TimePad::Zero,
            },
            cache: true,
            bias: false,
        },
        Case {
            name: "causal stride 2 everywhere, one lane",
            boxes: vec![Box3::new(5, 6, 6)],
            c_in: 8,
            c_out: 24,
            conv: k3(2, [2, 1, 1], true, TimePad::Replicate),
            cache: false,
            bias: true,
        },
        Case {
            name: "conv2d (kt = 1) over three-channel images, C_out 5",
            boxes: vec![Box3::new(1, 6, 7), Box3::new(1, 5, 5)],
            c_in: 3,
            c_out: 5,
            conv: Conv3d::conv2d([3, 3], [1, 1], [1, 1]),
            cache: false,
            bias: true,
        },
        Case {
            name: "conv2d stride 2 with a channel count off the 8-wide vector",
            boxes: vec![Box3::new(1, 9, 8)],
            c_in: 12,
            c_out: 20,
            conv: Conv3d::conv2d([3, 3], [2, 2], [1, 1]),
            cache: false,
            bias: false,
        },
    ]
}

#[test]
fn the_conv3d_answers_the_cpu_reference_every_case() {
    the_direct_kernel_answers_the_reference();
    the_tensor_core_kernel_answers_the_reference();
    the_tensor_core_kernel_refuses_a_channel_count_it_cannot_vectorise();
}

fn the_direct_kernel_answers_the_reference() {
    for case in cases() {
        check(&case, ConvPath::Direct);
    }
}

fn the_tensor_core_kernel_answers_the_reference() {
    for case in cases().iter().filter(|c| c.c_in % 8 == 0) {
        check(case, ConvPath::TensorCore);
    }
}

fn the_tensor_core_kernel_refuses_a_channel_count_it_cannot_vectorise() {
    let case = cases()
        .into_iter()
        .find(|c| c.c_in == 3)
        .expect("the three-channel case");
    let (grid, rows) = table(&case.boxes);
    let outs = out_boxes(&case.conv, &case.boxes);
    let (o_grid, rows_out) = table(&outs);
    let mut gpu = Gpu::open();
    let x_at = gpu.zeros(rows * case.c_in * 2);
    let grid_at = gpu.up(&grid);
    let o_grid_at = gpu.up(&o_grid);
    let w_at = gpu.zeros(case.c_out * case.c_in * 9 * 2);
    let o_at = gpu.zeros(rows_out * case.c_out * 2);
    let ctx = gpu.ctx();
    let mut o = Tensor::new(o_at, rows_out as u32, case.c_out as u32, Dtype::Bf16);
    let refused = conv3d_on(
        &ctx,
        ConvPath::TensorCore,
        Tensor::new(x_at, rows as u32, case.c_in as u32, Dtype::Bf16),
        Tensor::new(grid_at, 2, 4, Dtype::I32),
        Tensor::new(w_at, case.c_out as u32, (case.c_in * 9) as u32, Dtype::Bf16),
        None,
        case.conv,
        None,
        &mut o,
        Tensor::new(o_grid_at, 2, 4, Dtype::I32),
    );
    assert!(refused.is_err(), "three channels do not vectorise");
}
