//! The voxel-grid convolutions and reshapes against host references:
//! conv3d on both kernels (f32 direct and the matrix core) over every
//! padding, stride and cache case; the pixel (un)shuffle, patchify,
//! upsampling and avg-down orderings; a trimmed shuffle over the grid the
//! device derives; and the causal frame cache's gather and store.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/spatial.rs"]
mod spatial;

use common::{Gpu, bf16, bf16s, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::conv::{
    Conv3d, ConvPath, TimePad, avg_down, cache_gather, cache_store, conv_weight_taps_major,
    conv3d_on, patchify, pixel_shuffle, pixel_unshuffle, unpatchify, upsample_nearest,
};
use kernels_rocm::elemwise::posenc::{GridRule, derive_grid};
use spatial::{
    Box3, avg_down_ref, conv3d_ref, near, out_boxes, pixel_shuffle_ref, pixel_unshuffle_ref, table,
    upsample_ref,
};

fn grid_of(gpu: &Gpu, boxes: &[Box3]) -> (Tensor, usize) {
    let (grid, rows) = table(boxes);
    (gpu.ints_shaped(&grid, boxes.len() as u32, 4), rows)
}

fn bits(gpu: &Gpu, t: &Tensor) -> Vec<u16> {
    gpu.read_as::<u16>(t)[..t.elements() as usize].to_vec()
}

struct Case {
    name: &'static str,
    boxes: Vec<Box3>,
    c_in: usize,
    c_out: usize,
    conv: Conv3d,
    cache: bool,
    bias: bool,
}

fn check(gpu: &Gpu, case: &Case, path: ConvPath) {
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
    let (grid, rows) = grid_of(gpu, boxes);
    let outs = out_boxes(conv, boxes);
    let (o_grid, rows_out) = grid_of(gpu, &outs);

    let seed = 0x5ea7 ^ rows as u64 ^ (c_out as u64) << 8;
    let x = bf16s(rows * c_in, seed, 1.0);
    let w = bf16s(c_out * c_in * taps, seed + 1, 1.0);
    let bias_vals = values(c_out, seed + 2, 1.0);
    let cache_rows: usize = boxes.iter().map(|b| conv.pad[0] as usize * b.plane()).sum();
    let cache_vals = bf16s(cache_rows * c_in, seed + 3, 1.0);
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

    let ctx = gpu.ctx();
    let k = (c_in * taps) as u32;
    let w_nat = gpu.tensor(&w, c_out as u32, k, Dtype::Bf16);
    let mut w_t = gpu.zeros(c_out as u32, k, Dtype::Bf16);
    conv_weight_taps_major(&ctx, w_nat, c_in as u32, taps as u32, &mut w_t)
        .expect("the relabelling fires");
    // Rows past the last lane start as NaN and must land zeros.
    let slack = 5usize;
    let mut o = gpu.tensor(
        &vec![f32::NAN; (rows_out + slack) * c_out],
        (rows_out + slack) as u32,
        c_out as u32,
        Dtype::Bf16,
    );
    conv3d_on(
        &ctx,
        path,
        gpu.tensor(&x, rows as u32, c_in as u32, Dtype::Bf16),
        grid,
        w_t,
        bias.then(|| gpu.tensor(&bias_vals, c_out as u32, 1, Dtype::F32)),
        *conv,
        cache.then(|| gpu.tensor(&cache_vals, cache_rows as u32, c_in as u32, Dtype::Bf16)),
        &mut o,
        o_grid,
        [0, 0],
    )
    .unwrap_or_else(|e| panic!("{name} on {path:?} fires: {e}"));
    let got = gpu.read(&o);
    assert!(
        got[rows_out * c_out..].iter().all(|&v| v == 0.0),
        "{name} on {path:?}: a padded row past the lanes did not land zeros"
    );
    let scale = (taps * c_in) as f32;
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
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
        pad,
        pad_back: pad,
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
        Case {
            name: "wide: 96 channels in, 80 out, past one 64x64 tile each way",
            boxes: vec![Box3::new(2, 9, 10), Box3::new(1, 7, 6)],
            c_in: 96,
            c_out: 80,
            conv: k3(1, [1, 1, 1], false, TimePad::Zero),
            cache: false,
            bias: true,
        },
    ]
}

/// Every case on the direct kernel, and those whose channels fill the
/// 8-wide vector on the matrix-core one; the matrix core refuses a channel
/// count it cannot vectorise.
#[test]
fn the_conv3d_answers_the_cpu_reference_every_case() {
    let Some(gpu) = Gpu::open() else { return };
    for case in cases() {
        check(&gpu, &case, ConvPath::Direct);
        if case.c_in % 8 == 0 {
            check(&gpu, &case, ConvPath::TensorCore);
        }
    }

    let case = cases()
        .into_iter()
        .find(|c| c.c_in == 3)
        .expect("the three-channel case");
    let (grid, rows) = grid_of(&gpu, &case.boxes);
    let outs = out_boxes(&case.conv, &case.boxes);
    let (o_grid, rows_out) = grid_of(&gpu, &outs);
    let mut o = gpu.zeros(rows_out as u32, case.c_out as u32, Dtype::Bf16);
    let refused = conv3d_on(
        &gpu.ctx(),
        ConvPath::TensorCore,
        gpu.zeros(rows as u32, case.c_in as u32, Dtype::Bf16),
        grid,
        gpu.zeros(case.c_out as u32, (case.c_in * 9) as u32, Dtype::Bf16),
        None,
        case.conv,
        None,
        &mut o,
        o_grid,
        [0, 0],
    );
    assert!(refused.is_err(), "three channels do not vectorise");
}

const BOXES: [Box3; 2] = [Box3::new(2, 4, 6), Box3::new(4, 2, 4)];

/// The hand-computed `[1, 2, 2]` shuffle torch lands, a shuffle and its
/// inverse against the references and each other, patchify's token
/// channels and its round trip, nearest upsampling either way, and avg-down
/// (with a short chunk padded in front).
#[test]
fn the_voxel_reshapes_land_the_reference_ordering_every_case() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();

    // Hand computed: two voxels of four channels into a 2x4 plane.
    let (g, _) = grid_of(&gpu, &[Box3::new(1, 1, 2)]);
    let (og, rows_out) = grid_of(&gpu, &[Box3::new(1, 2, 4)]);
    let mut o = gpu.zeros(rows_out as u32, 1, Dtype::Bf16);
    let x = gpu.tensor(&[0.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0], 2, 4, Dtype::Bf16);
    pixel_shuffle(&ctx, x, g, [1, 2, 2], 0, &mut o, og, [0, 0]).expect("the shuffle fires");
    assert_eq!(gpu.read(&o), [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);

    // A shuffle and its inverse.
    let (c, vol, r) = (3usize, 8usize, [2u32; 3]);
    let (g, rows) = grid_of(&gpu, &BOXES);
    let x = bf16s(rows * c * vol, 0x5a, 1.0);
    let (want_up, up_boxes) = pixel_shuffle_ref(&x, &BOXES, c, [2, 2, 2]);
    let (og, rows_up) = grid_of(&gpu, &up_boxes);
    let x_t = gpu.tensor(&x, rows as u32, (c * vol) as u32, Dtype::Bf16);
    let mut up = gpu.zeros(rows_up as u32, c as u32, Dtype::Bf16);
    pixel_shuffle(&ctx, x_t, g, r, 0, &mut up, og, [0, 0]).expect("the shuffle fires");
    let mut back = gpu.zeros(rows as u32, (c * vol) as u32, Dtype::Bf16);
    pixel_unshuffle(&ctx, up, og, r, &mut back, g, [0, 0]).expect("the unshuffle fires");
    assert_eq!(gpu.read(&up), want_up, "the shuffle's ordering");
    assert_eq!(bits(&gpu, &back), bits(&gpu, &x_t), "the round trip");

    // Patchify's `(c pt ph pw)` token channels, and back.
    let (c, p) = (4usize, [1u32, 2, 2]);
    let x = bf16s(rows * c, 0x9a7, 1.0);
    let (want_tok, token_boxes) = pixel_unshuffle_ref(&x, &BOXES, c, [1, 2, 2]);
    let (tg, tokens) = grid_of(&gpu, &token_boxes);
    let x_t = gpu.tensor(&x, rows as u32, c as u32, Dtype::Bf16);
    let mut tok = gpu.zeros(tokens as u32, (c * 4) as u32, Dtype::Bf16);
    patchify(&ctx, x_t, g, p, &mut tok, tg, [0, 0]).expect("patchify fires");
    let mut back = gpu.zeros(rows as u32, c as u32, Dtype::Bf16);
    unpatchify(&ctx, tok, tg, p, &mut back, g, [0, 0]).expect("unpatchify fires");
    assert_eq!(gpu.read(&tok), want_tok, "the token channels");
    assert_eq!(bits(&gpu, &back), bits(&gpu, &x_t), "the round trip");

    // Nearest upsampling repeats every frame, or keeps the first single.
    for keep_first in [false, true] {
        let (c, f) = (8usize, [2usize, 2, 3]);
        let x = bf16s(rows * c, 0x0b, 1.0);
        let (want, out_boxes) = upsample_ref(&x, &BOXES, c, f, keep_first);
        let (og, rows_out) = grid_of(&gpu, &out_boxes);
        let mut o = gpu.zeros(rows_out as u32, c as u32, Dtype::Bf16);
        upsample_nearest(
            &ctx,
            gpu.tensor(&x, rows as u32, c as u32, Dtype::Bf16),
            g,
            [2, 2, 3],
            keep_first,
            &mut o,
            og,
            [0, 0],
        )
        .expect("the upsample fires");
        assert_eq!(gpu.read(&o), want, "keep_first {keep_first}");
    }

    // Avg-down over a whole time block, and over a chunk short of one.
    for (boxes, c, r, group) in [
        (BOXES.to_vec(), 4usize, [2u32, 2, 2], 4u32),
        (
            vec![Box3::new(3, 4, 4), Box3::new(1, 2, 2)],
            6,
            [2, 2, 2],
            3,
        ),
    ] {
        let (g, rows) = grid_of(&gpu, &boxes);
        let x = bf16s(rows * c, 0xa7d ^ c as u64, 1.0);
        let ru = r.map(|v| v as usize);
        let (want, out_boxes) = avg_down_ref(&x, &boxes, c, ru, group as usize);
        let (og, rows_out) = grid_of(&gpu, &out_boxes);
        let c_out = c * ru.iter().product::<usize>() / group as usize;
        let mut o = gpu.zeros(rows_out as u32, c_out as u32, Dtype::Bf16);
        avg_down(
            &ctx,
            gpu.tensor(&x, rows as u32, c as u32, Dtype::Bf16),
            g,
            r,
            group,
            &mut o,
            og,
            [0, 0],
        )
        .expect("avg-down fires");
        for (i, (&got, &w)) in gpu.read(&o).iter().zip(&want).enumerate() {
            assert!(
                near(got, w, 1.0 / 128.0, 1e-3),
                "avg-down group {group}: element {i} is {got} against {w}"
            );
        }
    }
}

/// A shuffle trimmed of its first frame lands on the grid the device
/// derives for it, every live row the shuffled voxel a frame on.
#[test]
fn a_trimmed_pixel_shuffle_drops_its_anchor_frames() {
    let Some(gpu) = Gpu::open() else { return };
    const BOXES: [Box3; 2] = [Box3::new(3, 2, 3), Box3::new(2, 3, 2)];
    const R: [u32; 3] = [2, 2, 2];
    const TRIM: u32 = 1;
    let (c_out, vol) = (3usize, 8usize);
    let (g, rows) = grid_of(&gpu, &BOXES);
    let x = bf16s(rows * c_out * vol, 0x7a17, 1.0);
    let (plain, plain_boxes) = pixel_shuffle_ref(&x, &BOXES, c_out, [2, 2, 2]);
    let trimmed: Vec<Box3> = plain_boxes
        .iter()
        .map(|b| Box3::new(b.t - TRIM as usize, b.h, b.w))
        .collect();
    let (want_grid, rows_out) = table(&trimmed);
    let mut want = Vec::new();
    let mut src = 0usize;
    for (b, keep) in plain_boxes.iter().zip(&trimmed) {
        let skip = TRIM as usize * b.plane();
        want.extend_from_slice(&plain[(src + skip) * c_out..(src + skip + keep.voxels()) * c_out]);
        src += b.voxels();
    }
    let allocated: usize = plain_boxes.iter().map(|b| b.voxels()).sum();

    let ctx = gpu.ctx();
    let mut og = gpu.ints_shaped(&[0; 8], BOXES.len() as u32, 4);
    derive_grid(&ctx, g, GridRule::Shuffle { r: R, trim_t: TRIM }, &mut og)
        .expect("the rule fires");
    let mut o = gpu.zeros(allocated as u32, c_out as u32, Dtype::Bf16);
    pixel_shuffle(
        &ctx,
        gpu.tensor(&x, rows as u32, (c_out * vol) as u32, Dtype::Bf16),
        g,
        R,
        TRIM,
        &mut o,
        og,
        [0, 0],
    )
    .expect("the shuffle fires");
    let got_grid: Vec<i32> = gpu.read_as::<i32>(&og)[..8].to_vec();
    assert_eq!(got_grid, want_grid, "the device rule's boxes and offsets");
    assert_eq!(got_grid, vec![5, 4, 6, 0, 3, 6, 4, 120]);
    let got = gpu.read(&o);
    assert_eq!(&got[..rows_out * c_out], &want[..], "the live rows");
    assert!(
        got[rows_out * c_out..].iter().all(|&v| v == 0.0),
        "the rows past the trimmed claim stay zero"
    );
}

/// Every grid rule the device derives against the host's extents.
#[test]
fn the_derived_grids_are_the_hosts() {
    let Some(gpu) = Gpu::open() else { return };
    let boxes = [Box3::new(5, 6, 8), Box3::new(1, 4, 4), Box3::new(3, 2, 6)];
    let (g, _) = grid_of(&gpu, &boxes);
    let conv = Conv3d {
        k: [3, 3, 3],
        stride: [2, 2, 2],
        pad: [2, 1, 1],
        pad_back: [0, 1, 1],
        causal_t: true,
        time_pad: TimePad::Zero,
    };
    let each = |shape: fn(Box3) -> Box3| boxes.map(shape).to_vec();
    let cases = [
        (
            GridRule::Conv {
                k: conv.k,
                stride: conv.stride,
                pad: conv.pad,
                pad_back: conv.pad_back,
                causal_t: conv.causal_t,
            },
            out_boxes(&conv, &boxes),
        ),
        (
            GridRule::Upsample {
                factor: [2, 2, 2],
                keep_first_frame: true,
            },
            each(|b| Box3::new(1 + (b.t - 1) * 2, b.h * 2, b.w * 2)),
        ),
        (
            GridRule::Unshuffle { r: [1, 2, 2] },
            each(|b| Box3::new(b.t, b.h / 2, b.w / 2)),
        ),
        (
            GridRule::AvgDown { factor: [2, 2, 2] },
            each(|b| Box3::new(b.t.div_ceil(2), b.h / 2, b.w / 2)),
        ),
    ];
    for (rule, want) in cases {
        let mut og = gpu.ints_shaped(&[-1; 12], 3, 4);
        derive_grid(&gpu.ctx(), g, rule, &mut og).expect("the rule fires");
        assert_eq!(
            gpu.read_as::<i32>(&og)[..12].to_vec(),
            table(&want).0,
            "{rule:?}"
        );
    }
}

/// A fire's cache slots gathered lane after lane, and the next fire's
/// window stored back: its last `frames` frames, topped up from the cache
/// it read where the fire held fewer.
#[test]
fn the_frame_cache_gathers_and_stores_each_lanes_window() {
    let Some(gpu) = Gpu::open() else { return };
    let (c, frames) = (8usize, 2usize);
    // The second lane's fire holds a single frame: one stored frame comes
    // from its cache.
    let boxes = [Box3::new(3, 2, 3), Box3::new(1, 3, 2)];
    let slots = [2i32, 0];
    let stride = frames * 6 * c;
    let (g, rows) = grid_of(&gpu, &boxes);
    let cache_rows: usize = boxes.iter().map(|b| frames * b.plane()).sum();

    let slab_vals = bf16s(3 * stride, 0xcac, 1.0);
    let slab = gpu.tensor(&slab_vals, 3, stride as u32, Dtype::Bf16);
    let ids = gpu.ints(&slots);
    let ctx = gpu.ctx();
    let mut cache = gpu.zeros(cache_rows as u32, c as u32, Dtype::Bf16);
    cache_gather(&ctx, slab, ids, g, frames as u32, &mut cache).expect("the gather fires");
    let mut want_cache = Vec::new();
    for (b, &slot) in boxes.iter().zip(&slots) {
        let n = frames * b.plane() * c;
        let at = slot as usize * stride;
        want_cache.extend_from_slice(&slab_vals[at..at + n]);
    }
    let got_cache = gpu.read(&cache);
    assert_eq!(got_cache, want_cache, "each lane's slot, lane after lane");

    let x = bf16s(rows * c, 0x5707, 1.0);
    let mut out = gpu.tensor(&slab_vals, 3, stride as u32, Dtype::Bf16);
    cache_store(
        &ctx,
        gpu.tensor(&x, rows as u32, c as u32, Dtype::Bf16),
        cache,
        ids,
        g,
        frames as u32,
        &mut out,
        0,
    )
    .expect("the store fires");
    let mut want = slab_vals.clone();
    let (mut off, mut base) = (0usize, 0usize);
    for (b, &slot) in boxes.iter().zip(&slots) {
        let plane = b.plane();
        for f in 0..frames {
            let src_t = b.t as i64 - frames as i64 + f as i64;
            for hw in 0..plane {
                for ch in 0..c {
                    let v = if src_t >= 0 {
                        x[(off + src_t as usize * plane + hw) * c + ch]
                    } else {
                        got_cache[(base + (f + b.t) * plane + hw) * c + ch]
                    };
                    want[slot as usize * stride + (f * plane + hw) * c + ch] = v;
                }
            }
        }
        off += b.voxels();
        base += frames * plane;
    }
    assert_eq!(
        gpu.read(&out),
        want,
        "each slot holds its lane's last frames"
    );
    // The slot no lane names is left as it was.
    assert_eq!(
        bits(&gpu, &out)[stride..2 * stride],
        slab_vals[stride..2 * stride]
            .iter()
            .map(|&v| bf16(v))
            .collect::<Vec<_>>()
    );
}
