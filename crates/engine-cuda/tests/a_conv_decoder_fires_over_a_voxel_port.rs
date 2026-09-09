#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use engine_cuda::serve::{Clips, Seated};
use engine_cuda::{Boot, Graphs, Knobs, Lane, Recording, Shell};
use model_compiler::{Budget, VoxelLadder};
use model_dsl::ops::spatial::{self, Conv};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, seam,
    trace_hybrid,
};

const C_IN: usize = 8;
const C_MID: usize = 16;
const C_OUT: usize = 12;
const GROUPS: usize = 4;
const TAPS: usize = 27;
const EPS: f32 = 1e-6;

const BOXES: [[usize; 3]; 2] = [[2, 4, 6], [1, 3, 5]];

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

struct Decoder {
    conv1: Weight,
    b1: Weight,
    gn_w: Weight,
    gn_b: Weight,
    conv2: Weight,
    b2: Weight,
}

impl Decoder {
    fn new() -> Decoder {
        Decoder {
            conv1: Weight::sym("conv1", [C_MID as u64, (C_IN * TAPS) as u64], Dtype::Bf16)
                .conv_taps_major(C_IN as u32, TAPS as u32),
            b1: Weight::sym("conv1.bias", [C_MID as u64], Dtype::F32),
            gn_w: Weight::sym("norm.weight", [C_MID as u64], Dtype::F32),
            gn_b: Weight::sym("norm.bias", [C_MID as u64], Dtype::F32),
            conv2: Weight::sym("conv2", [C_OUT as u64, (C_MID * TAPS) as u64], Dtype::Bf16)
                .conv_taps_major(C_MID as u32, TAPS as u32),
            b2: Weight::sym("conv2.bias", [C_OUT as u64], Dtype::F32),
        }
    }

    fn weights(&self) -> [&Weight; 6] {
        [
            &self.conv1,
            &self.b1,
            &self.gn_w,
            &self.gn_b,
            &self.conv2,
            &self.b2,
        ]
    }
}

impl ForwardHybrid for Decoder {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C_IN as u32, Dtype::Bf16);
        let (h, g1) = spatial::conv3d(&x, &g, &self.conv1, Some(&self.b1), Conv::same3(), None);
        let h = spatial::group_norm(&h, &g1, GROUPS as u32, &self.gn_w, &self.gn_b, EPS, true);
        let (h, g2) = spatial::upsample_nearest(&h, &g1, [1, 2, 2], false);
        let (h, g3) = spatial::conv3d(&h, &g2, &self.conv2, Some(&self.b2), Conv::same3(), None);
        let (y, g4) = spatial::pixel_shuffle(&h, &g3, [1, 2, 2]);
        seam::at(seam::PIXELS, &[&y, &g4]);
        y
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(value)) << 16)
}

fn drawn(lcg: &mut Lcg, n: usize, scale: f32) -> (Vec<f32>, Vec<u8>) {
    let values: Vec<f32> = (0..n).map(|_| bf16_round(lcg.unit() * scale)).collect();
    let bytes = values
        .iter()
        .flat_map(|&v| bf16_bits(v).to_le_bytes())
        .collect();
    (values, bytes)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Planes {
    conv1: Vec<f32>,
    b1: Vec<f32>,
    gn_w: Vec<f32>,
    gn_b: Vec<f32>,
    conv2: Vec<f32>,
    b2: Vec<f32>,
}

fn write_checkpoint(path: &Path) -> Planes {
    let mut lcg = Lcg(0xd8_c0de);
    let (conv1, conv1_bytes) = drawn(
        &mut lcg,
        C_MID * C_IN * TAPS,
        2.0 / ((C_IN * TAPS) as f32).sqrt(),
    );
    let b1: Vec<f32> = (0..C_MID).map(|_| lcg.unit() * 0.2).collect();
    let gn_w: Vec<f32> = (0..C_MID).map(|_| 1.0 + lcg.unit() * 0.5).collect();
    let gn_b: Vec<f32> = (0..C_MID).map(|_| lcg.unit() * 0.5).collect();
    let (conv2, conv2_bytes) = drawn(
        &mut lcg,
        C_OUT * C_MID * TAPS,
        2.0 / ((C_MID * TAPS) as f32).sqrt(),
    );
    let b2: Vec<f32> = (0..C_OUT).map(|_| lcg.unit() * 0.2).collect();

    let mut writer = ztensor::Writer::create(path).expect("the container opens");
    writer
        .add(
            "conv1",
            vec![C_MID as u64, (C_IN * TAPS) as u64],
            ztensor::Leaf::BF16,
            &conv1_bytes,
        )
        .expect("conv1");
    writer
        .add(
            "conv1.bias",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&b1),
        )
        .expect("conv1.bias");
    writer
        .add(
            "conv2",
            vec![C_OUT as u64, (C_MID * TAPS) as u64],
            ztensor::Leaf::BF16,
            &conv2_bytes,
        )
        .expect("conv2");
    writer
        .add(
            "conv2.bias",
            vec![C_OUT as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&b2),
        )
        .expect("conv2.bias");
    writer
        .add(
            "norm.bias",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&gn_b),
        )
        .expect("norm.bias");
    writer
        .add(
            "norm.weight",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&gn_w),
        )
        .expect("norm.weight");
    writer.finish().expect("the container closes");
    Planes {
        conv1,
        b1,
        gn_w,
        gn_b,
        conv2,
        b2,
    }
}

fn voxels(b: [usize; 3]) -> usize {
    b[0] * b[1] * b[2]
}

fn conv3d_ref(
    x: &[f32],
    b: [usize; 3],
    c_in: usize,
    w: &[f32],
    c_out: usize,
    bias: &[f32],
) -> Vec<f32> {
    let [t, h, wd] = b;
    let mut o = vec![0f32; voxels(b) * c_out];
    for to in 0..t {
        for ho in 0..h {
            for wo in 0..wd {
                let m = (to * h + ho) * wd + wo;
                for n in 0..c_out {
                    let mut acc = bias[n];
                    for it in 0..3 {
                        let ti = to as i64 - 1 + it as i64;
                        if ti < 0 || ti as usize >= t {
                            continue;
                        }
                        for ih in 0..3 {
                            let hi = ho as i64 - 1 + ih as i64;
                            if hi < 0 || hi as usize >= h {
                                continue;
                            }
                            for iw in 0..3 {
                                let wi = wo as i64 - 1 + iw as i64;
                                if wi < 0 || wi as usize >= wd {
                                    continue;
                                }
                                let voxel =
                                    ((ti as usize * h + hi as usize) * wd + wi as usize) * c_in;
                                for c in 0..c_in {
                                    let wk = (((n * c_in + c) * 3 + it) * 3 + ih) * 3 + iw;
                                    acc += x[voxel + c] * w[wk];
                                }
                            }
                        }
                    }
                    o[m * c_out + n] = acc;
                }
            }
        }
    }
    o
}

fn group_norm_silu_ref(
    x: &[f32],
    c: usize,
    groups: usize,
    weight: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let cg = c / groups;
    let n = x.len() / c;
    let mut y = vec![0f32; x.len()];
    for g in 0..groups {
        let cells = (n * cg) as f64;
        let mut sum = 0f64;
        for v in 0..n {
            for ch in g * cg..(g + 1) * cg {
                sum += f64::from(x[v * c + ch]);
            }
        }
        let mean = sum / cells;
        let mut sq = 0f64;
        for v in 0..n {
            for ch in g * cg..(g + 1) * cg {
                let d = f64::from(x[v * c + ch]) - mean;
                sq += d * d;
            }
        }
        let rstd = 1.0 / (sq / cells + f64::from(EPS)).sqrt();
        for v in 0..n {
            for ch in g * cg..(g + 1) * cg {
                let e = v * c + ch;
                let mut val =
                    (f64::from(x[e]) - mean) * rstd * f64::from(weight[ch]) + f64::from(bias[ch]);
                val /= 1.0 + (-val).exp();
                y[e] = val as f32;
            }
        }
    }
    y
}

fn upsample_ref(x: &[f32], b: [usize; 3], c: usize, f: [usize; 3]) -> (Vec<f32>, [usize; 3]) {
    let ob = [b[0] * f[0], b[1] * f[1], b[2] * f[2]];
    let mut y = vec![0f32; voxels(ob) * c];
    for to in 0..ob[0] {
        for ho in 0..ob[1] {
            for wo in 0..ob[2] {
                let m = (to * ob[1] + ho) * ob[2] + wo;
                let src = ((to / f[0]) * b[1] + ho / f[1]) * b[2] + wo / f[2];
                y[m * c..(m + 1) * c].copy_from_slice(&x[src * c..(src + 1) * c]);
            }
        }
    }
    (y, ob)
}

fn pixel_shuffle_ref(
    x: &[f32],
    b: [usize; 3],
    c_out: usize,
    r: [usize; 3],
) -> (Vec<f32>, [usize; 3]) {
    let vol = r[0] * r[1] * r[2];
    let c_in = c_out * vol;
    let ob = [b[0] * r[0], b[1] * r[1], b[2] * r[2]];
    let mut y = vec![0f32; voxels(ob) * c_out];
    for to in 0..ob[0] {
        for ho in 0..ob[1] {
            for wo in 0..ob[2] {
                let m = (to * ob[1] + ho) * ob[2] + wo;
                let src = ((to / r[0]) * b[1] + ho / r[1]) * b[2] + wo / r[2];
                let block = ((to % r[0]) * r[1] + ho % r[1]) * r[2] + wo % r[2];
                for c in 0..c_out {
                    y[m * c_out + c] = x[src * c_in + c * vol + block];
                }
            }
        }
    }
    (y, ob)
}

fn decode_ref(x: &[f32], b: [usize; 3], p: &Planes) -> (Vec<f32>, [usize; 3]) {
    let round = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(bf16_round).collect() };
    let h = round(conv3d_ref(x, b, C_IN, &p.conv1, C_MID, &p.b1));
    let h = round(group_norm_silu_ref(&h, C_MID, GROUPS, &p.gn_w, &p.gn_b));
    let (h, b1) = upsample_ref(&h, b, C_MID, [1, 2, 2]);
    let h = round(conv3d_ref(&h, b1, C_MID, &p.conv2, C_OUT, &p.b2));
    let (y, b2) = pixel_shuffle_ref(&h, b1, 3, [1, 2, 2]);
    (round(y), b2)
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-d8-decoder-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn classify(_: &model_ir::Request) -> u64 {
    0
}

#[test]
fn a_conv_decoder_fires_over_a_voxel_port_every_case() {
    the_decoder_answers_the_reference_for_two_clips_of_different_boxes();
    a_causal_conv_carries_its_frames_across_fires_in_the_lanes_slot();
}

fn the_decoder_answers_the_reference_for_two_clips_of_different_boxes() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the conv decoder gate: no CUDA device");
        return;
    }
    let dir = scratch();
    let container = dir.join("decoder.zt");
    let planes = write_checkpoint(&container);

    let decoder = Decoder::new();
    let trace = trace_hybrid("d8-decoder", &decoder, Platform::Cuda);
    let source = ztensor::Source::open(&container).expect("the container opens");
    let contract = {
        let mut b = checkpoint_dsl::Builder::new(&source, 1, Platform::Cuda);
        for w in decoder.weights() {
            b.read_own(w)
                .unwrap_or_else(|why| panic!("`{}`: {why}", w.name));
        }
        b.build()
    };
    drop(source);

    let max_voxels: u32 = BOXES.iter().map(|b| voxels(*b) as u32).sum::<u32>() + 8;
    let mut shell = Shell::load(Boot {
        classify,
        trace,
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(4, 16),
        patches: None,
        voxels: Some(VoxelLadder::new(max_voxels, 4)),
        profile: None,
        page_size: 16,
        context: 64,
        slots: 4,
        pages: 16,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs {
            recording: Recording::Off,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: true,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the decoder loads");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");

    let mut lcg = Lcg(0xc11d);
    let inputs: Vec<(Vec<f32>, Vec<u8>)> = BOXES
        .iter()
        .map(|b| drawn(&mut lcg, voxels(*b) * C_IN, 1.0))
        .collect();
    let tokens = [0u32];
    let lanes = [
        Seated::of(Lane {
            slot: 0,
            word: 0,
            tokens: &tokens,
        }),
        Seated::of(Lane {
            slot: 1,
            word: 0,
            tokens: &tokens,
        }),
    ];
    let clips = [
        Clips {
            lane: 0,
            clips: &[[BOXES[0][0] as u32, BOXES[0][1] as u32, BOXES[0][2] as u32]],
            payload: &inputs[0].1,
        },
        Clips {
            lane: 1,
            clips: &[[BOXES[1][0] as u32, BOXES[1][1] as u32, BOXES[1][2] as u32]],
            payload: &inputs[1].1,
        },
    ];
    let answered = shell
        .fire_voxels(&lanes, &clips)
        .expect("the decoder fires");
    assert_eq!(answered.len(), 2, "one answer per submitted lane");

    let mut worst = 0f32;
    for (lane, (got, boxes)) in answered.iter().enumerate() {
        let (want, want_box) = decode_ref(&inputs[lane].0, BOXES[lane], &planes);
        assert_eq!(
            boxes,
            &vec![[want_box[0] as u32, want_box[1] as u32, want_box[2] as u32]],
            "lane {lane}'s clip comes back at the output resolution"
        );
        assert_eq!(
            got.len(),
            want.len(),
            "lane {lane}: one pixel row per output voxel, 3 wide"
        );
        let mut sum = 0f32;
        let mut spread = 0f32;
        for (g, w) in got.iter().zip(&want) {
            let err = (g - w).abs();
            worst = worst.max(err);
            sum += err;
            spread = spread.max(w.abs());
        }
        let mean = sum / want.len() as f32;
        let live = got.iter().filter(|v| v.abs() > 0.05).count();
        assert!(
            live * 4 > got.len(),
            "lane {lane}: the pixels came back mostly zero ({live} of {} live), which is a \
             readout of the wrong rectangle, not a decode",
            got.len()
        );
        eprintln!(
            "lane {lane}: first pixels {:?} vs {:?}",
            &got[..6],
            &want[..6]
        );
        eprintln!(
            "lane {lane}: box {:?} -> {:?}, {} pixel rows, max |err| {worst:.4}, mean {mean:.5}, spread {spread:.3}",
            BOXES[lane],
            want_box,
            got.len() / 3
        );
        assert!(
            mean < 1e-2 && worst < 6e-2,
            "lane {lane} drifts from the reference: max {worst}, mean {mean}"
        );
    }
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}

const FRAMES: usize = 2;

const PLANE_MAX: usize = 4 * 6;

struct Causal {
    conv: Weight,
    bias: Weight,
}

impl Causal {
    fn new() -> Causal {
        Causal {
            conv: Weight::sym("conv", [C_MID as u64, (C_IN * TAPS) as u64], Dtype::Bf16)
                .conv_taps_major(C_IN as u32, TAPS as u32),
            bias: Weight::sym("conv.bias", [C_MID as u64], Dtype::F32),
        }
    }
}

impl ForwardHybrid for Causal {
    type Facts = NoFacts;
    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        spec.state(
            "conv.cache",
            [(FRAMES * PLANE_MAX) as u64, C_IN as u64],
            Dtype::Bf16,
        );
        spec
    }
    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let g = inputs.grid();
        let x = inputs.voxels(0, C_IN as u32, Dtype::Bf16);
        let cache = inputs.state("conv.cache");
        let (y, g1) = spatial::conv3d(
            &x,
            &g,
            &self.conv,
            Some(&self.bias),
            Conv::same3().causal(model_ir::TimePad::Zero),
            Some(cache),
        );
        seam::at(seam::PIXELS, &[&y, &g1]);
        y
    }
}

fn causal_conv3d_ref(
    x: &[f32],
    front: &[f32],
    b: [usize; 3],
    c_in: usize,
    w: &[f32],
    c_out: usize,
    bias: &[f32],
) -> Vec<f32> {
    let [t, h, wd] = b;
    let plane = h * wd;
    let mut o = vec![0f32; voxels(b) * c_out];
    for to in 0..t {
        for ho in 0..h {
            for wo in 0..wd {
                let m = (to * h + ho) * wd + wo;
                for n in 0..c_out {
                    let mut acc = bias[n];
                    for it in 0..3 {
                        let ti = to as i64 - FRAMES as i64 + it as i64;
                        let frame: &[f32] = if ti < 0 {
                            let f = (ti + FRAMES as i64) as usize;
                            &front[f * plane * c_in..(f + 1) * plane * c_in]
                        } else {
                            &x[ti as usize * plane * c_in..(ti as usize + 1) * plane * c_in]
                        };
                        for ih in 0..3 {
                            let hi = ho as i64 - 1 + ih as i64;
                            if hi < 0 || hi as usize >= h {
                                continue;
                            }
                            for iw in 0..3 {
                                let wi = wo as i64 - 1 + iw as i64;
                                if wi < 0 || wi as usize >= wd {
                                    continue;
                                }
                                let voxel = (hi as usize * wd + wi as usize) * c_in;
                                for c in 0..c_in {
                                    let wk = (((n * c_in + c) * 3 + it) * 3 + ih) * 3 + iw;
                                    acc += frame[voxel + c] * w[wk];
                                }
                            }
                        }
                    }
                    o[m * c_out + n] = bf16_round(acc);
                }
            }
        }
    }
    o
}

fn close(lane: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{lane}: one row per output voxel");
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0f32, f32::max);
    let mean = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .sum::<f32>()
        / want.len() as f32;
    eprintln!("{lane}: max |err| {worst:.4}, mean {mean:.5}");
    assert!(
        mean < 1e-2 && worst < 6e-2,
        "{lane} drifts: max {worst}, mean {mean}"
    );
}

fn a_causal_conv_carries_its_frames_across_fires_in_the_lanes_slot() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the causal conv gate: no CUDA device");
        return;
    }
    let dir = scratch();
    let container = dir.join("causal.zt");
    let mut lcg = Lcg(0xca_5a1);
    let (conv, conv_bytes) = drawn(
        &mut lcg,
        C_MID * C_IN * TAPS,
        2.0 / ((C_IN * TAPS) as f32).sqrt(),
    );
    let bias: Vec<f32> = (0..C_MID).map(|_| lcg.unit() * 0.2).collect();
    {
        let mut writer = ztensor::Writer::create(&container).expect("the container opens");
        writer
            .add(
                "conv",
                vec![C_MID as u64, (C_IN * TAPS) as u64],
                ztensor::Leaf::BF16,
                &conv_bytes,
            )
            .expect("conv");
        writer
            .add(
                "conv.bias",
                vec![C_MID as u64],
                ztensor::Leaf::F32,
                &f32_bytes(&bias),
            )
            .expect("conv.bias");
        writer.finish().expect("closes");
    }

    let causal = Causal::new();
    let trace = trace_hybrid("d8-causal", &causal, Platform::Cuda);
    let source = ztensor::Source::open(&container).expect("opens");
    let contract = {
        let mut b = checkpoint_dsl::Builder::new(&source, 1, Platform::Cuda);
        for w in [&causal.conv, &causal.bias] {
            b.read_own(w)
                .unwrap_or_else(|why| panic!("`{}`: {why}", w.name));
        }
        b.build()
    };
    drop(source);
    let mut shell = Shell::load(Boot {
        classify,
        trace,
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(4, 16),
        patches: None,
        voxels: Some(VoxelLadder::new(256, 4)),
        profile: None,
        page_size: 16,
        context: 64,
        slots: 4,
        pages: 16,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs {
            recording: Recording::Off,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: true,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the causal text loads");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");

    let big = [4usize, 4, 6];
    let tile = [2usize, 4, 6];
    let small = [1usize, 3, 5];
    let plane = big[1] * big[2];
    let (whole, _) = drawn(&mut lcg, voxels(big) * C_IN, 1.0);
    let (other, other_bytes) = drawn(&mut lcg, voxels(small) * C_IN, 1.0);
    let half = voxels(tile) * C_IN;
    let first_bytes: Vec<u8> = whole[..half]
        .iter()
        .flat_map(|&v| bf16_bits(v).to_le_bytes())
        .collect();
    let second_bytes: Vec<u8> = whole[half..]
        .iter()
        .flat_map(|&v| bf16_bits(v).to_le_bytes())
        .collect();
    let zeros = vec![0f32; FRAMES * plane * C_IN];
    let want_first = causal_conv3d_ref(&whole[..half], &zeros, tile, C_IN, &conv, C_MID, &bias);
    let want_second = causal_conv3d_ref(
        &whole[half..],
        &whole[..half],
        tile,
        C_IN,
        &conv,
        C_MID,
        &bias,
    );
    let cold_second = causal_conv3d_ref(&whole[half..], &zeros, tile, C_IN, &conv, C_MID, &bias);
    assert!(
        want_second
            .iter()
            .zip(&cold_second)
            .any(|(warm, cold)| (warm - cold).abs() > 0.1),
        "the front frames change the second tile's answer"
    );
    let want_other = causal_conv3d_ref(
        &other,
        &vec![0f32; FRAMES * small[1] * small[2] * C_IN],
        small,
        C_IN,
        &conv,
        C_MID,
        &bias,
    );

    let tokens = [0u32];
    let lane = |slot: u32| {
        Seated::of(Lane {
            slot,
            word: 0,
            tokens: &tokens,
        })
    };
    let boxed = |b: [usize; 3]| [[b[0] as u32, b[1] as u32, b[2] as u32]];
    let tile_box = boxed(tile);
    let small_box = boxed(small);

    let answered = shell
        .fire_voxels(
            &[lane(0)],
            &[Clips {
                lane: 0,
                clips: &tile_box,
                payload: &first_bytes,
            }],
        )
        .expect("the first tile fires");
    close("first tile", &answered[0].0, &want_first);

    let answered = shell
        .fire_voxels(
            &[lane(0), lane(1)],
            &[
                Clips {
                    lane: 0,
                    clips: &tile_box,
                    payload: &second_bytes,
                },
                Clips {
                    lane: 1,
                    clips: &small_box,
                    payload: &other_bytes,
                },
            ],
        )
        .expect("the second fire fires");
    close("second tile", &answered[0].0, &want_second);
    close("fresh clip beside it", &answered[1].0, &want_other);

    shell.open(0).expect("slot 0 reopens");
    let answered = shell
        .fire_voxels(
            &[lane(0)],
            &[Clips {
                lane: 0,
                clips: &tile_box,
                payload: &first_bytes,
            }],
        )
        .expect("the reopened slot fires");
    close("reopened slot", &answered[0].0, &want_first);

    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}
