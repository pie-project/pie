//! A conv decoder served whole by the shell on the first wgpu adapter: two
//! lanes, a clip each of different boxes, through the voxel port, conv3d,
//! group norm with its SiLU, voxel attention, nearest upsample, a second
//! conv3d and a pixel shuffle, read back as the pixel plane and checked
//! against an f64 host reference rounded where the device stores bf16.

use std::path::{Path, PathBuf};

use engine_wgpu::api::DeviceBoot;
use engine_wgpu::serve::{Boot, Clips, Lane, Seated, Shell};
use poem::ops::attn;
use poem::ops::conv::{self, ConvShape};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Weight, trace_hybrid};
use poem_compiler::{Budget, VoxelLadder};

const C_IN: usize = 8;
const C_MID: usize = 16;
/// Four pixel channels out of the shuffle: a bf16 row lands whole words.
const C_OUT: usize = 16;
const GROUPS: usize = 4;
const TAPS: usize = 27;
const EPS: f32 = 1e-6;

const BOXES: [[usize; 3]; 2] = [[2, 4, 6], [1, 3, 5]];

fn sm_scale() -> f32 {
    1.0 / (C_MID as f32).sqrt()
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
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) {
        let g = inputs.grid();
        let x = inputs.input(
            "voxels",
            Dtype::Bf16,
            poem_ir::InputAxis::Voxel,
            C_IN as u32,
            None,
        );
        let (h, g1) = conv::conv3d(
            &x,
            &g,
            &self.conv1,
            Some(&self.b1),
            ConvShape::same3(),
            None,
        );
        let h = conv::group_norm(&h, &g1, GROUPS as u32, &self.gn_w, &self.gn_b, EPS, true);
        let h = attn::voxel(&h, &h, &h, &g1, sm_scale());
        let (h, g2) = conv::upsample_nearest(&h, &g1, [1, 2, 2], false);
        let (h, g3) = conv::conv3d(
            &h,
            &g2,
            &self.conv2,
            Some(&self.b2),
            ConvShape::same3(),
            None,
        );
        let (y, g4) = conv::pixel_shuffle(&h, &g3, [1, 2, 2]);
        poem::output("pixels", &y, Some(&g4));
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
    let planes: [(&str, Vec<u64>, ztensor::Leaf, &[u8]); 6] = [
        (
            "conv1",
            vec![C_MID as u64, (C_IN * TAPS) as u64],
            ztensor::Leaf::BF16,
            &conv1_bytes,
        ),
        (
            "conv1.bias",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&b1),
        ),
        (
            "conv2",
            vec![C_OUT as u64, (C_MID * TAPS) as u64],
            ztensor::Leaf::BF16,
            &conv2_bytes,
        ),
        (
            "conv2.bias",
            vec![C_OUT as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&b2),
        ),
        (
            "norm.bias",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&gn_b),
        ),
        (
            "norm.weight",
            vec![C_MID as u64],
            ztensor::Leaf::F32,
            &f32_bytes(&gn_w),
        ),
    ];
    for (name, shape, leaf, bytes) in planes {
        writer.add(name, shape, leaf, bytes).expect(name);
    }
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

/// A 3x3x3 convolution, padded by one each side, weights `[c_out, c_in, 3, 3, 3]`.
fn conv3d_ref(
    x: &[f32],
    b: [usize; 3],
    c_in: usize,
    w: &[f32],
    c_out: usize,
    bias: &[f32],
) -> Vec<f32> {
    let [t, h, wd] = b;
    let at = |i: usize, k: usize, n: usize| (i + k).checked_sub(1).filter(|&v| v < n);
    let mut o = vec![0f32; voxels(b) * c_out];
    for to in 0..t {
        for ho in 0..h {
            for wo in 0..wd {
                let m = (to * h + ho) * wd + wo;
                for n in 0..c_out {
                    let mut acc = f64::from(bias[n]);
                    for it in 0..3 {
                        let Some(ti) = at(to, it, t) else { continue };
                        for ih in 0..3 {
                            let Some(hi) = at(ho, ih, h) else { continue };
                            for iw in 0..3 {
                                let Some(wi) = at(wo, iw, wd) else {
                                    continue;
                                };
                                let voxel = ((ti * h + hi) * wd + wi) * c_in;
                                for c in 0..c_in {
                                    let wk = (((n * c_in + c) * 3 + it) * 3 + ih) * 3 + iw;
                                    acc += f64::from(x[voxel + c]) * f64::from(w[wk]);
                                }
                            }
                        }
                    }
                    o[m * c_out + n] = acc as f32;
                }
            }
        }
    }
    o
}

fn group_norm_silu_ref(x: &[f32], c: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let cg = c / GROUPS;
    let n = x.len() / c;
    let mut y = vec![0f32; x.len()];
    for g in 0..GROUPS {
        let cells = (n * cg) as f64;
        let at = |v: usize, ch: usize| f64::from(x[v * c + ch]);
        let mean = (0..n)
            .flat_map(|v| (g * cg..(g + 1) * cg).map(move |ch| (v, ch)))
            .map(|(v, ch)| at(v, ch))
            .sum::<f64>()
            / cells;
        let var = (0..n)
            .flat_map(|v| (g * cg..(g + 1) * cg).map(move |ch| (v, ch)))
            .map(|(v, ch)| (at(v, ch) - mean).powi(2))
            .sum::<f64>()
            / cells;
        let rstd = 1.0 / (var + f64::from(EPS)).sqrt();
        for v in 0..n {
            for ch in g * cg..(g + 1) * cg {
                let val = (at(v, ch) - mean) * rstd * f64::from(weight[ch]) + f64::from(bias[ch]);
                y[v * c + ch] = (val / (1.0 + (-val).exp())) as f32;
            }
        }
    }
    y
}

/// Every row of one clip attending every row of it, `q = k = v = x`.
fn voxel_attention_ref(x: &[f32], c: usize) -> Vec<f32> {
    let n = x.len() / c;
    let scale = f64::from(sm_scale());
    let mut o = vec![0f32; x.len()];
    for r in 0..n {
        let scores: Vec<f64> = (0..n)
            .map(|j| {
                (0..c)
                    .map(|d| f64::from(x[r * c + d]) * f64::from(x[j * c + d]))
                    .sum::<f64>()
                    * scale
            })
            .collect();
        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
        let total: f64 = w.iter().sum();
        for d in 0..c {
            o[r * c + d] =
                ((0..n).map(|j| w[j] * f64::from(x[j * c + d])).sum::<f64>() / total) as f32;
        }
    }
    o
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

fn decode_ref(x: &[f32], b: [usize; 3], p: &Planes) -> Vec<f32> {
    let round = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(bf16_round).collect() };
    let h = round(conv3d_ref(x, b, C_IN, &p.conv1, C_MID, &p.b1));
    let h = round(group_norm_silu_ref(&h, C_MID, &p.gn_w, &p.gn_b));
    let h = round(voxel_attention_ref(&h, C_MID));
    let (h, b1) = upsample_ref(&h, b, C_MID, [1, 2, 2]);
    let h = round(conv3d_ref(&h, b1, C_MID, &p.conv2, C_OUT, &p.b2));
    let (y, _) = pixel_shuffle_ref(&h, b1, C_OUT / 4, [1, 2, 2]);
    round(y)
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-wgpu-decoder-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

#[test]
fn a_conv_decoder_fires_over_a_voxel_input() {
    let device = DeviceBoot::default();
    if engine_wgpu::Context::bind(&device, 0.25).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = scratch();
    let container = dir.join("decoder.zt");
    let planes = write_checkpoint(&container);

    let decoder = Decoder::new();
    let trace = trace_hybrid("wgpu-decoder", &decoder, Platform::Wgpu);
    let source = ztensor::Source::open(&container).expect("the container opens");
    let contract = {
        let mut b = poem::import::Builder::new(&source, 1, Platform::Wgpu);
        for w in decoder.weights() {
            b.read_own(w)
                .unwrap_or_else(|why| panic!("`{}`: {why}", w.name));
        }
        b.build()
    };
    drop(source);

    let max_voxels: u32 = (BOXES.iter().map(|b| voxels(*b) as u32).sum::<u32>() + 8) * 16;
    let mut shell = Shell::load(Boot {
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
        runahead: engine::runahead::Runahead::F1,
        device: &device,
        residency: engine::load::Residency::default(),
        handed: None,
    })
    .expect("the decoder loads");
    assert_eq!(
        shell.pixels_width(),
        Some(4),
        "the plan reads out four channels"
    );
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");

    let mut lcg = Lcg(0xc11d);
    let inputs: Vec<(Vec<f32>, Vec<u8>)> = BOXES
        .iter()
        .map(|b| drawn(&mut lcg, voxels(*b) * C_IN, 1.0))
        .collect();
    let tokens = [0u32];
    let lanes = [0u32, 1].map(|slot| Seated {
        kv_less: true,
        ..Seated::of(Lane {
            slot,
            word: 0,
            tokens: &tokens,
        })
    });
    let boxes = BOXES.map(|b| [[b[0] as u32, b[1] as u32, b[2] as u32]]);
    let clips = [
        Clips {
            lane: 0,
            boxes: &boxes[0],
            payload: &inputs[0].1,
        },
        Clips {
            lane: 1,
            boxes: &boxes[1],
            payload: &inputs[1].1,
        },
    ];
    let plane = shell
        .fire_voxels(&lanes, &clips)
        .expect("the decoder fires");

    let mut at = 0;
    for (lane, b) in BOXES.iter().enumerate() {
        let want = decode_ref(&inputs[lane].0, *b, &planes);
        let got = plane
            .get(at..at + want.len())
            .unwrap_or_else(|| panic!("lane {lane}: the plane stops at {}", plane.len()));
        at += want.len();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        let mean = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .sum::<f32>()
            / want.len() as f32;
        let live = got.iter().filter(|v| v.abs() > 0.05).count();
        eprintln!(
            "lane {lane}: box {b:?}, {} pixel rows, max |err| {worst:.4}, mean {mean:.5}",
            want.len() / 4
        );
        assert!(
            live * 4 > got.len(),
            "lane {lane}: {live} of {} pixels live, a readout of the wrong rectangle",
            got.len()
        );
        assert!(
            mean < 1e-2 && worst < 6e-2,
            "lane {lane} drifts from the reference: max {worst}, mean {mean}"
        );
    }
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}
