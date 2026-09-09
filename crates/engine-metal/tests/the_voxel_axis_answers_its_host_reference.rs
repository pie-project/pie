#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::spatial::{attn, conv, norm, resample, rule};
use model_ir::Dtype;

type Case = (
    rule::GridRule,
    Box<dyn Fn([i32; 3]) -> [i32; 3]>,
    &'static str,
);

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Rig {
    fn open() -> Option<Self> {
        Some(Self {
            device: Context::bind().ok()?,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
        })
    }

    fn bf16(&self, data: &[f32]) -> (Buffer, u32) {
        self.raw(&bf16_bytes(data))
    }

    fn f32s(&self, data: &[f32]) -> (Buffer, u32) {
        self.raw(
            &data
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn i32s(&self, data: &[i32]) -> (Buffer, u32) {
        self.raw(
            &data
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn raw(&self, bytes: &[u8]) -> (Buffer, u32) {
        let mut b = Buffer::zeroed(&self.device, bytes.len().max(4) as u64).expect("a plane");
        b.write(0, bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn empty(&self, bytes: u64) -> (Buffer, u32) {
        let b = Buffer::zeroed(&self.device, bytes.max(4)).expect("a plane");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn read_bf16(&self, handle: u32, elements: usize) -> Vec<f32> {
        bf16_floats(
            &self
                .handles
                .read(handle, (elements * 2) as u64)
                .expect("read"),
        )
    }

    fn read_i32(&self, handle: u32, elements: usize) -> Vec<i32> {
        self.handles
            .read(handle, (elements * 4) as u64)
            .expect("read")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn fire(&self, f: impl FnOnce(&Sink<'_>)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        f(&sink);
        frame.commit().expect("the commit");
    }
}

fn within_a_bf16_ulp(what: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let ulp = (w.abs() * (1.0 / 128.0)).max(1.0 / 2560.0);
        let d = (w - g).abs() / ulp;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("{what}: worst {worst:.3} ulp at {at}");
    assert!(
        worst <= 1.0,
        "{what}: parts from the reference by {worst:.2} bf16 ulp at {at} \
         (want {}, got {})",
        want[at],
        got[at]
    );
}

const CLIPS: [[i32; 3]; 2] = [[1, 5, 4], [2, 3, 3]];

fn boxes(extents: &[[i32; 3]]) -> Vec<i32> {
    let mut table = Vec::new();
    let mut off = 0;
    for e in extents {
        table.extend_from_slice(&[e[0], e[1], e[2], off]);
        off += e[0] * e[1] * e[2];
    }
    table
}

fn voxels(extents: &[[i32; 3]]) -> usize {
    extents.iter().map(|e| (e[0] * e[1] * e[2]) as usize).sum()
}

#[test]
fn the_voxel_axis_answers_its_host_reference_every_case() {
    every_grid_rule_maps_the_box_its_host_twin_maps();
    the_convolution_answers_the_tap_walk();
    the_group_norm_answers_the_two_pass_reference();
    the_mid_block_attention_answers_the_plain_softmax();
    the_reshapes_land_the_reference_ordering();
    the_frame_cache_stores_the_frames_its_next_chunk_pads_with();
}

fn every_grid_rule_maps_the_box_its_host_twin_maps() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());
    let grid = boxes(&[[2, 8, 6], [3, 4, 4]]);
    let (_gb, hg) = rig.i32s(&grid);
    let (_ob, ho) = rig.empty(grid.len() as u64 * 4);
    let gt = Tensor::new(hg, 2, 4, Dtype::I32);
    let ot = Tensor::new(ho, 2, 4, Dtype::I32);

    let cases: Vec<Case> = vec![
        (
            rule::GridRule::Conv {
                k: [1, 3, 3],
                stride: [1, 1, 1],
                pad: [0, 1, 1],
                pad_back: [0, 1, 1],
                causal_t: false,
            },
            Box::new(|[t, h, w]| [t, h, w]),
            "same3",
        ),
        (
            rule::GridRule::Conv {
                k: [1, 3, 3],
                stride: [1, 2, 2],
                pad: [0, 0, 0],
                pad_back: [0, 1, 1],
                causal_t: false,
            },
            Box::new(|[t, h, w]| [t, (h + 1 - 3) / 2 + 1, (w + 1 - 3) / 2 + 1]),
            "downsample2d",
        ),
        (
            rule::GridRule::Upsample {
                factor: [1, 2, 2],
                keep_first_frame: false,
            },
            Box::new(|[t, h, w]| [t, h * 2, w * 2]),
            "upsample",
        ),
        (
            rule::GridRule::Upsample {
                factor: [2, 2, 2],
                keep_first_frame: true,
            },
            Box::new(|[t, h, w]| [1 + (t - 1) * 2, h * 2, w * 2]),
            "upsample_keep_first",
        ),
        (
            rule::GridRule::Shuffle {
                r: [1, 2, 2],
                trim_t: 0,
            },
            Box::new(|[t, h, w]| [t, h * 2, w * 2]),
            "shuffle",
        ),
        (
            rule::GridRule::Unshuffle { r: [1, 2, 2] },
            Box::new(|[t, h, w]| [t, h / 2, w / 2]),
            "unshuffle",
        ),
        (
            rule::GridRule::AvgDown { factor: [2, 2, 2] },
            Box::new(|[t, h, w]| [(t + 1) / 2, h / 2, w / 2]),
            "avg_down",
        ),
    ];

    for (r, twin, name) in cases {
        rig.fire(|s| rule::derive_grid(s, gt, r, ot).expect("the rule"));
        let got = rig.read_i32(ho, 8);
        let mut off = 0;
        for (clip, chunk) in grid.as_chunks::<4>().0.iter().enumerate() {
            let want = twin([chunk[0], chunk[1], chunk[2]]);
            let have = [got[clip * 4], got[clip * 4 + 1], got[clip * 4 + 2]];
            assert_eq!(want, have, "{name}: clip {clip}'s box");
            assert_eq!(got[clip * 4 + 3], off, "{name}: clip {clip}'s row offset");
            off += want[0] * want[1] * want[2];
        }
        eprintln!("{name}: {:?}", got);
    }
}

fn the_convolution_answers_the_tap_walk() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const C_IN: usize = 16;
    const C_OUT: usize = 12;
    const K: [usize; 3] = [1, 3, 3];
    let taps = K[0] * K[1] * K[2];
    let grid = boxes(&CLIPS);
    let o_grid = grid.clone();
    let rows = voxels(&CLIPS);

    let x: Vec<f32> = (0..(rows * C_IN) as u64)
        .map(|at| bf16_round(unit(at)))
        .collect();
    let w: Vec<f32> = (0..(C_OUT * taps * C_IN) as u64)
        .map(|at| bf16_round(0.2 * unit(at ^ 0x51)))
        .collect();
    let bias: Vec<f32> = (0..C_OUT as u64).map(|at| 0.1 * unit(at ^ 0x77)).collect();

    let (_xb, hx) = rig.bf16(&x);
    let (_wb, hw) = rig.bf16(&w);
    let (_bb, hb) = rig.f32s(&bias);
    let (_gb, hg) = rig.i32s(&grid);
    let (_ogb, hog) = rig.i32s(&o_grid);
    let (_yb, hy) = rig.empty((rows * C_OUT) as u64 * 2);

    let conv3d = conv::Conv3d {
        k: [1, 3, 3],
        stride: [1, 1, 1],
        pad: [0, 1, 1],
        pad_back: [0, 1, 1],
        causal_t: false,
        time_pad: conv::TimePad::Zero,
    };
    rig.fire(|s| {
        conv::conv3d(
            s,
            Tensor::new(hx, rows as u32, C_IN as u32, Dtype::Bf16),
            Tensor::new(hg, 2, 4, Dtype::I32),
            Tensor::new(hw, C_OUT as u32, (taps * C_IN) as u32, Dtype::Bf16),
            Some(Tensor::new(hb, C_OUT as u32, 1, Dtype::F32)),
            None,
            conv3d,
            Tensor::new(hog, 2, 4, Dtype::I32),
            Tensor::new(hy, rows as u32, C_OUT as u32, Dtype::Bf16),
        )
        .expect("the launch")
    });

    let mut want = vec![0.0f32; rows * C_OUT];
    for (clip, e) in CLIPS.iter().enumerate() {
        let (t, h, wd) = (e[0] as usize, e[1] as usize, e[2] as usize);
        let off = grid[clip * 4 + 3] as usize;
        for ot in 0..t {
            for oh in 0..h {
                for ow in 0..wd {
                    let row = off + (ot * h + oh) * wd + ow;
                    for n in 0..C_OUT {
                        let mut acc = 0.0f32;
                        for tap in 0..taps {
                            let ih = tap / K[2];
                            let iw = tap % K[2];
                            let hi = oh as i64 + ih as i64 - 1;
                            let wi = ow as i64 + iw as i64 - 1;
                            if hi < 0 || hi >= h as i64 || wi < 0 || wi >= wd as i64 {
                                continue;
                            }
                            let src = off + (ot * h + hi as usize) * wd + wi as usize;
                            for c in 0..C_IN {
                                acc += x[src * C_IN + c] * w[(n * taps + tap) * C_IN + c];
                            }
                        }
                        want[row * C_OUT + n] = bf16_round(acc + bias[n]);
                    }
                }
            }
        }
    }
    within_a_bf16_ulp("conv3d", &want, &rig.read_bf16(hy, rows * C_OUT));
}

fn the_group_norm_answers_the_two_pass_reference() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const C: usize = 32;
    const GROUPS: usize = 4;
    let grid = boxes(&CLIPS);
    let rows = voxels(&CLIPS);
    let x: Vec<f32> = (0..(rows * C) as u64)
        .map(|at| bf16_round(20.0 + 3.0 * unit(at ^ 0x2222)))
        .collect();
    let weight: Vec<f32> = (0..C as u64)
        .map(|at| 1.0 + 0.2 * unit(at ^ 0x33))
        .collect();
    let bias: Vec<f32> = (0..C as u64).map(|at| 0.1 * unit(at ^ 0x44)).collect();

    let (_xb, hx) = rig.bf16(&x);
    let (_wb, hw) = rig.f32s(&weight);
    let (_bb, hb) = rig.f32s(&bias);
    let (_gb, hg) = rig.i32s(&grid);
    let (_yb, hy) = rig.empty((rows * C) as u64 * 2);
    let floats = norm::scratch_floats(2, GROUPS as u32);
    let (_pb, hp) = rig.empty(floats * 4);

    let partials_floats = 2 * 32 * GROUPS as u64 * 4;
    let stats_floats = floats - partials_floats;
    let _ = stats_floats;
    let (_sb, hs) = rig.empty(2 * GROUPS as u64 * 2 * 4);

    rig.fire(|s| {
        norm::group_norm(
            s,
            Tensor::new(hx, rows as u32, C as u32, Dtype::Bf16),
            Tensor::new(hg, 2, 4, Dtype::I32),
            GROUPS as u32,
            Tensor::new(hw, C as u32, 1, Dtype::F32),
            Tensor::new(hb, C as u32, 1, Dtype::F32),
            1e-5,
            false,
            Tensor::new(hp, (partials_floats / 4) as u32, 4, Dtype::F32),
            Tensor::new(hs, (2 * GROUPS) as u32, 2, Dtype::F32),
            Tensor::new(hy, rows as u32, C as u32, Dtype::Bf16),
        )
        .expect("the launch")
    });

    let cg = C / GROUPS;
    let mut want = vec![0.0f32; rows * C];
    for (clip, e) in CLIPS.iter().enumerate() {
        let n = (e[0] * e[1] * e[2]) as usize;
        let off = grid[clip * 4 + 3] as usize;
        for g in 0..GROUPS {
            let mut values = Vec::with_capacity(n * cg);
            for v in 0..n {
                for c in g * cg..(g + 1) * cg {
                    values.push(x[(off + v) * C + c]);
                }
            }
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let var =
                values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32;
            let rstd = 1.0 / (var + 1e-5).sqrt();
            for v in 0..n {
                for c in g * cg..(g + 1) * cg {
                    let at = (off + v) * C + c;
                    want[at] = bf16_round((x[at] - mean) * rstd * weight[c] + bias[c]);
                }
            }
        }
    }
    within_a_bf16_ulp("group_norm", &want, &rig.read_bf16(hy, rows * C));
}

fn the_mid_block_attention_answers_the_plain_softmax() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const C: usize = 256;
    let grid = boxes(&CLIPS);
    let rows = voxels(&CLIPS);
    let q: Vec<f32> = (0..(rows * C) as u64)
        .map(|at| bf16_round(unit(at)))
        .collect();
    let k: Vec<f32> = (0..(rows * C) as u64)
        .map(|at| bf16_round(unit(at ^ 0xA1)))
        .collect();
    let v: Vec<f32> = (0..(rows * C) as u64)
        .map(|at| bf16_round(unit(at ^ 0xB2)))
        .collect();
    let scale = (C as f32).sqrt().recip();

    let (_qb, hq) = rig.bf16(&q);
    let (_kb, hk) = rig.bf16(&k);
    let (_vb, hv) = rig.bf16(&v);
    let (_gb, hg) = rig.i32s(&grid);
    let (_yb, hy) = rig.empty((rows * C) as u64 * 2);
    let plane = |h: u32| Tensor::new(h, rows as u32, C as u32, Dtype::Bf16);

    for (segment, name) in [
        (attn::Segment::Clip, "clip"),
        (attn::Segment::Frames(1), "frames1"),
    ] {
        rig.fire(|s| {
            attn::attention(
                s,
                plane(hq),
                plane(hk),
                plane(hv),
                Tensor::new(hg, 2, 4, Dtype::I32),
                segment,
                scale,
                plane(hy),
            )
            .expect("the launch")
        });

        let mut want = vec![0.0f32; rows * C];
        for (clip, e) in CLIPS.iter().enumerate() {
            let (t, h, wd) = (e[0] as usize, e[1] as usize, e[2] as usize);
            let off = grid[clip * 4 + 3] as usize;
            let plane_rows = h * wd;
            for row in off..off + t * plane_rows {
                let (begin, end) = match segment {
                    attn::Segment::Clip => (off, off + t * plane_rows),
                    attn::Segment::Frames(n) => {
                        let frame = (row - off) / plane_rows;
                        let first = frame / n as usize * n as usize;
                        let last = (first + n as usize).min(t);
                        (off + first * plane_rows, off + last * plane_rows)
                    }
                };
                let scores: Vec<f32> = (begin..end)
                    .map(|j| (0..C).map(|c| q[row * C + c] * k[j * C + c]).sum::<f32>() * scale)
                    .collect();
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let denom: f32 = weights.iter().sum();
                for c in 0..C {
                    let acc: f32 = weights
                        .iter()
                        .zip(begin..end)
                        .map(|(w, j)| w * v[j * C + c])
                        .sum();
                    want[row * C + c] = bf16_round(acc / denom);
                }
            }
        }
        let got = rig.read_bf16(hy, rows * C);
        let worst = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("spatial attention {name}: worst absolute {worst:.3e}");
        assert!(worst < 6e-3, "spatial attention {name}: {worst:.3e}");
    }
}

fn the_reshapes_land_the_reference_ordering() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const C: usize = 8;
    let grid = boxes(&CLIPS);
    let rows = voxels(&CLIPS);
    let x: Vec<f32> = (0..(rows * C * 8) as u64)
        .map(|at| bf16_round(unit(at)))
        .collect();
    let (_xb, hx) = rig.bf16(&x);
    let (_gb, hg) = rig.i32s(&grid);
    let gt = Tensor::new(hg, 2, 4, Dtype::I32);

    let up: Vec<[i32; 3]> = CLIPS.iter().map(|e| [e[0], e[1] * 2, e[2] * 2]).collect();
    let up_grid = boxes(&up);
    let up_rows = voxels(&up);
    let (_ub, hu) = rig.i32s(&up_grid);
    let (_yb, hy) = rig.empty((up_rows * C) as u64 * 2);
    rig.fire(|s| {
        resample::upsample_nearest(
            s,
            Tensor::new(hx, rows as u32, C as u32, Dtype::Bf16),
            gt,
            [1, 2, 2],
            false,
            Tensor::new(hu, 2, 4, Dtype::I32),
            Tensor::new(hy, up_rows as u32, C as u32, Dtype::Bf16),
        )
        .expect("upsample")
    });
    let mut want = vec![0.0f32; up_rows * C];
    for (clip, e) in CLIPS.iter().enumerate() {
        let (t, h, wd) = (e[0] as usize, e[1] as usize, e[2] as usize);
        let inoff = grid[clip * 4 + 3] as usize;
        let outoff = up_grid[clip * 4 + 3] as usize;
        for ot in 0..t {
            for oh in 0..h * 2 {
                for ow in 0..wd * 2 {
                    let dst = outoff + (ot * h * 2 + oh) * wd * 2 + ow;
                    let src = inoff + (ot * h + oh / 2) * wd + ow / 2;
                    for c in 0..C {
                        want[dst * C + c] = x[src * C + c];
                    }
                }
            }
        }
    }
    within_a_bf16_ulp("upsample_nearest", &want, &rig.read_bf16(hy, up_rows * C));

    let small: Vec<[i32; 3]> = CLIPS
        .iter()
        .map(|e| [e[0], e[1] / 2 * 2, e[2] / 2 * 2])
        .collect();
    let downed: Vec<[i32; 3]> = small.iter().map(|e| [e[0], e[1] / 2, e[2] / 2]).collect();
    let small_grid = boxes(&small);
    let down_grid = boxes(&downed);
    let small_rows = voxels(&small);
    let down_rows = voxels(&downed);
    let src: Vec<f32> = (0..(small_rows * C) as u64)
        .map(|at| bf16_round(unit(at ^ 0x99)))
        .collect();
    let (_sb, hs) = rig.bf16(&src);
    let (_sgb, hsg) = rig.i32s(&small_grid);
    let (_dgb, hdg) = rig.i32s(&down_grid);
    let (_db, hd) = rig.empty((down_rows * C * 4) as u64 * 2);
    rig.fire(|s| {
        resample::pixel_unshuffle(
            s,
            Tensor::new(hs, small_rows as u32, C as u32, Dtype::Bf16),
            Tensor::new(hsg, 2, 4, Dtype::I32),
            [1, 2, 2],
            Tensor::new(hdg, 2, 4, Dtype::I32),
            Tensor::new(hd, down_rows as u32, (C * 4) as u32, Dtype::Bf16),
        )
        .expect("unshuffle")
    });
    let (_bb, hback) = rig.empty((small_rows * C) as u64 * 2);
    rig.fire(|s| {
        resample::pixel_shuffle(
            s,
            Tensor::new(hd, down_rows as u32, (C * 4) as u32, Dtype::Bf16),
            Tensor::new(hdg, 2, 4, Dtype::I32),
            [1, 2, 2],
            0,
            Tensor::new(hsg, 2, 4, Dtype::I32),
            Tensor::new(hback, small_rows as u32, C as u32, Dtype::Bf16),
        )
        .expect("shuffle")
    });
    within_a_bf16_ulp(
        "unshuffle then shuffle",
        &src,
        &rig.read_bf16(hback, small_rows * C),
    );
}

fn the_frame_cache_stores_the_frames_its_next_chunk_pads_with() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const C: usize = 3;
    const FRAMES: i32 = 2;
    let grid = boxes(&CLIPS);
    let rows = voxels(&CLIPS);

    let cache_rows: usize = CLIPS.iter().map(|e| (FRAMES * e[1] * e[2]) as usize).sum();
    let stride: usize = CLIPS
        .iter()
        .map(|e| (FRAMES * e[1] * e[2]) as usize)
        .max()
        .expect("a clip");

    let x: Vec<f32> = bf16_floats(&bf16_bytes(
        &(0..rows * C)
            .map(|i| unit(i as u64) * 4.0)
            .collect::<Vec<_>>(),
    ));
    let old: Vec<f32> = bf16_floats(&bf16_bytes(
        &(0..cache_rows * C)
            .map(|i| unit(i as u64 + 7717) * 4.0)
            .collect::<Vec<_>>(),
    ));
    let slots: Vec<i32> = (0..CLIPS.len() as i32).rev().collect();

    let (_xb, hx) = rig.bf16(&x);
    let (_cb, hc) = rig.bf16(&old);
    let (_sb, hs) = rig.i32s(&slots);
    let (_gb, hg) = rig.i32s(&grid);
    let (_lb, hslab) = rig.empty((slots.len() * stride * C) as u64 * 2);
    rig.fire(|s| {
        resample::cache_store(
            s,
            Tensor::new(hx, rows as u32, C as u32, Dtype::Bf16),
            Tensor::new(hc, cache_rows as u32, C as u32, Dtype::Bf16),
            Tensor::new(hs, slots.len() as u32, 1, Dtype::I32),
            Tensor::new(hg, CLIPS.len() as u32, 4, Dtype::I32),
            FRAMES as u32,
            Tensor::new(hslab, stride as u32, C as u32, Dtype::Bf16),
        )
        .expect("cache store")
    });

    let mut want = vec![0.0f32; slots.len() * stride * C];
    let (mut voxel_at, mut cache_at) = (0usize, 0usize);
    for (clip, e) in CLIPS.iter().enumerate() {
        let (t, plane) = (e[0], (e[1] * e[2]) as usize);
        for f in 0..FRAMES as usize {
            for hw in 0..plane {
                for c in 0..C {
                    let src_t = t - FRAMES + f as i32;
                    let value = if src_t >= 0 {
                        x[(voxel_at + src_t as usize * plane + hw) * C + c]
                    } else {
                        old[(cache_at + (f + t as usize) * plane + hw) * C + c]
                    };
                    let local = (f * plane + hw) * C + c;
                    want[slots[clip] as usize * stride * C + local] = value;
                }
            }
        }
        voxel_at += (t as usize) * plane;
        cache_at += FRAMES as usize * plane;
    }
    let got = rig.read_bf16(hslab, slots.len() * stride * C);
    assert_eq!(
        want, got,
        "a frame-cache store moves its rows and rounds nothing"
    );
    eprintln!("cache store: bit-exact over {} slabbed elements", got.len());

    let (_rb, hround) = rig.empty((cache_rows * C) as u64 * 2);
    rig.fire(|s| {
        resample::cache_gather(
            s,
            Tensor::new(hslab, stride as u32, C as u32, Dtype::Bf16),
            Tensor::new(hs, slots.len() as u32, 1, Dtype::I32),
            Tensor::new(hg, CLIPS.len() as u32, 4, Dtype::I32),
            FRAMES as u32,
            Tensor::new(hround, cache_rows as u32, C as u32, Dtype::Bf16),
        )
        .expect("cache gather")
    });
    let mut back = vec![0.0f32; cache_rows * C];
    let mut at = 0usize;
    for (clip, e) in CLIPS.iter().enumerate() {
        let rows = (FRAMES * e[1] * e[2]) as usize;
        for local in 0..rows * C {
            back[at * C + local] = want[slots[clip] as usize * stride * C + local];
        }
        at += rows;
    }
    let round = rig.read_bf16(hround, cache_rows * C);
    assert_eq!(
        back, round,
        "a gather is the store's inverse, slot for slot"
    );
    eprintln!(
        "cache gather: bit-exact over {} cached elements",
        round.len()
    );
}
