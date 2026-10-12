#![allow(dead_code)]

use kernels_cuda::spatial::{Conv3d, TimePad};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Box3 {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl Box3 {
    pub const fn new(t: usize, h: usize, w: usize) -> Self {
        Self { t, h, w }
    }

    pub const fn voxels(self) -> usize {
        self.t * self.h * self.w
    }

    pub const fn plane(self) -> usize {
        self.h * self.w
    }
}

pub fn table(boxes: &[Box3]) -> (Vec<i32>, usize) {
    let mut rows = 0usize;
    let mut out = Vec::with_capacity(boxes.len() * 4);
    for b in boxes {
        out.extend([b.t as i32, b.h as i32, b.w as i32, rows as i32]);
        rows += b.voxels();
    }
    (out, rows)
}

pub fn out_boxes(conv: &Conv3d, boxes: &[Box3]) -> Vec<Box3> {
    boxes
        .iter()
        .map(|b| {
            let [t, h, w] = conv
                .out_extent([b.t as u32, b.h as u32, b.w as u32])
                .expect("the box holds the kernel");
            Box3::new(t as usize, h as usize, w as usize)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_ref(
    x: &[f32],
    boxes: &[Box3],
    c_in: usize,
    w_natural: &[f32],
    c_out: usize,
    bias: Option<&[f32]>,
    conv: &Conv3d,
    cache: Option<&[f32]>,
) -> Vec<f32> {
    let [kt, kh, kw] = conv.k.map(|v| v as usize);
    let [st, sh, sw] = conv.stride.map(|v| v as usize);
    let [pt, ph, pw] = conv.pad.map(|v| v as i64);
    let outs = out_boxes(conv, boxes);
    let rows_out: usize = outs.iter().map(|b| b.voxels()).sum();
    let mut o = vec![0f32; rows_out * c_out];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    let mut cache_off = 0usize;
    for (l, b) in boxes.iter().enumerate() {
        let ob = outs[l];
        for to in 0..ob.t {
            for ho in 0..ob.h {
                for wo in 0..ob.w {
                    let m = out_off + (to * ob.h + ho) * ob.w + wo;
                    for n in 0..c_out {
                        let mut acc = bias.map_or(0.0, |b| b[n]);
                        for it in 0..kt {
                            let mut ti = (to * st) as i64 - pt + it as i64;
                            let mut frame: Option<&[f32]> = None;
                            let replicate = conv.time_pad == TimePad::Replicate;
                            if ti < 0 {
                                if let (true, Some(cache)) = (conv.causal_t, cache) {
                                    let f = (ti + pt) as usize;
                                    let start = cache_off + f * b.plane();
                                    frame = Some(&cache[start * c_in..(start + b.plane()) * c_in]);
                                } else if replicate {
                                    ti = 0;
                                } else {
                                    continue;
                                }
                            }
                            if frame.is_none() {
                                if ti as usize >= b.t {
                                    if conv.causal_t || !replicate {
                                        continue;
                                    }
                                    ti = b.t as i64 - 1;
                                }
                                let start = in_off + ti as usize * b.plane();
                                frame = Some(&x[start * c_in..(start + b.plane()) * c_in]);
                            }
                            let frame = frame.expect("resolved");
                            for ih in 0..kh {
                                let hi = (ho * sh) as i64 - ph + ih as i64;
                                if hi < 0 || hi as usize >= b.h {
                                    continue;
                                }
                                for iw in 0..kw {
                                    let wi = (wo * sw) as i64 - pw + iw as i64;
                                    if wi < 0 || wi as usize >= b.w {
                                        continue;
                                    }
                                    let voxel = (hi as usize * b.w + wi as usize) * c_in;
                                    for c in 0..c_in {
                                        let wk = (((n * c_in + c) * kt + it) * kh + ih) * kw + iw;
                                        acc += frame[voxel + c] * w_natural[wk];
                                    }
                                }
                            }
                        }
                        o[m * c_out + n] = acc;
                    }
                }
            }
        }
        in_off += b.voxels();
        out_off += ob.voxels();
        cache_off += pt as usize * b.plane();
    }
    o
}

#[allow(clippy::too_many_arguments)]
pub fn group_norm_ref(
    x: &[f32],
    boxes: &[Box3],
    c: usize,
    groups: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
    silu: bool,
) -> Vec<f32> {
    let cg = c / groups;
    let mut y = vec![0f32; x.len()];
    let mut off = 0usize;
    for b in boxes {
        let n = b.voxels();
        for g in 0..groups {
            let mut sum = 0f64;
            let mut count = 0f64;
            for v in 0..n {
                for ch in g * cg..(g + 1) * cg {
                    sum += f64::from(x[(off + v) * c + ch]);
                    count += 1.0;
                }
            }
            let mean = sum / count;
            let mut sq = 0f64;
            for v in 0..n {
                for ch in g * cg..(g + 1) * cg {
                    let d = f64::from(x[(off + v) * c + ch]) - mean;
                    sq += d * d;
                }
            }
            let rstd = 1.0 / (sq / count + f64::from(eps)).sqrt();
            for v in 0..n {
                for ch in g * cg..(g + 1) * cg {
                    let e = (off + v) * c + ch;
                    let mut val = (f64::from(x[e]) - mean) * rstd * f64::from(weight[ch])
                        + f64::from(bias[ch]);
                    if silu {
                        val /= 1.0 + (-val).exp();
                    }
                    y[e] = val as f32;
                }
            }
        }
        off += n;
    }
    y
}

pub fn pixel_shuffle_ref(
    x: &[f32],
    boxes: &[Box3],
    c_out: usize,
    r: [usize; 3],
) -> (Vec<f32>, Vec<Box3>) {
    let vol = r[0] * r[1] * r[2];
    let c_in = c_out * vol;
    let outs: Vec<Box3> = boxes
        .iter()
        .map(|b| Box3::new(b.t * r[0], b.h * r[1], b.w * r[2]))
        .collect();
    let rows_out: usize = outs.iter().map(|b| b.voxels()).sum();
    let mut y = vec![0f32; rows_out * c_out];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for (l, b) in boxes.iter().enumerate() {
        let ob = outs[l];
        for to in 0..ob.t {
            for ho in 0..ob.h {
                for wo in 0..ob.w {
                    let m = out_off + (to * ob.h + ho) * ob.w + wo;
                    let src = in_off + ((to / r[0]) * b.h + ho / r[1]) * b.w + wo / r[2];
                    let block = ((to % r[0]) * r[1] + ho % r[1]) * r[2] + wo % r[2];
                    for c in 0..c_out {
                        y[m * c_out + c] = x[src * c_in + c * vol + block];
                    }
                }
            }
        }
        in_off += b.voxels();
        out_off += ob.voxels();
    }
    (y, outs)
}

pub fn avg_down_ref(
    x: &[f32],
    boxes: &[Box3],
    c: usize,
    r: [usize; 3],
    group: usize,
) -> (Vec<f32>, Vec<Box3>) {
    let vol = r[0] * r[1] * r[2];
    let c_out = c * vol / group;
    let outs: Vec<Box3> = boxes
        .iter()
        .map(|b| Box3::new(b.t.div_ceil(r[0]), b.h / r[1], b.w / r[2]))
        .collect();
    let rows_out: usize = outs.iter().map(|b| b.voxels()).sum();
    let mut y = vec![0f32; rows_out * c_out];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for (l, b) in boxes.iter().enumerate() {
        let ob = outs[l];
        let pad_t = (r[0] - b.t % r[0]) % r[0];
        for to in 0..ob.t {
            for ho in 0..ob.h {
                for wo in 0..ob.w {
                    let m = out_off + (to * ob.h + ho) * ob.w + wo;
                    for n in 0..c_out {
                        let mut acc = 0f32;
                        for j in 0..group {
                            let q = n * group + j;
                            let ch = q / vol;
                            let block = q % vol;
                            let i1 = block / (r[1] * r[2]);
                            let i2 = (block / r[2]) % r[1];
                            let i3 = block % r[2];
                            let ti = (to * r[0] + i1) as i64 - pad_t as i64;
                            if ti < 0 {
                                continue;
                            }
                            let src = in_off
                                + ((ti as usize) * b.h + ho * r[1] + i2) * b.w
                                + wo * r[2]
                                + i3;
                            acc += x[src * c + ch];
                        }
                        y[m * c_out + n] = acc / group as f32;
                    }
                }
            }
        }
        in_off += b.voxels();
        out_off += ob.voxels();
    }
    (y, outs)
}

pub fn pixel_unshuffle_ref(
    x: &[f32],
    boxes: &[Box3],
    c: usize,
    r: [usize; 3],
) -> (Vec<f32>, Vec<Box3>) {
    let vol = r[0] * r[1] * r[2];
    let c_out = c * vol;
    let outs: Vec<Box3> = boxes
        .iter()
        .map(|b| Box3::new(b.t / r[0], b.h / r[1], b.w / r[2]))
        .collect();
    let rows_out: usize = outs.iter().map(|b| b.voxels()).sum();
    let mut y = vec![0f32; rows_out * c_out];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for (l, b) in boxes.iter().enumerate() {
        let ob = outs[l];
        for to in 0..ob.t {
            for ho in 0..ob.h {
                for wo in 0..ob.w {
                    let m = out_off + (to * ob.h + ho) * ob.w + wo;
                    for col in 0..c_out {
                        let ch = col / vol;
                        let block = col % vol;
                        let i1 = block / (r[1] * r[2]);
                        let i2 = (block / r[2]) % r[1];
                        let i3 = block % r[2];
                        let src = in_off
                            + ((to * r[0] + i1) * b.h + ho * r[1] + i2) * b.w
                            + wo * r[2]
                            + i3;
                        y[m * c_out + col] = x[src * c + ch];
                    }
                }
            }
        }
        in_off += b.voxels();
        out_off += ob.voxels();
    }
    (y, outs)
}

pub fn upsample_ref(
    x: &[f32],
    boxes: &[Box3],
    c: usize,
    f: [usize; 3],
    keep_first: bool,
) -> (Vec<f32>, Vec<Box3>) {
    let outs: Vec<Box3> = boxes
        .iter()
        .map(|b| {
            let t = if keep_first {
                1 + (b.t - 1) * f[0]
            } else {
                b.t * f[0]
            };
            Box3::new(t, b.h * f[1], b.w * f[2])
        })
        .collect();
    let rows_out: usize = outs.iter().map(|b| b.voxels()).sum();
    let mut y = vec![0f32; rows_out * c];
    let mut in_off = 0usize;
    let mut out_off = 0usize;
    for (l, b) in boxes.iter().enumerate() {
        let ob = outs[l];
        for to in 0..ob.t {
            let ti = if keep_first {
                if to == 0 { 0 } else { (to - 1) / f[0] + 1 }
            } else {
                to / f[0]
            };
            for ho in 0..ob.h {
                for wo in 0..ob.w {
                    let m = out_off + (to * ob.h + ho) * ob.w + wo;
                    let src = in_off + (ti * b.h + ho / f[1]) * b.w + wo / f[2];
                    y[m * c..(m + 1) * c].copy_from_slice(&x[src * c..(src + 1) * c]);
                }
            }
        }
        in_off += b.voxels();
        out_off += ob.voxels();
    }
    (y, outs)
}

pub fn near(got: f32, want: f32, rel: f32, abs: f32) -> bool {
    (got - want).abs() <= rel * want.abs() + abs
}
