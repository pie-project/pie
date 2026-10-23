//! The resamplers on the first Vulkan device, beside the hand-written
//! kernels, against a host reference: nearest upsampling (the first frame
//! kept alone or not), pixel shuffle (a frame trimmed) and unshuffle, in bf16
//! and in f32 (an odd row there), and the averaged downsample (time padded),
//! over two clips, with rows past every clip landing zero.

use kernels_vulkan::Tensor;
use kernels_vulkan::spatial::resample;
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

/// Two clips: (t, h, w).
const CLIPS: [[usize; 3]; 2] = [[3, 4, 4], [1, 2, 2]];
const EXTRA: usize = 2;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Upsample { keep_first: bool },
    Shuffle { trim: usize },
    Unshuffle,
    AvgDown { group: usize },
}

const R: [usize; 3] = [2, 2, 2];

impl Op {
    fn out_clip(self, c: [usize; 3]) -> [usize; 3] {
        match self {
            Op::Upsample { keep_first } => [
                if keep_first && c[0] > 0 {
                    1 + (c[0] - 1) * R[0]
                } else {
                    c[0] * R[0]
                },
                c[1] * R[1],
                c[2] * R[2],
            ],
            Op::Shuffle { trim } => [c[0] * R[0] - trim, c[1] * R[1], c[2] * R[2]],
            Op::Unshuffle => [c[0] / R[0], c[1] / R[1], c[2] / R[2]],
            Op::AvgDown { .. } => [c[0].div_ceil(R[0]), c[1] / R[1], c[2] / R[2]],
        }
    }

    /// Clips that divide: the unshuffle needs whole time blocks.
    fn clips(self) -> Vec<[usize; 3]> {
        match self {
            Op::Unshuffle => vec![[2, 4, 4], [2, 2, 2]],
            _ => CLIPS.to_vec(),
        }
    }

    fn out_width(self, c_in: usize) -> usize {
        let r = R[0] * R[1] * R[2];
        match self {
            Op::Upsample { .. } => c_in,
            Op::Shuffle { .. } => c_in / r,
            Op::Unshuffle => c_in * r,
            Op::AvgDown { group } => c_in * r / group,
        }
    }

    /// Output column `col` of voxel `(t, h, w)` from clip `c` at row `off`.
    fn value(
        self,
        x: &[f64],
        c_in: usize,
        c: [usize; 3],
        off: usize,
        v: [usize; 3],
        col: usize,
    ) -> f64 {
        let at = |t: usize, h: usize, w: usize, ch: usize| {
            x[(off + (t * c[1] + h) * c[2] + w) * c_in + ch]
        };
        let r = R[0] * R[1] * R[2];
        match self {
            Op::Upsample { keep_first } => {
                let t = if keep_first {
                    if v[0] == 0 { 0 } else { (v[0] - 1) / R[0] + 1 }
                } else {
                    v[0] / R[0]
                };
                at(t, v[1] / R[1], v[2] / R[2], col)
            }
            Op::Shuffle { trim } => {
                let ot = v[0] + trim;
                let block = ((ot % R[0]) * R[1] + v[1] % R[1]) * R[2] + v[2] % R[2];
                at(ot / R[0], v[1] / R[1], v[2] / R[2], col * r + block)
            }
            Op::Unshuffle => {
                let (cin, block) = (col / r, col % r);
                let (i1, i2, i3) = (block / (R[1] * R[2]), block / R[2] % R[1], block % R[2]);
                at(v[0] * R[0] + i1, v[1] * R[1] + i2, v[2] * R[2] + i3, cin)
            }
            Op::AvgDown { group } => {
                let pad = (R[0] - c[0] % R[0]) % R[0];
                let mut acc = 0.0;
                for j in 0..group {
                    let q = col * group + j;
                    let (cin, block) = (q / r, q % r);
                    let (i1, i2, i3) = (block / (R[1] * R[2]), block / R[2] % R[1], block % R[2]);
                    let ti = (v[0] * R[0] + i1) as isize - pad as isize;
                    if ti < 0 {
                        continue;
                    }
                    acc += at(ti as usize, v[1] * R[1] + i2, v[2] * R[2] + i3, cin);
                }
                acc / group as f64
            }
        }
    }
}

fn table(clips: &[[usize; 3]]) -> Vec<i32> {
    let mut off = 0;
    let mut v = Vec::new();
    for c in clips {
        v.extend([c[0] as i32, c[1] as i32, c[2] as i32, off as i32]);
        off += c[0] * c[1] * c[2];
    }
    v
}

#[test]
fn the_resamplers_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let cases = [
        (Op::Upsample { keep_first: true }, 4, Dtype::Bf16),
        (Op::Upsample { keep_first: false }, 3, Dtype::F32),
        (Op::Shuffle { trim: 1 }, 16, Dtype::Bf16),
        (Op::Shuffle { trim: 0 }, 24, Dtype::F32),
        (Op::Unshuffle, 2, Dtype::Bf16),
        (Op::Unshuffle, 3, Dtype::F32),
        (Op::AvgDown { group: 2 }, 2, Dtype::Bf16),
    ];
    for (i, (op, c_in, dtype)) in cases.into_iter().enumerate() {
        let clips = op.clips();
        let rows_in: usize = clips.iter().map(|c| c[0] * c[1] * c[2]).sum();
        let raw = noise(i as u64 + 1, rows_in * c_in);
        let (xb, xr) = if dtype == Dtype::Bf16 {
            let (b, r) = gpu.up(&raw);
            (b, r)
        } else {
            (gpu.up_f32(&raw), raw.clone())
        };
        let xr: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
        let outs: Vec<[usize; 3]> = clips.iter().map(|c| op.out_clip(*c)).collect();
        let width = op.out_width(c_in);
        let mut want = Vec::new();
        let mut off = 0;
        for (c, o) in clips.iter().zip(&outs) {
            for ot in 0..o[0] {
                for oh in 0..o[1] {
                    for ow in 0..o[2] {
                        for col in 0..width {
                            want.push(op.value(&xr, c_in, *c, off, [ot, oh, ow], col));
                        }
                    }
                }
            }
            off += c[0] * c[1] * c[2];
        }
        want.extend(std::iter::repeat_n(0.0, EXTRA * width));
        let rows_out = want.len() / width;
        let gb = gpu.up_i32(&table(&clips));
        let ob = gpu.up_i32(&table(&outs));
        let x = t(xb, rows_in, c_in, dtype);
        let grid = t(gb, clips.len(), 4, Dtype::I32);
        let o_grid = t(ob, clips.len(), 4, Dtype::I32);
        let y = |buf: u32| t(buf, rows_out, width, dtype);
        let r = R.map(|n| n as u32);
        let n = rows_out * width;
        let f32 = dtype == Dtype::F32;
        both(
            &mut gpu,
            &format!("{op:?} {dtype:?}"),
            &|g| {
                let filled = vec![7.0; n];
                vec![(
                    if f32 {
                        g.up_f32(&filled)
                    } else {
                        g.up(&filled).0
                    },
                    n,
                    f32,
                )]
            },
            &[want],
            &|g, o| {
                g.run(|c| match op {
                    Op::Upsample { keep_first } => {
                        resample::upsample_nearest(c, x, grid, r, keep_first, o_grid, y(o[0].0))
                    }
                    Op::Shuffle { trim } => {
                        resample::pixel_shuffle(c, x, grid, r, trim as u32, o_grid, y(o[0].0))
                    }
                    Op::Unshuffle => resample::pixel_unshuffle(c, x, grid, r, o_grid, y(o[0].0)),
                    Op::AvgDown { group } => {
                        resample::avg_down(c, x, grid, r, group as u32, o_grid, y(o[0].0))
                    }
                })
            },
            &|g, o| {
                use kernels_slang::conv::resample as sr;
                let (x, grid, o_grid, y) = (st(x), st(grid), st(o_grid), st(y(o[0].0)));
                g.run_slang(|c| match op {
                    Op::Upsample { keep_first } => {
                        sr::upsample_nearest(c, x, grid, r, keep_first, o_grid, y)
                    }
                    Op::Shuffle { trim } => {
                        sr::pixel_shuffle(c, x, grid, r, trim as u32, o_grid, y)
                    }
                    Op::Unshuffle => sr::pixel_unshuffle(c, x, grid, r, o_grid, y),
                    Op::AvgDown { group } => sr::avg_down(c, x, grid, r, group as u32, o_grid, y),
                })
            },
        );
    }
}
