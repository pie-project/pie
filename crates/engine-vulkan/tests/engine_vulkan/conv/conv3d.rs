//! The 3-D convolution on the first Vulkan device, against an f64 host
//! reference: two clips (a video and an image), strided and padded, time
//! padding zero-causal or repeated, with and without a bias, through the
//! strip kernel (`c_in` 6, a depthwise one, fewer than 16 channels out) and
//! the tiled one at its 64- and 32-channel tiles, taps spread in time, and
//! output rows past every clip landing zero.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

/// Two clips: (t, h, w).
const CLIPS: [[usize; 3]; 2] = [[3, 4, 5], [1, 3, 3]];
const K: [usize; 3] = [3, 3, 3];
const STRIDE: [usize; 3] = [1, 2, 2];

/// A clip table: each clip's extent and first row.
fn table(clips: &[[usize; 3]]) -> Vec<i32> {
    let mut off = 0;
    let mut v = Vec::new();
    for c in clips {
        v.extend([c[0] as i32, c[1] as i32, c[2] as i32, off as i32]);
        off += c[0] * c[1] * c[2];
    }
    v
}

struct Case {
    c_in: usize,
    c_out: usize,
    causal: bool,
    replicate: bool,
    bias: bool,
    dilation: usize,
    depthwise: bool,
}

impl Case {
    /// The kernel's extent, its time taps spread by `dilation`.
    fn spread(&self) -> [usize; 3] {
        [(K[0] - 1) * self.dilation + 1, K[1], K[2]]
    }

    fn pad(&self) -> ([usize; 3], [usize; 3]) {
        let t = self.spread()[0];
        if self.causal {
            ([t - 1, 1, 1], [0, 1, 1])
        } else {
            ([t / 2, 1, 1], [t / 2, 1, 1])
        }
    }

    /// Each output channel's weight row: every tap's input channels, or one
    /// a tap for a depthwise convolution.
    fn per(&self) -> usize {
        if self.depthwise { 1 } else { self.c_in }
    }

    fn out_clips(&self) -> Vec<[usize; 3]> {
        let (front, back) = self.pad();
        CLIPS
            .iter()
            .map(|c| {
                let mut o = [0; 3];
                let k = self.spread();
                for a in 0..3 {
                    o[a] = (c[a] + front[a] + back[a] - k[a]) / STRIDE[a] + 1;
                }
                o
            })
            .collect()
    }

    /// The reference rows, `extra` zero rows past the clips.
    fn reference(&self, x: &[f64], w: &[f64], bias: &[f64], extra: usize) -> Vec<f64> {
        let (front, _) = self.pad();
        let taps = K[0] * K[1] * K[2];
        let mut y = Vec::new();
        let mut in_off = 0;
        for (c, o) in CLIPS.iter().zip(self.out_clips()) {
            for t in 0..o[0] {
                for h in 0..o[1] {
                    for ww in 0..o[2] {
                        for n in 0..self.c_out {
                            let mut acc = 0.0;
                            for tap in 0..taps {
                                let (it, ih, iw) =
                                    (tap / (K[1] * K[2]), tap / K[2] % K[1], tap % K[2]);
                                let hi = (h * STRIDE[1] + ih) as isize - front[1] as isize;
                                let wi = (ww * STRIDE[2] + iw) as isize - front[2] as isize;
                                if hi < 0 || hi >= c[1] as isize || wi < 0 || wi >= c[2] as isize {
                                    continue;
                                }
                                let mut ti = (t * STRIDE[0] + it * self.dilation) as isize
                                    - front[0] as isize;
                                if ti < 0 {
                                    if !self.replicate {
                                        continue;
                                    }
                                    ti = 0;
                                }
                                if ti >= c[0] as isize {
                                    if self.causal || !self.replicate {
                                        continue;
                                    }
                                    ti = c[0] as isize - 1;
                                }
                                let row = in_off
                                    + (ti as usize * c[1] + hi as usize) * c[2]
                                    + wi as usize;
                                if self.depthwise {
                                    acc += x[row * self.c_in + n] * w[n * taps + tap];
                                    continue;
                                }
                                for ch in 0..self.c_in {
                                    acc += x[row * self.c_in + ch]
                                        * w[n * taps * self.c_in + tap * self.c_in + ch];
                                }
                            }
                            y.push(acc + if self.bias { bias[n] } else { 0.0 });
                        }
                    }
                }
            }
            in_off += c[0] * c[1] * c[2];
        }
        y.extend(std::iter::repeat_n(0.0, extra * self.c_out));
        y
    }
}

#[test]
fn the_conv3d_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let rows_in: usize = CLIPS.iter().map(|c| c[0] * c[1] * c[2]).sum();
    let taps = K[0] * K[1] * K[2];
    let case = |c_in, c_out, causal, replicate, bias| Case {
        c_in,
        c_out,
        causal,
        replicate,
        bias,
        dilation: 1,
        depthwise: false,
    };
    let cases = [
        case(6, 6, false, true, true),
        case(6, 6, true, false, false),
        case(16, 6, false, true, false),
        case(16, 64, false, true, false),
        case(16, 32, true, false, true),
        Case {
            dilation: 2,
            ..case(16, 64, true, false, true)
        },
        Case {
            dilation: 2,
            ..case(6, 6, true, false, false)
        },
        Case {
            depthwise: true,
            ..case(6, 6, true, false, true)
        },
    ];
    for (i, case) in cases.iter().enumerate() {
        let seed = 10 * i as u64;
        let (xb, xr) = gpu.up(&noise(seed + 1, rows_in * case.c_in));
        let (wb, wr) = gpu.up(&noise(seed + 2, case.c_out * taps * case.per()));
        let bias = noise(seed + 3, case.c_out);
        let bb = gpu.up_f32(&bias);
        let f64s = |v: &[f32]| v.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
        let extra = 2;
        let want = case.reference(&f64s(&xr), &f64s(&wr), &f64s(&bias), extra);
        let rows_out = want.len() / case.c_out;
        let gb = gpu.up_i32(&table(&CLIPS));
        let ob = gpu.up_i32(&table(&case.out_clips()));
        let x = t(xb, rows_in, case.c_in, Dtype::Bf16);
        let grid = t(gb, CLIPS.len(), 4, Dtype::I32);
        let o_grid = t(ob, CLIPS.len(), 4, Dtype::I32);
        let w = t(wb, case.c_out, taps * case.per(), Dtype::Bf16);
        let b = case.bias.then(|| t(bb, 1, case.c_out, Dtype::F32));
        let (front, back) = case.pad();
        let u3 = |v: [usize; 3]| v.map(|n| n as u32);
        let conv = kernels_slang::conv::conv3d::Conv3d {
            k: u3(K),
            stride: u3(STRIDE),
            pad: u3(front),
            pad_back: u3(back),
            causal_t: case.causal,
            time_pad: if case.replicate {
                kernels_slang::conv::conv3d::TimePad::Replicate
            } else {
                kernels_slang::conv::conv3d::TimePad::Zero
            },
            dilation: [case.dilation as u32, 1, 1],
            depthwise: case.depthwise,
        };
        let n = rows_out * case.c_out;
        landed(
            &mut gpu,
            &format!(
                "conv3d {} -> {}, causal {}, bias {}, dilation {}, depthwise {}",
                case.c_in, case.c_out, case.causal, case.bias, case.dilation, case.depthwise
            ),
            &|g| vec![(g.up(&vec![7.0; n]).0, n, false)],
            &[want],
            &|g, o| {
                g.run(|c| {
                    kernels_slang::conv::conv3d::conv3d(
                        c,
                        x,
                        grid,
                        w,
                        b,
                        conv,
                        o_grid,
                        t(o[0].0, rows_out, case.c_out, Dtype::Bf16),
                        [0, 0],
                    )
                })
            },
        );
    }
}
