//! The clip-table rules on the first wgpu adapter, against a host reference,
//! exactly: conv (causal and not),
//! upsample (the first frame kept alone or not), shuffle (a frame trimmed),
//! unshuffle and avg-down, over three clips one of which a rule cannot shape
//! (it lands empty) and the running row offsets after it.

use kernels_slang::Tensor;
use kernels_slang::conv::rule as slang;
use poem_ir::Dtype;

use crate::gpu::Gpu;

const CLIPS: [[i32; 3]; 3] = [[3, 4, 6], [1, 3, 3], [2, 2, 4]];

fn conv_axis(n: i32, k: i32, s: i32, front: i32, back: i32) -> Option<i32> {
    let span = n + front + back - k;
    (span >= 0).then(|| span / s.max(1) + 1)
}

/// The rule's clip `(t, h, w)`, `None` when it cannot shape it.
fn shaped(rule: slang::GridRule, [t, h, w]: [i32; 3]) -> Option<[i32; 3]> {
    let i = |v: [u32; 3]| v.map(|n| n as i32);
    match rule {
        slang::GridRule::Conv {
            k,
            stride,
            pad,
            pad_back,
            causal_t,
        } => {
            let (k, s, f, b) = (i(k), i(stride), i(pad), i(pad_back));
            let back_t = if causal_t { 0 } else { b[0] };
            Some([
                conv_axis(t, k[0], s[0], f[0], back_t)?,
                conv_axis(h, k[1], s[1], f[1], b[1])?,
                conv_axis(w, k[2], s[2], f[2], b[2])?,
            ])
        }
        slang::GridRule::Upsample {
            factor,
            keep_first_frame,
        } => {
            let a = i(factor);
            let ot = if keep_first_frame && t > 0 {
                1 + (t - 1) * a[0]
            } else {
                t * a[0]
            };
            Some([ot, h * a[1], w * a[2]])
        }
        slang::GridRule::Shuffle { r, trim_t } => {
            let a = i(r);
            let ot = t * a[0] - trim_t as i32;
            (ot > 0).then_some([ot, h * a[1], w * a[2]])
        }
        slang::GridRule::Unshuffle { r } => {
            let a = i(r);
            (t % a[0] == 0 && h % a[1] == 0 && w % a[2] == 0).then_some([
                t / a[0],
                h / a[1],
                w / a[2],
            ])
        }
        slang::GridRule::AvgDown { factor } => {
            let a = i(factor);
            (h % a[1] == 0 && w % a[2] == 0).then_some([(t + a[0] - 1) / a[0], h / a[1], w / a[2]])
        }
    }
}

fn ints(gpu: &Gpu, buf: u32, n: usize) -> Vec<i32> {
    gpu.down_f32(buf, n)
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect()
}

#[test]
fn the_clip_rules_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let mut table = Vec::new();
    let mut off = 0;
    for c in CLIPS {
        table.extend([c[0], c[1], c[2], off]);
        off += c[0] * c[1] * c[2];
    }
    let gb = gpu.up_i32(&table);
    let t = |buf: u32| Tensor::new(buf, CLIPS.len() as u32, 4, Dtype::I32);
    let rules = [
        slang::GridRule::Conv {
            k: [3, 3, 3],
            stride: [1, 2, 2],
            pad: [1, 1, 1],
            pad_back: [1, 1, 1],
            causal_t: false,
        },
        slang::GridRule::Conv {
            k: [3, 5, 5],
            stride: [2, 1, 1],
            pad: [2, 0, 0],
            pad_back: [5, 0, 0],
            causal_t: true,
        },
        slang::GridRule::Upsample {
            factor: [2, 2, 2],
            keep_first_frame: true,
        },
        slang::GridRule::Upsample {
            factor: [1, 2, 3],
            keep_first_frame: false,
        },
        slang::GridRule::Shuffle {
            r: [2, 2, 2],
            trim_t: 1,
        },
        slang::GridRule::Shuffle {
            r: [1, 2, 2],
            trim_t: 1,
        },
        slang::GridRule::Unshuffle { r: [1, 2, 2] },
        slang::GridRule::AvgDown { factor: [2, 2, 2] },
    ];
    for rule in rules {
        let mut want = Vec::new();
        let mut off = 0;
        for c in CLIPS {
            let [ot, oh, ow] = shaped(rule, c).unwrap_or([0; 3]);
            want.extend([ot, oh, ow, off]);
            off += ot * oh * ow;
        }
        let ob = gpu.up_i32(&[-7; 4 * CLIPS.len()]);
        gpu.run(|c| slang::derive_grid(c, t(gb), rule, t(ob)));
        assert_eq!(ints(&gpu, ob, 4 * CLIPS.len()), want, "{rule:?}");
    }
}
