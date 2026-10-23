//! The attention epilogues on the first Vulkan device, beside the hand-written
//! kernels, against an f64 host reference: two readings merged by their
//! log-sum-exps (one side empty in places), the sink rescale, and the logit
//! softcap over an odd count.

use kernels_vulkan::{Tensor, attn};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[test]
fn the_attention_epilogues_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let (rows, heads, hd) = (3usize, 4usize, 64usize);
    let width = heads * hd;
    let n = rows * width;
    let b = |buf: u32| Tensor::new(buf, rows as u32, width as u32, Dtype::Bf16);
    let f = |buf: u32| Tensor::new(buf, rows as u32, heads as u32, Dtype::F32);

    // merge: column 1 has an empty second reading, column 6 an empty first.
    let (o1, r1) = gpu.up(&noise(1, n));
    let (o2, r2) = gpu.up(&noise(2, n));
    let (r1, r2) = (f64s(&r1), f64s(&r2));
    let mut l1: Vec<f32> = noise(3, rows * heads).iter().map(|v| v * 3.0).collect();
    let mut l2: Vec<f32> = noise(4, rows * heads).iter().map(|v| v * 3.0).collect();
    l2[1] = f32::NEG_INFINITY;
    l1[6] = f32::NEG_INFINITY;
    let (lse1, lse2) = (gpu.up_f32(&l1), gpu.up_f32(&l2));
    let mut want_o = vec![0.0; n];
    let mut want_l = vec![0.0; rows * heads];
    for col in 0..rows * heads {
        let (a, c) = (f64::from(l1[col]), f64::from(l2[col]));
        let (w1, w2, l) = if !c.is_finite() {
            (1.0, 0.0, a)
        } else if !a.is_finite() {
            (0.0, 1.0, c)
        } else {
            let m = a.max(c);
            let (w1, w2) = ((a - m).exp2(), (c - m).exp2());
            (w1 / (w1 + w2), w2 / (w1 + w2), m + (w1 + w2).log2())
        };
        want_l[col] = l;
        for d in 0..hd {
            let i = col * hd + d;
            want_o[i] = r1[i] * w1 + r2[i] * w2;
        }
    }
    both(
        &mut gpu,
        "merge lse",
        &|g| {
            vec![
                (g.zeros(n), n, false),
                (g.zeros(2 * rows * heads), rows * heads, true),
            ]
        },
        &[want_o, want_l],
        &|g, o| {
            g.run(|c| {
                attn::merge_lse(
                    c,
                    b(o1),
                    f(lse1),
                    b(o2),
                    f(lse2),
                    heads as u32,
                    hd as u32,
                    b(o[0].0),
                    f(o[1].0),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::merge::merge_lse(
                    c,
                    st(b(o1)),
                    st(f(lse1)),
                    st(b(o2)),
                    st(f(lse2)),
                    heads as u32,
                    hd as u32,
                    st(b(o[0].0)),
                    st(f(o[1].0)),
                )
            })
        },
    );

    // sink: column 2's reading is empty and keeps its output.
    let sv = noise(5, heads);
    let (sinks, sr) = gpu.up(&sv);
    let mut lv: Vec<f32> = noise(6, rows * heads).iter().map(|v| v * 2.0).collect();
    lv[2] = f32::INFINITY;
    let lse = gpu.up_f32(&lv);
    let want: Vec<f64> = (0..n)
        .map(|i| {
            let col = i / hd;
            let l = f64::from(lv[col]);
            let r = if l.is_finite() {
                1.0 / (1.0 + (-(l * std::f64::consts::LN_2 - f64::from(sr[col % heads]))).exp())
            } else {
                1.0
            };
            r1[i] * r
        })
        .collect();
    let sink_t = Tensor::new(sinks, 1, heads as u32, Dtype::Bf16);
    both(
        &mut gpu,
        "sink",
        &|g| vec![(g.up(&noise(1, n)).0, n, false)],
        &[want],
        &|g, o| g.run(|c| attn::sink(c, b(o[0].0), f(lse), sink_t, hd as u32)),
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::merge::sink(
                    c,
                    st(b(o[0].0)),
                    st(f(lse)),
                    st(sink_t),
                    hd as u32,
                )
            })
        },
    );

    // softcap over 3 x 7 = 21 values, a sentinel past them.
    let mut xv: Vec<f32> = noise(7, 21).iter().map(|v| v * 40.0).collect();
    xv.push(5.0);
    let held = f64s(&gpu.up(&xv).1);
    let cap = 30.0f64;
    let mut want: Vec<f64> = held.iter().map(|v| cap * (v / cap).tanh()).collect();
    want[21] = 5.0;
    let xt = |buf: u32| Tensor::new(buf, 3, 7, Dtype::Bf16);
    both(
        &mut gpu,
        "logit softcap",
        &|g| vec![(g.up(&xv).0, 22, false)],
        &[want],
        &|g, o| g.run(|c| attn::logit_softcap(c, xt(o[0].0), cap as f32)),
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::merge::logit_softcap(c, st(xt(o[0].0)), cap as f32)
            })
        },
    );
}
