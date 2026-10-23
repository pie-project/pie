//! The gated linear units on the first wgpu adapter, beside the hand-written kernels,
//! against an f64 host reference: the packed `[gate | up]` forms (SwiGLU,
//! GeGLU, clamped, GPT-OSS, SiTU) and the split ones over an odd count, a
//! sentinel past it.

use kernels_wgpu::{Tensor, linear::mlp};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

/// A unit of the gate and the up, as the reference reads it.
type Unit = Box<dyn Fn(f64, f64) -> f64>;

fn sig(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

fn gelu(v: f64) -> f64 {
    0.5 * v * (1.0 + (0.797_884_560_802_865_4 * (v + 0.044715 * v * v * v)).tanh())
}

#[test]
fn the_glus_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let b = |buf: u32, r: usize, w: usize| Tensor::new(buf, r as u32, w as u32, Dtype::Bf16);
    let (limit, alpha, beta, cap) = (2.5f32, 1.702f32, 3.0f32, 4.0f32);
    let (l, a, be, c) = (
        f64::from(limit),
        f64::from(alpha),
        f64::from(beta),
        f64::from(cap),
    );
    let rows = 3usize;
    for inter in [64usize, 1000] {
        let (p, pr) = gpu.up(&noise(inter as u64, rows * 2 * inter)
            .iter()
            .map(|v| v * 2.0)
            .collect::<Vec<_>>());
        let pr: Vec<f64> = pr.iter().map(|v| f64::from(*v)).collect();
        let at = |i: usize| {
            (
                pr[(i / inter) * 2 * inter + i % inter],
                pr[(i / inter) * 2 * inter + inter + i % inter],
            )
        };
        let n = rows * inter;
        let forms: [(&str, Unit); 5] = [
            ("swiglu", Box::new(|g, u| g * sig(g) * u)),
            ("geglu packed", Box::new(|g, u| gelu(g) * u)),
            (
                "swiglu clamp",
                Box::new(move |g, u| g.min(l) * sig(g.min(l)) * u.clamp(-l, l)),
            ),
            (
                "swiglu clamp alpha",
                Box::new(move |g, u| g.min(l) * sig(a * g.min(l)) * (u.clamp(-l, l) + 1.0)),
            ),
            (
                "situ",
                Box::new(move |g, u| be * (g / be).tanh() * sig(g) * c * (u / c).tanh()),
            ),
        ];
        for (k, (what, f)) in forms.iter().enumerate() {
            let want: Vec<f64> = (0..n)
                .map(|i| {
                    let (g, u) = at(i);
                    f(g, u)
                })
                .collect();
            let (pt, iu) = (b(p, rows, 2 * inter), inter as u32);
            both(
                &mut gpu,
                &format!("{what} {inter}"),
                &|g| vec![(g.zeros(n), n, false)],
                &[want],
                &|g, o| {
                    let y = b(o[0].0, rows, inter);
                    g.run(|cx| match k {
                        0 => mlp::swiglu(cx, pt, iu, y),
                        1 => mlp::geglu_tanh_packed(cx, pt, iu, y),
                        2 => mlp::swiglu_clamp(cx, pt, iu, limit, y),
                        3 => mlp::swiglu_clamp_alpha(cx, pt, iu, limit, alpha, y),
                        _ => mlp::situ(cx, pt, iu, beta, Some(cap), y),
                    })
                },
                &|g, o| {
                    let (pt, y) = (st(pt), st(b(o[0].0, rows, inter)));
                    g.run_slang(|cx| match k {
                        0 => kernels_slang::linear::glu::swiglu(cx, pt, iu, y),
                        1 => kernels_slang::linear::glu::geglu_tanh_packed(cx, pt, iu, y),
                        2 => kernels_slang::linear::glu::swiglu_clamp(cx, pt, iu, limit, y),
                        3 => kernels_slang::linear::glu::swiglu_clamp_alpha(
                            cx, pt, iu, limit, alpha, y,
                        ),
                        _ => kernels_slang::linear::glu::situ(cx, pt, iu, beta, Some(cap), y),
                    })
                },
            );
        }
    }
    // The split forms over 3 x 7, a sentinel past the 21 values.
    let (r, w) = (3usize, 7usize);
    let n = r * w;
    let mut gv: Vec<f32> = noise(5, n).iter().map(|v| v * 2.0).collect();
    gv.push(0.0);
    let (gate, gr) = gpu.up(&gv);
    let (up, ur) = gpu.up(&noise(6, n + 1));
    let (gr, ur): (Vec<f64>, Vec<f64>) = (
        gr.iter().map(|v| f64::from(*v)).collect(),
        ur.iter().map(|v| f64::from(*v)).collect(),
    );
    let forms: [(&str, Unit); 3] = [
        ("geglu split", Box::new(|g, u| gelu(g) * u)),
        (
            "swiglu clamp split",
            Box::new(move |g, u| g.min(l) * sig(g.min(l)) * u.clamp(-l, l)),
        ),
        ("gelu", Box::new(|g, _| gelu(g))),
    ];
    for (k, (what, f)) in forms.iter().enumerate() {
        let mut want: Vec<f64> = (0..n).map(|i| f(gr[i], ur[i])).collect();
        want.push(7.0);
        let (gt, ut) = (b(gate, r, w), b(up, r, w));
        both(
            &mut gpu,
            what,
            &|g| vec![(g.up(&vec![7.0; n + 1]).0, n + 1, false)],
            &[want],
            &|g, o| {
                let y = b(o[0].0, r, w);
                g.run(|cx| match k {
                    0 => mlp::geglu_tanh(cx, gt, ut, y),
                    1 => mlp::swiglu_clamp_split(cx, gt, ut, limit, y),
                    _ => mlp::gelu_tanh(cx, gt, y),
                })
            },
            &|g, o| {
                let (gt, ut, y) = (st(gt), st(ut), st(b(o[0].0, r, w)));
                g.run_slang(|cx| match k {
                    0 => kernels_slang::linear::glu::geglu_tanh(cx, gt, ut, y),
                    1 => kernels_slang::linear::glu::swiglu_clamp_split(cx, gt, ut, limit, y),
                    _ => kernels_slang::linear::glu::gelu_tanh(cx, gt, y),
                })
            },
        );
    }
}
