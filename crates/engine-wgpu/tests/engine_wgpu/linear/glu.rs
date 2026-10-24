//! The gated linear units on the first wgpu adapter, against an f64 host
//! reference: the packed `[gate | up]` forms (SwiGLU, GeGLU, clamped,
//! GPT-OSS, SiTU) and the split ones over an odd count, a sentinel past it.

use kernels_slang::Tensor;
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

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
            landed(
                &mut gpu,
                &format!("{what} {inter}"),
                &|g| vec![(g.zeros(n), n, false)],
                &[want],
                &|g, o| {
                    let (pt, y) = (pt, b(o[0].0, rows, inter));
                    g.run(|cx| match k {
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
        landed(
            &mut gpu,
            what,
            &|g| vec![(g.up(&vec![7.0; n + 1]).0, n + 1, false)],
            &[want],
            &|g, o| {
                let (gt, ut, y) = (gt, ut, b(o[0].0, r, w));
                g.run(|cx| match k {
                    0 => kernels_slang::linear::glu::geglu_tanh(cx, gt, ut, y),
                    1 => kernels_slang::linear::glu::swiglu_clamp_split(cx, gt, ut, limit, y),
                    _ => kernels_slang::linear::glu::gelu_tanh(cx, gt, y),
                })
            },
        );
    }
}

/// Legs the pack left apart: each form over a gate and an up plane lands
/// what it lands over the `[gate | up]` rows of the same values, bit for
/// bit, an unclamped one clamped at the largest finite f32.
#[test]
fn a_gate_and_an_up_apart_land_what_they_land_packed() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let b = |buf: u32, r: usize, w: usize| Tensor::new(buf, r as u32, w as u32, Dtype::Bf16);
    let (rows, inter) = (3usize, 1000usize);
    let n = rows * inter;
    let gv: Vec<f32> = noise(11, n).iter().map(|v| v * 4.0).collect();
    let uv: Vec<f32> = noise(12, n).iter().map(|v| v * 4.0).collect();
    let pv: Vec<f32> = (0..rows)
        .flat_map(|r| {
            gv[r * inter..(r + 1) * inter]
                .iter()
                .chain(&uv[r * inter..(r + 1) * inter])
                .copied()
                .collect::<Vec<_>>()
        })
        .collect();
    let (gate, _) = gpu.up(&gv);
    let (up, _) = gpu.up(&uv);
    let (packed, _) = gpu.up(&pv);
    let (gt, ut, pt) = (
        b(gate, rows, inter),
        b(up, rows, inter),
        b(packed, rows, 2 * inter),
    );
    let iu = inter as u32;
    let (limit, alpha, beta, cap) = (2.5f32, 1.702f32, 3.0f32, 4.0f32);
    let inf = f32::MAX;
    let forms = [
        "swiglu",
        "swiglu clamp",
        "swiglu clamp alpha",
        "swiglu alpha",
        "situ",
        "situ capped",
        "geglu",
    ];
    for (k, what) in forms.iter().enumerate() {
        let (apart, joined) = (gpu.zeros(n), gpu.zeros(n));
        let (ya, yj) = (b(apart, rows, inter), b(joined, rows, inter));
        gpu.run(|cx| {
            use kernels_slang::linear::glu as g;
            match k {
                0 => g::swiglu_clamp_split(cx, gt, ut, inf, ya)?,
                1 => g::swiglu_clamp_split(cx, gt, ut, limit, ya)?,
                2 => g::swiglu_clamp_alpha_split(cx, gt, ut, limit, alpha, ya)?,
                3 => g::swiglu_clamp_alpha_split(cx, gt, ut, inf, alpha, ya)?,
                4 => g::situ_split(cx, gt, ut, beta, None, ya)?,
                5 => g::situ_split(cx, gt, ut, beta, Some(cap), ya)?,
                _ => g::geglu_tanh(cx, gt, ut, ya)?,
            }
            match k {
                0 => g::swiglu(cx, pt, iu, yj),
                1 => g::swiglu_clamp(cx, pt, iu, limit, yj),
                2 => g::swiglu_clamp_alpha(cx, pt, iu, limit, alpha, yj),
                3 => g::swiglu_clamp_alpha(cx, pt, iu, inf, alpha, yj),
                4 => g::situ(cx, pt, iu, beta, None, yj),
                5 => g::situ(cx, pt, iu, beta, Some(cap), yj),
                _ => g::geglu_tanh_packed(cx, pt, iu, yj),
            }
        });
        let (a, j) = (gpu.down(apart, n), gpu.down(joined, n));
        let off = a
            .iter()
            .zip(&j)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(
            off, 0,
            "{what}: {off} of {n} values differ apart and joined"
        );
    }
}
