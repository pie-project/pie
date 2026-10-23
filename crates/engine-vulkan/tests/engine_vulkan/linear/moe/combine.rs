//! The mixture-of-experts combines on the first Vulkan device, beside the
//! hand-written kernels, against an f64 host reference: the weighted sum,
//! the bias sum and the sigmoid-gated add over an odd width.

use kernels_vulkan::{Tensor, linear::moe};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

fn sig(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

#[test]
fn the_moe_combines_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |b: u32, r: usize, w: usize, d: Dtype| Tensor::new(b, r as u32, w as u32, d);
    let (rows, experts, k) = (5usize, 64usize, 6usize);
    // The combines over 5 rows of 33 (an odd count).
    let w = 33usize;
    let n = rows * w;
    let (routed, rr) = gpu.up(&noise(4, rows * k * w));
    let rr = f64s(&rr);
    let rw: Vec<f32> = noise(5, rows * k);
    let rwb = gpu.up_f32(&rw);
    let want: Vec<f64> = (0..n)
        .map(|i| {
            (0..k)
                .map(|e| f64::from(rw[(i / w) * k + e]) * rr[((i / w) * k + e) * w + i % w])
                .sum()
        })
        .collect();
    let b16 = |b: u32, r: usize, wd: usize| t(b, r, wd, Dtype::Bf16);
    both(
        &mut gpu,
        "moe weighted sum",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::weighted_sum(
                    c,
                    b16(routed, rows * k, w),
                    t(rwb, rows, k, Dtype::F32),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::weighted_sum(
                    c,
                    st(b16(routed, rows * k, w)),
                    st(t(rwb, rows, k, Dtype::F32)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
    let (x, xr) = gpu.up(&noise(6, n));
    let (eb, ebr) = gpu.up(&noise(7, experts * w));
    let (xr, ebr) = (f64s(&xr), f64s(&ebr));
    let routes: Vec<i32> = (0..rows * k)
        .map(|i| {
            if i % 5 == 3 {
                -1
            } else {
                ((i * 11) % experts) as i32
            }
        })
        .collect();
    let rb = gpu.up_i32(&routes);
    let want: Vec<f64> = (0..n)
        .map(|i| {
            let r = i / w;
            xr[i]
                + (0..k)
                    .filter(|e| routes[r * k + e] >= 0)
                    .map(|e| f64::from(rw[r * k + e]) * ebr[routes[r * k + e] as usize * w + i % w])
                    .sum::<f64>()
        })
        .collect();
    both(
        &mut gpu,
        "moe bias sum",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::bias_sum(
                    c,
                    b16(x, rows, w),
                    b16(eb, experts, w),
                    t(rb, rows, k, Dtype::I32),
                    t(rwb, rows, k, Dtype::F32),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::bias_sum(
                    c,
                    st(b16(x, rows, w)),
                    st(b16(eb, experts, w)),
                    st(t(rb, rows, k, Dtype::I32)),
                    st(t(rwb, rows, k, Dtype::F32)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
    let (sh, shr) = gpu.up(&noise(8, n));
    let (gt, gtr) = gpu.up(&noise(9, rows));
    let (shr, gtr) = (f64s(&shr), f64s(&gtr));
    let want: Vec<f64> = (0..n).map(|i| xr[i] + sig(gtr[i / w]) * shr[i]).collect();
    both(
        &mut gpu,
        "moe sigmoid gate add",
        &|g| vec![(g.zeros(n + 1), n, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                moe::sigmoid_gate_add(
                    c,
                    b16(x, rows, w),
                    b16(sh, rows, w),
                    b16(gt, rows, 1),
                    b16(o[0].0, rows, w),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::linear::moe::combine::sigmoid_gate_add(
                    c,
                    st(b16(x, rows, w)),
                    st(b16(sh, rows, w)),
                    st(b16(gt, rows, 1)),
                    st(b16(o[0].0, rows, w)),
                )
            })
        },
    );
}
