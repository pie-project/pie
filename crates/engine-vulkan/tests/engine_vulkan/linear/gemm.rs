//! The dense matmuls on the first Vulkan device, beside the hand-written
//! kernels, against an f64 host reference: the vector kernels below four
//! rows (an even contraction in eight-wide loads, an odd one), the tiled
//! kernel over ragged tiles, the cooperative-matrix tier where the device
//! has one, and the f32 lane form at every weight and result dtype.

use kernels_slang::linear::gemm::{self, Tactic};
use kernels_vulkan::{Arg, Encode, Fire, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

/// `v` rounded to the nearest IEEE half: the cooperative-matrix tier
/// multiplies halves, so its reference does too.
fn half(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let quantum = 2f64.powi((v.abs().log2().floor() as i32).max(-14) - 10);
    (v / quantum).round_ties_even() * quantum
}

fn reference(
    x: &[f32],
    w: &[f32],
    (m, n, k): (usize, usize, usize),
    op: fn(f64) -> f64,
) -> Vec<f64> {
    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            (0..k)
                .map(|j| op(f64::from(x[r * k + j])) * op(f64::from(w[c * k + j])))
                .sum()
        })
        .collect()
}

#[test]
fn the_dense_matmuls_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let coopmat = gpu.device.chip().coopmat;
    // (rows, columns, contraction, tier, the hand-written entry, groups, workgroup)
    let mut cases = vec![
        (
            1usize,
            24usize,
            64usize,
            kernels_slang::Tier::Baseline,
            "dense_gemv_t_v_bf16",
            [3u32, 1, 1],
            [32u32, 8, 1],
        ),
        (
            3,
            21,
            60,
            kernels_slang::Tier::Baseline,
            "dense_gemv_t_bf16",
            [1, 3, 3],
            [32, 8, 1],
        ),
        (
            40,
            50,
            70,
            kernels_slang::Tier::Baseline,
            "dense_gemm_t_bf16",
            [2, 2, 1],
            [32, 2, 2],
        ),
    ];
    if coopmat {
        cases.push((
            70,
            64,
            96,
            kernels_slang::Tier::Coopmat,
            "dense_gemm_t_cm_bf16",
            [1, 2, 1],
            [32, 4, 1],
        ));
    }
    for (m, n, k, tier, hand, groups, local) in cases {
        let (x, xr) = gpu.up(&noise(1 + m as u64, m * k));
        let (w, wr) = gpu.up(&noise(2 + n as u64, n * k));
        let (xt, wt) = (t(x, m, k, Dtype::Bf16), t(w, n, k, Dtype::Bf16));
        let op: fn(f64) -> f64 = if tier == kernels_slang::Tier::Coopmat {
            half
        } else {
            |v| v
        };
        let want = reference(&xr, &wr, (m, n, k), op);
        both(
            &mut gpu,
            &format!("gemm {m} x {n} x {k} at {}", tier.name()),
            &|g| vec![(g.zeros(m * n), m * n, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, Dtype::Bf16);
                g.run(|c| {
                    c.fire(
                        Fire::at("gemm/dense.slang", hand)
                            .groups(groups)
                            .group(local),
                        &[
                            xt.arg(),
                            wt.arg(),
                            y.arg_mut(),
                            (m as i32).arg(),
                            (n as i32).arg(),
                            (k as i32).arg(),
                        ],
                    )
                });
            },
            &|g, o| {
                let y = st(t(o[0].0, m, n, Dtype::Bf16));
                g.run_slang(|c| gemm::matmul(c, Tactic { tier }, st(xt), st(wt), y));
            },
        );
    }
    // The f32 lane form.
    let (m, n, k) = (3usize, 20usize, 33usize);
    let xv = noise(3, m * k);
    let x = gpu.up_f32(&xv);
    let xt = t(x, m, k, Dtype::F32);
    for (wd, yd, hand) in [
        (Dtype::Bf16, Dtype::F32, "lane_gemm_f32_bf16_f32"),
        (Dtype::F32, Dtype::F32, "lane_gemm_f32_f32_f32"),
        (Dtype::Bf16, Dtype::Bf16, "lane_gemm_f32_bf16_bf16"),
        (Dtype::F32, Dtype::Bf16, "lane_gemm_f32_f32_bf16"),
    ] {
        let wv = noise(4, n * k);
        let (w, held) = if wd == Dtype::Bf16 {
            gpu.up(&wv)
        } else {
            (gpu.up_f32(&wv), wv.clone())
        };
        let wt = t(w, n, k, wd);
        let want = reference(&xv, &held, (m, n, k), |v| v);
        let f32_out = yd == Dtype::F32;
        both(
            &mut gpu,
            &format!("gemm lane {wd:?} weight into {yd:?}"),
            &|g| vec![(g.zeros(2 * m * n), m * n, f32_out)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, yd);
                g.run(|c| {
                    c.fire(
                        Fire::at("gemm/lane_gemm.slang", hand)
                            .groups([1, 1, 1])
                            .group([64, 4, 1]),
                        &[
                            xt.arg(),
                            wt.arg(),
                            y.arg_mut(),
                            (m as i32).arg(),
                            (n as i32).arg(),
                            (k as i32).arg(),
                        ],
                    )
                });
            },
            &|g, o| {
                let y = st(t(o[0].0, m, n, yd));
                g.run_slang(|c| gemm::lane(c, "linear.matmul", st(xt), st(wt), y));
            },
        );
    }
}
