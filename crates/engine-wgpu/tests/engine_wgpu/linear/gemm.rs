//! The dense matmuls on the first wgpu adapter, beside the hand-written
//! kernels, against an f64 host reference: the vector kernel below four
//! rows (at the subgroup tier too where the adapter has one), the 16-row
//! tile below 48 rows and the 64-row tile above, over ragged tiles and an
//! odd contraction.

use kernels_slang::Tier;
use kernels_slang::linear::gemm::{self, Tactic};
use kernels_wgpu::{Arg, Encode, Fire, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

#[test]
fn the_dense_matmuls_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let subgroups = gpu.device.chip().subgroups_32();
    // (rows, columns, contraction, tier, the hand-written entry, groups, workgroup)
    let mut cases = vec![
        (
            3usize,
            22usize,
            64usize,
            Tier::Baseline,
            "dense_gemv_t_bf16",
            [1u32, 3, 3],
            [32u32, 8, 1],
        ),
        (
            2,
            22,
            61,
            Tier::Baseline,
            "dense_gemv_t_bf16",
            [1, 3, 2],
            [32, 8, 1],
        ),
        (
            40,
            50,
            70,
            Tier::Baseline,
            "dense_gemm_t16_bf16",
            [1, 3, 1],
            [16, 16, 1],
        ),
        (
            70,
            66,
            33,
            Tier::Baseline,
            "dense_gemm_t_bf16",
            [2, 2, 1],
            [16, 16, 1],
        ),
    ];
    if subgroups {
        cases.push((
            3,
            22,
            64,
            Tier::Subgroup,
            "dense_gemv_t_bf16",
            [1, 3, 3],
            [32, 8, 1],
        ));
    }
    for (m, n, k, tier, hand, groups, local) in cases {
        let (x, xr) = gpu.up(&noise(1 + m as u64, m * k));
        let (w, wr) = gpu.up(&noise(2 + n as u64, n * k));
        let (xt, wt) = (t(x, m, k, Dtype::Bf16), t(w, n, k, Dtype::Bf16));
        let want: Vec<f64> = (0..m * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                (0..k)
                    .map(|j| f64::from(xr[r * k + j]) * f64::from(wr[c * k + j]))
                    .sum()
            })
            .collect();
        both(
            &mut gpu,
            &format!("gemm {m} x {n} x {k} at {}", tier.name()),
            &|g| vec![(g.zeros(m * n), m * n, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, Dtype::Bf16);
                g.run(|c| {
                    c.fire(
                        Fire::at("gemm/dense.wgsl", hand)
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
}
