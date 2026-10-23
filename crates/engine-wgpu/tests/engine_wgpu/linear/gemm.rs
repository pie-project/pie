//! The dense matmuls on the first wgpu adapter against an f64 host
//! reference: the vector kernel below four
//! rows (at the subgroup tier too where the adapter has one), the 16-row
//! tile below 48 rows and the 64-row tile above, over ragged tiles and an
//! odd contraction.

use kernels_slang::Tensor;
use kernels_slang::Tier;
use kernels_slang::linear::gemm::{self, Tactic};
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

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
    // (rows, columns, contraction, tier)
    let mut cases = vec![
        (3usize, 22usize, 64usize, Tier::Baseline),
        (2, 22, 61, Tier::Baseline),
        (40, 50, 70, Tier::Baseline),
        (70, 66, 33, Tier::Baseline),
    ];
    if subgroups {
        cases.push((3, 22, 64, Tier::Subgroup));
    }
    for (m, n, k, tier) in cases {
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
        landed(
            &mut gpu,
            &format!("gemm {m} x {n} x {k} at {}", tier.name()),
            &|g| vec![(g.zeros(m * n), m * n, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, Dtype::Bf16);
                g.run(|c| gemm::matmul(c, Tactic { tier }, xt, wt, y));
            },
        );
    }
}
