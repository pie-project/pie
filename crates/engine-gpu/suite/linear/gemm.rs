//! The dense matmuls on the first device against an f64 host reference: the
//! vector kernels at 1 to 16 rows on each workgroup's rows (an even
//! contraction in eight-wide loads, an odd one; odd column counts on
//! SPIR-V; at the subgroup tier too where every subgroup is 32 lanes), the
//! tiled kernel over ragged tiles, the register tile part-filled and past one at its
//! 64- and 32-column forms, the cooperative-matrix tier where the device has
//! one, and the f32 lane form at every weight and result dtype (f32 results
//! only on WGSL).

use kernels_slang::Tensor;
use kernels_slang::linear::gemm::{self, Tactic};
use poem_ir::Dtype;

use crate::gpu::{Gpu, landed, noise};

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
    // (rows, columns, contraction, tier, vector rows a workgroup (0 tiled),
    // the wide tile)
    let mut cases = vec![
        (3usize, 21usize, 60usize, kernels_slang::Tier::Baseline, 4u32, false),
        (2, 22, 61, kernels_slang::Tier::Baseline, 4, false),
        // The plain vector kernel past four rows.
        (9, 22, 61, kernels_slang::Tier::Baseline, 16, false),
        (40, 50, 70, kernels_slang::Tier::Baseline, 0, false),
        (70, 66, 33, kernels_slang::Tier::Baseline, 0, false),
        // The register tile, part-filled and past one tile: its 64- and
        // 32-column forms; the 32 x 32 tile past one register tile.
        (130, 70, 48, kernels_slang::Tier::Baseline, 0, true),
        (5, 96, 32, kernels_slang::Tier::Baseline, 0, true),
        (300, 96, 32, kernels_slang::Tier::Baseline, 0, true),
        (300, 66, 32, kernels_slang::Tier::Baseline, 0, false),
    ];
    // The eight-column kernel at 1 to 16 rows on each workgroup's rows:
    // a block of them, the last part-filled, past them.
    for m in 1..=16 {
        for vector in kernels_slang::linear::gemm::VECTOR_ROWS {
            cases.push((m, 22, 64, kernels_slang::Tier::Baseline, vector, false));
        }
    }
    if gpu.subgroup() >= 32 {
        cases.push((3, 22, 64, kernels_slang::Tier::Subgroup, 4, false));
    }
    if gpu.coopmat() {
        cases.push((70, 64, 96, kernels_slang::Tier::Coopmat, 0, false));
    }
    for (m, n, k, tier, vector, wide) in cases {
        if !crate::gpu::lands(n) {
            continue;
        }
        let (x, xr) = gpu.up(&noise(1 + m as u64, m * k));
        let (w, wr) = gpu.up(&noise(2 + n as u64, n * k));
        let (xt, wt) = (t(x, m, k, Dtype::Bf16), t(w, n, k, Dtype::Bf16));
        let op: fn(f64) -> f64 = if tier == kernels_slang::Tier::Coopmat {
            half
        } else {
            |v| v
        };
        let want = reference(&xr, &wr, (m, n, k), op);
        landed(
            &mut gpu,
            &format!(
                "gemm {m} x {n} x {k} at {} (vector {vector}, wide {wide})",
                tier.name()
            ),
            &|g| vec![(g.zeros(m * n), m * n, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, Dtype::Bf16);
                let tactic = Tactic { tier, vector, wide };
                g.run(|c| gemm::matmul(c, tactic, xt, wt, y, kernels_slang::ssm::RowSpan::whole(xt)));
            },
        );
    }
    // The f32 lane form.
    let (m, n, k) = (3usize, 20usize, 33usize);
    let xv = noise(3, m * k);
    let x = gpu.up_f32(&xv);
    let xt = t(x, m, k, Dtype::F32);
    for (wd, yd) in [
        (Dtype::Bf16, Dtype::F32),
        (Dtype::F32, Dtype::F32),
        (Dtype::Bf16, Dtype::Bf16),
        (Dtype::F32, Dtype::Bf16),
    ] {
        // The bf16 lane outputs are SPIR-V's.
        if yd == Dtype::Bf16 && !crate::backend::SPIRV {
            continue;
        }
        let wv = noise(4, n * k);
        let (w, held) = if wd == Dtype::Bf16 {
            gpu.up(&wv)
        } else {
            (gpu.up_f32(&wv), wv.clone())
        };
        let wt = t(w, n, k, wd);
        let want = reference(&xv, &held, (m, n, k), |v| v);
        let f32_out = yd == Dtype::F32;
        landed(
            &mut gpu,
            &format!("gemm lane {wd:?} weight into {yd:?}"),
            &|g| vec![(g.zeros(2 * m * n), m * n, f32_out)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, m, n, yd);
                g.run(|c| gemm::lane(c, "linear.matmul", xt, wt, y));
            },
        );
    }
}
