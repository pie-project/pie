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
        (
            3usize,
            21usize,
            60usize,
            kernels_slang::Tier::Baseline,
            4u32,
            false,
        ),
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
                g.run(|c| gemm::matmul(c, Tactic { tier, vector, wide }, xt, wt, y));
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

/// The vector kernels at every rows a workgroup carries land the same bits
/// at every row count (each row's sum the same order: a lane's strided
/// eight-wide steps, four a loop, then the workgroup's tree), and they and
/// the tiled kernels stay within the reference's tolerance at a decode's
/// and a short prompt's rows over projection-shaped planes; each kernel's
/// worst error against the f64 reference is said.
#[test]
fn the_vector_kernels_agree_bit_for_bit_and_with_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let tier = kernels_slang::Tier::Baseline;
    for (n, k) in [(1024usize, 1024usize), (3072, 1024), (1024, 3072)] {
        let (w, wr) = gpu.up(&noise(7 + n as u64 + k as u64, n * k));
        let wt = t(w, n, k, Dtype::Bf16);
        for m in [1usize, 2, 3, 8, 16, 32] {
            let (x, xr) = gpu.up(&noise(11 + m as u64, m * k));
            let xt = t(x, m, k, Dtype::Bf16);
            let want = reference(&xr, &wr, (m, n, k), |v| v);
            let run = |gpu: &mut Gpu, vector: u32, wide: bool| {
                let y = gpu.zeros(m * n);
                let yt = t(y, m, n, Dtype::Bf16);
                gpu.run(|c| gemm::matmul(c, Tactic { tier, vector, wide }, xt, wt, yt));
                gpu.down(y, m * n)
            };
            let worst = |got: &[f32]| {
                got.iter()
                    .zip(&want)
                    .map(|(&g, &w)| (f64::from(g) - w).abs())
                    .fold(0.0, f64::max)
            };
            let one = run(&mut gpu, 1, false);
            let mut said = vec![format!("x1 {:.3e}", worst(&one))];
            for vector in [4, 8, 16] {
                let got = run(&mut gpu, vector, false);
                assert!(
                    got.iter()
                        .zip(&one)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{m} x {n} x {k}: the {vector}-row vector kernel's bits differ from the 1-row one's"
                );
                said.push(format!("x{vector} {:.3e}", worst(&got)));
            }
            for (wide, name) in [(false, "tiled"), (true, "wide")] {
                let got = run(&mut gpu, 0, wide);
                said.push(format!("{name} {:.3e}", worst(&got)));
                crate::gpu::close(&got, &want, &format!("{name} {m} x {n} x {k}"));
            }
            crate::gpu::close(&one, &want, &format!("vector {m} x {n} x {k}"));
            eprintln!("{m} x {n} x {k}: worst |error| vs f64: {}", said.join(", "));
        }
    }
}
