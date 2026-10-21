//! y = act · wᵀ, bf16 in and out, f32 accumulation: the gemv at one row
//! (both of its tactics) and rocBLAS above it, against a host reference.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::linear::gemm;

fn reference(a: &[f32], w: &[f32], bias: Option<&[f32]>, m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0; m * n];
    for r in 0..m {
        for c in 0..n {
            let dot: f64 = (0..k)
                .map(|i| f64::from(a[r * k + i]) * f64::from(w[c * k + i]))
                .sum();
            // The device rounds the product, then adds the bias and rounds again.
            y[r * n + c] = match bias {
                Some(b) => round_bf16(dot as f32) + b[c],
                None => dot as f32,
            };
        }
    }
    y
}

#[test]
fn linear_matmul() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // (m, n, k): one row with a short N (split K) and a long one (a
    // wavefront per row); the few-row gemv, its activation in LDS and (at
    // 8 x 3584) not; and prefill-shaped blocks for hipBLASLt.
    for (m, n, k) in [
        (1u32, 1024u32, 2048u32),
        (1, 6144, 1024),
        (2, 1000, 1024),
        (4, 1024, 2048),
        (8, 6144, 1024),
        (8, 1024, 3584),
        // The skinny kernels' few rows past four (whole products on the
        // matrix core), a ragged N, and an activation past LDS (read from L2).
        (3, 1000, 2048),
        (5, 1030, 1024),
        (6, 4100, 5376),
        (2, 520, 17408),
        (8, 1024, 21504),
        (37, 1000, 1024),
        (64, 1024, 2048),
        // The decode batch's kernel: unsplit, split K, past 128 rows (the
        // narrower stage), a ragged N and M.
        (128, 4096, 1024),
        (128, 512, 4096),
        (200, 1024, 2048),
        (17, 784, 1024),
        (256, 96, 2048),
    ] {
        let a: Vec<f32> = values((m * k) as usize, 1, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let w: Vec<f32> = values((n * k) as usize, 2, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let at = gpu.tensor(&a, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, n, k, Dtype::Bf16);
        let mut y = gpu.zeros(m, n, Dtype::Bf16);
        gemm::matmul(&ctx, at, wt, &mut y).unwrap();
        // f32 accumulation over K terms in a different order: a couple of
        // bf16 steps of the result.
        close_bf16(
            &format!("matmul {m}x{n}x{k}"),
            &gpu.read(&y),
            &reference(&a, &w, None, m as usize, n as usize, k as usize),
            2.0,
        );

        let b: Vec<f32> = values(n as usize, 3, 0.5)
            .into_iter()
            .map(round_bf16)
            .collect();
        let bt = gpu.tensor(&b, 1, n, Dtype::Bf16);
        let mut y = gpu.zeros(m, n, Dtype::Bf16);
        gemm::matmul_bias(&ctx, at, wt, bt, &mut y).unwrap();
        // The sum can cancel: hold it to the operands' scale, two bf16
        // roundings (the product's and the sum's) of the larger operand.
        let got = gpu.read(&y);
        let dots = reference(&a, &w, None, m as usize, n as usize, k as usize);
        let want = reference(&a, &w, Some(&b), m as usize, n as usize, k as usize);
        for (i, (h, w)) in got.iter().zip(&want).enumerate() {
            let scale = dots[i].abs().max(b[i % n as usize].abs());
            assert!(
                (h - w).abs() <= 3.0 * scale / 256.0 + 1e-6,
                "matmul_bias {m}x{n}x{k}[{i}] = {h}, want {w}"
            );
        }
    }
}

/// The decode MLP's fused first half lands what the matmul and the swiglu
/// land apart: the same dots in the same order, rounded where the packed
/// plane would be.
#[test]
fn the_fused_swiglu_is_the_matmul_then_the_swiglu() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (m, i, k) in [
        (1u32, 3584u32, 1024u32),
        (4, 3584, 1024),
        (3, 1000, 2048),
        (8, 512, 1024),
        // Banks of at least 8192 rows (whole products) and
        // a K past LDS.
        (3, 4200, 2048),
        (4, 4600, 5120),
        (2, 300, 17408),
    ] {
        let a: Vec<f32> = values((m * k) as usize, 11, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let w: Vec<f32> = values((2 * i * k) as usize, 12, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let at = gpu.tensor(&a, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, 2 * i, k, Dtype::Bf16);
        let mut h = gpu.zeros(m, i, Dtype::Bf16);
        assert!(
            gemm::matmul_swiglu(&ctx, at, wt, &mut h).unwrap(),
            "{m}x{i}x{k} fused"
        );
        let mut packed = gpu.zeros(m, 2 * i, Dtype::Bf16);
        gemm::matmul(&ctx, at, wt, &mut packed).unwrap();
        let mut h2 = gpu.zeros(m, i, Dtype::Bf16);
        kernels_rocm::linear::mlp::swiglu(&ctx, packed, i, 1, &mut h2).unwrap();
        let (got, want) = (gpu.read(&h), gpu.read(&h2));
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "{m}x{i}x{k}: {diff} cells differ");
    }
}

/// The batch engine's projection (forced over every row count it covers):
/// one item and several, a ragged N, a K short of a whole stage, and the
/// fused SwiGLU against its own matmul then the swiglu.
#[test]
fn the_batch_engine_projection() {
    let Some(gpu) = Gpu::open() else { return };
    let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
    t.dense_batch_from = 1;
    t.dense_batch_until = 4096;
    t.dense_batch_weight_mi = 0;
    t.dense_batch_weight_max_mi = 65_536;
    let ctx = gpu.ctx().with_tuning(t);
    for (m, n, k) in [
        (8u32, 1000u32, 1024u32),
        (17, 784, 1032),
        (32, 4096, 2048),
        (64, 1024, 2048),
        (100, 520, 1096),
        (256, 192, 512),
    ] {
        let a: Vec<f32> = values((m * k) as usize, 21, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let w: Vec<f32> = values((n * k) as usize, 22, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let at = gpu.tensor(&a, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, n, k, Dtype::Bf16);
        let mut y = gpu.zeros(m, n, Dtype::Bf16);
        gemm::matmul(&ctx, at, wt, &mut y).unwrap();
        close_bf16(
            &format!("batch matmul {m}x{n}x{k}"),
            &gpu.read(&y),
            &reference(&a, &w, None, m as usize, n as usize, k as usize),
            2.0,
        );
    }
    for (m, i, k) in [(8u32, 512u32, 1024u32), (40, 1000, 2048), (200, 136, 1032)] {
        let a: Vec<f32> = values((m * k) as usize, 23, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let w: Vec<f32> = values((2 * i * k) as usize, 24, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let at = gpu.tensor(&a, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, 2 * i, k, Dtype::Bf16);
        let mut h = gpu.zeros(m, i, Dtype::Bf16);
        assert!(
            gemm::matmul_swiglu(&ctx, at, wt, &mut h).unwrap(),
            "{m}x{i}x{k} fused"
        );
        let mut packed = gpu.zeros(m, 2 * i, Dtype::Bf16);
        gemm::matmul(&ctx, at, wt, &mut packed).unwrap();
        let mut h2 = gpu.zeros(m, i, Dtype::Bf16);
        kernels_rocm::linear::mlp::swiglu(&ctx, packed, i, 1, &mut h2).unwrap();
        let (got, want) = (gpu.read(&h), gpu.read(&h2));
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "batch {m}x{i}x{k}: {diff} cells differ");
    }
}

#[test]
fn the_f32_projection_lands_unrounded() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (m, n, k) in [(1u32, 32u32, 4096u32), (200, 288, 6144)] {
        let a: Vec<f32> = values((m * k) as usize, 31, 1.0)
            .into_iter()
            .map(round_bf16)
            .collect();
        let w: Vec<f32> = values((n * k) as usize, 32, 0.05)
            .into_iter()
            .map(round_bf16)
            .collect();
        let at = gpu.tensor(&a, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, n, k, Dtype::Bf16);
        let mut y = gpu.zeros(m, n, Dtype::F32);
        gemm::matmul_f32(&ctx, at, wt, &mut y).unwrap();
        let want = reference(&a, &w, None, m as usize, n as usize, k as usize);
        for (i, (h, w)) in gpu.read(&y).iter().zip(&want).enumerate() {
            assert!(
                (h - w).abs() <= 1e-4 * w.abs().max(1.0),
                "matmul_f32 {m}x{n}x{k}[{i}] = {h}, want {w}"
            );
        }
    }
}
