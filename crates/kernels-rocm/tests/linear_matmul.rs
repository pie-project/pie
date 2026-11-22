//! y = act · wᵀ, bf16 in and out, f32 accumulation: the gemv at one row
//! (both of its tactics) and rocBLAS above it, against a host reference.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
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


/// The table that answers the turned kernel wherever it and the skinny
/// kernel both cover a product: a test of the turned kernel runs under it.
fn turned_table() -> std::sync::Arc<kernels_rocm::linear::routes::Routes> {
    std::sync::Arc::new(kernels_rocm::linear::routes::Routes::all(kernels_rocm::linear::routes::Kernel::Turned as u32))
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
        gpu.laid(&ctx, |ctx| gemm::matmul_bias(ctx, at, wt, bt, &mut y))
            .unwrap();
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
        kernels_rocm::linear::glu::swiglu(&ctx, packed, i, 1, &mut h2).unwrap();
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
        gpu.laid(&ctx, |ctx| gemm::matmul(ctx, at, wt, &mut y))
            .unwrap();
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
        kernels_rocm::linear::glu::swiglu(&ctx, packed, i, 1, &mut h2).unwrap();
        let (got, want) = (gpu.read(&h), gpu.read(&h2));
        let diff = got
            .iter()
            .zip(&want)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "batch {m}x{i}x{k}: {diff} cells differ");
    }
}

/// The turned kernel (forced over every row count it covers): K of one to
/// eight slices, one and two activation blocks, a ragged N, the f32 landing
/// and the fused swiglu.
#[test]
fn the_turned_projection() {
    let Some(gpu) = Gpu::open() else { return };
    let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
    t.gemm_turn_from = 1;
    t.gemm_turn_wide_until = 32;
    t.gemm_skinny_batch_min_n = 0;
    let ctx = gpu.ctx().with_tuning(t).with_routes(turned_table());
    for (m, n, k) in [
        (1u32, 2056u32, 256u32),
        (8, 4096, 2048),
        (13, 24, 1280),
        (24, 3000, 2048),
        (32, 1024, 768),
    ] {
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
        let want = reference(&a, &w, None, m as usize, n as usize, k as usize);
        let mut y = gpu.zeros(m, n, Dtype::Bf16);
        gemm::matmul(&ctx, at, wt, &mut y).unwrap();
        close_bf16(&format!("turned {m}x{n}x{k}"), &gpu.read(&y), &want, 2.0);
        let mut yf = gpu.zeros(m, n, Dtype::F32);
        gemm::matmul_f32(&ctx, at, wt, &mut yf).unwrap();
        let got = gpu.read(&yf);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0.0, f32::max);
        assert!(worst < 1e-3, "turned f32 {m}x{n}x{k}: {worst}");
        // The fused swiglu over the bank as [gate | up] lands its matmul's bits.
        let i = n / 2;
        let mut h = gpu.zeros(m, i, Dtype::Bf16);
        assert!(
            gemm::matmul_swiglu(&ctx, at, wt, &mut h).unwrap(),
            "turned {m}x{i}x{k} fused"
        );
        let mut h2 = gpu.zeros(m, i, Dtype::Bf16);
        kernels_rocm::linear::glu::swiglu(&ctx, y, i, 1, &mut h2).unwrap();
        let diff = gpu
            .read(&h)
            .iter()
            .zip(&gpu.read(&h2))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert_eq!(diff, 0, "turned {m}x{i}x{k}: {diff} cells differ");
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
        gpu.laid(&ctx, |ctx| gemm::matmul_f32(ctx, at, wt, &mut y))
            .unwrap();
        let want = reference(&a, &w, None, m as usize, n as usize, k as usize);
        for (i, (h, w)) in gpu.read(&y).iter().zip(&want).enumerate() {
            assert!(
                (h - w).abs() <= 1e-4 * w.abs().max(1.0),
                "matmul_f32 {m}x{n}x{k}[{i}] = {h}, want {w}"
            );
        }
    }
}

/// The lm head over a body's padded readouts multiplies only the live rows
/// (a device word), whichever of its three launches takes the count, and
/// leaves the padding's rows as they were.
#[test]
fn the_live_lm_head_multiplies_the_live_rows() {
    let Some(gpu) = Gpu::open() else { return };
    if kernels_rocm::hip::lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
    t.gemm_live_from = 129;
    let ctx = gpu.ctx().with_tuning(t);
    let (m, n, k) = (192u32, 3000u32, 512u32);
    let a: Vec<f32> = values((m * k) as usize, 31, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let w: Vec<f32> = values((n * k) as usize, 32, 0.05)
        .into_iter()
        .map(round_bf16)
        .collect();
    let (at, wt) = (
        gpu.tensor(&a, m, k, Dtype::Bf16),
        gpu.tensor(&w, n, k, Dtype::Bf16),
    );
    let want = reference(&a, &w, None, m as usize, n as usize, k as usize);
    for live in [7u32, 40, 150, 192] {
        let count = gpu.upload(&[live], 1, 1, Dtype::I32);
        let mut y = gpu.tensor(&vec![3.0; (m * n) as usize], m, n, Dtype::Bf16);
        gpu.laid(&ctx, |ctx| {
            gemm::lm_head_live(ctx, at, wt, count.ptr, &mut y)
        })
        .unwrap();
        let got = gpu.read(&y);
        let rows = (live * n) as usize;
        close_bf16(&format!("live {live}"), &got[..rows], &want[..rows], 2.0);
        assert!(
            got[rows..].iter().all(|&v| v == 3.0),
            "live {live}: a padding row was written"
        );
    }
}

/// The live lm head over a vocabulary past 2^16 rows whose size is no
/// multiple of a launch's macro tile (Qwen3's 151936: 64 rows past MT0 192,
/// 48 past MT0 176) writes every column of every live row, as the dense lm
/// head does: the grouped kernels drop such a partial last tile, so each
/// launch covers it with a group of its own.
#[test]
fn the_live_lm_head_writes_the_last_columns_of_a_wide_vocabulary() {
    let Some(gpu) = Gpu::open() else { return };
    if kernels_rocm::hip::lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
    t.gemm_live_from = 129;
    let ctx = gpu.ctx().with_tuning(t);
    let (m, n, k) = (192u32, 151_936u32, 256u32);
    let a: Vec<f32> = values((m * k) as usize, 41, 1.0)
        .into_iter()
        .map(round_bf16)
        .collect();
    let w: Vec<f32> = values((n * k) as usize, 42, 0.05)
        .into_iter()
        .map(round_bf16)
        .collect();
    let (at, wt) = (
        gpu.tensor(&a, m, k, Dtype::Bf16),
        gpu.tensor(&w, n, k, Dtype::Bf16),
    );
    let mut dense = gpu.tensor(&vec![0.0; (m * n) as usize], m, n, Dtype::Bf16);
    gpu.laid(&gpu.ctx(), |ctx| gemm::lm_head(ctx, at, wt, &mut dense))
        .unwrap();
    let want = gpu.read(&dense);
    // A count each launch takes: up to MT1 32, up to 128, past it.
    for live in [7u32, 100, 150, 192] {
        let count = gpu.upload(&[live], 1, 1, Dtype::I32);
        let mut y = gpu.tensor(&vec![3.0; (m * n) as usize], m, n, Dtype::Bf16);
        gpu.laid(&ctx, |ctx| {
            gemm::lm_head_live(ctx, at, wt, count.ptr, &mut y)
        })
        .unwrap();
        let got = gpu.read(&y);
        let rows = (live * n) as usize;
        // Two kernels' roundings of the same products; an unwritten
        // column holds 3.0.
        close_abs(
            &format!("live {live} of a {n}-wide vocabulary"),
            &got[..rows],
            &want[..rows],
            0.1,
        );
        assert!(
            got[rows..].iter().all(|&v| v == 3.0),
            "live {live}: a padding row was written"
        );
    }
}

/// The residual norm folded into the skinny product
/// (`ResidualAddRmsnormMatmul`): the sum lands in its own plane, the product
/// is the normed rows' as the norm would land them, where the measured
/// rates say the fold pays; where they say it does not, nothing lands.
#[test]
fn the_normed_skinny_product_is_the_norm_then_the_product() {
    let Some(gpu) = Gpu::open() else { return };
    // (m, n, k, plus_one, cached GB/s, folds): priced at device memory's
    // rate (no cached rate), one row past a 304-block grid reads less than a
    // launch's worth again, eight rows do not; at a cached rate every row
    // count the kernel stages folds, and lands the unfused path's bits.
    // Turned (the products from two rows on the turned kernel), the turned
    // kernel stages them in its turn, the product apart taking it too: 16
    // rows of 1,024 and their sums outgrow the turn and stay apart; a K past
    // 2048 takes the skinny kernel still.
    let (s, turned) = (false, true);
    for (m, n, k, plus_one, cached, folds, turn) in [
        (1u32, 4096u32, 1024u32, false, 0u32, true, s),
        (1, 3000, 1024, true, 0, true, s),
        (2, 6144, 1024, false, 0, true, s),
        (8, 4096, 1024, false, 0, false, s),
        (3, 4096, 1024, false, 1_000_000, true, s),
        (4, 4096, 1024, true, 1_000_000, true, s),
        (4, 6144, 1024, false, 1_000_000, true, s),
        (5, 2048, 1024, false, 1_000_000, true, s),
        (6, 2048, 1024, true, 1_000_000, true, s),
        (8, 2048, 1024, false, 1_000_000, true, s),
        (4, 2048, 3072, false, 1_000_000, true, s),
        (2, 6144, 1024, false, 1_000_000, true, turned),
        (3, 4096, 1024, true, 1_000_000, true, turned),
        (4, 4096, 1024, false, 1_000_000, true, turned),
        (8, 4096, 1024, true, 1_000_000, true, turned),
        (12, 4096, 1024, false, 1_000_000, true, turned),
        (16, 4096, 1024, false, 1_000_000, false, turned),
        (7, 2048, 2048, false, 1_000_000, true, turned),
        (4, 3000, 512, false, 1_000_000, true, turned),
        (4, 2048, 3072, false, 1_000_000, true, turned),
    ] {
        let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
        t.launch_ns = 1536;
        t.hbm_gbps = 4090;
        t.cached_gbps = cached;
        t.gemm_turn_from = if turn { 2 } else { 0 };
        let ctx = if turn { gpu.ctx().with_tuning(t).with_routes(turned_table()) } else { gpu.ctx().with_tuning(t) };
        let x: Vec<f32> = values((m * k) as usize, 41, 1.0).into_iter().map(round_bf16).collect();
        let y: Vec<f32> = values((m * k) as usize, 42, 1.0).into_iter().map(round_bf16).collect();
        let nw: Vec<f32> = values(k as usize, 43, 0.5).into_iter().map(round_bf16).collect();
        let w: Vec<f32> = values((n * k) as usize, 44, 0.05).into_iter().map(round_bf16).collect();
        let (xt, yt, nwt, wt) = (
            gpu.tensor(&x, m, k, Dtype::Bf16),
            gpu.tensor(&y, m, k, Dtype::Bf16),
            // The norm's weight as a load lays it: k rows of one.
            gpu.tensor(&nw, k, 1, Dtype::Bf16),
            gpu.tensor(&w, n, k, Dtype::Bf16),
        );
        let mut y_out = gpu.tensor(&vec![7.0; (m * k) as usize], m, k, Dtype::Bf16);
        let mut z = gpu.tensor(&vec![7.0; (m * n) as usize], m, n, Dtype::Bf16);
        let mut folded_normed = gpu.tensor(&vec![7.0; (m * k) as usize], m, k, Dtype::Bf16);
        let landed = gpu
            .laid(&ctx, |ctx| {
                gemm::residual_add_rmsnorm_matmul(
                    ctx, xt, yt, nwt, plus_one, 1e-6, wt, &mut y_out, &mut folded_normed, &mut z,
                )
            })
            .unwrap();
        assert_eq!(landed, folds, "{m}x{n}x{k}: the fold's choice");
        if !folds {
            assert!(gpu.read(&z).iter().all(|&v| v == 7.0), "{m}x{n}x{k}: nothing lands");
            continue;
        }
        let sums: Vec<f32> = y.iter().zip(&x).map(|(y, x)| round_bf16(y + x)).collect();
        assert_eq!(gpu.read(&y_out), sums, "{m}x{n}x{k}: the sum lands as it is");
        // The norm then the product apart, on the same kernels: the same bits.
        let mut apart_sum = gpu.zeros(m, k, Dtype::Bf16);
        let mut normed = gpu.zeros(m, k, Dtype::Bf16);
        let mut apart = gpu.zeros(m, n, Dtype::Bf16);
        gpu.laid(&ctx, |ctx| {
            kernels_rocm::elemwise::norm_residual::residual_add_rmsnorm_into(
                ctx, xt, yt, &mut apart_sum, nwt, plus_one, 1e-6, &mut normed,
            )?;
            gemm::lm_head(ctx, normed, wt, &mut apart)
        })
        .unwrap();
        assert_eq!(gpu.read(&apart_sum), sums, "{m}x{n}x{k}: the norm's sum");
        assert_eq!(
            gpu.read(&folded_normed),
            gpu.read(&normed),
            "{m}x{n}x{k}: the normed rows land for their other readers"
        );
        assert_eq!(
            gpu.read(&z),
            gpu.read(&apart),
            "{m}x{n}x{k} plus_one={plus_one}: the folded product is the norm's then the product's"
        );
        let want = reference(&gpu.read(&normed), &w, None, m as usize, n as usize, k as usize);
        close_abs(&format!("{m}x{n}x{k} against the host"), &gpu.read(&z), &want, 2e-2);
    }
}

/// The residual norm folded into a dense MLP's gate-up product
/// (`ResidualAddRmsnormMatmulSwigluMatmul`'s first half): the gated columns
/// are the norm's then the gated product's, bit for bit; the sum and the
/// normed rows land for their readers.
#[test]
fn the_normed_gated_skinny_product_is_the_norm_then_the_gated_product() {
    let Some(gpu) = Gpu::open() else { return };
    let mut t = kernels_rocm::tuning::Tuning::DEFAULT;
    t.launch_ns = 1536;
    t.hbm_gbps = 4090;
    for (m, i, k, plus_one, cached, turn) in [
        (1u32, 3072u32, 1024u32, false, 0u32, false),
        (2, 1536, 1024, true, 0, false),
        (4, 3072, 1024, false, 1_000_000, false),
        (2, 3072, 1024, true, 1_000_000, true),
        (4, 3072, 1024, false, 1_000_000, true),
        (8, 1536, 1024, false, 1_000_000, true),
        (12, 3072, 1024, true, 1_000_000, true),
    ] {
        let mut t = t;
        t.cached_gbps = cached;
        t.gemm_turn_from = if turn { 2 } else { 0 };
        let ctx = if turn { gpu.ctx().with_tuning(t).with_routes(turned_table()) } else { gpu.ctx().with_tuning(t) };
        let x: Vec<f32> = values((m * k) as usize, 51, 1.0).into_iter().map(round_bf16).collect();
        let y: Vec<f32> = values((m * k) as usize, 52, 1.0).into_iter().map(round_bf16).collect();
        let nw: Vec<f32> = values(k as usize, 53, 0.5).into_iter().map(round_bf16).collect();
        let w: Vec<f32> = values((2 * i * k) as usize, 54, 0.05).into_iter().map(round_bf16).collect();
        let (xt, yt, nwt, wt) = (
            gpu.tensor(&x, m, k, Dtype::Bf16),
            gpu.tensor(&y, m, k, Dtype::Bf16),
            gpu.tensor(&nw, k, 1, Dtype::Bf16),
            gpu.tensor(&w, 2 * i, k, Dtype::Bf16),
        );
        let mut y_out = gpu.zeros(m, k, Dtype::Bf16);
        let mut folded_normed = gpu.zeros(m, k, Dtype::Bf16);
        let mut h = gpu.tensor(&vec![7.0; (m * i) as usize], m, i, Dtype::Bf16);
        let landed = gpu
            .laid(&ctx, |ctx| {
                gemm::residual_add_rmsnorm_matmul_swiglu(
                    ctx, xt, yt, nwt, plus_one, 1e-6, wt, &mut y_out, &mut folded_normed, &mut h,
                )
            })
            .unwrap();
        assert!(landed, "{m}x{i}x{k}: one or two rows fold");
        let mut apart_sum = gpu.zeros(m, k, Dtype::Bf16);
        let mut normed = gpu.zeros(m, k, Dtype::Bf16);
        let mut apart = gpu.zeros(m, i, Dtype::Bf16);
        let gated = gpu
            .laid(&ctx, |ctx| {
                kernels_rocm::elemwise::norm_residual::residual_add_rmsnorm_into(
                    ctx, xt, yt, &mut apart_sum, nwt, plus_one, 1e-6, &mut normed,
                )?;
                gemm::matmul_swiglu(ctx, normed, wt, &mut apart)
            })
            .unwrap();
        assert!(gated, "{m}x{i}x{k}: the gated product apart takes the skinny kernel");
        assert_eq!(gpu.read(&y_out), gpu.read(&apart_sum), "{m}x{i}x{k}: the sum");
        assert_eq!(gpu.read(&folded_normed), gpu.read(&normed), "{m}x{i}x{k}: the normed rows");
        assert_eq!(gpu.read(&h), gpu.read(&apart), "{m}x{i}x{k} plus_one={plus_one}: the gated columns");
    }
}
