#![cfg(feature = "cuda")]

// The few-rows GEMM (`kernel.gemm.rows_m`), one launch for a decode's 2..=8
// rows, lands each row's dot with every weight row and the bias as the
// gemv lands it, at every K split it picks (wide N one warp a row, narrow N
// eight warps a row) and ragged N. Paced beside cuBLAS at DeepSeek-V4-Flash's
// narrow bf16 projections (ignored, a measurement):
//
//     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//         the_few_rows_gemm_paces -- --ignored --nocapture

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use kernels_cuda::linear::gemm_rows_bf16;
use std::time::Instant;

fn lands(m: usize, n: usize, k: usize, with_bias: bool) {
    let mut gpu = Gpu::open();
    let mut rng = Lcg::seeded((m * 31 + n * 7 + k) as u64);
    let (act_raw, act) = rng.row(m * k);
    let (w_raw, w) = rng.row(n * k);
    let (b_raw, b) = rng.row(n);
    let (a_at, w_at) = (gpu.up(&act_raw), gpu.up(&w_raw));
    let b_at = if with_bias { gpu.up(&b_raw) } else { 0 };
    let y_at = gpu.zeros(m * n * 2);
    let ctx = gpu.ctx();
    let fire = || {
        gemm_rows_bf16(
            &ctx, w_at, a_at, b_at, y_at, m as i32, n as i32, k as i32, false,
        )
        .expect("the few-rows GEMM fires");
    };
    fire();
    gpu.sync();
    let y: Vec<u16> = gpu.down(y_at, m * n);
    for r in 0..m {
        for c in 0..n {
            let dot: f32 = (0..k).map(|i| act[r * k + i] * w[c * k + i]).sum();
            let mut want = from_bf16(to_bf16(dot));
            if with_bias {
                want = from_bf16(to_bf16(want + b[c]));
            }
            let got = from_bf16(y[r * n + c]);
            assert!(
                close(got, want),
                "{m}x{n}x{k} row {r} column {c}: {got} against {want}"
            );
        }
    }
    // The slices add in a fixed order: a second launch lands the same bits.
    fire();
    gpu.sync();
    assert_eq!(gpu.down::<u16>(y_at, m * n), y, "{m}x{n}x{k} lands again");
}

#[test]
fn the_few_rows_gemm_lands_the_dot() {
    lands(8, 512, 4096, false);
    lands(8, 64, 4096, true);
    lands(2, 1024, 4096, true);
    lands(5, 300, 1032, false);
    lands(3, 7, 64, true);
    lands(8, 2048, 256, false);
}

#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_few_rows_gemm_paces_dsv4() {
    use kernels_cuda::linear::gemm::matmul;
    use kernels_cuda::tensor::Tensor;
    let mut gpu = Gpu::open();
    println!("| m | n | k | cuBLAS (tuned) us | few rows us |");
    println!("|---|---|---|---|---|");
    for (m, n, k) in [
        (8usize, 512usize, 4096usize),
        (8, 256, 4096),
        (8, 64, 4096),
        (8, 1024, 4096),
        (2, 512, 4096),
        (4, 512, 4096),
    ] {
        // Planes past the L2, cycled, so each fire reads its weight cold.
        let copies = (256 << 20) / (n * k * 2) + 1;
        let mut rng = Lcg::seeded(0x5e7 + n as u64);
        let (w_raw, _) = rng.row(n * k);
        let ws: Vec<u64> = (0..copies).map(|_| gpu.up(&w_raw)).collect();
        let (a_raw, _) = rng.row(m * k);
        let a_at = gpu.up(&a_raw);
        let y_at = gpu.zeros(m * n * 2);
        let ctx = gpu.ctx();
        let time = |f: &dyn Fn(usize)| {
            const REPS: usize = 400;
            (0..32).for_each(&f);
            gpu.sync();
            let t0 = Instant::now();
            (0..REPS).for_each(&f);
            gpu.sync();
            t0.elapsed().as_secs_f64() * 1e6 / REPS as f64
        };
        let cublas = time(&|i| {
            let act = Tensor::new(a_at, m as u32, k as u32, dtype::Dtype::Bf16);
            let w = Tensor::new(ws[i % ws.len()], n as u32, k as u32, dtype::Dtype::Bf16);
            let mut y = Tensor::new(y_at, m as u32, n as u32, dtype::Dtype::Bf16);
            matmul(&ctx, act, w, &mut y).expect("cuBLAS fires");
        });
        let rows = time(&|i| {
            gemm_rows_bf16(
                &ctx,
                ws[i % ws.len()],
                a_at,
                0,
                y_at,
                m as i32,
                n as i32,
                k as i32,
                false,
            )
            .expect("the few-rows GEMM fires");
        });
        println!("| {m} | {n} | {k} | {cublas:.1} | {rows:.1} |");
    }
}
