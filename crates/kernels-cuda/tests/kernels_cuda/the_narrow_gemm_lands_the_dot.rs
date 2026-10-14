#![cfg(feature = "cuda")]

// The narrow GEMM (`kernel.gemm.narrow_n`), a block a row, lands each row's
// dot with each of at most eight weight rows, and the bias as the gemv lands
// it: Qwen3.8-Flash-Next's hyper-connection inject (four columns over 10240)
// and narrower and odd widths.

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use kernels_cuda::linear::gemm_narrow_bf16;

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
    gemm_narrow_bf16(&ctx, w_at, a_at, b_at, y_at, m as i32, n as i32, k as i32)
        .expect("the narrow GEMM fires");
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
}

#[test]
fn the_narrow_gemm_lands_the_dot() {
    lands(64, 4, 10240, false);
    lands(7, 1, 2560, true);
    lands(33, 8, 264, true);
    lands(5, 3, 8, false);
}
