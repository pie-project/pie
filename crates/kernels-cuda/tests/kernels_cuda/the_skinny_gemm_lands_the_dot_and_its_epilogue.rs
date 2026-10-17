#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::Tensor;
use kernels_cuda::elemwise::norm::add_bias;
use kernels_cuda::linear::gemm::{matmul, matmul_bias};
use kernels_cuda::linear::skinny::{Epilogue, ROWS, covers, skinny_bf16};

fn round(v: f32) -> f32 {
    from_bf16(to_bf16(v))
}

fn gelu_tanh(g: f32) -> f32 {
    let inner = 0.797_884_6_f32 * (g + 0.044_715 * g * g * g);
    0.5 * g * (1.0 + inner.tanh())
}

fn check(epilogue: Epilogue, m: usize, n: usize, k: usize) {
    let weight_rows = match epilogue {
        Epilogue::Geglu => 2 * n,
        _ => n,
    };
    let mut lcg = Lcg::seeded(0x51 ^ ((m as u64) << 8) ^ n as u64);
    let (w_raw, w) = lcg.row(weight_rows * k);
    let (a_raw, a) = lcg.row(m * k);
    let tall = m + 1;
    let (y_raw, _) = lcg.row(tall * n);

    let mut gpu = Gpu::open();
    let w_at = gpu.up(&w_raw);
    let a_at = gpu.up(&a_raw);
    let y_at = gpu.up(&y_raw);
    let ctx = gpu.ctx();
    skinny_bf16(
        &ctx, w_at, a_at, y_at, m as i32, n as i32, k as i32, epilogue,
    )
    .unwrap_or_else(|e| panic!("{epilogue:?} m={m} n={n} k={k}: {e}"));
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, tall * n);

    let dot = |r: usize, c: usize| -> f32 { (0..k).map(|i| a[r * k + i] * w[c * k + i]).sum() };
    for r in 0..m {
        for c in 0..n {
            let noise = |v: f32| v.abs() * (1.0 / 128.0) + 2e-3 * (k as f32).sqrt() / 16.0;
            let (want, slack) = match epilogue {
                Epilogue::Store => {
                    let d = dot(r, c);
                    (d, noise(d))
                }
                Epilogue::Softcap(cap) => {
                    let d = round(dot(r, c));
                    (cap * (d / cap).tanh(), noise(d) + d.abs() * (1.0 / 128.0))
                }
                Epilogue::Geglu => {
                    let g = round(dot(r, c));
                    let u = round(dot(r, n + c));
                    let v = gelu_tanh(g) * u;
                    (
                        v,
                        noise(g) * u.abs() + noise(u) * g.abs() + v.abs() * (1.0 / 64.0) + 1e-3,
                    )
                }
            };
            let g = from_bf16(got[r * n + c]);
            assert!(
                (g - want).abs() <= slack,
                "{epilogue:?} m={m} n={n} k={k}: y[{r}][{c}] = {g}, want {want} (slack {slack})"
            );
        }
    }
    assert_eq!(
        &got[m * n..],
        &y_raw[m * n..],
        "{epilogue:?} m={m} n={n} k={k}: the row past m moved"
    );
}

#[test]
fn the_skinny_gemm_lands_the_dot_and_its_epilogue_every_case() {
    // One row, a ragged count and a full tile, at each epilogue.
    let full = ROWS as usize;
    for (epilogue, m, n, k) in [
        (Epilogue::Store, 1, 192, 128),
        (Epilogue::Store, 5, 192, 128),
        (Epilogue::Store, full, 192, 128),
        (Epilogue::Store, 64, 640, 2560),
        (Epilogue::Softcap(30.0), 1, 256, 256),
        (Epilogue::Softcap(30.0), 64, 256, 256),
        (Epilogue::Geglu, 1, 96, 256),
        (Epilogue::Geglu, 5, 96, 256),
        (Epilogue::Geglu, 64, 320, 1280),
    ] {
        check(epilogue, m, n, k);
    }
    // A folded bias lands the add after the projection's bits; under
    // `gemm_bias_epilogue` a cuBLASLt epilogue's within a rounding of it.
    for (m, n, k) in [(1, 512, 256), (1, 4104, 256), (3, 512, 256)] {
        bias(m, n, k, false);
    }
    for (m, n, k) in [(64, 5120, 2880), (64, 2880, 4096), (8, 640, 256)] {
        bias(m, n, k, true);
    }
    a_shape_the_block_does_not_divide_is_refused_without_firing();
}

fn bias(m: usize, n: usize, k: usize, epilogue: bool) {
    let mut lcg = Lcg::seeded(if epilogue { 0x5b } else { 0x7b } ^ n as u64);
    let (w_raw, _) = lcg.row(n * k);
    let (a_raw, _) = lcg.row(m * k);
    let (b_raw, _) = lcg.row(n);
    let mut gpu = Gpu::open();
    let w = Tensor::new(gpu.up(&w_raw), n as u32, k as u32, Dtype::Bf16);
    let a = Tensor::new(gpu.up(&a_raw), m as u32, k as u32, Dtype::Bf16);
    let b = Tensor::new(gpu.up(&b_raw), 1, n as u32, Dtype::Bf16);
    let mut fused = Tensor::new(gpu.zeros(m * n * 2), m as u32, n as u32, Dtype::Bf16);
    let mut apart = Tensor::new(gpu.zeros(m * n * 2), m as u32, n as u32, Dtype::Bf16);
    let tuning = gpu.ctx().tuning();
    let ctx = gpu.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        gemm_bias_epilogue: epilogue,
        ..tuning
    });
    // Twice: the first call picks the shape's tactic, the second runs it.
    for _ in 0..2 {
        matmul_bias(&ctx, a, w, b, &mut fused).expect("the biased projection fires");
    }
    matmul(&ctx, a, w, &mut apart).expect("the projection fires");
    add_bias(&ctx, b, &mut apart).expect("the add fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(fused.ptr, m * n);
    let want: Vec<u16> = gpu.down(apart.ptr, m * n);
    if !epilogue {
        assert_eq!(got, want, "m={m} n={n} k={k}");
        return;
    }
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let (g, w) = (from_bf16(*g), from_bf16(*w));
        assert!(
            (g - w).abs() <= 1e-2 * w.abs().max(1.0),
            "m={m} n={n} k={k} at {i}: {g} with the epilogue, {w} with the add"
        );
    }
}

fn a_shape_the_block_does_not_divide_is_refused_without_firing() {
    assert!(
        !covers(1, 96, 128, Epilogue::Store, ROWS),
        "n=96 is not whole 64s"
    );
    assert!(
        !covers(65, 64, 128, Epilogue::Store, ROWS),
        "m=65 is past the tile"
    );
    assert!(
        !covers(1, 64, 192, Epilogue::Store, ROWS),
        "k=192 is not whole 128s"
    );
    assert!(
        !covers(1, 48, 128, Epilogue::Geglu, ROWS),
        "I=48 is not whole 32s"
    );
    assert!(covers(64, 32, 128, Epilogue::Geglu, ROWS));
    let mut gpu = Gpu::open();
    let w = gpu.up(&vec![0u16; 96 * 128]);
    let a = gpu.up(&vec![0u16; 128]);
    let y = gpu.zeros(96 * 2);
    let ctx = gpu.ctx();
    assert!(skinny_bf16(&ctx, w, a, y, 1, 96, 128, Epilogue::Store).is_err());
}
