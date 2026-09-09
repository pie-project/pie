#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, from_bf16, to_bf16};
use kernels_cuda::linear::skinny::{Epilogue, ROWS, covers, skinny_bf16};

fn round(v: f32) -> f32 {
    from_bf16(to_bf16(v))
}

fn gelu_tanh(g: f32) -> f32 {
    let inner = 0.797_884_560_802_865_4_f32 * (g + 0.044_715 * g * g * g);
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
    skinny_bf16(&ctx, w_at, a_at, y_at, m as i32, n as i32, k as i32, epilogue)
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
                    (v, noise(g) * u.abs() + noise(u) * g.abs() + v.abs() * (1.0 / 64.0) + 1e-3)
                }
            };
            let g = from_bf16(got[r * n + c]);
            assert!(
                (g - want).abs() <= slack,
                "{epilogue:?} m={m} n={n} k={k}: y[{r}][{c}] = {g}, want {want} (slack {slack})"
            );
        }
    }
    assert_eq!(&got[m * n..], &y_raw[m * n..], "{epilogue:?} m={m} n={n} k={k}: the row past m moved");
}

#[test]
fn the_skinny_gemm_lands_the_dot_and_its_epilogue_every_case() {
    the_plain_projection_answers_the_dot_at_one_ragged_and_full_rows();
    the_softcap_epilogue_lands_the_capped_logit();
    the_geglu_epilogue_lands_the_gated_product();
    a_shape_the_block_does_not_divide_is_refused_without_firing();
}

fn the_plain_projection_answers_the_dot_at_one_ragged_and_full_rows() {
    for m in [1usize, 5, ROWS as usize] {
        check(Epilogue::Store, m, 192, 128);
    }
    check(Epilogue::Store, 64, 640, 2560);
}

fn the_softcap_epilogue_lands_the_capped_logit() {
    for m in [1usize, 64] {
        check(Epilogue::Softcap(30.0), m, 256, 256);
    }
}

fn the_geglu_epilogue_lands_the_gated_product() {
    for m in [1usize, 5, 64] {
        check(Epilogue::Geglu, m, 96, 256);
    }
    check(Epilogue::Geglu, 64, 320, 1280);
}

fn a_shape_the_block_does_not_divide_is_refused_without_firing() {
    assert!(!covers(1, 96, 128, Epilogue::Store), "n=96 is not whole 64s");
    assert!(!covers(65, 64, 128, Epilogue::Store), "m=65 is past the tile");
    assert!(!covers(1, 64, 192, Epilogue::Store), "k=192 is not whole 128s");
    assert!(!covers(1, 48, 128, Epilogue::Geglu), "I=48 is not whole 32s");
    assert!(covers(64, 32, 128, Epilogue::Geglu));
    let mut gpu = Gpu::open();
    let w = gpu.up(&vec![0u16; 96 * 128]);
    let a = gpu.up(&vec![0u16; 128]);
    let y = gpu.zeros(96 * 2);
    let ctx = gpu.ctx();
    assert!(skinny_bf16(&ctx, w, a, y, 1, 96, 128, Epilogue::Store).is_err());
}
