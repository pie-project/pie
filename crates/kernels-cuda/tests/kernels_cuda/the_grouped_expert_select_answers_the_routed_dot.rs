#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

#[derive(Clone, Copy)]
enum Spread {
    Mixed,
    Dominant,
}

fn expert_of(token: usize, j: usize, experts: usize, spread: Spread) -> i32 {
    let e = match spread {
        Spread::Mixed => (token * 3 + j * 5 + token / 7) % experts,
        Spread::Dominant if j == 0 => 0,
        Spread::Dominant => 1 + token % (experts - 1),
    };
    i32::try_from(e).expect("an expert id inside i32")
}

fn routed_dot(
    x: &[f32],
    bank: &[f32],
    routes: &[i32],
    top_k: usize,
    k: usize,
    n: usize,
    by_token: bool,
) -> Vec<f32> {
    let mut y = vec![0f32; routes.len() * n];
    for (route, &expert) in routes.iter().enumerate() {
        let row = if by_token { route / top_k } else { route };
        let expert = expert as usize;
        for col in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc += x[row * k + i] * bank[(expert * n + col) * k + i];
            }
            y[route * n + col] = acc;
        }
    }
    y
}

/// Each case on the old arm and on the tcgen05 one (where the device has it).
#[allow(clippy::too_many_arguments)]
fn check(
    experts: usize,
    tokens: usize,
    top_k: usize,
    k: usize,
    n: usize,
    by_token: bool,
    spread: Spread,
) {
    for tuning in [Tuning::DEFAULT, crate::common::TC5] {
        check_tuned(tuning, experts, tokens, top_k, k, n, by_token, spread);
    }
}

#[allow(clippy::too_many_arguments)]
fn check_tuned(
    tuning: Tuning,
    experts: usize,
    tokens: usize,
    top_k: usize,
    k: usize,
    n: usize,
    by_token: bool,
    spread: Spread,
) {
    let routes: Vec<i32> = (0..tokens)
        .flat_map(|t| (0..top_k).map(move |j| expert_of(t, j, experts, spread)))
        .collect();
    let route_count = routes.len();
    let x_rows = if by_token { tokens } else { route_count };

    let mut rng = Lcg::seeded(0x5eed_1234);
    let (x_raw, x_exact) = rng.row(x_rows * k);
    let (bank_raw, bank_exact) = rng.row(experts * n * k);
    let want = routed_dot(&x_exact, &bank_exact, &routes, top_k, k, n, by_token);

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let bank_at = gpu.up(&bank_raw);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(route_count * n * 2);

    let ctx = gpu.ctx().with_tuning(tuning);
    let x = Tensor::new(
        x_at,
        u32::try_from(x_rows).unwrap(),
        u32::try_from(k).unwrap(),
        Dtype::Bf16,
    );
    let bank = Tensor::new(
        bank_at,
        u32::try_from(experts).unwrap(),
        u32::try_from(n * k).unwrap(),
        Dtype::Bf16,
    );
    let routes_t = Tensor::new(
        routes_at,
        u32::try_from(tokens).unwrap(),
        u32::try_from(top_k).unwrap(),
        Dtype::I32,
    );
    let mut y = Tensor::new(
        y_at,
        u32::try_from(route_count).unwrap(),
        u32::try_from(n).unwrap(),
        Dtype::Bf16,
    );

    matmul_select(&ctx, x, bank, routes_t, &mut y, ExpertTable::RESIDENT)
        .expect("the routed select answers this shape");
    gpu.sync();

    let got: Vec<u16> = gpu.down(y_at, route_count * n);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (&raw, &want)) in got.iter().zip(want.iter()).enumerate() {
        let miss = (from_bf16(raw) - want).abs();
        if miss > worst {
            worst = miss;
            at = i;
        }
    }
    assert!(
        close(from_bf16(got[at]), want[at]),
        "route {} column {} of a {} fire answered {} where the dot says {}",
        at / n,
        at % n,
        if by_token { "token-read" } else { "route-read" },
        from_bf16(got[at]),
        want[at],
    );
}

#[test]
fn the_grouped_expert_select_answers_the_routed_dot_every_case() {
    use Spread::{Dominant, Mixed};
    // (experts, tokens, top_k, k, n, by_token, spread): the token- and
    // route-read legs, a decode-width fire, a dominant expert inside the
    // block budget, a ragged expert count; then widths the grouped tcgen05
    // GEMM takes, masked (40 routes) and contiguous, one past 65535 routes.
    for (experts, tokens, top_k, k, n, by_token, spread) in [
        (8, 48, 2, 64, 96, true, Mixed),
        (8, 48, 2, 64, 96, false, Mixed),
        (8, 1, 2, 64, 96, true, Mixed),
        (8, 48, 2, 64, 96, true, Dominant),
        (48, 40, 3, 64, 96, true, Mixed),
        (16, 20, 2, 256, 128, true, Mixed),
        (16, 64, 4, 256, 192, false, Mixed),
        (16, 70, 4, 256, 192, true, Mixed),
        (16, 300, 4, 512, 320, true, Dominant),
        (8, 33_000, 2, 64, 64, true, Mixed),
    ] {
        check(experts, tokens, top_k, k, n, by_token, spread);
    }
    // A route run past grid.y's 65535 rows on the GEMV itself (the grouped
    // leg off): DeepSeek-V4's grouped o-projection at 8192 rows.
    let gemv = Tuning {
        moe_grouped: false,
        ..Tuning::DEFAULT
    };
    check_tuned(gemv, 8, 33_000, 2, 32, 32, true, Mixed);
    check_tuned(gemv, 8, 8_192, 8, 32, 32, false, Mixed);
}
