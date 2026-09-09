#![cfg(feature = "cuda")]

mod common;

use common::{Gpu, Lcg, close, from_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{ExpertTable, matmul_select};
use kernels_cuda::tensor::Tensor;

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

fn check(
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

    let ctx = gpu.ctx();
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
    the_token_read_leg_answers_the_dot();
    the_route_read_leg_answers_the_dot();
    a_fire_wider_than_the_gemvs_grid_is_still_answered();
    a_decode_width_fire_is_still_served();
    a_dominant_expert_still_fits_the_block_budget();
    a_ragged_expert_count_still_lands();
}

fn the_token_read_leg_answers_the_dot() {
    check(8, 48, 2, 64, 96, true, Spread::Mixed);
}

fn the_route_read_leg_answers_the_dot() {
    check(8, 48, 2, 64, 96, false, Spread::Mixed);
}

fn a_fire_wider_than_the_gemvs_grid_is_still_answered() {
    check(8, 33_000, 2, 32, 32, true, Spread::Mixed);
}

fn a_decode_width_fire_is_still_served() {
    check(8, 1, 2, 64, 96, true, Spread::Mixed);
}

fn a_dominant_expert_still_fits_the_block_budget() {
    check(8, 48, 2, 64, 96, true, Spread::Dominant);
}

fn a_ragged_expert_count_still_lands() {
    check(48, 40, 3, 64, 96, true, Spread::Mixed);
}
