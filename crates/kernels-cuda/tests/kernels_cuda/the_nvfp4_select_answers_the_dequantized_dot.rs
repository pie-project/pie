#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::nvfp4::{matmul, matmul_select};
use kernels_cuda::tensor::Tensor;

const GROUP: usize = 16;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

fn e4m3(b: u8) -> f32 {
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    let v = if e == 0 {
        m / 512.0
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if b & 0x80 != 0 { -v } else { v }
}

struct Bank {
    codes: Vec<u8>,
    scales: Vec<u8>,
    tensor: Vec<f32>,
    dequant: Vec<f32>,
}

/// `slabs` slabs of `n` rows of `k` random codes, each group's e4m3 scale
/// between 2^-3 and 2^3 and each row's tensor scale its own.
fn bank(lcg: &mut Lcg, slabs: usize, n: usize, k: usize) -> Bank {
    let groups = k / GROUP;
    let mut codes = vec![0u8; slabs * n * k / 2];
    let mut scales = vec![0u8; slabs * n * groups];
    let mut tensor = vec![0f32; slabs * n];
    let mut dequant = vec![0f32; slabs * n * k];
    let mut byte = || ((lcg.unit() + 1.0) * 127.5) as u8;
    for row in 0..slabs * n {
        tensor[row] = 0.01 + f32::from(byte()) / 2550.0;
        for g in 0..groups {
            let scale = ((byte() % 48) + 32) & 0x7F;
            scales[row * groups + g] = scale;
            let gain = e4m3(scale) * tensor[row];
            for j in 0..GROUP / 2 {
                let b = byte();
                let at = row * k / 2 + g * GROUP / 2 + j;
                codes[at] = b;
                dequant[row * k + g * GROUP + 2 * j] = E2M1[usize::from(b & 0xF)] * gain;
                dequant[row * k + g * GROUP + 2 * j + 1] = E2M1[usize::from(b >> 4)] * gain;
            }
        }
    }
    Bank {
        codes,
        scales,
        tensor,
        dequant,
    }
}

fn routed(
    experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    top_k: usize,
    by_token: bool,
    seed: u64,
) {
    let mut lcg = Lcg::seeded(seed);
    let bank = bank(&mut lcg, experts, n, k);
    let act_rows = if by_token { tokens } else { tokens * top_k };
    let (x_raw, x) = lcg.row(act_rows * k);
    let routes: Vec<i32> = (0..tokens * top_k)
        .map(|r| ((r / top_k * 37 + r % top_k * 17 + 3) % experts) as i32)
        .collect();

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&bank.codes);
    let scales_at = gpu.up(&bank.scales);
    let tensor_at = gpu.up(&bank.tensor);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(tokens * top_k * n * 2);
    let ctx = gpu.ctx();
    let slab =
        |at: u64, bytes: usize, dtype: Dtype| Tensor::new(at, experts as u32, bytes as u32, dtype);
    let mut y_t = Tensor::new(y_at, (tokens * top_k) as u32, n as u32, Dtype::Bf16);
    matmul_select(
        &ctx,
        Tensor::new(x_at, act_rows as u32, k as u32, Dtype::Bf16),
        slab(codes_at, n * k / 2, Dtype::Nvfp4),
        slab(scales_at, n * k / GROUP, Dtype::U8),
        slab(tensor_at, n * 4, Dtype::U8),
        Tensor::new(routes_at, tokens as u32, top_k as u32, Dtype::I32),
        &mut y_t,
        GroupSeat::RESIDENT,
    )
    .expect("the select fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, tokens * top_k * n);
    let want = |route: usize, r: usize| {
        let e = routes[route] as usize;
        let xr = if by_token { route / top_k } else { route };
        (0..k)
            .map(|c| bank.dequant[(e * n + r) * k + c] * x[xr * k + c])
            .sum::<f32>()
    };
    verdict(tokens * top_k, n, &got, want);
}

fn dense(n: usize, k: usize, rows: usize, seed: u64) {
    let mut lcg = Lcg::seeded(seed);
    let bank = bank(&mut lcg, 1, n, k);
    let (x_raw, x) = lcg.row(rows * k);
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&bank.codes);
    let scales_at = gpu.up(&bank.scales);
    let tensor_at = gpu.up(&bank.tensor);
    let y_at = gpu.zeros(rows * n * 2);
    let ctx = gpu.ctx();
    let mut y_t = Tensor::new(y_at, rows as u32, n as u32, Dtype::Bf16);
    matmul(
        &ctx,
        "linear.matmul",
        Tensor::new(x_at, rows as u32, k as u32, Dtype::Bf16),
        Tensor::new(codes_at, n as u32, (k / 2) as u32, Dtype::Nvfp4),
        Tensor::new(scales_at, n as u32, (k / GROUP) as u32, Dtype::U8),
        Tensor::new(tensor_at, n as u32, 4, Dtype::U8),
        &mut y_t,
        GroupSeat::RESIDENT,
    )
    .expect("the dense product fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, rows * n);
    // Past one row the product reads a bf16 tile of the weights.
    let tiled = |w: f32| if rows > 1 { from_bf16(to_bf16(w)) } else { w };
    let want = |row: usize, r: usize| {
        (0..k)
            .map(|c| tiled(bank.dequant[r * k + c]) * x[row * k + c])
            .sum::<f32>()
    };
    verdict(rows, n, &got, want);
}

fn verdict(rows: usize, n: usize, got: &[u16], want: impl Fn(usize, usize) -> f32) {
    let mut bad = 0usize;
    let mut worst = 0f32;
    for row in 0..rows {
        for r in 0..n {
            let (g, w) = (from_bf16(got[row * n + r]), want(row, r));
            worst = worst.max((g - w).abs() / w.abs().max(1.0));
            if !close(g, w) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("row {row} column {r}: got {g} want {w}");
                }
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "{bad} of {} outputs differ from the dequantized dot (worst relative error {worst:.4})",
        rows * n
    );
    eprintln!("ok: worst relative error {worst:.5}");
}

#[test]
fn the_nvfp4_select_answers_the_dequantized_dot_every_case() {
    the_gate_up_leg_read_by_token_route_by_route();
    the_down_leg_read_by_route_route_by_route();
    a_wide_fire_takes_the_grouped_kernel_on_the_gate_up_leg();
    a_wide_fire_takes_the_grouped_kernel_on_the_down_leg_with_a_ragged_row_tile();
    a_dense_projection_answers_at_one_row_and_at_many();
}

fn the_gate_up_leg_read_by_token_route_by_route() {
    routed(16, 1280, 2560, 3, 10, true, 0x71);
}

fn the_down_leg_read_by_route_route_by_route() {
    routed(16, 2560, 640, 2, 10, false, 0x72);
}

fn a_wide_fire_takes_the_grouped_kernel_on_the_gate_up_leg() {
    routed(16, 1280, 2560, 48, 10, true, 0x73);
}

fn a_wide_fire_takes_the_grouped_kernel_on_the_down_leg_with_a_ragged_row_tile() {
    routed(16, 200, 640, 48, 10, false, 0x74);
}

fn a_dense_projection_answers_at_one_row_and_at_many() {
    dense(320, 2560, 1, 0x75);
    dense(320, 2560, 7, 0x76);
}
