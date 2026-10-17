#![cfg(feature = "cuda")]

use crate::common::tc5::{e2m1, e4m3};
use crate::common::{Gpu, Lcg, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::GroupSeat;
use kernels_cuda::linear::nvfp4::{matmul, matmul_select};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

const GROUP: usize = 16;

/// The register-dequant MMA kernel on, and the MXFP4 e2m1-convert kernel,
/// which an NVFP4 bank must not take.
const MMA: Tuning = Tuning {
    nvfp4_mma_until: 65_536,
    mxfp4_cvt: true,
    ..crate::common::TC5
};

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
                dequant[row * k + g * GROUP + 2 * j] = e2m1(b & 0xF) * gain;
                dequant[row * k + g * GROUP + 2 * j + 1] = e2m1(b >> 4) * gain;
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

/// Each case on the old arm, on the tcgen05 one (where the device has it),
/// on the register-dequant MMA kernel, and at B200's own decisions.
#[allow(clippy::too_many_arguments)]
fn routed(
    experts: usize,
    n: usize,
    k: usize,
    tokens: usize,
    top_k: usize,
    by_token: bool,
    seed: u64,
) {
    for tuning in [
        Tuning::DEFAULT,
        crate::common::TC5,
        MMA,
        crate::common::SM100,
    ] {
        routed_tuned(tuning, experts, n, k, tokens, top_k, by_token, seed);
    }
}

#[allow(clippy::too_many_arguments)]
fn routed_tuned(
    tuning: Tuning,
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
    let ctx = gpu.ctx().with_tuning(tuning);
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
    for row in 0..rows {
        for r in 0..n {
            let (g, w) = (from_bf16(got[row * n + r]), want(row, r));
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
        "{bad} of {} outputs differ from the dequantized dot",
        rows * n
    );
}

#[test]
fn the_nvfp4_select_answers_the_dequantized_dot_every_case() {
    // (experts, n, k, tokens, top_k, by_token): the gate-up leg read by
    // token and the down leg by route, at decode widths and at a fire wide
    // enough for the grouped kernel (the down leg's 200 rows a ragged tile).
    for (i, (experts, n, k, tokens, top_k, by_token)) in [
        (16, 1280, 2560, 3, 10, true),
        (16, 2560, 640, 2, 10, false),
        (16, 1280, 2560, 48, 10, true),
        (16, 200, 640, 48, 10, false),
    ]
    .into_iter()
    .enumerate()
    {
        routed(experts, n, k, tokens, top_k, by_token, 0x71 + i as u64);
    }
    // A dense projection at one row and at many.
    dense(320, 2560, 1, 0x75);
    dense(320, 2560, 7, 0x76);
    the_register_dequant_gate_up_lands_its_swiglu(MMA);
    the_register_dequant_gate_up_lands_its_swiglu(crate::common::SM100);
}

/// The gate-up select with its SwiGLU on the register-dequant kernel: the
/// bank's gate rows, then its up rows, an expert.
fn the_register_dequant_gate_up_lands_its_swiglu(tuning: Tuning) {
    use kernels_cuda::linear::moe::{ExpertTable, GluBank, select_glu};
    use kernels_cuda::linear::tc5::Glu;
    let (experts, inter, k, tokens, top_k) = (16usize, 640usize, 2560usize, 48usize, 10usize);
    let n = 2 * inter;
    let mut lcg = Lcg::seeded(0x77);
    let bank = bank(&mut lcg, experts, n, k);
    let (x_raw, x) = lcg.row(tokens * k);
    let routes: Vec<i32> = (0..tokens * top_k)
        .map(|r| ((r / top_k * 37 + r % top_k * 17 + 3) % experts) as i32)
        .collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x_raw);
    let codes_at = gpu.up(&bank.codes);
    let scales_at = gpu.up(&bank.scales);
    let tensor_at = gpu.up(&bank.tensor);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(tokens * top_k * inter * 2);
    let ctx = gpu.ctx().with_tuning(tuning);
    let slab =
        |at: u64, bytes: usize, dtype: Dtype| Tensor::new(at, experts as u32, bytes as u32, dtype);
    let mut y_t = Tensor::new(y_at, (tokens * top_k) as u32, inter as u32, Dtype::Bf16);
    let took = select_glu(
        &ctx,
        Tensor::new(x_at, tokens as u32, k as u32, Dtype::Bf16),
        GluBank::Nvfp4 {
            codes: slab(codes_at, n * k / 2, Dtype::Nvfp4),
            scales: slab(scales_at, n * k / GROUP, Dtype::U8),
            tensor_scales: slab(tensor_at, n * 4, Dtype::U8),
        },
        Tensor::new(routes_at, tokens as u32, top_k as u32, Dtype::I32),
        &mut y_t,
        Glu::Swiglu,
        ExpertTable::RESIDENT,
    )
    .expect("the select fires");
    assert!(
        took,
        "the register-dequant kernel takes the gate-up and its GLU"
    );
    gpu.sync();
    let got: Vec<u16> = gpu.down(y_at, tokens * top_k * inter);
    let dot = |route: usize, r: usize| {
        let e = routes[route] as usize;
        (0..k)
            .map(|c| bank.dequant[(e * n + r) * k + c] * x[route / top_k * k + c])
            .sum::<f32>()
    };
    let want = |route: usize, c: usize| {
        let (g, u) = (dot(route, c), dot(route, inter + c));
        g / (1.0 + (-g).exp()) * u
    };
    verdict(tokens * top_k, inter, &got, want);
}
