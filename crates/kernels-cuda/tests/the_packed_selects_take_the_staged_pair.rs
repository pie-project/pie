//! TEMPORARY win-seat smoke — deleted after the run. Staged geometry on the
//! two packed selects: armed at `(count, start)` against a pre-sliced
//! reference, on `staged_rows.rs` gate (f)'s terms.

mod common;

use common::{Gpu, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{GroupSeat, matmul_select_bias, matmul_select_quant};
use kernels_cuda::tensor::Tensor;

const TOKENS: u32 = 4;
const TOP_K: u32 = 2;
const EXPERTS: u32 = 3;
const K: u32 = 64;
const N: u32 = 4;
const START: u32 = 1;
const LIVE: u32 = 2;
const SENTINEL: u16 = 0xAAAA;

fn codes() -> Vec<u8> {
    (0..(EXPERTS * N * (K / 2)) as usize)
        .map(|i| ((i * 37 + 11) % 251) as u8)
        .collect()
}

fn compare(what: &str, got: &[u16], want: &[u16]) {
    for row in 0..(TOKENS * TOP_K) as usize {
        let inside = row >= (START * TOP_K) as usize
            && row < ((START + LIVE) * TOP_K) as usize;
        for c in 0..N as usize {
            let at = row * N as usize + c;
            if inside {
                let w = (row - (START * TOP_K) as usize) * N as usize + c;
                assert!(
                    close(from_bf16(got[at]), from_bf16(want[w])),
                    "{what} row {row} col {c}: {} against pre-sliced {}",
                    from_bf16(got[at]),
                    from_bf16(want[w])
                );
            } else {
                assert_eq!(got[at], SENTINEL, "{what} row {row} col {c} was written");
            }
        }
    }
}

#[test]
#[ignore = "temporary smoke; run with -- --ignored"]
fn the_packed_selects_convert_the_pair_to_their_own_axis() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();

    let x_rows: Vec<u16> = (0..(TOKENS * TOP_K * K) as usize)
        .map(|i| to_bf16(((i % 13) as f32 - 6.0) / 8.0))
        .collect();
    let x_at = gpu.up(&x_rows);
    let codes_at = gpu.up(&codes());
    // mxfp4 e8m0 exponents, kept near 1.0 so nothing overflows.
    let mx_scales: Vec<u8> = (0..(EXPERTS * N * (K / 32)) as usize)
        .map(|i| (125 + (i % 5)) as u8)
        .collect();
    let mx_scales_at = gpu.up(&mx_scales);
    // mlxu4 bf16 scales and zero points, one pair per 64-code group.
    let aff: Vec<u16> = (0..(EXPERTS * N * (K / 64)) as usize)
        .map(|i| to_bf16(0.25 + (i % 4) as f32 * 0.125))
        .collect();
    let aff_scales_at = gpu.up(&aff);
    let aff_biases_at = gpu.up(&aff);
    let bias: Vec<u16> = (0..(EXPERTS * N) as usize)
        .map(|i| to_bf16((i % 3) as f32 * 0.5))
        .collect();
    let bias_at = gpu.up(&bias);
    let routes_at = gpu.up(&[0_i32, 1, 2, 0, 1, 2, 2, 1]);

    let codes_t = Tensor::new(codes_at, EXPERTS * N, K / 2, Dtype::U8);
    let mx_scales_t = Tensor::new(mx_scales_at, EXPERTS * N, K / 32, Dtype::U8);
    let aff_scales_t = Tensor::new(aff_scales_at, EXPERTS * N, 2, Dtype::U8);
    let aff_biases_t = Tensor::new(aff_biases_at, EXPERTS * N, 2, Dtype::U8);
    let bias_t = Tensor::new(bias_at, EXPERTS, N, Dtype::Bf16);
    let staged_at = gpu.up(&[LIVE, START, 0, 0]);

    // ── (1) the bias leg, mxfp4, by TOKEN: `x` has one row per token.
    let y_at = gpu.up(&vec![SENTINEL; (TOKENS * TOP_K * N) as usize]);
    let mut y = Tensor::new(y_at, TOKENS * TOP_K, N, Dtype::Bf16);
    ctx.arm_stage(staged_at);
    matmul_select_bias(
        &ctx,
        Tensor::new(x_at, TOKENS, K, Dtype::Bf16),
        codes_t,
        mx_scales_t,
        bias_t,
        Tensor::new(routes_at, TOKENS, TOP_K, Dtype::I32),
        &mut y,
        GroupSeat::RESIDENT,
    )
    .expect("the armed bias leg enqueues");
    ctx.disarm_stage();
    let want_at = gpu.up(&vec![SENTINEL; (LIVE * TOP_K * N) as usize]);
    let mut want = Tensor::new(want_at, LIVE * TOP_K, N, Dtype::Bf16);
    matmul_select_bias(
        &ctx,
        Tensor::new(x_at + u64::from(START * K) * 2, LIVE, K, Dtype::Bf16),
        codes_t,
        mx_scales_t,
        bias_t,
        Tensor::new(routes_at + u64::from(START * TOP_K) * 4, LIVE, TOP_K, Dtype::I32),
        &mut want,
        GroupSeat::RESIDENT,
    )
    .expect("the pre-sliced bias leg enqueues");
    gpu.sync();
    compare(
        "mxfp4 bias leg",
        &gpu.down::<u16>(y_at, (TOKENS * TOP_K * N) as usize),
        &gpu.down::<u16>(want_at, (LIVE * TOP_K * N) as usize),
    );

    // ── (2) the down leg, mxfp4, by ROUTE: `x` has one row per route.
    let y2_at = gpu.up(&vec![SENTINEL; (TOKENS * TOP_K * N) as usize]);
    let mut y2 = Tensor::new(y2_at, TOKENS * TOP_K, N, Dtype::Bf16);
    ctx.arm_stage(staged_at);
    matmul_select_quant(
        &ctx,
        Tensor::new(x_at, TOKENS * TOP_K, K, Dtype::Bf16),
        codes_t,
        mx_scales_t,
        None,
        Tensor::new(routes_at, TOKENS, TOP_K, Dtype::I32),
        &mut y2,
        GroupSeat::RESIDENT,
    )
    .expect("the armed down leg enqueues");
    ctx.disarm_stage();
    let want2_at = gpu.up(&vec![SENTINEL; (LIVE * TOP_K * N) as usize]);
    let mut want2 = Tensor::new(want2_at, LIVE * TOP_K, N, Dtype::Bf16);
    matmul_select_quant(
        &ctx,
        Tensor::new(x_at + u64::from(START * TOP_K * K) * 2, LIVE * TOP_K, K, Dtype::Bf16),
        codes_t,
        mx_scales_t,
        None,
        Tensor::new(routes_at + u64::from(START * TOP_K) * 4, LIVE, TOP_K, Dtype::I32),
        &mut want2,
        GroupSeat::RESIDENT,
    )
    .expect("the pre-sliced down leg enqueues");
    gpu.sync();
    compare(
        "mxfp4 down leg",
        &gpu.down::<u16>(y2_at, (TOKENS * TOP_K * N) as usize),
        &gpu.down::<u16>(want2_at, (LIVE * TOP_K * N) as usize),
    );

    // ── (3) the affine twin, by ROUTE.
    let y3_at = gpu.up(&vec![SENTINEL; (TOKENS * TOP_K * N) as usize]);
    let mut y3 = Tensor::new(y3_at, TOKENS * TOP_K, N, Dtype::Bf16);
    ctx.arm_stage(staged_at);
    matmul_select_quant(
        &ctx,
        Tensor::new(x_at, TOKENS * TOP_K, K, Dtype::Bf16),
        codes_t,
        aff_scales_t,
        Some(aff_biases_t),
        Tensor::new(routes_at, TOKENS, TOP_K, Dtype::I32),
        &mut y3,
        GroupSeat::RESIDENT,
    )
    .expect("the armed affine twin enqueues");
    ctx.disarm_stage();
    let want3_at = gpu.up(&vec![SENTINEL; (LIVE * TOP_K * N) as usize]);
    let mut want3 = Tensor::new(want3_at, LIVE * TOP_K, N, Dtype::Bf16);
    matmul_select_quant(
        &ctx,
        Tensor::new(x_at + u64::from(START * TOP_K * K) * 2, LIVE * TOP_K, K, Dtype::Bf16),
        codes_t,
        aff_scales_t,
        Some(aff_biases_t),
        Tensor::new(routes_at + u64::from(START * TOP_K) * 4, LIVE, TOP_K, Dtype::I32),
        &mut want3,
        GroupSeat::RESIDENT,
    )
    .expect("the pre-sliced affine twin enqueues");
    gpu.sync();
    compare(
        "mlxu4 down leg",
        &gpu.down::<u16>(y3_at, (TOKENS * TOP_K * N) as usize),
        &gpu.down::<u16>(want3_at, (LIVE * TOP_K * N) as usize),
    );
}
