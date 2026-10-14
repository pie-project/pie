#![cfg(feature = "cuda")]

use crate::common::{Gpu, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{
    ExpertTable, GluBank, GroupSeat, matmul_select_bias, matmul_select_quant, select_glu,
};
use kernels_cuda::linear::tc5::Glu;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;
use std::time::Instant;

const EXPERTS: usize = 8;
const TOP_K: usize = 4;

const BLOCK: usize = 32;

fn bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as u8
        })
        .collect()
}

fn scale_bytes(seed: u64, count: usize) -> Vec<u8> {
    bytes(seed, count)
        .into_iter()
        .map(|b| 124 + (b % 7))
        .collect()
}

fn routes_of(tokens: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| {
            (0..TOP_K).map(move |j| {
                i32::try_from((t * 3 + j * 5 + t / 5) % EXPERTS).expect("an expert id in i32")
            })
        })
        .collect()
}

const FP32: Tuning = Tuning {
    mxfp4_wmma: false,
    ..Tuning::DEFAULT
};

fn check(tokens: usize, n: usize, k: usize, by_token: bool, biased: bool, tuning: Tuning) {
    assert_eq!(k % BLOCK, 0, "K is whole mxfp4 blocks");
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };

    let codes = bytes(0x11 + n as u64, EXPERTS * n * k / 2);
    let scales = scale_bytes(0x22 + k as u64, EXPERTS * n * groups);
    let act: Vec<u16> = bytes(0x33, act_rows * k * 2)
        .chunks(2)
        .map(|p| to_bf16((f32::from(p[0]) / 128.0 - 1.0) * 0.5))
        .collect();
    let bias: Vec<u16> = bytes(0x44, EXPERTS * n)
        .into_iter()
        .map(|b| to_bf16((f32::from(b) / 128.0 - 1.0) * 120.0))
        .collect();

    let mut gpu = Gpu::open();
    let act_at = gpu.up(&act);
    let codes_at = gpu.up(&codes);
    let scales_at = gpu.up(&scales);
    let bias_at = gpu.up(&bias);
    let routes_at = gpu.up(&routes);
    let grouped_at = gpu.zeros(route_count * n * 2);
    let per_route_at = gpu.zeros(route_count * n * 2);

    let grouped = gpu.ctx().with_tuning(tuning);
    let bare = gpu.ctx().with_tuning(Tuning {
        mxfp4_grouped: false,
        moe_tc5: false,
        mxfp4_mma_until: 0,
        ..tuning
    });
    let x = Tensor::new(act_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, EXPERTS as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, EXPERTS as u32, (n * groups) as u32, Dtype::U8);
    let bias_t = Tensor::new(bias_at, EXPERTS as u32, n as u32, Dtype::Bf16);
    let routes_t = Tensor::new(routes_at, tokens as u32, TOP_K as u32, Dtype::I32);
    let fire = |ctx: &kernels_cuda::Ctx, into: u64| {
        let mut y = Tensor::new(into, route_count as u32, n as u32, Dtype::Bf16);
        if biased {
            matmul_select_bias(
                ctx,
                x,
                codes_t,
                scales_t,
                bias_t,
                routes_t,
                &mut y,
                GroupSeat::RESIDENT,
            )
        } else {
            matmul_select_quant(
                ctx,
                x,
                codes_t,
                scales_t,
                None,
                routes_t,
                &mut y,
                GroupSeat::RESIDENT,
            )
        }
        .expect("the mxfp4 select fires");
    };
    fire(&grouped, grouped_at);
    fire(&bare, per_route_at);
    gpu.sync();

    let grouped: Vec<u16> = gpu.down(grouped_at, route_count * n);
    let per_route: Vec<u16> = gpu.down(per_route_at, route_count * n);
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, p)) in grouped.iter().zip(&per_route).enumerate() {
        let miss = (from_bf16(*g) - from_bf16(*p)).abs();
        if miss > worst {
            worst = miss;
            at = i;
        }
    }
    assert!(
        close(from_bf16(grouped[at]), from_bf16(per_route[at])),
        "route {} column {} of a {} {} fire: grouped {} where the per-route select says {}",
        at / n,
        at % n,
        if by_token { "token-read" } else { "route-read" },
        if biased { "biased" } else { "bare" },
        from_bf16(grouped[at]),
        from_bf16(per_route[at]),
    );
}

#[test]
fn the_grouped_mxfp4_select_answers_the_per_route_one_every_case() {
    the_token_read_leg_with_a_bias_agrees();
    the_route_read_leg_agrees();
    the_token_read_leg_without_a_bias_agrees();
    the_route_read_leg_with_a_bias_agrees();
    a_rectangle_that_divides_neither_tile_agrees();
    a_row_longer_than_two_blocks_a_lane_agrees();
    the_gpt_oss_legs_are_priced();
    the_fp32_form_agrees_at_both_readings();
    the_fp32_form_agrees_on_a_ragged_rectangle();
    the_tcgen05_form_agrees_at_gpt_oss_widths();
    the_register_dequant_form_agrees();
}

/// The register-dequant MMA kernel: one, two and four 8-route blocks a
/// slice, ragged rows, a K that ends mid-chunk, and scales past 2^1.
fn the_register_dequant_form_agrees() {
    let mma = Tuning {
        mxfp4_mma_until: 65_536,
        mxfp4_mma_from: 1,
        ..Tuning::DEFAULT
    };
    check(1, 200, 2880, true, true, mma);
    check(3, 576, 2880, false, true, mma);
    check(12, 384, 2880, true, false, mma);
    check(40, 576, 2880, true, true, mma);
    check(100, 384, 2880, false, true, mma);
    check(64, 256, 256, false, false, mma);
    check_glu(1, 192, 2880, true, true);
    check_glu(16, 2880, 2880, true, true);
    check_glu(40, 160, 2880, false, false);
}

/// The register-dequant kernel's GLU: the gate rows' and the up rows'
/// products (the per-route select's) through gpt-oss's clamped GLU.
fn check_glu(tokens: usize, inter: usize, k: usize, by_token: bool, biased: bool) {
    let n = 2 * inter;
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };
    let codes = bytes(0x51 + n as u64, EXPERTS * n * k / 2);
    let scales = scale_bytes(0x52 + k as u64, EXPERTS * n * groups);
    let act: Vec<u16> = bytes(0x53, act_rows * k * 2)
        .chunks(2)
        .map(|p| to_bf16((f32::from(p[0]) / 128.0 - 1.0) * 0.5))
        .collect();
    let bias: Vec<u16> = bytes(0x54, EXPERTS * n)
        .into_iter()
        .map(|b| to_bf16((f32::from(b) / 128.0 - 1.0) * 2.0))
        .collect();
    let mut gpu = Gpu::open();
    let x = Tensor::new(gpu.up(&act), act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(
        gpu.up(&codes),
        EXPERTS as u32,
        (n * k / 2) as u32,
        Dtype::U8,
    );
    let scales_t = Tensor::new(
        gpu.up(&scales),
        EXPERTS as u32,
        (n * groups) as u32,
        Dtype::U8,
    );
    let bias_t = Tensor::new(gpu.up(&bias), EXPERTS as u32, n as u32, Dtype::Bf16);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, TOP_K as u32, Dtype::I32);
    let full_at = gpu.zeros(route_count * n * 2);
    let glu_at = gpu.zeros(route_count * inter * 2);
    let bare = gpu.ctx().with_tuning(Tuning {
        mxfp4_grouped: false,
        ..Tuning::DEFAULT
    });
    let mma = gpu.ctx().with_tuning(Tuning {
        mxfp4_mma_until: 65_536,
        mxfp4_mma_from: 1,
        ..Tuning::DEFAULT
    });
    let mut full = Tensor::new(full_at, route_count as u32, n as u32, Dtype::Bf16);
    let bias_of = biased.then_some(bias_t);
    match bias_of {
        Some(b) => matmul_select_bias(
            &bare,
            x,
            codes_t,
            scales_t,
            b,
            routes_t,
            &mut full,
            GroupSeat::RESIDENT,
        ),
        None => matmul_select_quant(
            &bare,
            x,
            codes_t,
            scales_t,
            None,
            routes_t,
            &mut full,
            GroupSeat::RESIDENT,
        ),
    }
    .expect("the per-route select fires");
    let (limit, alpha) = (7.0f32, 1.702f32);
    let mut y = Tensor::new(glu_at, route_count as u32, inter as u32, Dtype::Bf16);
    let fused = select_glu(
        &mma,
        x,
        GluBank::Mxfp4 {
            codes: codes_t,
            scales: scales_t,
            bias: bias_of,
        },
        routes_t,
        &mut y,
        Glu::SwigluClampAlpha { limit, alpha },
        ExpertTable::RESIDENT,
    )
    .expect("the fused select fires");
    assert!(fused, "the register-dequant kernel lands the GLU");
    gpu.sync();
    let full: Vec<u16> = gpu.down(full_at, route_count * n);
    let got: Vec<u16> = gpu.down(glu_at, route_count * inter);
    for r in 0..route_count {
        for j in 0..inter {
            let g = from_bf16(full[r * n + j]).min(limit);
            let u = from_bf16(full[r * n + inter + j]).clamp(-limit, limit);
            let want = (u + 1.0) * g / (1.0 + (-alpha * g).exp());
            let have = from_bf16(got[r * inter + j]);
            assert!(
                close(have, want),
                "route {r} column {j} of a GLU {inter} wide: {have} where the select and the GLU say {want}"
            );
        }
    }
}

/// The grouped tcgen05 GEMM (W4A16, on a device that has it) at a gpt-oss
/// leg's K, biased and bare, masked and contiguous.
fn the_tcgen05_form_agrees_at_gpt_oss_widths() {
    check(40, 576, 2880, true, true, crate::common::TC5);
    check(100, 384, 2880, true, false, crate::common::TC5);
    check(100, 384, 2880, false, true, crate::common::TC5);
}

fn the_token_read_leg_with_a_bias_agrees() {
    check(64, 256, 256, true, true, Tuning::DEFAULT);
}

fn the_route_read_leg_agrees() {
    check(64, 256, 256, false, false, Tuning::DEFAULT);
}

fn the_token_read_leg_without_a_bias_agrees() {
    check(64, 256, 256, true, false, Tuning::DEFAULT);
}

fn the_route_read_leg_with_a_bias_agrees() {
    check(64, 256, 256, false, true, Tuning::DEFAULT);
}

fn a_rectangle_that_divides_neither_tile_agrees() {
    check(48, 200, 160, true, true, Tuning::DEFAULT);
}

fn a_row_longer_than_two_blocks_a_lane_agrees() {
    check(24, 96, 2240, true, true, Tuning::DEFAULT);
}

fn price(n: usize, k: usize, by_token: bool, label: &str) {
    let tokens = 256usize;
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };

    let mut gpu = Gpu::open();
    let act: Vec<u16> = bytes(0x33, act_rows * k * 2)
        .chunks(2)
        .map(|p| to_bf16((f32::from(p[0]) / 128.0 - 1.0) * 0.5))
        .collect();
    let act_at = gpu.up(&act);
    let codes_at = gpu.zeros(EXPERTS * n * k / 2);
    let scales_at = gpu.zeros(EXPERTS * n * groups);
    let routes_at = gpu.up(&routes);
    let y_at = gpu.zeros(route_count * n * 2);

    // The grouped WMMA kernel against the per-route select, and the
    // tcgen05 GEMM (where the device has it) beside them.
    let grouped = gpu.ctx().with_tuning(Tuning {
        moe_tc5: false,
        ..Tuning::DEFAULT
    });
    let bare = gpu.ctx().with_tuning(Tuning {
        mxfp4_grouped: false,
        moe_tc5: false,
        ..Tuning::DEFAULT
    });
    let tc5 = gpu.ctx().with_tuning(crate::common::TC5);
    let x = Tensor::new(act_at, act_rows as u32, k as u32, Dtype::Bf16);
    let codes_t = Tensor::new(codes_at, EXPERTS as u32, (n * k / 2) as u32, Dtype::U8);
    let scales_t = Tensor::new(scales_at, EXPERTS as u32, (n * groups) as u32, Dtype::U8);
    let routes_t = Tensor::new(routes_at, tokens as u32, TOP_K as u32, Dtype::I32);
    let mut y = Tensor::new(y_at, route_count as u32, n as u32, Dtype::Bf16);

    let mut run = |ctx: &kernels_cuda::Ctx| {
        const REPS: usize = 10;
        let mut fire = || {
            matmul_select_quant(
                ctx,
                x,
                codes_t,
                scales_t,
                None,
                routes_t,
                &mut y,
                GroupSeat::RESIDENT,
            )
            .expect("the mxfp4 select fires");
        };
        (0..3).for_each(|_| fire());
        gpu.sync();
        let t0 = Instant::now();
        (0..REPS).for_each(|_| fire());
        gpu.sync();
        t0.elapsed().as_secs_f64() * 1e3 / REPS as f64
    };
    let grouped_ms = run(&grouped);
    let per_route_ms = run(&bare);
    let tc5_ms = run(&tc5);

    let flop = 2.0 * route_count as f64 * n as f64 * k as f64;
    eprintln!(
        "{label:9} n={n:5} k={k:5}: grouped {grouped_ms:7.3} ms ({:5.1} TFLOP/s)  \
         per-route {per_route_ms:7.3} ms ({:5.1} TFLOP/s)  {:.1}x; tcgen05 {tc5_ms:7.3} ms \
         ({:5.1} TFLOP/s)",
        flop / grouped_ms / 1e9,
        flop / per_route_ms / 1e9,
        per_route_ms / grouped_ms,
        flop / tc5_ms / 1e9,
    );
    assert!(
        grouped_ms < per_route_ms,
        "{label}: grouping took {grouped_ms:.3} ms against the per-route select's \
         {per_route_ms:.3}; this is the shape it exists for"
    );
}

fn the_gpt_oss_legs_are_priced() {
    price(5760, 2880, true, "gate/up");
    price(2880, 2880, false, "down");
}

fn the_fp32_form_agrees_at_both_readings() {
    check(64, 256, 256, true, true, FP32);
    check(64, 256, 256, false, false, FP32);
}

fn the_fp32_form_agrees_on_a_ragged_rectangle() {
    check(48, 200, 160, true, true, FP32);
}
