#![cfg(feature = "cuda")]

use crate::common::{Gpu, close, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::linear::moe::{
    ExpertTable, GluBank, GroupSeat, matmul_select_bias, matmul_select_quant, select_glu,
};
use kernels_cuda::linear::tc5::Glu;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;
use std::cell::Cell;
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

thread_local! {
    /// Whether the banks this thread builds leave f16's range: one row in
    /// 40 scaled 2^-27 .. 2^23 throughout.
    static WIDE: Cell<bool> = const { Cell::new(false) };
    /// Whether the activations this thread builds do: one in 97 at 2^16.
    static BIG: Cell<bool> = const { Cell::new(false) };
}

/// Block scales 2^-3 .. 2^3; with `WIDE`, a row in 40 (rows `groups`
/// blocks long) all one scale far outside (a row's own scale, so its sums
/// do not cancel past what bf16 holds).
fn scale_bytes(seed: u64, count: usize, groups: usize) -> Vec<u8> {
    let wide = WIDE.with(Cell::get);
    let row_scale = bytes(seed ^ 0x77, count / groups);
    bytes(seed, count)
        .into_iter()
        .enumerate()
        .map(|(i, b)| {
            let row = row_scale[i / groups];
            if wide && row % 40 == 3 {
                100 + (row % 51)
            } else {
                124 + (b % 7)
            }
        })
        .collect()
}

/// Activations in -0.5 .. 0.5 (with `BIG`, one in 97 at +-2^16).
fn activations(seed: u64, count: usize) -> Vec<u16> {
    let wide = BIG.with(Cell::get);
    bytes(seed, count * 2)
        .chunks(2)
        .enumerate()
        .map(|(i, p)| {
            let v = (f32::from(p[0]) / 128.0 - 1.0) * 0.5;
            to_bf16(if wide && i % 97 == 5 {
                v.signum() * 65536.0
            } else {
                v
            })
        })
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
    let scales = scale_bytes(0x22 + k as u64, EXPERTS * n * groups, groups);
    let act = activations(0x33, act_rows * k);
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
    // The host's sum, in f64, and the sum of its terms' magnitudes.
    let host = |at: usize| {
        const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let (route, col) = (at / n, at % n);
        let e = routes[route] as usize;
        let row = if by_token { route / TOP_K } else { route };
        let b = if biased {
            f64::from(from_bf16(bias[e * n + col]))
        } else {
            0.0
        };
        let (mut sum, mut mag) = (b, b.abs());
        for i in 0..k {
            let byte = codes[(e * n + col) * k / 2 + i / 2];
            let q = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
            let v = E2M1[usize::from(q & 7)] * if q & 8 != 0 { -1.0 } else { 1.0 };
            let s = f64::from(scales[(e * n + col) * groups + i / BLOCK]);
            let term = v * (s - 127.0).exp2() * f64::from(from_bf16(act[row * k + i]));
            sum += term;
            mag += term.abs();
        }
        (sum, mag)
    };
    // Activations past f16's range sum terms of 2^16 that cancel past
    // what any f32 accumulation keeps: those fires answer the host to a
    // part in 2^10 of their terms' magnitudes instead.
    if BIG.with(Cell::get) {
        for (i, g) in grouped.iter().enumerate() {
            let (want, mag) = host(i);
            let have = f64::from(from_bf16(*g));
            assert!(
                (have - want).abs() <= want.abs() / 256.0 + mag / 1024.0,
                "route {} column {}: {have} where the host says {want} (terms {mag})",
                i / n,
                i % n,
            );
        }
        return;
    }
    let host = host(at).0;
    assert!(
        close(from_bf16(grouped[at]), from_bf16(per_route[at])),
        "route {} column {} of a {} {} fire: grouped {} where the per-route select says {} \
         (the host {host})",
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
    the_e2m1_convert_form_agrees();
}

/// The register-dequant MMA kernel: one, two and four 8-route blocks a
/// slice, ragged rows, a K that ends mid-chunk, and scales past 2^1.
fn the_register_dequant_form_agrees() {
    register_dequant_cases(MMA);
}

const MMA: Tuning = Tuning {
    mxfp4_mma_until: 65_536,
    mxfp4_mma_from: 1,
    ..Tuning::DEFAULT
};

fn register_dequant_cases(mma: Tuning) {
    check(1, 200, 2880, true, true, mma);
    check(3, 576, 2880, false, true, mma);
    check(12, 384, 2880, true, false, mma);
    check(40, 576, 2880, true, true, mma);
    check(100, 384, 2880, false, true, mma);
    check(64, 256, 256, false, false, mma);
    check(5, 128, 4096, true, false, mma);
    check_glu(1, 192, 2880, true, true, mma);
    check_glu(16, 2880, 2880, true, true, mma);
    check_glu(40, 160, 2880, false, false, mma);
}

/// The register-dequant kernel on the e2m1 converts (where the device has
/// them): the same cases, then again with scales past f16's range, which
/// take the bf16 arithmetic, and activations past it, which walk K again
/// on it.
fn the_e2m1_convert_form_agrees() {
    if !kernels_cuda::linear::tc5::supported() {
        return;
    }
    let cvt = Tuning {
        mxfp4_cvt: true,
        mxfp4_a8_until: 65_536,
        ..MMA
    };
    register_dequant_cases(cvt);
    WIDE.with(|w| w.set(true));
    check(3, 576, 2880, false, true, cvt);
    check(40, 384, 2880, true, false, cvt);
    check_glu(1, 192, 2880, true, true, cvt);
    WIDE.with(|w| w.set(false));
    BIG.with(|w| w.set(true));
    check(3, 576, 2880, false, true, cvt);
    check(12, 384, 2880, true, false, cvt);
    BIG.with(|w| w.set(false));
    for keep in [false, true] {
        check_atoms(8, 256, 2048, cvt, keep);
        check_atoms(12, 384, 4096, cvt, keep);
        check_atoms(1, 384, 4096, cvt, keep);
    }
    // B200's own decisions: the arm its profile picks at each width (its
    // GLU cases where the register-dequant kernel spans them).
    let sm100 = crate::common::SM100;
    check(1, 200, 2880, true, true, sm100);
    check(12, 384, 2880, true, false, sm100);
    check(40, 576, 2880, true, true, sm100);
    check(100, 384, 2880, false, true, sm100);
    check_glu(16, 2880, 2880, true, true, sm100);
    check_glu(40, 160, 2880, false, false, sm100);
}

/// The byte of scale `(row, col)` in tcgen05's atom layout of `cols`
/// columns (`atom_offset` in `linear/act_quant.cuh`).
fn atom_offset(row: usize, col: usize, cols: usize) -> usize {
    (row / 128 * cols.div_ceil(4) + col / 4) * 512 + (row % 32) * 16 + (row / 32 % 4) * 4 + col % 4
}

/// The select over a W4A8 bank's atom-laid scales (rows counted across the
/// experts) answers the row-major one; with `keep`, the first keeps its
/// route order (`keeping_order`) and the second takes it rather than lay
/// its own.
fn check_atoms(tokens: usize, n: usize, k: usize, cvt: Tuning, keep: bool) {
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let codes = bytes(0x61 + n as u64, EXPERTS * n * k / 2);
    let scales = scale_bytes(0x62 + k as u64, EXPERTS * n * groups, groups);
    let mut atoms = vec![0u8; (EXPERTS * n).div_ceil(128) * groups.div_ceil(4) * 512];
    for row in 0..EXPERTS * n {
        for col in 0..groups {
            atoms[atom_offset(row, col, groups)] = scales[row * groups + col];
        }
    }
    let act = activations(0x63, tokens * k);
    let mut gpu = Gpu::open();
    let x = Tensor::new(gpu.up(&act), tokens as u32, k as u32, Dtype::Bf16);
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
    let atoms_at = gpu.up(&atoms);
    let routes_t = Tensor::new(gpu.up(&routes), tokens as u32, TOP_K as u32, Dtype::I32);
    let laid_at = gpu.zeros(route_count * n * 2);
    let plain_at = gpu.zeros(route_count * n * 2);
    let ctx = gpu.ctx().with_tuning(cvt);
    let mut laid = Tensor::new(laid_at, route_count as u32, n as u32, Dtype::Bf16);
    let mut first = || {
        kernels_cuda::linear::moe::select_mxfp4_atoms(&ctx, x, codes_t, atoms_at, routes_t, &mut laid)
    };
    let fired = if keep {
        kernels_cuda::linear::moe::keeping_order(first)
    } else {
        first()
    };
    assert!(
        fired.expect("the atom-laid select fires"),
        "the e2m1-convert kernel takes {route_count} routes"
    );
    let mut plain = Tensor::new(plain_at, route_count as u32, n as u32, Dtype::Bf16);
    matmul_select_quant(
        &ctx,
        x,
        codes_t,
        scales_t,
        None,
        routes_t,
        &mut plain,
        GroupSeat::RESIDENT,
    )
    .expect("the row-major select fires");
    gpu.sync();
    let laid: Vec<u16> = gpu.down(laid_at, route_count * n);
    let plain: Vec<u16> = gpu.down(plain_at, route_count * n);
    assert_eq!(
        laid, plain,
        "the atom-laid scales answer the row-major ones bit for bit (kept order: {keep})"
    );
}

/// The register-dequant kernel's GLU: the gate rows' and the up rows'
/// products (the per-route select's) through gpt-oss's clamped GLU.
fn check_glu(tokens: usize, inter: usize, k: usize, by_token: bool, biased: bool, mma: Tuning) {
    let n = 2 * inter;
    let groups = k / BLOCK;
    let routes = routes_of(tokens);
    let route_count = routes.len();
    let act_rows = if by_token { tokens } else { route_count };
    let codes = bytes(0x51 + n as u64, EXPERTS * n * k / 2);
    let scales = scale_bytes(0x52 + k as u64, EXPERTS * n * groups, groups);
    let act = activations(0x53, act_rows * k);
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
    let mma = gpu.ctx().with_tuning(mma);
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

    // The down select right after takes the GLU select's route order; it
    // answers what the per-route select does from the same GLU.
    let down_n = 160;
    let down_codes = bytes(0x55, EXPERTS * down_n * inter / 2);
    let down_scales = scale_bytes(0x56, EXPERTS * down_n * inter / BLOCK, inter / BLOCK);
    let dc = Tensor::new(
        gpu.up(&down_codes),
        EXPERTS as u32,
        (down_n * inter / 2) as u32,
        Dtype::U8,
    );
    let ds = Tensor::new(
        gpu.up(&down_scales),
        EXPERTS as u32,
        (down_n * inter / BLOCK) as u32,
        Dtype::U8,
    );
    let shared_at = gpu.zeros(route_count * down_n * 2);
    let apart_at = gpu.zeros(route_count * down_n * 2);
    let fire_down = |ctx: &kernels_cuda::Ctx, into: u64| {
        let mut out = Tensor::new(into, route_count as u32, down_n as u32, Dtype::Bf16);
        matmul_select_quant(
            ctx,
            y,
            dc,
            ds,
            None,
            routes_t,
            &mut out,
            GroupSeat::RESIDENT,
        )
        .expect("the down select fires");
    };
    let mut again = Tensor::new(glu_at, route_count as u32, inter as u32, Dtype::Bf16);
    select_glu(
        &mma,
        x,
        GluBank::Mxfp4 {
            codes: codes_t,
            scales: scales_t,
            bias: bias_of,
        },
        routes_t,
        &mut again,
        Glu::SwigluClampAlpha { limit, alpha },
        ExpertTable::RESIDENT,
    )
    .expect("the fused select fires");
    fire_down(&mma, shared_at);
    fire_down(&bare, apart_at);
    gpu.sync();
    let shared: Vec<u16> = gpu.down(shared_at, route_count * down_n);
    let apart: Vec<u16> = gpu.down(apart_at, route_count * down_n);
    for (i, (a, b)) in shared.iter().zip(&apart).enumerate() {
        assert!(
            close(from_bf16(*a), from_bf16(*b)),
            "route {} column {}: the down select on the kept order says {}, apart {}",
            i / down_n,
            i % down_n,
            from_bf16(*a),
            from_bf16(*b)
        );
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
