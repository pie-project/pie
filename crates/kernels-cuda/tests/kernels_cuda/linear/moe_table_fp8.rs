//! The bf16 and block-FP8 routed selects through an expert table
//! ([`super::moe_table::same_bits`]): the register-held, tiled and wide bf16
//! GEMMs, the grouped mma.sync select, cuBLAS's grouped leg, and the W8A16
//! and W8A8 FP8 selects land the contiguous bank's bits from a scattered
//! pool, and a swapped table does not. The Hopper (wgmma, TMA) arms run only
//! on sm_90.

use super::moe_table::{routes, same_bits};
use crate::common::expert_pool::Plane;
use crate::common::linear_ref::{t, tab};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::linear::gemm_sm100::Glu;
use kernels_cuda::linear::moe::{self, ExpertTable};
use kernels_cuda::tuning::Tuning;
use std::time::Instant;

/// `count` random bytes: e4m3 codes (no NaN), or exponents near 127.
fn bytes(seed: &mut u64, count: usize, exps: bool) -> Vec<u8> {
    tab(count, |_| {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let b = (*seed >> 33) as u8;
        if exps { 122 + b % 8 } else { b & 0xF7 }
    })
}

/// An fp8b128 / W8A16 bank: `experts x n x k` e4m3 codes and their ue8m0
/// exponents a `gran` of K laid in atoms; the two planes.
fn fp8_bank(g: &mut Gpu, seed: u64, [experts, n, k]: [usize; 3], gran: usize) -> [Plane; 2] {
    let mut s = seed | 1;
    let codes = g.up(&bytes(&mut s, experts * n * k, false));
    let atoms_k = (k / gran).div_ceil(4);
    let stride = n / 128 * atoms_k * 512;
    let atoms = g.up(&bytes(&mut s, experts * stride, true));
    [
        Plane::of(codes, (n * k) as u64),
        Plane::of(atoms, stride as u64),
    ]
}

/// Activations `[rows, k]` bf16, routes `[tokens, top]`, and `y` for the
/// routes' `n` columns.
struct Fire {
    x: u64,
    r: u64,
    y: u64,
}

fn fire_of(
    g: &mut Gpu,
    lcg: &mut Lcg,
    [ex, tokens, top, n, k]: [usize; 5],
    by_token: bool,
) -> Fire {
    let rows = if by_token { tokens } else { tokens * top };
    let (xr, _) = lcg.row(rows * k);
    Fire {
        x: g.up(&xr),
        r: g.up(&routes(tokens, top, ex)),
        y: g.zeros(tokens * top * n * 2),
    }
}

/// (experts, tokens, top, n, k, by_token, glu, tuning, arm).
type BfCase = (
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
    Option<Glu>,
    Tuning,
    &'static str,
);

#[test]
fn the_bf16_selects_land_the_same_bits_through_any_table() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x0bf1_67ab);
    // (experts, tokens, top, n, k, by_token, glu, arm tuning): the
    // tiled GEMM at a few routes an expert and at 64 (with and without its
    // GLU), its wide tile (128 routes an expert), the grouped mma.sync
    // select, and the leg `matmul_select` prices cheapest.
    let d = Tuning::DEFAULT;
    let priced = Tuning {
        roof: crate::common::roof(),
        ..d
    };
    #[rustfmt::skip]
    let cases: [BfCase; 7] = [
        (8, 4, 2, 128, 256, true, None, d, "tiled few"),
        (8, 6, 2, 96, 512, false, Some(Glu::Swiglu), d, "tiled glu"),
        (8, 160, 2, 256, 256, true, None, d, "tiled"),
        (4, 288, 2, 128, 256, true, Some(Glu::Swiglu), d, "wide glu"),
        (8, 24, 4, 128, 256, true, None, d, "mma"),
        (8, 20, 2, 136, 512, false, None, d, "mma tail"),
        (8, 40, 2, 128, 256, true, None, priced, "priced"),
    ];
    for (ex, tokens, top, n, k, by_token, glu, tuning, arm) in cases {
        let rows = if glu.is_some() { 2 * n } else { n };
        let (raw, _) = lcg.row(ex * rows * k);
        let bank = g.up(&raw);
        let f = fire_of(&mut g, &mut lcg, [ex, tokens, top, n, k], by_token);
        let ctx = g.ctx().with_tuning(tuning);
        let xr = if by_token { tokens } else { tokens * top };
        let what = format!("bf16 {arm} {ex}x{tokens}x{top} {n}x{k} {glu:?}");
        let plane = [Plane::of(bank, (rows * k * 2) as u64)];
        same_bits(
            &mut g,
            &what,
            ex,
            &plane,
            (f.y, tokens * top * n * 2),
            |table| {
                let (x, r) = (t(f.x, xr, k, Bf16), t(f.r, tokens, top, I32));
                let mut y = t(f.y, tokens * top, n, Bf16);
                let plane = t(bank, ex, n * k, Bf16);
                if arm.starts_with("tiled") || arm.starts_with("wide") {
                    let took = moe::select_bf16_tiled(
                        &ctx, "test", x, bank, ex as u32, r, &mut y, glu, table,
                    )
                    .expect("the select fires");
                    assert!(took, "{arm}: the tiled select takes the fire");
                } else if arm.starts_with("mma") {
                    moe::matmul_select_on(&ctx, moe::Leg::Mma, x, plane, r, &mut y, table)
                        .expect("the select fires");
                } else {
                    moe::matmul_select(&ctx, x, plane, r, &mut y, table).expect("the select fires");
                }
            },
        );
    }
}

#[test]
fn the_w8a16_selects_land_the_same_bits_through_any_table() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0xf8a16);
    // The register-held select (gran 128 and 32, a GLU), and from 8192
    // routes the tiled GEMM.
    #[rustfmt::skip]
    let cases = [
        (8usize, 3usize, 2usize, 128usize, 512usize, true, 128usize, None),
        (8, 5, 4, 256, 256, false, 32, None),
        (8, 6, 2, 128, 256, true, 128, Some(Glu::Swiglu)),
        (16, 4096, 2, 128, 256, true, 128, None),
    ];
    for (ex, tokens, top, n, k, by_token, gran, glu) in cases {
        let rows = if glu.is_some() { 2 * n } else { n };
        let planes = fp8_bank(&mut g, 0xa16 ^ (n + k + gran) as u64, [ex, rows, k], gran);
        let f = fire_of(&mut g, &mut lcg, [ex, tokens, top, n, k], by_token);
        let ctx = g.ctx();
        let xr = if by_token { tokens } else { tokens * top };
        let what = format!("W8A16 {ex}x{tokens}x{top} {n}x{k} gran {gran} {glu:?}");
        same_bits(
            &mut g,
            &what,
            ex,
            &planes,
            (f.y, tokens * top * n * 2),
            |table| {
                let (x, r) = (t(f.x, xr, k, Bf16), t(f.r, tokens, top, I32));
                let mut y = t(f.y, tokens * top, n, Bf16);
                moe::select_fp8_w8a16(
                    &ctx,
                    "test",
                    x,
                    planes[0].base,
                    planes[1].base,
                    ex as u32,
                    gran as u32,
                    r,
                    &mut y,
                    glu,
                    table,
                )
                .expect("the W8A16 select fires");
            },
        );
    }
}

#[test]
fn the_w8a8_selects_land_the_same_bits_through_any_table() {
    if !moe::fp8_w8a8_capable() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0xf8a8);
    // A decode's slices (nb 1), a wider expert's (nb 8), the 256-row wide
    // tile (past 64 routes an expert), and its GLU form; on Hopper the
    // wgmma, warp-specialized and TMA arms by tuning.
    let ws = Tuning {
        fp8a8_ws_from: 1,
        ..Tuning::DEFAULT
    };
    let tma = Tuning {
        fp8a8_tma: true,
        ..ws
    };
    #[rustfmt::skip]
    let mut cases = vec![
        (8usize, 3usize, 2usize, 128usize, 512usize, true, false, Tuning::DEFAULT),
        (8, 24, 2, 256, 512, false, false, Tuning::DEFAULT),
        (4, 160, 2, 256, 512, true, false, Tuning::DEFAULT),
        (4, 160, 2, 128, 512, true, true, Tuning::DEFAULT),
    ];
    if kernels_cuda::jit::cache::arch() == Some("sm_90") {
        cases.push((4, 160, 2, 256, 512, true, false, ws));
        cases.push((4, 160, 2, 256, 512, true, false, tma));
    }
    for (ex, tokens, top, n, k, by_token, glu, tuning) in cases {
        let rows = if glu { 2 * n } else { n };
        let planes = fp8_bank(&mut g, 0xa8 ^ (n + k) as u64, [ex, rows, k], 128);
        let f = fire_of(&mut g, &mut lcg, [ex, tokens, top, n, k], by_token);
        let ctx = g.ctx().with_tuning(tuning);
        let xr = if by_token { tokens } else { tokens * top };
        let what = format!(
            "W8A8 {ex}x{tokens}x{top} {n}x{k} glu {glu} tma {}",
            tuning.fp8a8_tma
        );
        // The TMA arm maps the window: an expert's codes whole K-byte rows past it.
        let planes = [planes[0].aligned(k as u64), planes[1]];
        same_bits(
            &mut g,
            &what,
            ex,
            &planes,
            (f.y, tokens * top * n * 2),
            |table| {
                let (x, r) = (t(f.x, xr, k, Bf16), t(f.r, tokens, top, I32));
                let mut y = t(f.y, tokens * top, n, Bf16);
                let (codes, atoms) = (planes[0].base, planes[1].base);
                if glu {
                    let took = moe::select_fp8_w8a8_glu(
                        &ctx,
                        x,
                        codes,
                        atoms,
                        ex as u32,
                        r,
                        &mut y,
                        Glu::Swiglu,
                        table,
                    )
                    .expect("the W8A8 GLU select fires");
                    assert!(took, "{ex}x{tokens}: the W8A8 GLU select takes the fire");
                } else {
                    moe::select_fp8_w8a8(&ctx, x, codes, atoms, ex as u32, r, &mut y, table)
                        .expect("the W8A8 select fires");
                }
            },
        );
    }
}

#[test]
fn the_tma_fed_w8a8_select_lands_the_same_bits_through_any_table() {
    if kernels_cuda::jit::cache::arch() != Some("sm_90") {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7a8);
    let tuning = Tuning {
        fp8a8_wgtma: true,
        ..Tuning::DEFAULT
    };
    #[rustfmt::skip]
    let cases = [(16usize, 3usize, 8usize, 256usize, 1024usize, true), (8, 6, 4, 384, 512, false),
        (4, 24, 4, 256, 768, true), (4, 50, 4, 128, 512, true)];
    for (ex, tokens, top, n, k, by_token) in cases {
        let planes = fp8_bank(&mut g, 0x7a8 ^ (n + k) as u64, [ex, n, k], 128);
        let planes = [planes[0].aligned(k as u64), planes[1]];
        let f = fire_of(&mut g, &mut lcg, [ex, tokens, top, n, k], by_token);
        let ctx = g.ctx().with_tuning(tuning);
        let xr = if by_token { tokens } else { tokens * top };
        let what = format!("TMA-fed W8A8 {ex}x{tokens}x{top} {n}x{k}");
        same_bits(
            &mut g,
            &what,
            ex,
            &planes,
            (f.y, tokens * top * n * 2),
            |table| {
                let (x, r) = (t(f.x, xr, k, Bf16), t(f.r, tokens, top, I32));
                let mut y = t(f.y, tokens * top, n, Bf16);
                moe::select_fp8_w8a8(
                    &ctx,
                    x,
                    planes[0].base,
                    planes[1].base,
                    ex as u32,
                    r,
                    &mut y,
                    table,
                )
                .expect("the W8A8 select fires");
            },
        );
    }
}

/// The indirection's cost: the W8A16 decode select and the bf16 GEMV timed
/// with no table and with a scattered one (`PIE_TABLE_TIMING=1` prints).
#[test]
fn an_expert_table_costs_a_select_next_to_nothing() {
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x71e);
    let (ex, tokens, top, n, k) = (32usize, 8usize, 4usize, 1024usize, 2048usize);
    let planes = fp8_bank(&mut g, 0x71e, [ex, n, k], 128);
    let f = fire_of(&mut g, &mut lcg, [ex, tokens, top, n, k], true);
    let pool = crate::common::expert_pool::scatter(&mut g, ex, &planes, 7);
    let ctx = g.ctx();
    let fire = |table: ExpertTable| {
        let (x, r) = (t(f.x, tokens, k, Bf16), t(f.r, tokens, top, I32));
        let mut y = t(f.y, tokens * top, n, Bf16);
        g.laid(&ctx, |ctx| {
            moe::select_fp8_w8a16(
                ctx,
                "test",
                x,
                planes[0].base,
                planes[1].base,
                ex as u32,
                128,
                r,
                &mut y,
                None,
                table,
            )
        })
        .expect("the W8A16 select fires");
    };
    let time = |g: &Gpu, table: ExpertTable| {
        for _ in 0..20 {
            fire(table);
        }
        g.sync();
        let at = Instant::now();
        for _ in 0..300 {
            fire(table);
        }
        g.sync();
        at.elapsed().as_secs_f64() * 1e6 / 300.0
    };
    let mut best = (f64::MAX, f64::MAX);
    for _ in 0..3 {
        best.0 = best.0.min(time(&g, ExpertTable::RESIDENT));
        best.1 = best
            .1
            .min(time(&g, ExpertTable::located(pool.table, pool.window())));
    }
    if std::env::var("PIE_TABLE_TIMING").is_ok_and(|v| v == "1") {
        println!(
            "TABLE W8A16 {ex}x{tokens}x{top} {n}x{k}: contiguous {:.1} us, table {:.1} us",
            best.0, best.1
        );
    }
    // Generous: a shared card's noise, not the indirection, sets the bound.
    assert!(best.1 < best.0 * 1.5 + 20.0, "the table costs {best:?} us");
}
