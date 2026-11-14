//! Every routed select (and the fused routed MLPs) through an expert table:
//! the contiguous bank with no table, with its own (degenerate) table, with
//! its experts scattered across a pool in a shuffled order with garbage
//! between them (the bank itself garbled after), and with a mixed table
//! (even experts at the bank, odd ones in the pool) all land the same bits,
//! and the table's counter counts each expert's routes; a table with two
//! experts swapped lands other bits. A table that may name mapped host
//! memory (not located) takes the per-route stream, landing its bits, or the
//! arm a located one takes (an affine bank's), landing the same bits, or is
//! refused where a codec has no stream.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/expert_pool.rs"]
mod expert_pool;

use common::{Gpu, bf16s, values};
use dtype::Dtype;
use expert_pool::{Plane, degenerate, fill, garble, mixed, scatter, swapped, table};
use kernels_rocm::hip::lt_grouped;
use kernels_rocm::linear::moe::{self, ExpertTable};
use kernels_rocm::tuning::Tuning;
use kernels_rocm::{Ctx, Error, Tensor};

/// Routes naming every expert, experts 0 and 1 among the first token's, no
/// token's twice; every 17th from the fourth dead (-1) when `dead`.
fn routes(tokens: usize, top: usize, experts: usize, dead: bool) -> Vec<i32> {
    (0..tokens * top)
        .map(|r| {
            let (t, j) = (r / top, r % top);
            if dead && r % 17 == 3 {
                -1
            } else {
                ((t * 3 + t / 7 + j) % experts) as i32
            }
        })
        .collect()
}

/// The routes each expert serves.
fn served(routes: &[i32], experts: usize) -> Vec<u32> {
    let mut want = vec![0u32; experts];
    for &e in routes {
        if e >= 0 {
            want[e as usize] += 1;
        }
    }
    want
}

fn tuned(g: &Gpu, f: impl FnOnce(&mut Tuning)) -> Ctx {
    let mut t = Tuning::DEFAULT;
    f(&mut t);
    g.ctx().with_tuning(t)
}

/// `count` random bytes in `lo..=hi`.
fn bytes(count: usize, seed: u64, lo: u8, hi: u8) -> Vec<u8> {
    let span = u32::from(hi - lo) + 1;
    values(count, seed, 1.0)
        .into_iter()
        .map(|v| lo + (((v + 1.0) * 0.5 * span as f32) as u32).min(span - 1) as u8)
        .collect()
}

/// A routed fire under test: its banks (`(planes, experts)` each), the
/// routes they serve, and the result plane.
struct Case<'a> {
    what: String,
    banks: Vec<(&'a [Plane], usize)>,
    routes: &'a [i32],
    y: Tensor,
}

/// What a table that may name mapped host memory takes.
#[derive(Clone, Copy)]
enum Host<'a> {
    /// The per-route stream: the bits a context whose fire takes the
    /// stream lands.
    Stream(&'a Ctx),
    /// The arm a located table takes: the same bits.
    Same,
    /// No arm: the fire declines it.
    Refused,
}

impl<'a> Host<'a> {
    fn of(stream: Option<&'a Ctx>) -> Host<'a> {
        stream.map_or(Host::Refused, Host::Stream)
    }
}

/// `fire` (whether the arm took the fire) under each table: no table, the
/// banks' own, a mixed one and the scattered pool's land one result bit for
/// bit, and each counter the routes its experts serve; a table with bank
/// 0's experts 0 and 1 swapped lands another. A table that may name host
/// memory lands what `host` says.
fn same_bits(
    g: &Gpu,
    c: &Case,
    ctx: &Ctx,
    host: Host<'_>,
    fire: impl Fn(&Ctx, &[ExpertTable]) -> Result<bool, Error>,
) {
    let what = &c.what;
    let y_bytes = (c.y.rows * c.y.width) as usize * 2;
    let run = |at: &Ctx, tables: &[ExpertTable]| -> Vec<u16> {
        fill(g, c.y.ptr, 0x7f, y_bytes);
        let took = g
            .laid(at, |at| fire(at, tables))
            .unwrap_or_else(|why| panic!("{what}: {why}"));
        assert!(took, "{what}: the arm declined the fire");
        g.read_as::<u16>(&c.y)
    };
    let counters = || -> Vec<Tensor> {
        c.banks
            .iter()
            .map(|&(_, experts)| g.zeros(experts as u32, 1, Dtype::U32))
            .collect()
    };
    let counted = |hits: &[Tensor], how: &str| {
        let want: Vec<Vec<u32>> = c
            .banks
            .iter()
            .map(|&(_, experts)| served(c.routes, experts))
            .collect();
        let have: Vec<Vec<u32>> = hits.iter().map(|h| g.read_as::<u32>(h)).collect();
        assert_eq!(have, want, "{what}: {how} counts each expert's routes");
    };
    let resident = vec![ExpertTable::RESIDENT; c.banks.len()];
    let want = run(ctx, &resident);
    assert!(
        want.iter().any(|&v| v != 0x7f7f),
        "{what}: the fire landed nothing"
    );
    let stream_want = match host {
        Host::Stream(s) => Some(run(s, &resident)),
        Host::Same | Host::Refused => None,
    };

    let own: Vec<ExpertTable> = c
        .banks
        .iter()
        .map(|&(planes, experts)| {
            ExpertTable::located(table(g, &degenerate(experts, planes)), (0, 0))
        })
        .collect();
    assert!(
        run(ctx, &own) == want,
        "{what}: the bank's own table moved a bit"
    );

    let pools: Vec<_> = c
        .banks
        .iter()
        .enumerate()
        .map(|(i, &(planes, experts))| {
            scatter(g, experts, planes, 0x51ce ^ (experts * 7 + i) as u64)
        })
        .collect();
    let hits = counters();
    let mixed_t: Vec<ExpertTable> = c
        .banks
        .iter()
        .zip(&pools)
        .zip(&hits)
        .map(|((&(planes, experts), pool), h)| {
            ExpertTable::located(table(g, &mixed(experts, planes, pool)), pool.window())
                .counting(h.ptr)
        })
        .collect();
    assert!(
        run(ctx, &mixed_t) == want,
        "{what}: the mixed table moved a bit"
    );
    counted(&hits, "the mixed table");

    for &(planes, experts) in &c.banks {
        garble(g, experts, planes);
    }
    let hits = counters();
    let scattered: Vec<ExpertTable> = pools
        .iter()
        .zip(&hits)
        .map(|(pool, h)| {
            ExpertTable::located(table(g, &pool.entries), pool.window()).counting(h.ptr)
        })
        .collect();
    assert!(
        run(ctx, &scattered) == want,
        "{what}: the scattered table moved a bit"
    );
    counted(&hits, "the scattered table");

    let hits = counters();
    let host_t: Vec<ExpertTable> = scattered
        .iter()
        .zip(&hits)
        .map(|(t, h)| ExpertTable {
            located: false,
            ..t.counting(h.ptr)
        })
        .collect();
    match (host, stream_want) {
        (Host::Stream(_), Some(stream_want)) => {
            assert!(
                run(ctx, &host_t) == stream_want,
                "{what}: a table that may name host memory moved off the stream's bits"
            );
            counted(&hits, "the stream");
        }
        (Host::Same, _) => {
            assert!(
                run(ctx, &host_t) == want,
                "{what}: a table that may name host memory moved a bit"
            );
            counted(&hits, "the host table");
        }
        _ => assert!(
            !matches!(fire(ctx, &host_t), Ok(true)),
            "{what}: an arm with no stream took a table that may name host memory"
        ),
    }

    let mut wrong = scattered.clone();
    let count = c.banks[0].0.len();
    wrong[0] = ExpertTable::located(
        table(g, &swapped(&pools[0].entries, count, 0, 1)),
        pools[0].window(),
    );
    assert!(
        run(ctx, &wrong) != want,
        "{what}: a table with two experts swapped landed the same bits"
    );
}

/// The tuning the dense stream alone takes.
fn stream_only(t: &mut Tuning) {
    t.moe_batch = false;
    t.moe_grouped = false;
    t.moe_dense_batch_until = 0;
    t.mxfp4_grouped = false;
    t.affine_grouped = false;
    t.moe_lt_grouped = false;
    t.moe_ordered_from = 0;
}

#[test]
fn a_dense_select_lands_the_same_bits_through_any_table() {
    let Some(g) = Gpu::open() else { return };
    // (experts, tokens, top, n, k, arm): the decode stream, the batch leg,
    // the wide leg, the GEMM tile (past 64 routes an expert), the grouped
    // matrix-core leg and its tall block, and rocBLAS (a K the native legs
    // cannot read).
    #[rustfmt::skip]
    let cases = [
        (8usize, 3usize, 2usize, 96usize, 512usize, "stream", Dtype::Bf16),
        (4, 3, 2, 72, 1024, "stream", Dtype::F16),
        (8, 40, 4, 520, 264, "batch", Dtype::Bf16),
        (4, 40, 4, 136, 512, "batch", Dtype::F16),
        (8, 64, 4, 520, 256, "wide", Dtype::Bf16),
        (2, 220, 2, 72, 2048, "gemm", Dtype::Bf16),
        (8, 40, 4, 520, 256, "grouped", Dtype::Bf16),
        (4, 128, 4, 512, 2048, "grouped", Dtype::Bf16),
        (3, 20, 2, 50, 60, "blas", Dtype::Bf16),
    ];
    for (ex, tokens, top, n, k, arm, dt) in cases {
        let ctx = tuned(&g, |t| {
            t.moe_grouped = matches!(arm, "grouped" | "blas");
            t.moe_batch = matches!(arm, "batch" | "wide");
            t.moe_batch_from = 1;
            t.moe_dense_batch_until = match arm {
                "batch" => 64,
                "wide" | "gemm" => 4096,
                _ => 0,
            };
            t.moe_dense_wide_from = u32::from(arm == "wide");
        });
        let stream = tuned(&g, stream_only);
        for by_token in [true, false] {
            let bank = g.tensor(
                &bf16s(ex * n * k, (n + k) as u64, 0.1),
                ex as u32,
                (n * k) as u32,
                dt,
            );
            let planes = [Plane::of(bank.ptr, (n * k * 2) as u64)];
            let r = routes(tokens, top, ex, arm != "blas");
            let rows = if by_token { tokens } else { tokens * top };
            let x = g.tensor(&bf16s(rows * k, 12, 1.0), rows as u32, k as u32, dt);
            let rt = g.upload(&r, tokens as u32, top as u32, Dtype::I32);
            let y = g.zeros((tokens * top) as u32, n as u32, dt);
            let case = Case {
                what: format!("{dt:?} {arm} {ex}x{tokens}x{top} {n}x{k} by_token {by_token}"),
                banks: vec![(&planes, ex)],
                routes: &r,
                y,
            };
            // rocBLAS reads no table that may name host memory, and a K
            // off 16-byte loads has no stream to leave it to.
            let stream = (arm != "blas").then_some(&stream);
            same_bits(&g, &case, &ctx, Host::of(stream), |at, tables| {
                let mut y = y;
                moe::matmul_select(at, x, bank, rt, &mut y, tables[0]).map(|()| true)
            });
        }
    }
}

/// An MXFP4 bank (`[codes, scales, bias]`) or an affine one (`[codes,
/// scales, zeros]`) of `ex` experts of `n` x `k`; `bits` 0 is MXFP4.
fn packed_bank(g: &Gpu, ex: usize, n: usize, k: usize, bits: usize, group: usize) -> [Tensor; 3] {
    let seed = (ex * 31 + n + k + bits) as u64;
    if bits == 0 {
        let codes = g.upload(
            &bytes(ex * n * k / 2, seed, 0, 255),
            ex as u32,
            (n * k / 2) as u32,
            Dtype::U8,
        );
        let scales = g.upload(
            &bytes(ex * n * k / 32, seed + 1, 120, 126),
            ex as u32,
            (n * k / 32) as u32,
            Dtype::U8,
        );
        let bias = g.tensor(
            &bf16s(ex * n, seed + 2, 1.0),
            ex as u32,
            n as u32,
            Dtype::Bf16,
        );
        return [codes, scales, bias];
    }
    let factors = |seed: u64, lo: f32, span: f32| -> Vec<u8> {
        values(ex * n * k / group, seed, 1.0)
            .into_iter()
            .flat_map(|v| common::bf16(lo + span * (v + 1.0) * 0.5).to_le_bytes())
            .collect()
    };
    let row = 2 * n * k / group;
    [
        g.upload(
            &bytes(ex * n * k * bits / 8, seed, 0, 255),
            ex as u32,
            (n * k * bits / 8) as u32,
            Dtype::U8,
        ),
        g.upload(
            &factors(seed + 1, 0.01, 0.02),
            ex as u32,
            row as u32,
            Dtype::U8,
        ),
        g.upload(
            &factors(seed + 2, -0.05, 0.1),
            ex as u32,
            row as u32,
            Dtype::U8,
        ),
    ]
}

#[test]
fn an_fp4_or_affine_select_lands_the_same_bits_through_any_table() {
    let Some(g) = Gpu::open() else { return };
    // (bits (0: MXFP4), group, experts, tokens, top, n, k, arm, biased): the
    // stream, the batch leg, the grouped matrix-core and f32 legs.
    #[rustfmt::skip]
    let cases = [
        (0usize, 32usize, 8usize, 3usize, 4usize, 256usize, 1024usize, "stream", false),
        (0, 32, 8, 3, 4, 256, 1024, "stream", true),
        (0, 32, 8, 50, 4, 384, 512, "batch", true),
        (0, 32, 4, 2, 2, 100, 7168, "batch", false),
        (0, 32, 8, 50, 4, 384, 512, "wmma", true),
        (0, 32, 8, 50, 4, 200, 512, "simt", true),
        (4, 64, 6, 2, 3, 640, 2560, "stream", false),
        (8, 32, 4, 1, 2, 70, 1024, "stream", false),
        (4, 64, 8, 40, 4, 264, 512, "batch", false),
        (4, 64, 8, 40, 4, 264, 512, "wmma", false),
        (4, 32, 8, 40, 4, 136, 384, "simt", false),
    ];
    for (bits, group, ex, tokens, top, n, k, arm, biased) in cases {
        let ctx = tuned(&g, |t| {
            t.mxfp4_grouped = matches!(arm, "wmma" | "simt");
            t.affine_grouped = matches!(arm, "wmma" | "simt");
            t.affine_grouped_from = 1;
            t.mxfp4_wmma = arm == "wmma";
            t.affine_wmma = arm == "wmma";
            t.moe_batch = arm == "batch";
            t.moe_batch_from = 1;
        });
        let stream = tuned(&g, stream_only);
        for by_token in [true, false] {
            let [codes, scales, third] = packed_bank(&g, ex, n, k, bits, group);
            let row = |t: Tensor| u64::from(t.width) * t.dtype.bytes_ceil();
            let bias_at = if bits == 0 && !biased { 0 } else { third.ptr };
            let planes = [
                Plane::of(codes.ptr, row(codes)),
                Plane::of(scales.ptr, row(scales)),
                Plane::of(bias_at, row(third)),
            ];
            let r = routes(tokens, top, ex, true);
            let rows = if by_token { tokens } else { tokens * top };
            let x = g.tensor(
                &bf16s(rows * k, 42, 1.0),
                rows as u32,
                k as u32,
                Dtype::Bf16,
            );
            let rt = g.upload(&r, tokens as u32, top as u32, Dtype::I32);
            let y = g.zeros((tokens * top) as u32, n as u32, Dtype::Bf16);
            let kind = if bits == 0 {
                "mxfp4".to_string()
            } else {
                format!("u{bits}g{group}")
            };
            let case = Case {
                what: format!(
                    "{kind} {arm} {ex}x{tokens}x{top} {n}x{k} bias {biased} by_token {by_token}"
                ),
                banks: vec![(&planes, ex)],
                routes: &r,
                y,
            };
            // An affine or MXFP4 bank's arms read a host table as a located
            // one; eight-bit codes have the stream alone.
            let host = if bits == 8 {
                Host::Stream(&stream)
            } else {
                Host::Same
            };
            same_bits(&g, &case, &ctx, host, |at, tables| {
                let mut y = y;
                if bits == 0 && biased {
                    moe::matmul_select_bias(at, x, codes, scales, third, rt, &mut y, tables[0])
                } else {
                    let zeros = (bits != 0).then_some(third);
                    moe::matmul_select_quant(at, x, codes, scales, zeros, rt, &mut y, tables[0])
                }
                .map(|()| true)
            });
        }
    }
}

#[test]
fn a_lane_order_or_fp8_select_lands_the_same_bits_through_any_table() {
    let Some(g) = Gpu::open() else { return };
    let ctx = g.ctx();
    // An MXFP4 bank in lane order: by route (one token), in expert order.
    for (ex, tokens, top, n, k) in [
        (6usize, 1usize, 6usize, 208usize, 640usize),
        (8, 50, 4, 384, 512),
    ] {
        let [codes, scales, _] = packed_bank(&g, ex, n, k, 0, 32);
        let lc = g.zeros(ex as u32, (n * k / 2) as u32, Dtype::U8);
        let ls = g.zeros(ex as u32, (n * k / 32) as u32, Dtype::U8);
        moe::mxfp4_lanes(
            &ctx,
            "test",
            codes.ptr,
            lc.ptr,
            scales.ptr,
            ls.ptr,
            (ex * n) as u64,
            k as u64,
        )
        .expect("the bank goes to lane order");
        let planes = [
            Plane::of(lc.ptr, (n * k / 2) as u64),
            Plane::of(ls.ptr, (n * k / 32) as u64),
        ];
        let r = routes(tokens, top, ex, true);
        let x = g.tensor(
            &bf16s(tokens * k, 7, 1.0),
            tokens as u32,
            k as u32,
            Dtype::Bf16,
        );
        let rt = g.upload(&r, tokens as u32, top as u32, Dtype::I32);
        let y = g.zeros((tokens * top) as u32, n as u32, Dtype::Bf16);
        let case = Case {
            what: format!("mxfp4 lanes {ex}x{tokens}x{top} {n}x{k}"),
            banks: vec![(&planes, ex)],
            routes: &r,
            y,
        };
        same_bits(&g, &case, &ctx, Host::Same, |at, tables| {
            let mut y = y;
            moe::matmul_select_mxfp4_lanes(at, x, lc, ls, rt, &mut y, tables[0]).map(|()| true)
        });
    }
    // An `e4m3tile128` bank: the decode tile, the prefill tile (past 64
    // routes an expert).
    for (tokens, top, n, k) in [
        (1usize, 4usize, 256usize, 512usize),
        (24, 6, 384, 1024),
        (200, 4, 320, 512),
    ] {
        let ex = 6usize;
        let (nb, kb) = (n.div_ceil(128), k / 128);
        // Codes off 0x80 (fnuz's NaN), factors near 2^-8.
        let codes = g.upload(
            &bytes(ex * n * k, (n + k) as u64, 0, 127),
            ex as u32,
            (n * k) as u32,
            Dtype::U8,
        );
        let factors: Vec<u8> = values(ex * nb * kb, 3, 1.0)
            .into_iter()
            .flat_map(|v| (0.004 + 0.002 * v).to_le_bytes())
            .collect();
        let factors = g.upload(&factors, ex as u32, (4 * nb * kb) as u32, Dtype::U8);
        let planes = [
            Plane::of(codes.ptr, (n * k) as u64),
            Plane::of(factors.ptr, (4 * nb * kb) as u64),
        ];
        let r = routes(tokens, top, ex, false);
        let x = g.tensor(
            &bf16s(tokens * k, 43, 2.0),
            tokens as u32,
            k as u32,
            Dtype::Bf16,
        );
        let rt = g.upload(&r, tokens as u32, top as u32, Dtype::I32);
        let y = g.zeros((tokens * top) as u32, n as u32, Dtype::Bf16);
        let case = Case {
            what: format!("fp8 {ex}x{tokens}x{top} {n}x{k}"),
            banks: vec![(&planes, ex)],
            routes: &r,
            y,
        };
        same_bits(&g, &case, &ctx, Host::Refused, |at, tables| {
            let mut y = y;
            moe::matmul_select_fp8_mfma(at, x, codes, factors, rt, &mut y, tables[0]).map(|()| true)
        });
    }
}

/// A routed swiglu MLP of `ex` experts, its planes.
fn routed_mlp(
    g: &Gpu,
    r: &[i32],
    [tokens, top, ex, hidden, inter]: [usize; 5],
) -> (moe::RoutedMlp, Tensor) {
    let dt = Dtype::Bf16;
    let count = tokens * top;
    let plane = |n: usize, seed: u64, scale: f32, rows: usize| {
        g.tensor(&bf16s(n, seed, scale), rows as u32, (n / rows) as u32, dt)
    };
    let w: Vec<f32> = values(count, 4, 0.5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let p = moe::RoutedMlp {
        x: plane(tokens * hidden, 1, 1.0, tokens),
        routes: g.upload(r, tokens as u32, top as u32, Dtype::I32),
        weights: g.tensor(&w, tokens as u32, top as u32, Dtype::F32),
        gate_up: plane(ex * 2 * inter * hidden, 2, 0.05, ex),
        gate_up_experts: ExpertTable::RESIDENT,
        down: plane(ex * hidden * inter, 3, 0.05, ex),
        down_experts: ExpertTable::RESIDENT,
        intermediate: inter as u32,
        limit: None,
        fan: top as u32,
        packed: g.zeros(count as u32, (2 * inter) as u32, dt),
        h: g.zeros(count as u32, inter as u32, dt),
        routed: g.zeros(count as u32, hidden as u32, dt),
    };
    (p, g.zeros(tokens as u32, hidden as u32, dt))
}

#[test]
fn a_routed_mlp_lands_the_same_bits_through_any_table() {
    let Some(g) = Gpu::open() else { return };
    let lt = lt_grouped::library(Dtype::Bf16).is_ok();
    // (tokens, top, experts, hidden, inter, path): the one-token decode
    // launches, the decode batch legs, the expert-ordered MLP (its gate-up
    // on the tall block), the hipBLASLt sorted MLP.
    #[rustfmt::skip]
    let cases = [
        (1usize, 8usize, 32usize, 1024usize, 512usize, "decode"),
        (32, 8, 32, 1024, 512, "decode"),
        (1024, 8, 32, 2048, 256, "ordered"),
        (600, 4, 8, 320, 192, "sorted"),
    ];
    for (tokens, top, ex, hidden, inter, path) in cases {
        if path == "sorted" && !lt {
            continue;
        }
        let ctx = tuned(&g, |t| match path {
            "ordered" => t.moe_ordered_from = 1,
            "sorted" => {
                t.moe_lt_grouped = true;
                t.moe_lt_mlp_from = 1;
                t.moe_lt_deep_k = 64;
            }
            _ => {}
        });
        let stream = tuned(&g, stream_only);
        let r = routes(tokens, top, ex, true);
        let (p, y) = routed_mlp(&g, &r, [tokens, top, ex, hidden, inter]);
        let gate_up = [Plane::of(p.gate_up.ptr, (2 * inter * hidden * 2) as u64)];
        let down = [Plane::of(p.down.ptr, (hidden * inter * 2) as u64)];
        let case = Case {
            what: format!("{path} mlp {ex}x{tokens}x{top} {hidden}x{inter}"),
            banks: vec![(&gate_up, ex), (&down, ex)],
            routes: &r,
            y,
        };
        let fired = moe::lt_grouped_fires();
        let decode = path == "decode";
        same_bits(
            &g,
            &case,
            &ctx,
            Host::of((!decode).then_some(&stream)),
            |at, tables| {
                let p = moe::RoutedMlp {
                    gate_up_experts: tables[0],
                    down_experts: tables[1],
                    ..p
                };
                let mut y = y;
                if decode {
                    moe::decode_mlp(at, &p, None, &mut y)
                } else {
                    moe::mlp_swiglu(at, &p, &mut y).map(|()| true)
                }
            },
        );
        if path == "sorted" {
            assert!(
                moe::lt_grouped_fires() > fired,
                "the sorted MLP did not run"
            );
        }
    }
}
