//! The routed select through hipBLASLt's grouped GEMM kernels
//! (`hip::lt_grouped`, the prefill grouped leg of `moe::matmul_select`)
//! against a host reference: bf16 and f16, rows by token and by route, dead
//! routes left as they were, and the leg captured in a graph. Run these on a
//! ROCm bump: they are what holds the argument contract.
//! `cargo test --release --features rocm --test lt_grouped -- --nocapture`
//! (the benchmark: `-- --ignored`).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::{self, lt_grouped};
use kernels_rocm::linear::moe;
use kernels_rocm::tuning::Tuning;

/// Forces the leg for a scope; one scope at a time, the switch being the
/// process's.
struct Forced(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl Forced {
    fn on(on: bool) -> Self {
        static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let guard = ONE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        moe::force_lt_grouped(Some(on));
        Forced(guard)
    }
}

impl Drop for Forced {
    fn drop(&mut self) {
        moe::force_lt_grouped(None);
    }
}

fn f16_bits(v: f32) -> u16 {
    // Round-to-nearest-even f32 → f16 for the normal range the tests use.
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    let mut h = (u32::from(sign) << 16 >> 16) | ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    if exp == 0 {
        sign * man * 2f32.powi(-24)
    } else {
        sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
    }
}

fn bits(dtype: Dtype, v: f32) -> u16 {
    match dtype {
        Dtype::Bf16 => common::bf16(v),
        Dtype::F16 => f16_bits(v),
        _ => unreachable!(),
    }
}

fn widen(dtype: Dtype, b: u16) -> f32 {
    match dtype {
        Dtype::Bf16 => common::from_bf16(b),
        Dtype::F16 => from_f16(b),
        _ => unreachable!(),
    }
}

/// Device planes of 16-bit values, read back as they are.
#[derive(Default)]
struct Planes {
    held: Vec<hip::Buffer>,
}

impl Planes {
    fn up(&mut self, v: &[u16], rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let b = hip::Buffer::from_slice(v).unwrap();
        let t = Tensor::new(b.device_ptr(), rows, width, dtype);
        self.held.push(b);
        t
    }
    fn ints(&mut self, v: &[i32], rows: u32, width: u32) -> Tensor {
        let b = hip::Buffer::from_slice(v).unwrap();
        let t = Tensor::new(b.device_ptr(), rows, width, Dtype::I32);
        self.held.push(b);
        t
    }
    fn read(&self, gpu: &Gpu, t: &Tensor) -> Vec<u16> {
        gpu.sync();
        let b = self.held.iter().find(|b| b.device_ptr() == t.ptr).unwrap();
        b.read::<u16>().unwrap()[..t.elements() as usize].to_vec()
    }
}

/// Distinct top-k routes over `experts`, every `dead`-th one -1.
fn routes(tokens: usize, top_k: usize, experts: usize, seed: u64, dead: usize) -> Vec<i32> {
    let keys = values(tokens * experts, seed, 1.0);
    let mut out = Vec::with_capacity(tokens * top_k);
    let mut ids: Vec<usize> = (0..experts).collect();
    for t in 0..tokens {
        let row = &keys[t * experts..(t + 1) * experts];
        ids.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        out.extend(ids[..top_k].iter().map(|&e| e as i32));
    }
    if dead > 0 {
        for i in (5..out.len()).step_by(dead) {
            out[i] = -1;
        }
    }
    // Under test, experts 3 and 9 take nothing: empty groups among the
    // full ones.
    if dead > 0 {
        for e in &mut out {
            if *e == 3 || *e == 9 {
                *e += 1;
            }
        }
    }
    out
}

/// One routed select through the forced leg against the host.
#[allow(clippy::too_many_arguments)]
fn check(
    gpu: &Gpu,
    dtype: Dtype,
    tokens: usize,
    top_k: usize,
    experts: usize,
    n: usize,
    k: usize,
    by_token: bool,
    graph: bool,
) {
    let mut planes = Planes::default();
    let r = routes(tokens, top_k, experts, 7, 13);
    let count = r.len();
    let x_rows = if by_token { tokens } else { count };
    let xh: Vec<u16> = values(x_rows * k, 1, 1.0)
        .iter()
        .map(|&v| bits(dtype, v))
        .collect();
    let wh: Vec<u16> = values(experts * n * k, 2, 0.1)
        .iter()
        .map(|&v| bits(dtype, v))
        .collect();
    let x = planes.up(&xh, x_rows as u32, k as u32, dtype);
    let bank = planes.up(&wh, experts as u32, (n * k) as u32, dtype);
    let rt = planes.ints(&r, tokens as u32, top_k as u32);
    // y starts as a pattern: a dead route's row must keep it.
    let fill = bits(dtype, 3.0);
    let mut y = planes.up(&vec![fill; count * n], count as u32, n as u32, dtype);
    let ctx = gpu.ctx();
    let _forced = Forced::on(true);
    let fired = moe::lt_grouped_fires();
    let mut fire = || {
        moe::matmul_select(&ctx, x, bank, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
    };
    if graph {
        gpu.time_graph(2, &mut fire);
    } else {
        fire();
    }
    assert!(
        moe::lt_grouped_fires() > fired,
        "the hipBLASLt leg did not fire"
    );
    let have = planes.read(gpu, &y);
    for (route, &e) in r.iter().enumerate() {
        let row = &have[route * n..(route + 1) * n];
        if e < 0 {
            assert!(
                row.iter().all(|&v| v == fill),
                "dead route {route} was written"
            );
            continue;
        }
        let src = if by_token { route / top_k } else { route };
        for j in 0..n {
            let mut want = 0f64;
            for i in 0..k {
                want += f64::from(widen(dtype, xh[src * k + i]))
                    * f64::from(widen(dtype, wh[e as usize * n * k + j * k + i]));
            }
            let got = widen(dtype, row[j]);
            let tol = want.abs() as f32 / 128.0 + 0.01;
            assert!(
                (got - want as f32).abs() <= tol,
                "{dtype:?} by_token={by_token} y[{route}][{j}] = {got}, want {want}"
            );
        }
    }
}

#[test]
fn the_library_opens_at_a_known_version() {
    let Some(_gpu) = Gpu::open() else { return };
    let Ok(version) = lt_grouped::version() else {
        eprintln!("no hipBLASLt; skipping");
        return;
    };
    eprintln!("hipBLASLt version {version}");
    for dtype in [Dtype::Bf16, Dtype::F16] {
        match lt_grouped::library(dtype) {
            Ok(lib) => {
                assert!(!lib.solutions.is_empty());
                eprintln!("{dtype:?}: {} grouped solutions", lib.solutions.len());
            }
            Err(why) => eprintln!("{dtype:?}: not offered: {why}"),
        }
    }
}

#[test]
fn the_grouped_leg_answers_the_host() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        eprintln!("hipBLASLt's grouped library is not offered; skipping");
        return;
    }
    // Past the decode batch legs (over 160 routes an expert): ragged and
    // empty experts, a width short of every tile, dead routes.
    check(&gpu, Dtype::Bf16, 1200, 3, 16, 328, 264, true, false);
    check(&gpu, Dtype::Bf16, 400, 4, 8, 512, 128, false, false);
    if lt_grouped::library(Dtype::F16).is_ok() {
        check(&gpu, Dtype::F16, 600, 2, 6, 200, 136, true, false);
    }
}

#[test]
fn the_grouped_leg_replays_in_a_graph() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    check(&gpu, Dtype::Bf16, 600, 4, 12, 256, 192, true, true);
}

/// One routed swiglu MLP's planes: the bank pair, the routes and weights,
/// and the steps' intermediate planes.
struct Mlp {
    p: moe::RoutedMlp,
    y: Tensor,
}

#[allow(clippy::too_many_arguments)]
fn mlp(
    planes: &mut Planes,
    dt: Dtype,
    tokens: usize,
    top_k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    dead: usize,
) -> Mlp {
    let r = routes(tokens, top_k, experts, 11, dead);
    let count = r.len();
    let up = |v: Vec<f32>| -> Vec<u16> {
        v.iter()
            .map(|&v| bits(dt, if v == 0.0 { 1e-3 } else { v }))
            .collect()
    };
    let x = planes.up(
        &up(values(tokens * hidden, 1, 1.0)),
        tokens as u32,
        hidden as u32,
        dt,
    );
    let gate_up = planes.up(
        &up(values(experts * 2 * inter * hidden, 2, 0.05)),
        experts as u32,
        (2 * inter * hidden) as u32,
        dt,
    );
    let down = planes.up(
        &up(values(experts * hidden * inter, 3, 0.05)),
        experts as u32,
        (hidden * inter) as u32,
        dt,
    );
    let rt = planes.ints(&r, tokens as u32, top_k as u32);
    let w: Vec<f32> = values(count, 4, 0.5)
        .iter()
        .map(|v| v.abs() + 0.1)
        .collect();
    let wb = hip::Buffer::from_slice(&w).unwrap();
    let weights = Tensor::new(wb.device_ptr(), tokens as u32, top_k as u32, Dtype::F32);
    planes.held.push(wb);
    let zeros = |rows: usize, w: usize| vec![0u16; rows * w];
    let packed = planes.up(
        &zeros(count, 2 * inter),
        count as u32,
        (2 * inter) as u32,
        dt,
    );
    let h = planes.up(&zeros(count, inter), count as u32, inter as u32, dt);
    let routed = planes.up(&zeros(count, hidden), count as u32, hidden as u32, dt);
    let y = planes.up(&zeros(tokens, hidden), tokens as u32, hidden as u32, dt);
    Mlp {
        p: moe::RoutedMlp {
            x,
            routes: rt,
            weights,
            gate_up,
            gate_up_experts: moe::ExpertTable::RESIDENT,
            down,
            down_experts: moe::ExpertTable::RESIDENT,
            intermediate: inter as u32,
            limit: None,
            fan: top_k as u32,
            packed,
            h,
            routed,
        },
        y,
    }
}

/// The tuning a path's profile would name: `lt` the hipBLASLt legs' (the
/// per-route select from `select`, the sorted MLP from `sorted`), `ordered`
/// the native expert-ordered MLP's, at the tiles given.
fn tuning(select: u32, sorted: u32, ordered: u32, tiles: Option<(u32, u32)>) -> Tuning {
    let mut t = Tuning::DEFAULT;
    t.moe_lt_grouped = select > 0 || sorted > 0;
    t.moe_lt_select_from = select;
    t.moe_lt_mlp_from = sorted;
    t.moe_ordered_from = ordered;
    if let Some((deep, shallow)) = tiles {
        t.moe_lt_tile_deep = deep;
        t.moe_lt_tile_shallow = shallow;
    }
    t
}

#[test]
fn the_sorted_routed_mlp_answers_the_steps() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let dt = Dtype::Bf16;
    let mut planes = Planes::default();
    let (tokens, hidden) = (600usize, 320usize);
    let m = mlp(&mut planes, dt, tokens, 4, 8, hidden, 192, 9);
    let mut want = m.y;
    let mut have = planes.up(&vec![0; tokens * hidden], tokens as u32, hidden as u32, dt);
    let steps = gpu.ctx().with_tuning(tuning(0, 0, 0, None));
    moe::mlp_swiglu(&steps, &m.p, &mut want).unwrap();
    let sorted = gpu.ctx().with_tuning(tuning(0, 1, 0, None));
    let fired = moe::lt_grouped_fires();
    moe::mlp_swiglu(&sorted, &m.p, &mut have).unwrap();
    assert_eq!(
        moe::lt_grouped_fires(),
        fired + 2,
        "the sorted MLP did not run"
    );
    let (want, have) = (planes.read(&gpu, &want), planes.read(&gpu, &have));
    for (i, (&h, &w)) in have.iter().zip(&want).enumerate() {
        let (h, w) = (widen(dt, h), widen(dt, w));
        assert!(
            (h - w).abs() <= w.abs() / 32.0 + 0.02,
            "y[{}][{}] = {h}, steps {w}",
            i / hidden,
            i % hidden
        );
    }
}

// ---- the profile's measurements ------------------------------------------------

/// The models' routed MLPs: (name, experts, top-k, hidden, intermediate).
const MODELS: [(&str, usize, usize, usize, usize); 3] = [
    ("glm53-flash", 32, 8, 4096, 2048),
    ("glm53", 8, 8, 6144, 2048),
    ("qwen38-flash-next", 16, 10, 2560, 640),
];

/// Every tile of the grouped library on the models' gate-up and down
/// projections at an 8192- and a 2048-token prefill, through the routed
/// select's hipBLASLt leg: the measurement behind `kernel.moe.lt_tile_*`.
/// `LT_MODEL=<name>` narrows it.
#[test]
#[ignore = "benchmark"]
fn bench_lt_tiles() {
    let Some(gpu) = Gpu::open() else { return };
    let Ok(lib) = lt_grouped::library(Dtype::Bf16) else {
        return;
    };
    let mut tiles: Vec<u32> = lib
        .solutions
        .iter()
        .filter(|s| s.mt[0] * s.mt[1] >= 128 * 64)
        .map(|s| s.mt[0] * 1_000_000 + s.mt[1] * 1_000 + s.mt[2])
        .collect();
    tiles.sort_unstable();
    tiles.dedup();
    for (name, experts, top_k, hidden, inter) in MODELS {
        if std::env::var("LT_MODEL").is_ok_and(|m| m != name) {
            continue;
        }
        for tokens in [8192usize, 2048] {
            let mut planes = Planes::default();
            let m = mlp(
                &mut planes,
                Dtype::Bf16,
                tokens,
                top_k,
                experts,
                hidden,
                inter,
                0,
            );
            let count = (tokens * top_k) as f64;
            for (leg, x, bank, mut y, n, k) in [
                ("gate-up", m.p.x, m.p.gate_up, m.p.packed, 2 * inter, hidden),
                ("down", m.p.h, m.p.down, m.p.routed, hidden, inter),
            ] {
                let flops = 2.0 * count * n as f64 * k as f64;
                let mut timed: Vec<(f64, u32)> = Vec::new();
                for &tile in &tiles {
                    let ctx = gpu.ctx().with_tuning(tuning(1, 0, 0, Some((tile, tile))));
                    let us = gpu.time(3, || {
                        moe::matmul_select(
                            &ctx,
                            x,
                            bank,
                            m.p.routes,
                            &mut y,
                            moe::ExpertTable::RESIDENT,
                        )
                        .unwrap();
                    });
                    timed.push((us, tile));
                }
                timed.sort_by(|a, b| a.0.total_cmp(&b.0));
                let native = {
                    let ctx = gpu.ctx().with_tuning(tuning(0, 0, 0, None));
                    gpu.time(5, || {
                        moe::matmul_select(
                            &ctx,
                            x,
                            bank,
                            m.p.routes,
                            &mut y,
                            moe::ExpertTable::RESIDENT,
                        )
                        .unwrap();
                    })
                };
                let best: Vec<String> = timed
                    .iter()
                    .take(5)
                    .map(|(us, t)| format!("{t:09} {:.0}", flops / us / 1e6))
                    .collect();
                println!(
                    "{name:18} {tokens:5} {leg:7} n={n} k={k}: native {:.0} TFLOP/s; leg by tile {}",
                    flops / native / 1e6,
                    best.join(", ")
                );
            }
        }
    }
}

/// The routed MLP's paths at the models' shapes over a prefill's token
/// counts: the steps (the native grouped legs), the native legs in expert
/// order, the steps with hipBLASLt's select leg, and the MLP kept by expert
/// over hipBLASLt's GEMM: the measurement behind `kernel.moe.ordered_from`,
/// `kernel.moe.lt_select_from` and `kernel.moe.lt_mlp_from`.
/// `LT_MODEL=<name>` narrows it; `LT_TILES=deep,shallow` names the tiles.
#[test]
#[ignore = "benchmark"]
fn bench_moe_mlp_paths() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let tiles = std::env::var("LT_TILES").ok().map(|v| {
        let mut it = v
            .split(',')
            .map(|t| t.replace('_', "").parse::<u32>().unwrap());
        (it.next().unwrap(), it.next().unwrap())
    });
    for (name, experts, top_k, hidden, inter) in MODELS {
        if std::env::var("LT_MODEL").is_ok_and(|m| m != name) {
            continue;
        }
        for tokens in [256usize, 512, 1024, 2048, 4096, 8192, 16384] {
            let mut planes = Planes::default();
            let mut m = mlp(
                &mut planes,
                Dtype::Bf16,
                tokens,
                top_k,
                experts,
                hidden,
                inter,
                0,
            );
            let per = tokens * top_k / experts;
            let flops = 2.0 * (tokens * top_k) as f64 * (3 * inter * hidden) as f64;
            let mut row = Vec::new();
            for (path, t) in [
                ("steps", tuning(0, 0, 0, tiles)),
                ("ordered", tuning(0, 0, 1, tiles)),
                ("lt-select", tuning(1, 0, 0, tiles)),
                ("lt-sorted", tuning(0, 1, 0, tiles)),
            ] {
                let ctx = gpu.ctx().with_tuning(t);
                let us = gpu.time(5, || moe::mlp_swiglu(&ctx, &m.p, &mut m.y).unwrap());
                row.push(format!("{path} {us:8.1} us ({:3.0})", flops / us / 1e6));
            }
            println!(
                "{name:18} {tokens:5} tokens ({per:5}/expert): {}",
                row.join("  ")
            );
        }
    }
}
