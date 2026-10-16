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

/// The routed MLP apart: the gate-up select, the swiglu, the down select and
/// the weighted fold.
#[allow(clippy::too_many_arguments)]
fn mlp_apart(
    ctx: &kernels_rocm::Ctx,
    x: Tensor,
    up: Tensor,
    down: Tensor,
    rt: Tensor,
    weights: Tensor,
    packed: &mut Tensor,
    hidden: &mut Tensor,
    routed: &mut Tensor,
    y: &mut Tensor,
    inter: u32,
) {
    let resident = moe::ExpertTable::RESIDENT;
    moe::matmul_select(ctx, x, up, rt, packed, resident).unwrap();
    kernels_rocm::linear::mlp::swiglu(ctx, *packed, inter, 1, hidden).unwrap();
    moe::matmul_select(ctx, *hidden, down, rt, routed, resident).unwrap();
    moe::weighted_sum(ctx, *routed, weights, y).unwrap();
}

#[test]
fn the_sorted_routed_mlp_answers_the_ops_apart() {
    let Some(gpu) = Gpu::open() else { return };
    if lt_grouped::library(Dtype::Bf16).is_err() {
        return;
    }
    let dt = Dtype::Bf16;
    let (tokens, top_k, experts, hidden_w, inter) = (600usize, 4usize, 8usize, 320usize, 192usize);
    let packed_w = 2 * inter;
    let mut planes = Planes::default();
    let r = routes(tokens, top_k, experts, 11, 9);
    let count = r.len();
    let up = |v: Vec<f32>| v.iter().map(|&v| bits(dt, v)).collect::<Vec<u16>>();
    let x = planes.up(
        &up(values(tokens * hidden_w, 1, 1.0)),
        tokens as u32,
        hidden_w as u32,
        dt,
    );
    let w_up = planes.up(
        &up(values(experts * packed_w * hidden_w, 2, 0.1)),
        experts as u32,
        (packed_w * hidden_w) as u32,
        dt,
    );
    let w_down = planes.up(
        &up(values(experts * hidden_w * inter, 3, 0.1)),
        experts as u32,
        (hidden_w * inter) as u32,
        dt,
    );
    let rt = planes.ints(&r, tokens as u32, top_k as u32);
    let wb = hip::Buffer::from_slice(
        &values(count, 4, 0.5)
            .iter()
            .map(|v| v.abs() + 0.1)
            .collect::<Vec<f32>>(),
    )
    .unwrap();
    let weights = Tensor::new(wb.device_ptr(), tokens as u32, top_k as u32, Dtype::F32);
    let zeros = |rows: usize, w: usize| vec![0u16; rows * w];
    let mut packed = planes.up(&zeros(count, packed_w), count as u32, packed_w as u32, dt);
    let mut hid = planes.up(&zeros(count, inter), count as u32, inter as u32, dt);
    let mut routed = planes.up(&zeros(count, hidden_w), count as u32, hidden_w as u32, dt);
    let mut y_native = planes.up(&zeros(tokens, hidden_w), tokens as u32, hidden_w as u32, dt);
    let mut y_lt = planes.up(&zeros(tokens, hidden_w), tokens as u32, hidden_w as u32, dt);
    let mut y_sorted = planes.up(&zeros(tokens, hidden_w), tokens as u32, hidden_w as u32, dt);
    let ctx = gpu.ctx();
    {
        let _f = Forced::on(false);
        mlp_apart(
            &ctx,
            x,
            w_up,
            w_down,
            rt,
            weights,
            &mut packed,
            &mut hid,
            &mut routed,
            &mut y_native,
            inter as u32,
        );
    }
    let _f = Forced::on(true);
    // Apart through the hipBLASLt leg, which also picks its kernels.
    mlp_apart(
        &ctx,
        x,
        w_up,
        w_down,
        rt,
        weights,
        &mut packed,
        &mut hid,
        &mut routed,
        &mut y_lt,
        inter as u32,
    );
    let fired = moe::lt_grouped_fires();
    let mut glu = |ctx: &kernels_rocm::Ctx, p: Tensor, h: &mut Tensor| {
        kernels_rocm::linear::mlp::swiglu(ctx, p, inter as u32, 1, h)
    };
    let took = moe::experts_sorted(
        &ctx,
        x,
        w_up,
        moe::ExpertTable::RESIDENT,
        rt,
        weights,
        w_down,
        moe::ExpertTable::RESIDENT,
        packed_w as u32,
        inter as u32,
        &mut y_sorted,
        &mut glu,
    )
    .unwrap();
    assert!(
        took && moe::lt_grouped_fires() > fired,
        "the sorted MLP did not run"
    );
    let want = planes.read(&gpu, &y_native);
    for (what, t) in [("apart", &y_lt), ("sorted", &y_sorted)] {
        let have = planes.read(&gpu, t);
        for (i, (&h, &w)) in have.iter().zip(&want).enumerate() {
            let (h, w) = (widen(dt, h), widen(dt, w));
            assert!(
                (h - w).abs() <= w.abs() / 32.0 + 0.02,
                "{what} y[{}][{}] = {h}, native {w}",
                i / hidden_w,
                i % hidden_w
            );
        }
    }
}

// ---- benchmark -------------------------------------------------------------

#[test]
#[ignore = "benchmark"]
fn bench_grouped_leg() {
    let Some(gpu) = Gpu::open() else { return };
    let mut planes = Planes::default();
    for (name, tokens, top_k, experts, n, k) in [
        (
            "glm gate/up",
            8192usize,
            8usize,
            32usize,
            4096usize,
            4096usize,
        ),
        ("qwen-next gate/up", 8192, 10, 16, 2560, 1280),
        ("qwen-next down", 8192, 10, 16, 1280, 1280),
        ("glm 2k gate/up", 2048, 8, 32, 4096, 4096),
    ] {
        if std::env::var("LT_ONLY").is_ok_and(|o| !name.starts_with(&o)) {
            continue;
        }
        let r = routes(tokens, top_k, experts, 3, 0);
        let count = r.len();
        let xh: Vec<u16> = values(tokens * k, 1, 1.0)
            .iter()
            .map(|&v| common::bf16(v))
            .collect();
        // Random weights, no zeros.
        let wh: Vec<u16> = values(experts * n * k, 2, 0.05)
            .iter()
            .map(|&v| common::bf16(if v == 0.0 { 1e-3 } else { v }))
            .collect();
        let x = planes.up(&xh, tokens as u32, k as u32, Dtype::Bf16);
        let bank = planes.up(&wh, experts as u32, (n * k) as u32, Dtype::Bf16);
        let rt = planes.ints(&r, tokens as u32, top_k as u32);
        let mut y = planes.up(&vec![0; count * n], count as u32, n as u32, Dtype::Bf16);
        let flops = 2.0 * count as f64 * n as f64 * k as f64;
        let ctx = gpu.ctx();
        for (leg, force) in [("hipblaslt", true), ("native", false), ("hipblaslt", true)] {
            let _forced = Forced::on(force);
            let mut fire = || {
                moe::matmul_select(&ctx, x, bank, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
            };
            let us = gpu.time(10, &mut fire);
            let g = gpu.time_graph(10, &mut fire);
            println!(
                "{name:18} {tokens}x{top_k}/{experts} n={n} k={k} {leg:9}: {us:8.1} us {:6.1} TFLOP/s (graph {:6.1})",
                flops / us / 1e6,
                flops / g / 1e6
            );
        }
        // Calibration: hipBLASLt dense at the routed rows.
        let xs = planes.up(
            &vec![common::bf16(0.5); count * k],
            count as u32,
            k as u32,
            Dtype::Bf16,
        );
        let ws = hip::Buffer::alloc(hip::lt::WORKSPACE).unwrap();
        let shape = hip::lt::Shape {
            m: count as u32,
            n: n as u32,
            k: k as u32,
            bias: false,
            groups: 1,
        };
        let us = gpu.time(10, || {
            // SAFETY: xs is [count, k], the bank's first block [n, k], y [count, n].
            let ok = unsafe {
                hip::lt::matmul(
                    shape,
                    xs.ptr,
                    bank.ptr,
                    0,
                    y.ptr,
                    ws.device_ptr(),
                    gpu.stream.raw(),
                )
            };
            assert!(matches!(ok, Ok(true)));
        });
        println!(
            "{name:18} hipBLASLt dense M={count}: {:6.1} TFLOP/s",
            flops / us / 1e6
        );
        planes = Planes::default();
    }
}
