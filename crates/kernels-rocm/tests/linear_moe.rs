//! Mixture of experts against host references: the routers (softmax,
//! sigmoid with and without sinks, sqrt-softplus, the hash table, the group
//! fan-out), the folds back into token rows, and the expert matmuls over a
//! dense, an mxfp4 and an MLX affine bank on each of their legs (the decode
//! stream, the batch leg, the grouped matrix-core leg, the off-core grouped
//! leg, rocBLAS).

#![cfg(feature = "rocm")]

mod common;

use std::cell::RefCell;

use common::{Gpu, bf16, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::hip;
use kernels_rocm::linear::{moe, moe_route};
use kernels_rocm::tuning::Tuning;
use kernels_rocm::{Ctx, Tensor};

/// Device planes the harness does not carry: raw bytes and i32s, read back
/// as they are.
#[derive(Default)]
struct Planes {
    held: RefCell<Vec<hip::Buffer>>,
}

impl Planes {
    fn keep(&self, buffer: hip::Buffer, rows: u32, width: u32, dtype: Dtype) -> Tensor {
        let ptr = buffer.device_ptr();
        self.held.borrow_mut().push(buffer);
        Tensor::new(ptr, rows, width, dtype)
    }
    fn bytes(&self, b: &[u8], rows: u32, width: u32) -> Tensor {
        self.keep(hip::Buffer::from_slice(b).unwrap(), rows, width, Dtype::U8)
    }
    fn i32s(&self, v: &[i32], rows: u32, width: u32) -> Tensor {
        assert_eq!(v.len(), (rows * width) as usize);
        self.keep(hip::Buffer::from_slice(v).unwrap(), rows, width, Dtype::I32)
    }
    fn i64s(&self, v: &[i64], rows: u32, width: u32) -> Tensor {
        self.keep(hip::Buffer::from_slice(v).unwrap(), rows, width, Dtype::I64)
    }
    fn read_i32(&self, gpu: &Gpu, t: &Tensor) -> Vec<i32> {
        gpu.sync();
        let held = self.held.borrow();
        let b = held.iter().find(|b| b.device_ptr() == t.ptr).unwrap();
        b.read::<i32>().unwrap()[..t.elements() as usize].to_vec()
    }
}

fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

/// Distinct logits, exact in bf16: a shuffle of k / 32 for k in -E/2..E/2,
/// so no two experts tie however the device rounds a score.
fn distinct_logits(rows: usize, experts: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * experts);
    for r in 0..rows {
        let keys = values(experts, seed + r as u64, 1.0);
        let mut ids: Vec<usize> = (0..experts).collect();
        ids.sort_by(|&a, &b| keys[a].total_cmp(&keys[b]));
        let mut row = vec![0.0; experts];
        for (rank, &e) in ids.iter().enumerate() {
            row[e] = (rank as f32 - experts as f32 / 2.0) / 32.0;
        }
        out.extend(row);
    }
    out
}

/// The top `k` of `rank` (larger first, the lower index on a tie).
fn top(rank: &[f64], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..rank.len()).collect();
    ids.sort_by(|&a, &b| rank[b].total_cmp(&rank[a]).then(a.cmp(&b)));
    ids.truncate(k);
    ids
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sqrt_softplus(x: f64) -> f64 {
    let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
    sp.max(0.0).sqrt()
}

fn check_routes(what: &str, have_ids: &[i32], have_w: &[f32], want: &[(i32, f64)]) {
    let ids: Vec<i32> = want.iter().map(|w| w.0).collect();
    assert_eq!(have_ids, &ids[..], "{what}: picks");
    let ws: Vec<f32> = want.iter().map(|w| w.1 as f32).collect();
    close_abs(&format!("{what}: weights"), have_w, &ws, 2e-6);
}

#[test]
fn moe_routers() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let planes = Planes::default();

    for (rows, experts, k) in [
        (3u32, 8u32, 2u32),
        (5, 60, 4),
        (4, 128, 8),
        (2, 512, 10),
        (3, 384, 6),
        (2, 256, 16),
    ] {
        let (r, e, kk) = (rows as usize, experts as usize, k as usize);
        let logits = distinct_logits(r, e, u64::from(experts));
        let lt = gpu.tensor(&logits, rows, experts, Dtype::Bf16);
        let gain = bf16s(e, 7, 2.0);
        let gt = gpu.tensor(&gain, 1, experts, Dtype::Bf16);
        let bias: Vec<f32> = (0..e)
            .map(|i| ((i * 7919) % e) as f32 / e as f32 * 0.25)
            .collect();
        let bt = gpu.tensor(&bias, 1, experts, Dtype::F32);

        // softmax, plain and with the per-expert gain.
        for scaled in [false, true] {
            let mut ids = planes.i32s(&vec![0; r * kk], rows, k);
            let mut ws = gpu.zeros(rows, k, Dtype::F32);
            if scaled {
                moe::topk_softmax_scaled(&ctx, lt, gt, experts, k, &mut ids, &mut ws).unwrap();
            } else {
                moe::topk_softmax(&ctx, lt, experts, k, &mut ids, &mut ws).unwrap();
            }
            let (hi, hw) = (planes.read_i32(&gpu, &ids), gpu.read(&ws));
            for row in 0..r {
                let x = &logits[row * e..(row + 1) * e];
                let m = x.iter().fold(f64::MIN, |a, &v| a.max(f64::from(v)));
                let z: f64 = x.iter().map(|&v| (f64::from(v) - m).exp()).sum();
                let p: Vec<f64> = x.iter().map(|&v| (f64::from(v) - m).exp() / z).collect();
                let picks = top(&p, kk);
                let sum: f64 = picks.iter().map(|&i| p[i]).sum();
                let want: Vec<(i32, f64)> = picks
                    .iter()
                    .map(|&i| {
                        let g = if scaled { f64::from(gain[i]) } else { 1.0 };
                        (i as i32, p[i] / sum * g)
                    })
                    .collect();
                check_routes(
                    &format!(
                        "softmax{} {experts}x{k} row {row}",
                        if scaled { "_scaled" } else { "" }
                    ),
                    &hi[row * kk..(row + 1) * kk],
                    &hw[row * kk..(row + 1) * kk],
                    &want,
                );
            }
        }

        // sigmoid and sqrt-softplus, with and without the bias, renormalized
        // or scaled.
        for (sqrt, with_bias, renormalize) in [
            (false, false, true),
            (false, true, true),
            (false, true, false),
            (true, true, true),
            (true, true, false),
        ] {
            let mut ids = planes.i32s(&vec![0; r * kk], rows, k);
            let mut ws = gpu.zeros(rows, k, Dtype::F32);
            let scaling = 2.5;
            if sqrt {
                moe::topk_sqrt_softplus(
                    &ctx,
                    lt,
                    bt,
                    experts,
                    k,
                    renormalize,
                    scaling,
                    &mut ids,
                    &mut ws,
                )
                .unwrap();
            } else {
                let b = with_bias.then_some(bt);
                moe::topk_sigmoid(
                    &ctx,
                    lt,
                    b,
                    experts,
                    k,
                    renormalize,
                    scaling,
                    &mut ids,
                    &mut ws,
                )
                .unwrap();
            }
            let (hi, hw) = (planes.read_i32(&gpu, &ids), gpu.read(&ws));
            for row in 0..r {
                let x = &logits[row * e..(row + 1) * e];
                let s: Vec<f64> = x
                    .iter()
                    .map(|&v| {
                        if sqrt {
                            sqrt_softplus(f64::from(v))
                        } else {
                            sigmoid(f64::from(v))
                        }
                    })
                    .collect();
                let rank: Vec<f64> = s
                    .iter()
                    .zip(&bias)
                    .map(|(&s, &b)| if with_bias { s + f64::from(b) } else { s })
                    .collect();
                let picks = top(&rank, kk);
                let sum: f64 = picks.iter().map(|&i| s[i]).sum();
                let scale = if renormalize {
                    f64::from(scaling) / sum
                } else {
                    f64::from(scaling)
                };
                let want: Vec<(i32, f64)> =
                    picks.iter().map(|&i| (i as i32, s[i] * scale)).collect();
                check_routes(
                    &format!(
                        "sqrt={sqrt} bias={with_bias} renorm={renormalize} {experts}x{k} row {row}"
                    ),
                    &hi[row * kk..(row + 1) * kk],
                    &hw[row * kk..(row + 1) * kk],
                    &want,
                );
            }
        }
    }

    // sigmoid with sinks: E routed + S sink columns, a global scale, and a
    // fan-out past E (the spare picks land -1).
    for (rows, experts, sink, k) in [(3u32, 64u32, 2u32, 6u32), (2, 256, 1, 8), (2, 4, 2, 6)] {
        let (r, e, s, kk) = (rows as usize, experts as usize, sink as usize, k as usize);
        let w = e + s;
        let logits = distinct_logits(r, w, 99);
        let lt = gpu.tensor(&logits, rows, experts + sink, Dtype::Bf16);
        let bias: Vec<f32> = (0..e)
            .map(|i| ((i * 31) % e) as f32 / e as f32 * 0.1)
            .collect();
        let bt = gpu.tensor(&bias, 1, experts, Dtype::F32);
        let gs = gpu.tensor(&[0.75], 1, 1, Dtype::F32);
        let fan = k + sink;
        let mut ids = planes.i32s(&vec![0; r * (kk + s)], rows, fan);
        let mut ws = gpu.zeros(rows, fan, Dtype::F32);
        moe::topk_sigmoid_sink(
            &ctx,
            lt,
            Some(bt),
            Some(gs),
            experts,
            k,
            sink,
            1.5,
            &mut ids,
            &mut ws,
        )
        .unwrap();
        let (hi, hw) = (planes.read_i32(&gpu, &ids), gpu.read(&ws));
        for row in 0..r {
            let x = &logits[row * w..(row + 1) * w];
            let sg: Vec<f64> = x.iter().map(|&v| sigmoid(f64::from(v))).collect();
            let rank: Vec<f64> = (0..e).map(|i| sg[i] + f64::from(bias[i])).collect();
            let picks = top(&rank, kk.min(e));
            let sum: f64 = picks.iter().map(|&i| sg[i]).sum::<f64>() + sg[e..].iter().sum::<f64>();
            let scale = 1.5 * 0.75 / (sum + 1e-20);
            let mut want: Vec<(i32, f64)> =
                picks.iter().map(|&i| (i as i32, sg[i] * scale)).collect();
            want.resize(kk, (-1, 0.0));
            want.extend((0..s).map(|i| ((e + i) as i32, sg[e + i] * scale)));
            check_routes(
                &format!("sink {experts}+{sink}x{k} row {row}"),
                &hi[row * (kk + s)..(row + 1) * (kk + s)],
                &hw[row * (kk + s)..(row + 1) * (kk + s)],
                &want,
            );
        }
    }

    // A window: 6 rows launched, rows 2..5 live; the dead rows keep their
    // sentinel.
    let (rows, experts, k) = (6u32, 64u32, 4u32);
    let logits = distinct_logits(6, 64, 5);
    let lt = gpu.tensor(&logits, rows, experts, Dtype::Bf16);
    let mut ids = planes.i32s(&[-7; 24], rows, k);
    let mut ws = gpu.zeros(rows, k, Dtype::F32);
    let win = gpu.ints(&[3, 2, 0, 0]);
    ctx.arm_stage(win.ptr);
    moe::topk_sigmoid(&ctx, lt, None, experts, k, true, 1.0, &mut ids, &mut ws).unwrap();
    ctx.disarm_stage();
    let hi = planes.read_i32(&gpu, &ids);
    for row in 0..6 {
        let x = &logits[row * 64..(row + 1) * 64];
        let s: Vec<f64> = x.iter().map(|&v| sigmoid(f64::from(v))).collect();
        let want: Vec<i32> = if (2..5).contains(&row) {
            top(&s, 4).into_iter().map(|i| i as i32).collect()
        } else {
            vec![-7; 4]
        };
        assert_eq!(
            &hi[row * 4..(row + 1) * 4],
            &want[..],
            "windowed router row {row}"
        );
    }
}

#[test]
fn moe_route_tables() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let planes = Planes::default();

    // The hash route: ids past the vocabulary read row 0, an expert id past
    // the logits weighs nothing.
    let (tokens, vocab, experts, k) = (5u32, 7u32, 16u32, 3u32);
    let ids = [3, 0, 9, -1, 6];
    let table: Vec<i64> = (0..vocab as i64 * 3)
        .map(|i| if i == 5 { 40 } else { (i * 5) % 16 })
        .collect();
    let logits = bf16s((tokens * experts) as usize, 3, 4.0);
    let idt = planes.i32s(&ids, tokens, 1);
    let tt = planes.i64s(&table, vocab, k);
    let lt = gpu.tensor(&logits, tokens, experts, Dtype::Bf16);
    for renormalize in [false, true] {
        let mut routes = planes.i32s(&[0; 15], tokens, k);
        let mut ws = gpu.zeros(tokens, k, Dtype::F32);
        moe_route::hash_route(
            &ctx,
            idt,
            tt,
            lt,
            vocab,
            k,
            renormalize,
            1.5,
            &mut routes,
            &mut ws,
        )
        .unwrap();
        let (hr, hw) = (planes.read_i32(&gpu, &routes), gpu.read(&ws));
        for (n, &raw) in ids.iter().enumerate() {
            let tid = if (0..vocab as i32).contains(&raw) {
                raw as usize
            } else {
                0
            };
            let picks: Vec<i64> = table[tid * 3..tid * 3 + 3].to_vec();
            let w: Vec<f64> = picks
                .iter()
                .map(|&e| {
                    if (0..16).contains(&e) {
                        sqrt_softplus(f64::from(logits[n * 16 + e as usize]))
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = w.iter().sum();
            let scale = if renormalize && sum > 0.0 {
                1.5 / sum
            } else {
                1.5
            };
            let want: Vec<(i32, f64)> = picks
                .iter()
                .zip(&w)
                .map(|(&e, &w)| (e as i32, w * scale))
                .collect();
            check_routes(
                &format!("hash route {n} renorm={renormalize}"),
                &hr[n * 3..n * 3 + 3],
                &hw[n * 3..n * 3 + 3],
                &want,
            );
        }
    }

    // The group fan-out, whole and windowed.
    let mut routes = planes.i32s(&[-1; 4 * 6], 4, 6);
    moe_route::group_routes(&ctx, 6, &mut routes).unwrap();
    assert_eq!(
        planes.read_i32(&gpu, &routes),
        (0..24).map(|i| i % 6).collect::<Vec<_>>()
    );
    let mut routes = planes.i32s(&[-1; 4 * 6], 4, 6);
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    moe_route::group_routes(&ctx, 6, &mut routes).unwrap();
    ctx.disarm_stage();
    let want: Vec<i32> = (0..24)
        .map(|i| if (6..18).contains(&i) { i % 6 } else { -1 })
        .collect();
    assert_eq!(
        planes.read_i32(&gpu, &routes),
        want,
        "windowed group routes"
    );
}

#[test]
fn moe_folds() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let planes = Planes::default();
    let (tokens, k, hidden, experts) = (5u32, 3u32, 300u32, 4u32);
    let (t, kk, h) = (tokens as usize, k as usize, hidden as usize);
    let routed = bf16s(t * kk * h, 1, 1.0);
    let weights = values(t * kk, 2, 1.0);
    let rt = gpu.tensor(&routed, tokens * k, hidden, Dtype::Bf16);
    let wt = gpu.tensor(&weights, tokens, k, Dtype::F32);
    let mut y = gpu.zeros(tokens, hidden, Dtype::Bf16);
    moe::weighted_sum(&ctx, rt, wt, &mut y).unwrap();
    let want: Vec<f32> = (0..t * h)
        .map(|i| {
            let (n, c) = (i / h, i % h);
            (0..kk)
                .map(|j| weights[n * kk + j] * routed[(n * kk + j) * h + c])
                .sum()
        })
        .collect();
    close_bf16("weighted_sum", &gpu.read(&y), &want, 1.0);

    let x = bf16s(t * h, 3, 1.0);
    let bias = bf16s(experts as usize * h, 4, 1.0);
    let routes = [0, 3, -1, 2, 2, 1, -1, -1, -1, 3, 0, 1, 1, 1, 1];
    let xt = gpu.tensor(&x, tokens, hidden, Dtype::Bf16);
    let bt = gpu.tensor(&bias, experts, hidden, Dtype::Bf16);
    let routes_t = planes.i32s(&routes, tokens, k);
    let mut y = gpu.zeros(tokens, hidden, Dtype::Bf16);
    moe::bias_sum(&ctx, xt, bt, routes_t, wt, &mut y).unwrap();
    let want: Vec<f32> = (0..t * h)
        .map(|i| {
            let (n, c) = (i / h, i % h);
            x[i] + (0..kk)
                .filter(|&j| routes[n * kk + j] >= 0)
                .map(|j| weights[n * kk + j] * bias[routes[n * kk + j] as usize * h + c])
                .sum::<f32>()
        })
        .collect();
    close_bf16("bias_sum", &gpu.read(&y), &want, 1.0);

    // The shared expert behind its scalar gate, read at a pitch of 3.
    let shared = bf16s(t * h, 5, 1.0);
    let gate = bf16s(t * 3, 6, 3.0);
    let st = gpu.tensor(&shared, tokens, hidden, Dtype::Bf16);
    let gt = gpu.tensor(&gate, tokens, 3, Dtype::Bf16);
    let mut y = gpu.zeros(tokens, hidden, Dtype::Bf16);
    moe::sigmoid_gate_add(&ctx, xt, st, gt, &mut y).unwrap();
    let want: Vec<f32> = (0..t * h)
        .map(|i| x[i] + shared[i] * sigmoid(f64::from(gate[(i / h) * 3])) as f32)
        .collect();
    close_bf16("sigmoid_gate_add", &gpu.read(&y), &want, 1.0);
}

// ------------------------------------------------------------ matmuls --

/// A bank's experts as dense f64 rows: `w(e, n)` is expert e's row n.
struct Dense {
    n: usize,
    k: usize,
    w: Vec<f32>,
}

impl Dense {
    fn row(&self, e: usize, n: usize) -> &[f32] {
        let at = (e * self.n + n) * self.k;
        &self.w[at..at + self.k]
    }
}

/// y[route] = W[routes[route]]·x[token or route] (+ bias[expert]), with the
/// sum of |terms| a row (what an f32 accumulation's error scales with).
fn reference(
    bank: &Dense,
    x: &[f32],
    routes: &[i32],
    top_k: usize,
    by_token: bool,
    bias: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let (n, k) = (bank.n, bank.k);
    let mut y = vec![0.0; routes.len() * n];
    let mut mag = vec![0.0; routes.len() * n];
    for (r, &e) in routes.iter().enumerate() {
        if e < 0 {
            continue;
        }
        let xr = &x[(if by_token { r / top_k } else { r }) * k..][..k];
        for c in 0..n {
            let w = bank.row(e as usize, c);
            let dot: f64 = w
                .iter()
                .zip(xr)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let abs: f64 = w
                .iter()
                .zip(xr)
                .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                .sum();
            let b = bias.map_or(0.0, |b| f64::from(b[e as usize * n + c]));
            y[r * n + c] = (dot + b) as f32;
            mag[r * n + c] = abs as f32;
        }
    }
    (y, mag)
}

/// Within one bf16 rounding of the result plus `rel` of the terms'
/// magnitude.
fn close_dot(what: &str, have: &[f32], want: &[f32], mag: &[f32], rel: f32) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for i in 0..have.len() {
        let tol = want[i].abs() / 128.0 + mag[i] * rel + 1e-6;
        assert!(
            (have[i] - want[i]).abs() <= tol,
            "{what}[{i}] = {}, want {} (±{tol})",
            have[i],
            want[i]
        );
    }
}

/// Routes over `experts`: every token's picks distinct, a few dead (-1)
/// when asked.
fn make_routes(tokens: usize, top_k: usize, experts: usize, seed: u64, dead: bool) -> Vec<i32> {
    let keys = values(tokens * experts, seed, 1.0);
    let mut out = Vec::with_capacity(tokens * top_k);
    for t in 0..tokens {
        let row = &keys[t * experts..(t + 1) * experts];
        let mut ids: Vec<usize> = (0..experts).collect();
        ids.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        out.extend(ids[..top_k].iter().map(|&e| e as i32));
    }
    if dead {
        for i in (3..out.len()).step_by(17) {
            out[i] = -1;
        }
    }
    out
}

fn tuned(gpu: &Gpu, f: impl FnOnce(&mut Tuning)) -> Ctx {
    let mut t = Tuning::DEFAULT;
    f(&mut t);
    gpu.ctx().with_tuning(t)
}

#[test]
fn moe_matmul_dense() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    // (experts, tokens, top_k, n, k): the decode stream across short and long
    // K and a ragged N; the batch leg in 4- to 32-route items, K short of a
    // whole stage, and past its reach the grouped leg; fires wide enough to
    // group natively (in 32-, 64- and 128-route items), and a K the native
    // legs cannot read, which rocBLAS takes.
    for (experts, tokens, top_k, n, k, leg) in [
        (8u32, 1u32, 2u32, 1280u32, 2560u32, "stream"),
        (6, 3, 3, 1000, 640, "stream"),
        (4, 2, 2, 72, 7168, "stream"),
        (3, 2, 1, 13, 64, "stream"),
        (8, 1, 2, 1280, 2560, "batch"),
        (6, 3, 3, 1000, 640, "batch"),
        (4, 2, 2, 72, 7168, "batch"),
        (3, 2, 1, 13, 64, "batch"),
        (8, 40, 4, 520, 264, "batch"),
        (5, 64, 2, 136, 2312, "batch"),
        (4, 30, 4, 264, 520, "batch"),
        (3, 46, 2, 136, 1032, "batch"),
        (2, 60, 2, 200, 1024, "batch"),
        (4, 96, 4, 136, 512, "batch"),
        (3, 100, 2, 300, 264, "batch"),
        (8, 40, 4, 520, 256, "grouped"),
        (4, 48, 4, 136, 512, "grouped"),
        (5, 64, 2, 128, 2304, "grouped"),
        (4, 96, 4, 136, 512, "grouped"),
        // 128-route items over whole 512-row banks and a long K: the tall block.
        (4, 128, 4, 512, 2048, "grouped"),
        (3, 200, 2, 1024, 2048, "grouped"),
        (3, 20, 2, 50, 60, "blas"),
        // The wide leg in 64- to 256-route items, N ragged by 8 and by 2.
        (8, 64, 4, 520, 256, "wide"),
        (5, 65, 2, 130, 320, "wide"),
        (4, 100, 4, 136, 512, "wide"),
        (3, 140, 3, 200, 1024, "wide"),
        (2, 220, 2, 72, 2048, "wide"),
    ] {
        let (e, t, kk, nn, kd) = (
            experts as usize,
            tokens as usize,
            top_k as usize,
            n as usize,
            k as usize,
        );
        let bank = Dense {
            n: nn,
            k: kd,
            w: bf16s(e * nn * kd, 11, 0.1),
        };
        let bt = gpu.tensor(&bank.w, experts, n * k, Dtype::Bf16);
        let ctx = tuned(&gpu, |t| {
            t.moe_grouped = leg != "stream";
            t.moe_batch = leg == "batch" || leg == "wide";
            t.moe_batch_from = 1;
            // Up to 64 routes an expert, so a 64-route item fills.
            t.moe_dense_batch_until = if leg == "wide" { 4096 } else { 64 };
            t.moe_dense_wide_from = u32::from(leg == "wide");
        });
        for by_token in [true, false] {
            let routes = make_routes(t, kk, e, u64::from(n), leg != "blas");
            let x_rows = if by_token { t } else { t * kk };
            let x = bf16s(x_rows * kd, 12, 1.0);
            let xt = gpu.tensor(&x, x_rows as u32, k, Dtype::Bf16);
            let rt = planes.i32s(&routes, tokens, top_k);
            let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
            moe::matmul_select(&ctx, xt, bt, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
            let (want, mag) = reference(&bank, &x, &routes, kk, by_token, None);
            close_dot(
                &format!("dense {leg} {experts}e {tokens}x{top_k} {n}x{k} by_token={by_token}"),
                &gpu.read(&y),
                &want,
                &mag,
                1e-5,
            );
        }
    }

    // The stream and the batch leg under a window (rows 1..3 of 4 tokens
    // live), and the stream through an expert table, which also counts each
    // expert's routes.
    let (experts, tokens, top_k, n, k) = (4u32, 4u32, 2u32, 96u32, 512u32);
    let bank = Dense {
        n: 96,
        k: 512,
        w: bf16s(4 * 96 * 512, 21, 0.1),
    };
    let bt = gpu.tensor(&bank.w, experts, n * k, Dtype::Bf16);
    let routes = make_routes(4, 2, 4, 5, false);
    let x = bf16s(4 * 512, 22, 1.0);
    let xt = gpu.tensor(&x, tokens, k, Dtype::Bf16);
    let rt = planes.i32s(&routes, tokens, top_k);
    let (want, mag) = reference(&bank, &x, &routes, 2, true, None);
    let win = gpu.ints(&[2, 1, 0, 0]);
    for leg in ["stream", "batch", "wide"] {
        let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
        let ctx = tuned(&gpu, |t| {
            t.moe_batch = leg != "stream";
            t.moe_dense_wide_from = u32::from(leg == "wide");
        });
        ctx.arm_stage(win.ptr);
        moe::matmul_select(&ctx, xt, bt, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
        ctx.disarm_stage();
        let have = gpu.read(&y);
        let live = 2 * 96..6 * 96;
        close_dot(
            &format!("windowed {leg}"),
            &have[live.clone()],
            &want[live.clone()],
            &mag[live],
            1e-5,
        );
        assert!(
            have[..2 * 96]
                .iter()
                .chain(&have[6 * 96..])
                .all(|&v| v == 0.0),
            "{leg}: dead rows untouched"
        );
    }
    let ctx = gpu.ctx();

    let table: Vec<u64> = (0..4u64).map(|e| bt.ptr + e * 96 * 512 * 2).collect();
    let table_t = planes.keep(hip::Buffer::from_slice(&table).unwrap(), 4, 1, Dtype::U8);
    let hits = planes.i32s(&[0; 4], 1, 4);
    let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
    let streamed = moe::ExpertTable {
        table: table_t.ptr,
        hits: hits.ptr,
    };
    moe::matmul_select(&ctx, xt, bt, rt, &mut y, streamed).unwrap();
    close_dot("expert table", &gpu.read(&y), &want, &mag, 1e-5);
    let mut counted = [0; 4];
    for &e in &routes {
        counted[e as usize] += 1;
    }
    assert_eq!(planes.read_i32(&gpu, &hits), counted, "expert hits");
}

#[test]
fn moe_matmul_dense_wide_skewed() {
    // The wide leg where one expert takes more routes than an item: expert 0
    // in every token (100 routes over 64-route items), the others a few
    // each, so its routes split over two items, each reading the weight.
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let (experts, tokens, top_k, n, k) = (8u32, 100u32, 2u32, 192u32, 512u32);
    let bank = Dense {
        n: 192,
        k: 512,
        w: bf16s(8 * 192 * 512, 31, 0.1),
    };
    let bt = gpu.tensor(&bank.w, experts, n * k, Dtype::Bf16);
    let routes: Vec<i32> = (0..100).flat_map(|t| [0, 1 + t % 7]).collect();
    let x = bf16s(100 * 512, 32, 1.0);
    let xt = gpu.tensor(&x, tokens, k, Dtype::Bf16);
    let rt = planes.i32s(&routes, tokens, top_k);
    let (want, mag) = reference(&bank, &x, &routes, 2, true, None);
    let ctx = tuned(&gpu, |t| t.moe_dense_wide_from = 1);
    let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
    moe::matmul_select(&ctx, xt, bt, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
    close_dot("wide skewed", &gpu.read(&y), &want, &mag, 1e-5);
}

/// fp4 (e2m1) magnitudes by code.
const FP4: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// A random mxfp4 bank: codes (low nibble first), e8m0 scales, and its
/// dense expansion.
fn mxfp4_bank(experts: usize, n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Dense) {
    let raw = values(experts * n * k, seed, 1.0);
    let codes: Vec<u8> = raw.iter().map(|&v| ((v + 1.0) * 8.0) as u8 & 15).collect();
    let sraw = values(experts * n * k / 32, seed + 1, 1.0);
    let scales: Vec<u8> = sraw
        .iter()
        .map(|&v| (121.0 + (v + 1.0) * 3.0) as u8)
        .collect();
    let mut packed = vec![0u8; codes.len() / 2];
    let mut w = vec![0.0; codes.len()];
    for (i, &c) in codes.iter().enumerate() {
        packed[i / 2] |= c << (4 * (i % 2));
        let s = f32::from_bits(u32::from(scales[i / 32]) << 23);
        let m = FP4[(c & 7) as usize] * s;
        w[i] = if c & 8 != 0 { -m } else { m };
    }
    (packed, scales, Dense { n, k, w })
}

#[test]
fn moe_matmul_mxfp4() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    // Decode (the stream, and the batch leg in 16- and 32-route items, K
    // short of a whole stage), then grouped on the matrix core and off it;
    // a bias on the gate/up leg.
    for (experts, tokens, top_k, n, k, leg) in [
        (8u32, 1u32, 4u32, 1280u32, 2560u32, "stream"),
        (6, 2, 3, 2560, 640, "stream"),
        (4, 2, 2, 100, 7168, "stream"),
        (8, 1, 4, 1280, 2560, "batch"),
        (6, 2, 3, 2560, 640, "batch"),
        (4, 2, 2, 100, 7168, "batch"),
        (8, 50, 4, 384, 512, "batch"),
        (5, 33, 3, 136, 2336, "batch"),
        (4, 72, 4, 256, 512, "batch"),
        (3, 80, 2, 200, 2304, "batch"),
        (8, 50, 4, 384, 512, "wmma"),
        (4, 40, 4, 256, 512, "wmma"),
        (5, 33, 3, 136, 2304, "wmma"),
        (4, 72, 4, 256, 512, "wmma"),
        // Few items over a long K: the matrix-core leg slices K (4 slices,
        // an odd stage count; 2 slices of 32-route items).
        (4, 40, 4, 256, 4096, "wmma"),
        (4, 64, 2, 136, 4160, "wmma"),
        (3, 20, 2, 136, 4096, "wmma"),
        (8, 50, 4, 200, 512, "simt"),
    ] {
        let (e, t, kk, nn, kd) = (
            experts as usize,
            tokens as usize,
            top_k as usize,
            n as usize,
            k as usize,
        );
        let (codes, scales, bank) = mxfp4_bank(e, nn, kd, u64::from(n + k));
        let ct = planes.bytes(&codes, experts, n * k / 2);
        let st = planes.bytes(&scales, experts, n * k / 32);
        let bias = bf16s(e * nn, 31, 1.0);
        let bias_t = gpu.tensor(&bias, experts, n, Dtype::Bf16);
        let ctx = tuned(&gpu, |t| {
            t.mxfp4_grouped = leg != "stream";
            t.mxfp4_wmma = leg == "wmma";
            t.moe_batch = leg == "batch";
            t.moe_batch_from = 1;
        });
        for (by_token, biased) in [(true, false), (false, true)] {
            let routes = make_routes(t, kk, e, u64::from(k), true);
            let x_rows = if by_token { t } else { t * kk };
            let x = bf16s(x_rows * kd, 32, 1.0);
            let xt = gpu.tensor(&x, x_rows as u32, k, Dtype::Bf16);
            let rt = planes.i32s(&routes, tokens, top_k);
            let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
            if biased {
                moe::matmul_select_bias(
                    &ctx,
                    xt,
                    ct,
                    st,
                    bias_t,
                    rt,
                    &mut y,
                    moe::GroupSeat::RESIDENT,
                )
                .unwrap();
            } else {
                moe::matmul_select_quant(
                    &ctx,
                    xt,
                    ct,
                    st,
                    None,
                    rt,
                    &mut y,
                    moe::GroupSeat::RESIDENT,
                )
                .unwrap();
            }
            let (want, mag) = reference(
                &bank,
                &x,
                &routes,
                kk,
                by_token,
                biased.then_some(&bias[..]),
            );
            // A grouped leg writes only the live routes; a dead one stays zero.
            close_dot(
                &format!("mxfp4 {leg} {experts}e {tokens}x{top_k} {n}x{k} by_token={by_token}"),
                &gpu.read(&y),
                &want,
                &mag,
                1e-5,
            );
        }
    }
}

/// A random MLX affine bank: `bits`-bit codes in little-endian u32 words, a
/// bf16 scale and bias every `group` codes, and its dense expansion (bf16
/// rounded when `rounded`, as the matrix-core leg holds it).
fn affine_bank(
    experts: usize,
    n: usize,
    k: usize,
    bits: usize,
    group: usize,
    seed: u64,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Dense, Dense) {
    let levels = 1usize << bits;
    let raw = values(experts * n * k, seed, 1.0);
    let codes: Vec<u32> = raw
        .iter()
        .map(|&v| (((v + 1.0) / 2.0 * levels as f32) as usize).min(levels - 1) as u32)
        .collect();
    let groups = experts * n * k / group;
    let s = bf16s(groups, seed + 1, 0.02);
    let b = bf16s(groups, seed + 2, 0.1);
    let mut words = vec![0u32; experts * n * k * bits / 32];
    let mut exact = vec![0.0; codes.len()];
    let mut rounded = vec![0.0; codes.len()];
    for (i, &c) in codes.iter().enumerate() {
        words[i * bits / 32] |= c << ((i * bits) % 32);
        let v = (c as f32).mul_add(s[i / group], b[i / group]);
        exact[i] = v;
        rounded[i] = round_bf16(v);
    }
    let bytes = |v: &[f32]| {
        v.iter()
            .flat_map(|&x| bf16(x).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    (
        words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        bytes(&s),
        bytes(&b),
        Dense { n, k, w: exact },
        Dense { n, k, w: rounded },
    )
}

#[test]
fn moe_matmul_affine() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    for (bits, group) in [(4u32, 64u32), (4, 32), (4, 128), (2, 32), (8, 128), (8, 32)] {
        for (experts, tokens, top_k, n, k, leg) in [
            (6u32, 2u32, 3u32, 640u32, 2560u32, "stream"),
            (4, 1, 2, 70, 1024, "stream"),
            (6, 2, 3, 640, 2560, "batch"),
            (4, 1, 2, 70, 1024, "batch"),
            (6, 3, 2, 200, 1280, "batch"),
            (8, 40, 4, 264, 512, "batch"),
            (4, 24, 4, 136, 1280, "batch"),
            (4, 72, 4, 136, 256, "batch"),
            (3, 80, 2, 200, 640, "batch"),
            (8, 40, 4, 264, 512, "wmma"),
            (4, 72, 4, 136, 256, "wmma"),
            (4, 48, 4, 136, 256, "wmma"),
            (8, 40, 4, 136, 384, "simt"),
        ] {
            // The batch leg reads four-bit codes, an even count of groups a
            // row.
            if leg == "batch" && (bits != 4 || k % (2 * group) != 0) {
                continue;
            }
            let (e, t, kk, nn, kd) = (
                experts as usize,
                tokens as usize,
                top_k as usize,
                n as usize,
                k as usize,
            );
            let (codes, scales, biases, exact, rounded) = affine_bank(
                e,
                nn,
                kd,
                bits as usize,
                group as usize,
                u64::from(bits * 100 + group),
            );
            let ct = planes.bytes(&codes, experts, n * k * bits / 8);
            let st = planes.bytes(&scales, experts, 2 * n * k / group);
            let bt = planes.bytes(&biases, experts, 2 * n * k / group);
            let ctx = tuned(&gpu, |t| {
                t.affine_grouped = leg != "stream";
                t.affine_grouped_from = 1;
                t.affine_wmma = leg == "wmma";
                t.moe_batch = leg == "batch";
                t.moe_batch_from = 1;
            });
            for by_token in [true, false] {
                let routes = make_routes(t, kk, e, u64::from(n), true);
                let x_rows = if by_token { t } else { t * kk };
                let x = bf16s(x_rows * kd, 42, 1.0);
                let xt = gpu.tensor(&x, x_rows as u32, k, Dtype::Bf16);
                let rt = planes.i32s(&routes, tokens, top_k);
                let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
                moe::matmul_select_quant(
                    &ctx,
                    xt,
                    ct,
                    st,
                    Some(bt),
                    rt,
                    &mut y,
                    moe::GroupSeat::RESIDENT,
                )
                .unwrap();
                let bank = if leg == "wmma" { &rounded } else { &exact };
                let (want, mag) = reference(bank, &x, &routes, kk, by_token, None);
                close_dot(
                    &format!(
                        "affine{bits}/{group} {leg} {experts}e {tokens}x{top_k} {n}x{k} by_token={by_token}"
                    ),
                    &gpu.read(&y),
                    &want,
                    &mag,
                    2e-5,
                );
            }
        }
    }
}

/// f32 → f16 bits, round to nearest even, for values inside f16's range.
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = v.abs();
    if a < 6.103_515_6e-5 {
        // Subnormal: a count of 2^-24.
        return sign | (a / 5.960_464_5e-8).round_ties_even() as u16;
    }
    let e = ((b >> 23) & 0xff) - 127 + 15;
    let m = b & 0x7f_ffff;
    let mut r = (e << 10) | (m >> 13);
    let rest = m & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

fn from_f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 0x1f);
    let m = f32::from(h & 0x3ff);
    sign * if e == 0 {
        m * 2f32.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}

impl Planes {
    fn f16s(&self, v: &[f32], rows: u32, width: u32) -> Tensor {
        let bits: Vec<u16> = v.iter().map(|&x| f16_bits(x)).collect();
        self.keep(
            hip::Buffer::from_slice(&bits).unwrap(),
            rows,
            width,
            Dtype::F16,
        )
    }
    fn read_f16(&self, gpu: &Gpu, t: &Tensor) -> Vec<f32> {
        gpu.sync();
        let held = self.held.borrow();
        let b = held.iter().find(|b| b.device_ptr() == t.ptr).unwrap();
        b.read::<u16>().unwrap()[..t.elements() as usize]
            .iter()
            .map(|&h| from_f16(h))
            .collect()
    }
}

fn f16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale)
        .into_iter()
        .map(|v| from_f16(f16_bits(v)))
        .collect()
}

#[test]
fn moe_matmul_f16() {
    // The f16 instantiations: the dense stream, batch leg and both grouped
    // widths, the
    // mxfp4 stream and its off-core grouped leg, and the affine stream.
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    for (experts, tokens, top_k, n, k, leg) in [
        (6u32, 2u32, 3u32, 300u32, 640u32, "stream"),
        (6, 2, 3, 300, 648, "batch"),
        (8, 40, 4, 200, 256, "batch"),
        (3, 100, 2, 136, 264, "batch"),
        (8, 40, 4, 200, 256, "grouped"),
        (4, 96, 4, 136, 512, "grouped"),
        (4, 96, 4, 136, 512, "wide"),
    ] {
        let (e, t, kk, nn, kd) = (
            experts as usize,
            tokens as usize,
            top_k as usize,
            n as usize,
            k as usize,
        );
        let bank = Dense {
            n: nn,
            k: kd,
            w: f16s(e * nn * kd, 51, 0.1),
        };
        let bt = planes.f16s(&bank.w, experts, n * k);
        let ctx = tuned(&gpu, |t| {
            t.moe_grouped = leg != "stream";
            t.moe_batch = leg == "batch" || leg == "wide";
            t.moe_batch_from = 1;
            t.moe_dense_wide_from = u32::from(leg == "wide");
        });
        let routes = make_routes(t, kk, e, 52, true);
        let x = f16s(t * kd, 53, 1.0);
        let xt = planes.f16s(&x, tokens, k);
        let rt = planes.i32s(&routes, tokens, top_k);
        let mut y = planes.f16s(&vec![0.0; t * kk * nn], tokens * top_k, n);
        moe::matmul_select(&ctx, xt, bt, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
        let (want, mag) = reference(&bank, &x, &routes, kk, true, None);
        close_dot(
            &format!("f16 dense {leg} {n}x{k}"),
            &planes.read_f16(&gpu, &y),
            &want,
            &mag,
            1e-5,
        );
    }

    for (tokens, top_k, n, k, leg) in [
        (2u32, 3u32, 256u32, 640u32, "stream"),
        (40, 4, 200, 512, "simt"),
    ] {
        let (t, kk, nn, kd) = (tokens as usize, top_k as usize, n as usize, k as usize);
        let (codes, scales, bank) = mxfp4_bank(6, nn, kd, 61);
        let ct = planes.bytes(&codes, 6, n * k / 2);
        let st = planes.bytes(&scales, 6, n * k / 32);
        let ctx = tuned(&gpu, |t| {
            t.mxfp4_grouped = leg != "stream";
            t.moe_batch = false;
        });
        let routes = make_routes(t, kk, 6, 62, true);
        let x = f16s(t * kd, 63, 1.0);
        let xt = planes.f16s(&x, tokens, k);
        let rt = planes.i32s(&routes, tokens, top_k);
        let mut y = planes.f16s(&vec![0.0; t * kk * nn], tokens * top_k, n);
        moe::matmul_select_quant(&ctx, xt, ct, st, None, rt, &mut y, moe::GroupSeat::RESIDENT)
            .unwrap();
        let (want, mag) = reference(&bank, &x, &routes, kk, true, None);
        close_dot(
            &format!("f16 mxfp4 {leg} {n}x{k}"),
            &planes.read_f16(&gpu, &y),
            &want,
            &mag,
            1e-5,
        );
    }

    for (bits, group) in [(4u32, 64u32), (2, 32), (8, 128)] {
        let (tokens, top_k, n, k) = (2u32, 3u32, 136u32, 1024u32);
        let (t, kk, nn, kd) = (2usize, 3usize, 136usize, 1024usize);
        let (codes, scales, biases, exact, _) =
            affine_bank(6, nn, kd, bits as usize, group as usize, 71);
        let ct = planes.bytes(&codes, 6, n * k * bits / 8);
        let st = planes.bytes(&scales, 6, 2 * n * k / group);
        let bt = planes.bytes(&biases, 6, 2 * n * k / group);
        let ctx = tuned(&gpu, |t| {
            t.affine_grouped = false;
            t.moe_batch = false;
        });
        let routes = make_routes(t, kk, 6, 72, true);
        let x = f16s(t * kd, 73, 1.0);
        let xt = planes.f16s(&x, tokens, k);
        let rt = planes.i32s(&routes, tokens, top_k);
        let mut y = planes.f16s(&vec![0.0; t * kk * nn], tokens * top_k, n);
        moe::matmul_select_quant(
            &ctx,
            xt,
            ct,
            st,
            Some(bt),
            rt,
            &mut y,
            moe::GroupSeat::RESIDENT,
        )
        .unwrap();
        let (want, mag) = reference(&exact, &x, &routes, kk, true, None);
        close_dot(
            &format!("f16 affine{bits}/{group} stream"),
            &planes.read_f16(&gpu, &y),
            &want,
            &mag,
            2e-5,
        );
    }
}

/// The grouped projection (DeepSeek's grouped o-projection): every row's
/// group g through block g, x and y [tokens·groups] rows. Few tokens take
/// the routed legs; past the batch legs' reach, identity routes take one
/// strided-batched hipBLASLt launch, and the routed select is the
/// reference's equal either way.
#[test]
fn moe_matmul_grouped_projection() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let ctx = gpu.ctx();
    for (groups, n, k) in [(8u32, 128u32, 512u32), (4, 72, 264)] {
        let (g, nn, kd) = (groups as usize, n as usize, k as usize);
        let bank = Dense {
            n: nn,
            k: kd,
            w: bf16s(g * nn * kd, 31, 0.1),
        };
        let bt = gpu.tensor(&bank.w, groups, n * k, Dtype::Bf16);
        for tokens in [1u32, 8, 33, 64, 130] {
            let t = tokens as usize;
            let routes: Vec<i32> = (0..t * g).map(|i| (i % g) as i32).collect();
            let rows = tokens * groups;
            let x = bf16s(t * g * kd, 32 + u64::from(tokens), 1.0);
            let xt = gpu.tensor(&x, rows, k, Dtype::Bf16);
            let rt = planes.i32s(&routes, tokens, groups);
            let (want, mag) = reference(&bank, &x, &routes, g, false, None);
            for identity in [true, false] {
                let mut y = gpu.zeros(rows, n, Dtype::Bf16);
                moe::matmul_grouped(
                    &ctx,
                    xt,
                    bt,
                    rt,
                    &mut y,
                    moe::ExpertTable::RESIDENT,
                    identity,
                )
                .unwrap();
                close_dot(
                    &format!("grouped {groups}x{n}x{k} tokens {tokens} identity={identity}"),
                    &gpu.read(&y),
                    &want,
                    &mag,
                    1e-5,
                );
            }
        }
    }
}

#[test]
fn moe_matmul_windowed_grouped() {
    // The grouped legs order only the window's live routes: rows 1..3 of 4
    // tokens; at K 4096 the matrix-core leg slices K, and its sum lands only
    // the live routes too.
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    for k in [256u32, 4096] {
        let (experts, tokens, top_k, n) = (4u32, 4u32, 4u32, 128u32);
        let (codes, scales, bank) = mxfp4_bank(4, 128, k as usize, 77);
        let ct = planes.bytes(&codes, experts, n * k / 2);
        let st = planes.bytes(&scales, experts, n * k / 32);
        let routes = make_routes(4, 4, 4, 78, false);
        let x = bf16s(4 * k as usize, 79, 1.0);
        let xt = gpu.tensor(&x, tokens, k, Dtype::Bf16);
        let rt = planes.i32s(&routes, tokens, top_k);
        let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
        let ctx = tuned(&gpu, |t| {
            t.mxfp4_grouped_reuse = 0;
            t.moe_batch = false;
        });
        let win = gpu.ints(&[2, 1, 0, 0]);
        ctx.arm_stage(win.ptr);
        moe::matmul_select_quant(&ctx, xt, ct, st, None, rt, &mut y, moe::GroupSeat::RESIDENT)
            .unwrap();
        ctx.disarm_stage();
        let (want, mag) = reference(&bank, &x, &routes, 4, true, None);
        let have = gpu.read(&y);
        let live = 4 * 128..12 * 128;
        close_dot(
            &format!("windowed grouped k {k}"),
            &have[live.clone()],
            &want[live.clone()],
            &mag[live],
            1e-5,
        );
        assert!(
            have[..4 * 128]
                .iter()
                .chain(&have[12 * 128..])
                .all(|&v| v == 0.0),
            "k {k}: dead rows untouched"
        );
    }
}

// ------------------------------------------------------------- benches --

/// Device microseconds a call over `sets` route plans in turn, each plan
/// naming experts the others do not, so every call reads its weights cold
/// (from HBM, not the 256 MB last-level cache).
fn time_cold(gpu: &Gpu, sets: usize, reps: u32, mut f: impl FnMut(usize)) -> f64 {
    let mut i = 0;
    gpu.time_graph(reps, || {
        f(i % sets);
        i += 1;
    })
}

/// Routes for `sets` plans of `tokens` tokens over disjoint expert ranges.
fn cold_plans(
    planes: &Planes,
    sets: usize,
    tokens: usize,
    top_k: usize,
    per_set: usize,
) -> Vec<(Tensor, Vec<i32>)> {
    (0..sets)
        .map(|s| {
            let r: Vec<i32> = make_routes(tokens, top_k, per_set, s as u64 + 1, false)
                .into_iter()
                .map(|e| e + (s * per_set) as i32)
                .collect();
            (planes.i32s(&r, tokens as u32, top_k as u32), r)
        })
        .collect()
}

fn distinct(routes: &[i32]) -> usize {
    let mut v = routes.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe -- --ignored --nocapture"]
fn bench_moe_decode() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let ctx = gpu.ctx();
    // The roofline on this card as it is loaded: the dense one-row GEMV over
    // a 512 MB weight.
    {
        let (n, k) = (65_536u32, 4096u32);
        let w = Tensor::new(
            planes
                .keep(
                    hip::Buffer::alloc((n * k * 2) as usize).unwrap(),
                    1,
                    1,
                    Dtype::U8,
                )
                .ptr,
            n,
            k,
            Dtype::Bf16,
        );
        let x = gpu.tensor(&bf16s(k as usize, 1, 1.0), 1, k, Dtype::Bf16);
        let mut y = gpu.zeros(1, n, Dtype::Bf16);
        let us = gpu.time_graph(8, || {
            kernels_rocm::linear::gemm::matmul(&ctx, x, w, &mut y).unwrap()
        });
        println!(
            "dense gemv   512 MB roofline: {us:7.1} us  {:6.0} GB/s",
            f64::from(n) * f64::from(k) * 2.0 / us / 1e3
        );
    }
    // (model, top_k, per-set experts, n, k): bf16 Qwen3.8-Flash-Next gate/up
    // and down, GLM-5.3 gate/up, DeepSeek-V4.1-Flash down.
    for (name, top_k, per_set, n, k) in [
        (
            "qwen3.8-flash-next gate_up",
            10usize,
            16usize,
            1280u32,
            2560u32,
        ),
        ("qwen3.8-flash-next down", 10, 16, 2560, 640),
        ("glm-5.3 gate_up", 8, 8, 4096, 6144),
        ("deepseek-v4.1-flash down", 6, 8, 4096, 2304),
    ] {
        for tokens in [1usize, 4] {
            let sets = 8;
            let experts = (sets * per_set) as u32;
            let bytes = u64::from(experts) * u64::from(n) * u64::from(k) * 2;
            let bank = Tensor::new(
                planes
                    .keep(hip::Buffer::alloc(bytes as usize).unwrap(), 1, 1, Dtype::U8)
                    .ptr,
                experts,
                n * k,
                Dtype::Bf16,
            );
            let plans = cold_plans(&planes, sets, tokens, top_k, per_set);
            let x = gpu.tensor(
                &bf16s(tokens * k as usize, 1, 1.0),
                tokens as u32,
                k,
                Dtype::Bf16,
            );
            let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
            let us = time_cold(&gpu, sets, 32, |s| {
                moe::matmul_select(
                    &ctx,
                    x,
                    bank,
                    plans[s].0,
                    &mut y,
                    moe::ExpertTable::RESIDENT,
                )
                .unwrap();
            });
            let read = distinct(&plans[0].1) as f64 * f64::from(n) * f64::from(k) * 2.0;
            println!(
                "bf16 stream  {name:30} tokens {tokens}: {us:7.1} us  {:6.0} GB/s",
                read / us / 1e3
            );
        }
    }
    // mxfp4 Kimi K3 gate/up and down.
    for (name, top_k, per_set, n, k) in [
        ("kimi-k3 gate_up", 16usize, 16usize, 4096u32, 7168u32),
        ("kimi-k3 down", 16, 16, 7168, 2048),
    ] {
        for tokens in [1usize, 2] {
            let sets = 8;
            let experts = (sets * per_set) as u32;
            let code_bytes = experts as usize * (n * k / 2) as usize;
            let codes = planes.keep(
                hip::Buffer::alloc(code_bytes).unwrap(),
                experts,
                n * k / 2,
                Dtype::U8,
            );
            let scales = planes.bytes(
                &vec![127u8; experts as usize * (n * k / 32) as usize],
                experts,
                n * k / 32,
            );
            let plans = cold_plans(&planes, sets, tokens, top_k, per_set);
            let x = gpu.tensor(
                &bf16s(tokens * k as usize, 1, 1.0),
                tokens as u32,
                k,
                Dtype::Bf16,
            );
            let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
            let us = time_cold(&gpu, sets, 32, |s| {
                moe::matmul_select_quant(
                    &ctx,
                    x,
                    codes,
                    scales,
                    None,
                    plans[s].0,
                    &mut y,
                    moe::GroupSeat::RESIDENT,
                )
                .unwrap();
            });
            let read =
                distinct(&plans[0].1) as f64 * f64::from(n) * f64::from(k) * (0.5 + 1.0 / 32.0);
            println!(
                "mxfp4 stream {name:30} tokens {tokens}: {us:7.1} us  {:6.0} GB/s",
                read / us / 1e3
            );
        }
    }
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe -- --ignored --nocapture"]
fn bench_moe_prefill() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    // A 2048-token prefill over 64 experts of GLM-5.3-Flash's gate/up
    // (bf16) and Kimi K3's (mxfp4): the native grouped leg, rocBLAS, and the
    // stream for scale.
    let (tokens, top_k, experts, n, k) = (2048usize, 8usize, 64u32, 4096u32, 4096u32);
    let routes = make_routes(tokens, top_k, experts as usize, 3, false);
    let rt = planes.i32s(&routes, tokens as u32, top_k as u32);
    let x = gpu.tensor(
        &bf16s(tokens * k as usize, 1, 1.0),
        tokens as u32,
        k,
        Dtype::Bf16,
    );
    let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
    let bytes = u64::from(experts) * u64::from(n) * u64::from(k) * 2;
    let bank = Tensor::new(
        planes
            .keep(hip::Buffer::alloc(bytes as usize).unwrap(), 1, 1, Dtype::U8)
            .ptr,
        experts,
        n * k,
        Dtype::Bf16,
    );
    let flops = 2.0 * (tokens * top_k) as f64 * f64::from(n) * f64::from(k);
    for (leg, ctx) in [
        ("native", gpu.ctx()),
        ("stream", tuned(&gpu, |t| t.moe_grouped = false)),
    ] {
        let us = gpu.time(4, || {
            moe::matmul_select(&ctx, x, bank, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
        });
        println!(
            "bf16 {leg:8} {tokens}x{top_k} over {experts} experts {n}x{k}: {us:8.1} us  {:6.1} TFLOP/s  {:5.0} GB/s weights",
            flops / us / 1e6,
            bytes as f64 / us / 1e3
        );
    }

    let code_bytes = experts as usize * (n * k / 2) as usize;
    let codes = planes.keep(
        hip::Buffer::alloc(code_bytes).unwrap(),
        experts,
        n * k / 2,
        Dtype::U8,
    );
    let scales = planes.bytes(
        &vec![127u8; experts as usize * (n * k / 32) as usize],
        experts,
        n * k / 32,
    );
    for (leg, ctx) in [
        ("wmma", gpu.ctx()),
        ("simt", tuned(&gpu, |t| t.mxfp4_wmma = false)),
        ("stream", tuned(&gpu, |t| t.mxfp4_grouped = false)),
    ] {
        let us = gpu.time(4, || {
            moe::matmul_select_quant(
                &ctx,
                x,
                codes,
                scales,
                None,
                rt,
                &mut y,
                moe::GroupSeat::RESIDENT,
            )
            .unwrap();
        });
        println!(
            "mxfp4 {leg:7} {tokens}x{top_k} over {experts} experts {n}x{k}: {us:8.1} us  {:6.1} TFLOP/s",
            flops / us / 1e6
        );
    }
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe -- --ignored --nocapture"]
fn bench_moe_crossover() {
    // Where grouping overtakes the stream as an expert's routes grow: Qwen
    // 3.8-Flash-Next's gate/up (bf16, 128 of its experts) and Kimi K3's
    // (mxfp4, 64), at decode-batch fan-outs.
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let (experts, top_k, n, k) = (128u32, 8usize, 1280u32, 2560u32);
    let bytes = u64::from(experts) * u64::from(n) * u64::from(k) * 2;
    let bank = Tensor::new(
        planes
            .keep(hip::Buffer::alloc(bytes as usize).unwrap(), 1, 1, Dtype::U8)
            .ptr,
        experts,
        n * k,
        Dtype::Bf16,
    );
    for per_expert in [1usize, 2, 4, 8, 16] {
        let tokens = per_expert * experts as usize / top_k;
        let routes = make_routes(tokens, top_k, experts as usize, 9, false);
        let rt = planes.i32s(&routes, tokens as u32, top_k as u32);
        let x = gpu.tensor(
            &bf16s(tokens * k as usize, 1, 1.0),
            tokens as u32,
            k,
            Dtype::Bf16,
        );
        let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
        for (leg, ctx) in [
            ("stream", tuned(&gpu, |t| t.moe_grouped = false)),
            ("grouped", tuned(&gpu, |t| t.moe_group_at = 1)),
        ] {
            let us = gpu.time_graph(8, || {
                moe::matmul_select(&ctx, x, bank, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
            });
            println!("bf16  {per_expert:2} routes/expert {leg:8}: {us:8.1} us");
        }
    }
    let (experts, n, k) = (64u32, 4096u32, 7168u32);
    let codes = planes.keep(
        hip::Buffer::alloc(experts as usize * (n * k / 2) as usize).unwrap(),
        experts,
        n * k / 2,
        Dtype::U8,
    );
    let scales = planes.bytes(
        &vec![127u8; experts as usize * (n * k / 32) as usize],
        experts,
        n * k / 32,
    );
    for per_expert in [1usize, 2, 4, 8, 16] {
        let tokens = per_expert * experts as usize / top_k;
        let routes = make_routes(tokens, top_k, experts as usize, 9, false);
        let rt = planes.i32s(&routes, tokens as u32, top_k as u32);
        let x = gpu.tensor(
            &bf16s(tokens * k as usize, 1, 1.0),
            tokens as u32,
            k,
            Dtype::Bf16,
        );
        let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
        for (leg, ctx) in [
            ("stream", tuned(&gpu, |t| t.mxfp4_grouped = false)),
            ("wmma", tuned(&gpu, |t| t.mxfp4_grouped_reuse = 0)),
        ] {
            let us = gpu.time_graph(8, || {
                moe::matmul_select_quant(
                    &ctx,
                    x,
                    codes,
                    scales,
                    None,
                    rt,
                    &mut y,
                    moe::GroupSeat::RESIDENT,
                )
                .unwrap();
            });
            println!("mxfp4 {per_expert:2} routes/expert {leg:8}: {us:8.1} us");
        }
    }
}

/// One expert bank of a served model as the benches lay it out.
#[derive(Clone, Copy)]
enum Codec {
    Bf16,
    Mxfp4,
    Affine4,
}

impl Codec {
    /// Bytes an expert's n x k weight holds, codes and factors.
    fn bytes(self, n: u32, k: u32) -> u64 {
        let e = u64::from(n) * u64::from(k);
        match self {
            Codec::Bf16 => 2 * e,
            Codec::Mxfp4 => e / 2 + e / 32,
            Codec::Affine4 => e / 2 + 4 * e / 64,
        }
    }
}

/// A bank of `experts` experts of `codec` (uninitialised: the benches time
/// reads, not values).
fn bench_bank(planes: &Planes, codec: Codec, experts: u32, n: u32, k: u32) -> [Tensor; 3] {
    let alloc = |bytes: u64, width: u64| {
        planes.keep(
            hip::Buffer::alloc(bytes as usize).unwrap(),
            experts,
            u32::try_from(width).unwrap(),
            Dtype::U8,
        )
    };
    let e = u64::from(n) * u64::from(k);
    let rows = u64::from(experts);
    match codec {
        Codec::Bf16 => {
            let mut b = alloc(rows * e * 2, e);
            b.dtype = Dtype::Bf16;
            [b, b, b]
        }
        Codec::Mxfp4 => [
            alloc(rows * e / 2, e / 2),
            alloc(rows * e / 32, e / 32),
            alloc(1, 1),
        ],
        Codec::Affine4 => [
            alloc(rows * e / 2, e / 2),
            alloc(rows * e / 32, e / 32),
            alloc(rows * e / 32, e / 32),
        ],
    }
}

/// The bank's copy `copy` of `experts` experts as one bank.
fn bank_copy(t: Tensor, copy: u32, experts: u32) -> Tensor {
    let mut t = t;
    t.ptr += u64::from(copy) * u64::from(experts) * u64::from(t.width) * t.dtype.bytes_ceil();
    t.rows = experts;
    t
}

#[allow(clippy::too_many_arguments)]
fn bench_fire(
    ctx: &Ctx,
    codec: Codec,
    bank: [Tensor; 3],
    copy: u32,
    experts: u32,
    x: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) {
    let at = |t| bank_copy(t, copy, experts);
    match codec {
        Codec::Bf16 => {
            moe::matmul_select(ctx, x, at(bank[0]), routes, y, moe::ExpertTable::RESIDENT)
        }
        Codec::Mxfp4 => moe::matmul_select_quant(
            ctx,
            x,
            at(bank[0]),
            at(bank[1]),
            None,
            routes,
            y,
            moe::GroupSeat::RESIDENT,
        ),
        Codec::Affine4 => moe::matmul_select_quant(
            ctx,
            x,
            at(bank[0]),
            at(bank[1]),
            Some(at(bank[2])),
            routes,
            y,
            moe::GroupSeat::RESIDENT,
        ),
    }
    .unwrap();
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe -- --ignored --nocapture"]
fn bench_moe_models() {
    // The served models' expert matmuls at decode batches (GB/s of the
    // experts the routes touch) and prefill (TFLOP/s), random top-k over the
    // release's expert count, cold: the plans rotate over copies of the bank
    // at least 2 GB together. PIE_BENCH_MOE filters by name,
    // PIE_BENCH_MOE_TOKENS lists the token counts, PIE_BENCH_MOE_BATCH_FROM
    // and _UNTIL move the batch legs' reach.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = tuned(&gpu, |t| {
        if let Ok(v) = std::env::var("PIE_BENCH_MOE_BATCH_UNTIL") {
            t.moe_batch_until = v.parse().unwrap();
            t.moe_dense_batch_until = t.moe_batch_until;
        }
        if let Ok(v) = std::env::var("PIE_BENCH_MOE_BATCH_FROM") {
            t.moe_batch_from = v.parse().unwrap();
        }
    });
    let only = std::env::var("PIE_BENCH_MOE").unwrap_or_default();
    let token_list: Vec<usize> = std::env::var("PIE_BENCH_MOE_TOKENS").map_or_else(
        |_| vec![1, 8, 32, 128, 2048, 8192],
        |v| v.split(',').map(|t| t.parse().unwrap()).collect(),
    );
    for (name, codec, experts, top_k, n, k) in [
        (
            "glm53-mini gate_up u4",
            Codec::Affine4,
            32u32,
            8usize,
            4096u32,
            4096u32,
        ),
        ("glm53-mini down u4", Codec::Affine4, 32, 8, 4096, 2048),
        ("glm53-mini gate_up bf16", Codec::Bf16, 32, 8, 4096, 4096),
        ("glm53-mini down bf16", Codec::Bf16, 32, 8, 4096, 2048),
        ("kimi-k3 gate_up mxfp4", Codec::Mxfp4, 896, 16, 6144, 3584),
        ("kimi-k3 down mxfp4", Codec::Mxfp4, 896, 16, 3584, 3072),
        ("dsv41 gate mxfp4", Codec::Mxfp4, 384, 6, 2304, 5120),
        ("dsv41 down mxfp4", Codec::Mxfp4, 384, 6, 5120, 2304),
        // The mini's 16 experts: 128 tokens give each 48 routes.
        ("dsv41-mini gate mxfp4", Codec::Mxfp4, 16, 6, 2304, 5120),
        ("dsv41-mini down mxfp4", Codec::Mxfp4, 16, 6, 5120, 2304),
        // The grouped o-projection: each token's eight 4096-wide head groups
        // through their own 1024-row block, every group a route.
        ("dsv41 o_down bf16", Codec::Bf16, 8, 8, 1024, 4096),
        ("qwen3.8 gate_up bf16", Codec::Bf16, 512, 10, 1280, 2560),
        ("qwen3.8 down bf16", Codec::Bf16, 512, 10, 2560, 640),
        ("qwen3.8 gate_up u4", Codec::Affine4, 512, 10, 1280, 2560),
        ("qwen3.8 down u4", Codec::Affine4, 512, 10, 2560, 640),
    ] {
        if !only.is_empty() && !name.contains(only.as_str()) {
            continue;
        }
        let planes = Planes::default();
        let per = codec.bytes(n, k);
        let copies = (2u64 << 30).div_ceil(per * u64::from(experts)).clamp(1, 16) as u32;
        let bank = bench_bank(&planes, codec, experts * copies, n, k);
        for &tokens in &token_list {
            let routes_n = tokens * top_k;
            let plans = 8usize;
            let routes: Vec<(Tensor, Vec<i32>)> = (0..plans)
                .map(|s| {
                    let r = make_routes(tokens, top_k, experts as usize, 1000 + s as u64, false);
                    (planes.i32s(&r, tokens as u32, top_k as u32), r)
                })
                .collect();
            let x = planes.keep(
                hip::Buffer::alloc(tokens * k as usize * 2).unwrap(),
                tokens as u32,
                k,
                Dtype::Bf16,
            );
            let mut y = planes.keep(
                hip::Buffer::alloc(routes_n * n as usize * 2).unwrap(),
                routes_n as u32,
                n,
                Dtype::Bf16,
            );
            let reps = if tokens >= 2048 { 8 } else { 32 };
            let mut i = 0usize;
            let us = gpu.time_graph(reps, || {
                let s = i % plans;
                let copy = u32::try_from(s).unwrap() % copies;
                bench_fire(&ctx, codec, bank, copy, experts, x, routes[s].0, &mut y);
                i += 1;
            });
            let touched =
                routes.iter().map(|r| distinct(&r.1)).sum::<usize>() as f64 / plans as f64;
            println!(
                "{name:22} tokens {tokens:5} ({:6.1} routes/expert): {us:9.1} us  {:6.0} GB/s  {:6.1} TFLOP/s",
                routes_n as f64 / touched,
                touched * per as f64 / us / 1e3,
                2.0 * routes_n as f64 * f64::from(n) * f64::from(k) / us / 1e6
            );
        }
    }
}

#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe -- --ignored --nocapture"]
fn bench_grouped_projection() {
    // DeepSeek-V4's grouped o-projection: 8 groups of a 4096-wide head slice
    // through 1024-row blocks (a 64 MB bank), every token a route each; the
    // routed select against the strided-batched launch, cold over copies of
    // the bank 1 GB together. GB/s of the bank.
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let ctx = gpu.ctx();
    let (groups, n, k) = (8u32, 1024u32, 4096u32);
    let per = u64::from(groups) * u64::from(n) * u64::from(k) * 2;
    let copies = 16u32;
    let bank = planes.keep(
        hip::Buffer::alloc((per * u64::from(copies)) as usize).unwrap(),
        groups,
        n * k,
        Dtype::Bf16,
    );
    for tokens in [1u32, 8, 32, 64, 128, 256, 512] {
        let rows = tokens * groups;
        let routes: Vec<i32> = (0..rows).map(|i| (i % groups) as i32).collect();
        let rt = planes.i32s(&routes, tokens, groups);
        let x = planes.keep(
            hip::Buffer::alloc((rows * k * 2) as usize).unwrap(),
            rows,
            k,
            Dtype::Bf16,
        );
        let mut y = planes.keep(
            hip::Buffer::alloc((rows * n * 2) as usize).unwrap(),
            rows,
            n,
            Dtype::Bf16,
        );
        let mut line = format!("o_down {groups}x{n}x{k} tokens {tokens:4}:");
        for identity in [false, true] {
            let mut i = 0u32;
            let us = gpu.time_graph(32, || {
                let mut b = bank;
                b.ptr += u64::from(i % copies) * per;
                moe::matmul_grouped(&ctx, x, b, rt, &mut y, moe::ExpertTable::RESIDENT, identity)
                    .unwrap();
                i += 1;
            });
            line += &format!(
                "  {}: {us:7.1} us {:5.0} GB/s",
                if identity { "strided" } else { "routed" },
                per as f64 / us / 1e3
            );
        }
        println!("{line}");
    }
}

/// A prefill-sized routed matmul against the dense GEMM of the same
/// products (hipBLASLt, every route's row against one expert's weight), at
/// GLM-5.3-Flash's and Qwen3.8-Flash-Next's legs, or `MOE_PF="tokens,top_k,
/// experts,n,k"`. Weights cold enough: the bank is the experts' own.
#[test]
#[ignore = "benchmark: cargo test --release --features rocm --test linear_moe bench_moe_prefill_vs_dense -- --ignored --nocapture"]
fn bench_moe_prefill_vs_dense() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    let shapes: Vec<[usize; 5]> = match std::env::var("MOE_PF") {
        Ok(spec) => {
            let v: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
            vec![[v[0], v[1], v[2], v[3], v[4]]]
        }
        Err(_) => vec![
            [8192, 8, 32, 4096, 4096],
            [8192, 8, 32, 4096, 2048],
            [8192, 10, 16, 1280, 2560],
            [8192, 10, 16, 2560, 640],
        ],
    };
    for [tokens, top_k, experts, n, k] in shapes {
        let (experts, n, k) = (experts as u32, n as u32, k as u32);
        let routes = make_routes(tokens, top_k, experts as usize, 3, false);
        let rt = planes.i32s(&routes, tokens as u32, top_k as u32);
        let x = gpu.tensor(&bf16s(tokens * k as usize, 1, 1.0), tokens as u32, k, Dtype::Bf16);
        let mut y = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
        let bank = random_bank(&gpu, &planes, experts, n, k);
        let flops = 2.0 * (tokens * top_k) as f64 * f64::from(n) * f64::from(k);
        let ctx = gpu.ctx();
        let us = gpu.time(6, || {
            moe::matmul_select(&ctx, x, bank, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
        });
        // The routed leg the default took against the stream leg's products.
        let got = gpu.read(&y);
        let mut want_t = gpu.zeros((tokens * top_k) as u32, n, Dtype::Bf16);
        let stream = tuned(&gpu, |t| t.moe_grouped = false);
        moe::matmul_select(&stream, x, bank, rt, &mut want_t, moe::ExpertTable::RESIDENT).unwrap();
        let want = gpu.read(&want_t);
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs() / (w.abs() + 0.05));
        }
        assert!(worst < 0.05, "routed default leg off the stream leg by {worst}");
        let xr = gpu.tensor(&bf16s(tokens * top_k * k as usize, 2, 1.0), (tokens * top_k) as u32, k, Dtype::Bf16);
        let w = Tensor::new(bank.ptr, n, k, Dtype::Bf16);
        let dense = gpu.time(6, || {
            kernels_rocm::linear::gemm::matmul(&ctx, xr, w, &mut y).unwrap();
        });
        println!(
            "moe prefill {tokens}x{top_k} over {experts} experts {n}x{k}: routed {us:8.1} us {:6.1} TFLOP/s | dense {dense:8.1} us {:6.1} TFLOP/s",
            flops / us / 1e6,
            flops / dense / 1e6
        );
    }
}

/// DeepSeek's routed down-leg input: `min(g, l)·sigmoid(min(g, l)) ·
/// clamp(u, ±l)` times the row's route weight before it rounds, and the
/// routed rows (already weighted) summed in f32 with the shared row.
#[test]
fn the_routed_clamp_and_the_sum_add_answer_the_host() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (tokens, top_k, width, hidden) = (5usize, 3usize, 200usize, 136usize);
    let rows = tokens * top_k;
    let limit = 7.0f32;
    let g = values(rows * width, 1, 9.0).into_iter().map(round_bf16).collect::<Vec<_>>();
    let u = values(rows * width, 2, 9.0).into_iter().map(round_bf16).collect::<Vec<_>>();
    let w = values(rows, 3, 1.0);
    let mut y = gpu.zeros(rows as u32, width as u32, Dtype::Bf16);
    kernels_rocm::linear::mlp::swiglu_clamp_split_routed(
        &ctx,
        gpu.tensor(&g, rows as u32, width as u32, Dtype::Bf16),
        gpu.tensor(&u, rows as u32, width as u32, Dtype::Bf16),
        gpu.tensor(&w, rows as u32, 1, Dtype::F32),
        1,
        limit,
        &mut y,
    )
    .unwrap();
    let want: Vec<f32> = (0..rows * width)
        .map(|i| {
            let gg = g[i].min(limit);
            let uu = u[i].clamp(-limit, limit);
            round_bf16(gg / (1.0 + (-gg).exp()) * uu * w[i / width])
        })
        .collect();
    common::close_bf16("swiglu_clamp_split_routed", &gpu.read(&y), &want, 1.0);

    let routed = values(rows * hidden, 4, 2.0).into_iter().map(round_bf16).collect::<Vec<_>>();
    let shared = values(tokens * hidden, 5, 2.0).into_iter().map(round_bf16).collect::<Vec<_>>();
    let mut out = gpu.zeros(tokens as u32, hidden as u32, Dtype::Bf16);
    kernels_rocm::linear::moe::sum_add(
        &ctx,
        gpu.tensor(&routed, rows as u32, hidden as u32, Dtype::Bf16),
        gpu.tensor(&shared, tokens as u32, hidden as u32, Dtype::Bf16),
        &mut out,
    )
    .unwrap();
    let want: Vec<f32> = (0..tokens * hidden)
        .map(|i| {
            let (n, h) = (i / hidden, i % hidden);
            let s: f32 = (0..top_k).map(|k| routed[(n * top_k + k) * hidden + h]).sum();
            round_bf16(s + shared[i])
        })
        .collect();
    common::close_bf16("moe_sum_add", &gpu.read(&out), &want, 1.0);
}

/// An experts x (n x k) bf16 bank of random weights, not the allocation's
/// zeros: a matrix core fed zeros draws far less power and clocks higher than
/// real data lets it (hipBLASLt read 772 TFLOP/s on zeros, 636 on random
/// values).
fn random_bank(gpu: &Gpu, planes: &Planes, experts: u32, n: u32, k: u32) -> Tensor {
    let bytes = u64::from(experts) * u64::from(n) * u64::from(k) * 2;
    let bank = Tensor::new(
        planes.keep(hip::Buffer::alloc(bytes as usize).unwrap(), 1, 1, Dtype::U8).ptr,
        experts,
        n * k,
        Dtype::Bf16,
    );
    const CHUNK: usize = 16 << 20;
    let block = gpu.tensor(&bf16s(CHUNK / 2, 9, 0.1), 1, (CHUNK / 2) as u32, Dtype::Bf16);
    let mut at = 0usize;
    while at < bytes as usize {
        let len = CHUNK.min(bytes as usize - at);
        // SAFETY: both ranges are live device allocations of at least `len` bytes.
        unsafe {
            hip::rt::hipMemcpy(
                (bank.ptr as usize + at) as *mut _,
                block.ptr as usize as *const _,
                len,
                hip::rt::hipMemcpyDeviceToDevice,
            );
        }
        at += len;
    }
    gpu.sync();
    bank
}

/// The routed swiglu MLP at the prefill models' shapes: the default (the
/// expert-ordered pipeline where it measured faster) against the four routed
/// steps, both against dense GEMMs of the same products, and checked against
/// the steps.
#[test]
#[ignore = "benchmark"]
fn bench_moe_mlp_prefill() {
    let Some(gpu) = Gpu::open() else { return };
    let planes = Planes::default();
    // tokens, top_k, experts, hidden, intermediate, clamped
    let shapes: [(usize, usize, u32, u32, u32, bool); 4] = [
        (8192, 8, 32, 4096, 2048, true),
        (4096, 8, 32, 4096, 2048, true),
        (8192, 10, 16, 2560, 640, false),
        (4096, 10, 16, 2560, 640, false),
    ];
    for (tokens, top_k, experts, hidden, inter, clamped) in shapes {
        let r = (tokens * top_k) as u32;
        let routes = make_routes(tokens, top_k, experts as usize, 3, false);
        let rt = planes.i32s(&routes, tokens as u32, top_k as u32);
        let wt = gpu.tensor(&values(tokens * top_k, 2, 1.0), tokens as u32, top_k as u32, Dtype::F32);
        let x = gpu.tensor(&bf16s(tokens * hidden as usize, 1, 1.0), tokens as u32, hidden, Dtype::Bf16);
        let gate_up = random_bank(&gpu, &planes, experts, 2 * inter, hidden);
        let down = random_bank(&gpu, &planes, experts, hidden, inter);
        let p = moe::RoutedMlp {
            x,
            routes: rt,
            weights: wt,
            gate_up,
            gate_up_experts: moe::ExpertTable::RESIDENT,
            down,
            down_experts: moe::ExpertTable::RESIDENT,
            intermediate: inter,
            limit: clamped.then_some(7.0),
            fan: 1,
            packed: gpu.zeros(r, 2 * inter, Dtype::Bf16),
            h: gpu.zeros(r, inter, Dtype::Bf16),
            routed: gpu.zeros(r, hidden, Dtype::Bf16),
        };
        let ctx = gpu.ctx();
        let mut y = gpu.zeros(tokens as u32, hidden, Dtype::Bf16);
        let mut y_steps = gpu.zeros(tokens as u32, hidden, Dtype::Bf16);
        let fused = gpu.time(6, || moe::mlp_swiglu(&ctx, &p, &mut y).unwrap());
        let steps = gpu.time(6, || {
            let (mut packed, mut h, mut routed) = (p.packed, p.h, p.routed);
            moe::matmul_select(&ctx, x, gate_up, rt, &mut packed, moe::ExpertTable::RESIDENT).unwrap();
            match p.limit {
                Some(l) => kernels_rocm::linear::mlp::swiglu_clamp(&ctx, packed, inter, 1, l, &mut h).unwrap(),
                None => kernels_rocm::linear::mlp::swiglu(&ctx, packed, inter, 1, &mut h).unwrap(),
            }
            moe::matmul_select(&ctx, h, down, rt, &mut routed, moe::ExpertTable::RESIDENT).unwrap();
            moe::weighted_sum(&ctx, routed, wt, &mut y_steps).unwrap();
        });
        let (got, want) = (gpu.read(&y), gpu.read(&y_steps));
        // Both round the gate_up products and the activation to bf16, from
        // sums in different orders: an ulp apart there moves a few outputs by
        // a few percent of the typical magnitude, never the whole.
        let mag = want.iter().map(|v| v.abs()).sum::<f32>() / want.len() as f32;
        let (mut worst, mut err, mut norm) = (0f32, 0f64, 0f64);
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs() / (w.abs() + 0.1 * mag));
            err += f64::from(g - w).powi(2);
            norm += f64::from(*w).powi(2);
        }
        let rms = (err / norm).sqrt();
        assert!(
            rms < 5e-3 && worst < 0.2,
            "routed MLP off its four steps: rms {rms}, worst {worst}"
        );
        let xr = gpu.tensor(&bf16s(r as usize * hidden as usize, 2, 1.0), r, hidden, Dtype::Bf16);
        let hr = gpu.tensor(&bf16s(r as usize * inter as usize, 4, 1.0), r, inter, Dtype::Bf16);
        let mut packed = p.packed;
        let mut routed = p.routed;
        let w1 = Tensor::new(gate_up.ptr, 2 * inter, hidden, Dtype::Bf16);
        let w2 = Tensor::new(down.ptr, hidden, inter, Dtype::Bf16);
        let dense = gpu.time(6, || {
            kernels_rocm::linear::gemm::matmul(&ctx, xr, w1, &mut packed).unwrap();
            kernels_rocm::linear::gemm::matmul(&ctx, hr, w2, &mut routed).unwrap();
        });
        let flops = 2.0 * f64::from(r) * 3.0 * f64::from(inter) * f64::from(hidden);
        println!(
            "moe mlp {tokens}x{top_k} over {experts} experts {hidden}/{inter}: default {fused:8.1} us {:6.1} TF | steps {steps:8.1} us {:6.1} TF | dense gemms {dense:8.1} us {:6.1} TF | default/dense {:5.1}% | rms {rms:.1e} worst {worst:.3}",
            flops / fused / 1e6,
            flops / steps / 1e6,
            flops / dense / 1e6,
            100.0 * dense / fused,
        );
    }
}
