//! Mixture of experts against host references: the routers (softmax,
//! sigmoid with and without sinks, sqrt-softplus, the hash table, the group
//! fan-out), the folds back into token rows, and the expert matmuls over a
//! dense, an mxfp4 and an MLX affine bank on each of their legs (the decode
//! stream, the batch leg, the grouped matrix-core leg, the off-core grouped
//! leg, rocBLAS).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16, bf16s, close_abs, close_bf16, round_bf16, round_f16, values};
use dtype::Dtype;

use kernels_rocm::linear::{moe, moe_route};
use kernels_rocm::tuning::Tuning;
use kernels_rocm::{Ctx, Tensor};

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
            let mut ids = gpu.upload(&vec![0; r * kk], rows, k, Dtype::I32);
            let mut ws = gpu.zeros(rows, k, Dtype::F32);
            if scaled {
                moe::topk_softmax_scaled(&ctx, lt, gt, experts, k, &mut ids, &mut ws).unwrap();
            } else {
                moe::topk_softmax(&ctx, lt, experts, k, &mut ids, &mut ws).unwrap();
            }
            let (hi, hw) = (gpu.read_as::<i32>(&ids), gpu.read(&ws));
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
            let mut ids = gpu.upload(&vec![0; r * kk], rows, k, Dtype::I32);
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
            let (hi, hw) = (gpu.read_as::<i32>(&ids), gpu.read(&ws));
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
        let mut ids = gpu.upload(&vec![0; r * (kk + s)], rows, fan, Dtype::I32);
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
        let (hi, hw) = (gpu.read_as::<i32>(&ids), gpu.read(&ws));
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
    let mut ids = gpu.upload(&[-7; 24], rows, k, Dtype::I32);
    let mut ws = gpu.zeros(rows, k, Dtype::F32);
    let win = gpu.ints(&[3, 2, 0, 0]);
    ctx.arm_stage(win.ptr);
    moe::topk_sigmoid(&ctx, lt, None, experts, k, true, 1.0, &mut ids, &mut ws).unwrap();
    ctx.disarm_stage();
    let hi = gpu.read_as::<i32>(&ids);
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

    // The hash route: ids past the vocabulary read row 0, an expert id past
    // the logits weighs nothing.
    let (tokens, vocab, experts, k) = (5u32, 7u32, 16u32, 3u32);
    let ids = [3, 0, 9, -1, 6];
    let table: Vec<i64> = (0..vocab as i64 * 3)
        .map(|i| if i == 5 { 40 } else { (i * 5) % 16 })
        .collect();
    let logits = bf16s((tokens * experts) as usize, 3, 4.0);
    let idt = gpu.upload(&ids, tokens, 1, Dtype::I32);
    let tt = gpu.upload(&table, vocab, k, Dtype::I64);
    let lt = gpu.tensor(&logits, tokens, experts, Dtype::Bf16);
    for renormalize in [false, true] {
        let mut routes = gpu.upload(&[0; 15], tokens, k, Dtype::I32);
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
        let (hr, hw) = (gpu.read_as::<i32>(&routes), gpu.read(&ws));
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
    let mut routes = gpu.upload(&[-1; 4 * 6], 4, 6, Dtype::I32);
    moe_route::group_routes(&ctx, 6, &mut routes).unwrap();
    assert_eq!(
        gpu.read_as::<i32>(&routes),
        (0..24).map(|i| i % 6).collect::<Vec<_>>()
    );
    let mut routes = gpu.upload(&[-1; 4 * 6], 4, 6, Dtype::I32);
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    moe_route::group_routes(&ctx, 6, &mut routes).unwrap();
    ctx.disarm_stage();
    let want: Vec<i32> = (0..24)
        .map(|i| if (6..18).contains(&i) { i % 6 } else { -1 })
        .collect();
    assert_eq!(gpu.read_as::<i32>(&routes), want, "windowed group routes");
}

#[test]
fn moe_folds() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
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
    let routes_t = gpu.upload(&routes, tokens, k, Dtype::I32);
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
    // (experts, tokens, top_k, n, k): the decode stream across short and long
    // K and a ragged N; the batch leg in 4- to 32-route items, K short of a
    // whole stage, and past its reach the grouped leg; fires wide enough to
    // group natively (in 32-, 64- and 128-route items), and a K the native
    // legs cannot read, which rocBLAS takes.
    for (experts, tokens, top_k, n, k, leg) in [
        (8u32, 1u32, 2u32, 1280u32, 2560u32, "stream"),
        (4, 2, 2, 72, 7168, "stream"),
        (3, 2, 1, 13, 64, "stream"),
        (8, 1, 2, 1280, 2560, "batch"),
        (4, 2, 2, 72, 7168, "batch"),
        (8, 40, 4, 520, 264, "batch"),
        (5, 64, 2, 136, 2312, "batch"),
        (4, 96, 4, 136, 512, "batch"),
        (3, 100, 2, 300, 264, "batch"),
        (8, 40, 4, 520, 256, "grouped"),
        (5, 64, 2, 128, 2304, "grouped"),
        // 128-route items over whole 512-row banks and a long K: the tall block.
        (4, 128, 4, 512, 2048, "grouped"),
        (3, 200, 2, 1024, 2048, "grouped"),
        (3, 20, 2, 50, 60, "blas"),
        // The wide leg in 64- to 256-route items, N ragged by 8 and by 2.
        (8, 64, 4, 520, 256, "wide"),
        (5, 65, 2, 130, 320, "wide"),
        (2, 220, 2, 72, 2048, "wide"),
    ] {
        let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
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
            let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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
    let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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
    let table_t = gpu.upload(&table, 4, 1, Dtype::U8);
    let hits = gpu.upload(&[0; 4], 1, 4, Dtype::I32);
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
    assert_eq!(gpu.read_as::<i32>(&hits), counted, "expert hits");
}

#[test]
fn moe_matmul_dense_wide_skewed() {
    // The wide leg where one expert takes more routes than an item: expert 0
    // in every token (100 routes over 64-route items), the others a few
    // each, so its routes split over two items, each reading the weight.
    let Some(gpu) = Gpu::open() else { return };
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
    let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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
    // Decode (the stream, and the batch leg in 16- and 32-route items, K
    // short of a whole stage), then grouped on the matrix core and off it;
    // a bias on the gate/up leg.
    for (experts, tokens, top_k, n, k, leg) in [
        (8u32, 1u32, 4u32, 1280u32, 2560u32, "stream"),
        (4, 2, 2, 100, 7168, "stream"),
        (8, 1, 4, 1280, 2560, "batch"),
        (4, 2, 2, 100, 7168, "batch"),
        (8, 50, 4, 384, 512, "batch"),
        (3, 80, 2, 200, 2304, "batch"),
        (8, 50, 4, 384, 512, "wmma"),
        (4, 72, 4, 256, 512, "wmma"),
        // Few items over a long K: the matrix-core leg slices K (4 slices,
        // an odd stage count; 2 slices of 32-route items).
        (4, 40, 4, 256, 4096, "wmma"),
        (4, 64, 2, 136, 4160, "wmma"),
        (3, 20, 2, 136, 4096, "wmma"),
        (8, 50, 4, 200, 512, "simt"),
    ] {
        let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
        let (codes, scales, bank) = mxfp4_bank(e, nn, kd, u64::from(n + k));
        let ct = gpu.upload(&codes, experts, n * k / 2, Dtype::U8);
        let st = gpu.upload(&scales, experts, n * k / 32, Dtype::U8);
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
            let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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

#[test]
fn moe_matmul_mxfp4_lanes() {
    // A bank put in the matrix core's lane order at load, read by route (one
    // token) and in expert order (items of 16, 32 and 64 routes, a ragged
    // last strip block).
    let Some(gpu) = Gpu::open() else { return };
    let ctx = tuned(&gpu, |_| {});
    for (experts, tokens, top_k, n, k) in [
        (8u32, 1u32, 4u32, 1280u32, 2560u32),
        (6, 1, 6, 208, 640),
        (8, 50, 4, 384, 512),
        (3, 200, 2, 208, 1024),
    ] {
        let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
        let (codes, scales, bank) = mxfp4_bank(e, nn, kd, u64::from(n + k));
        let ct = gpu.upload(&codes, experts, n * k / 2, Dtype::U8);
        let st = gpu.upload(&scales, experts, n * k / 32, Dtype::U8);
        let lc = gpu.upload(&vec![0u8; codes.len()], experts, n * k / 2, Dtype::U8);
        let ls = gpu.upload(&vec![0u8; scales.len()], experts, n * k / 32, Dtype::U8);
        moe::mxfp4_lanes(
            &ctx,
            "test",
            ct.ptr,
            lc.ptr,
            st.ptr,
            ls.ptr,
            u64::from(experts * n),
            u64::from(k),
        )
        .unwrap();
        for by_token in [true, false] {
            let routes = make_routes(t, kk, e, u64::from(k), true);
            let x_rows = if by_token { t } else { t * kk };
            let x = bf16s(x_rows * kd, 32, 1.0);
            let xt = gpu.tensor(&x, x_rows as u32, k, Dtype::Bf16);
            let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
            let mut y = gpu.zeros(tokens * top_k, n, Dtype::Bf16);
            moe::matmul_select_mxfp4_lanes(&ctx, xt, lc, ls, rt, &mut y).unwrap();
            let (want, mag) = reference(&bank, &x, &routes, kk, by_token, None);
            close_dot(
                &format!("mxfp4 lanes {experts}e {tokens}x{top_k} {n}x{k} by_token={by_token}"),
                &gpu.read(&y),
                &want,
                &mag,
                1e-5,
            );
        }
    }
}

#[test]
fn mxfp4_lanes_guard() {
    // The lane-order tiles at a prefill's batches write no row past y's.
    let Some(gpu) = Gpu::open() else { return };
    let ctx = tuned(&gpu, |_| {});
    for (experts, tokens, top_k, n, k, by_token) in [
        (16u32, 16384u32, 6u32, 2304u32, 5120u32, true),
        (16, 16384, 6, 5120, 2304, false),
    ] {
        let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
        let codes: Vec<u8> = (0..e * nn * kd / 2)
            .map(|i| (i * 2654435761usize >> 13) as u8)
            .collect();
        let scales: Vec<u8> = (0..e * nn * kd / 32).map(|i| 120 + (i % 9) as u8).collect();
        let ct = gpu.upload(&codes, experts, n * k / 2, Dtype::U8);
        let st = gpu.upload(&scales, experts, n * k / 32, Dtype::U8);
        let routes = make_routes(t, kk, e, 5, true);
        let x_rows = if by_token { t } else { t * kk };
        let x = bf16s(x_rows * kd, 32, 1.0);
        let xt = gpu.tensor(&x, x_rows as u32, k, Dtype::Bf16);
        let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
        const GUARD: u32 = 64;
        let full = gpu.zeros(tokens * top_k + GUARD, n, Dtype::Bf16);
        let mut y = Tensor::new(full.ptr, tokens * top_k, n, Dtype::Bf16);
        moe::matmul_select_mxfp4_lanes(&ctx, xt, ct, st, rt, &mut y).unwrap();
        let got = gpu.read(&full);
        let live = (tokens * top_k * n) as usize;
        let past = got[live..].iter().filter(|v| **v != 0.0).count();
        assert_eq!(past, 0, "a write past y");
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
    for (bits, group) in [(4u32, 64u32), (4, 32), (4, 128), (2, 32), (8, 128), (8, 32)] {
        for (experts, tokens, top_k, n, k, leg) in [
            (6u32, 2u32, 3u32, 640u32, 2560u32, "stream"),
            (4, 1, 2, 70, 1024, "stream"),
            (6, 2, 3, 640, 2560, "batch"),
            (8, 40, 4, 264, 512, "batch"),
            (4, 24, 4, 136, 1280, "batch"),
            (3, 80, 2, 200, 640, "batch"),
            (8, 40, 4, 264, 512, "wmma"),
            (4, 48, 4, 136, 256, "wmma"),
            (8, 40, 4, 136, 384, "simt"),
        ] {
            // The batch leg reads four-bit codes, an even count of groups a
            // row.
            if leg == "batch" && (bits != 4 || k % (2 * group) != 0) {
                continue;
            }
            let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
            let (codes, scales, biases, exact, rounded) = affine_bank(
                e,
                nn,
                kd,
                bits as usize,
                group as usize,
                u64::from(bits * 100 + group),
            );
            let ct = gpu.upload(&codes, experts, n * k * bits / 8, Dtype::U8);
            let st = gpu.upload(&scales, experts, 2 * n * k / group, Dtype::U8);
            let bt = gpu.upload(&biases, experts, 2 * n * k / group, Dtype::U8);
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
                let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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

fn f16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_f16).collect()
}

#[test]
fn moe_matmul_f16() {
    // The f16 instantiations: the dense stream, batch leg and both grouped
    // widths, the mxfp4 stream and its off-core grouped leg, and the affine stream.
    let Some(gpu) = Gpu::open() else { return };
    for (experts, tokens, top_k, n, k, leg) in [
        (6u32, 2u32, 3u32, 300u32, 640u32, "stream"),
        (6, 2, 3, 300, 648, "batch"),
        (8, 40, 4, 200, 256, "batch"),
        (8, 40, 4, 200, 256, "grouped"),
        (4, 96, 4, 136, 512, "grouped"),
        (4, 96, 4, 136, 512, "wide"),
    ] {
        let [e, t, kk, nn, kd] = [experts, tokens, top_k, n, k].map(|v| v as usize);
        let bank = Dense {
            n: nn,
            k: kd,
            w: f16s(e * nn * kd, 51, 0.1),
        };
        let bt = gpu.tensor(&bank.w, experts, n * k, Dtype::F16);
        let ctx = tuned(&gpu, |t| {
            t.moe_grouped = leg != "stream";
            t.moe_batch = leg == "batch" || leg == "wide";
            t.moe_batch_from = 1;
            t.moe_dense_wide_from = u32::from(leg == "wide");
        });
        let routes = make_routes(t, kk, e, 52, true);
        let x = f16s(t * kd, 53, 1.0);
        let xt = gpu.tensor(&x, tokens, k, Dtype::F16);
        let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
        let mut y = gpu.zeros(tokens * top_k, n, Dtype::F16);
        moe::matmul_select(&ctx, xt, bt, rt, &mut y, moe::ExpertTable::RESIDENT).unwrap();
        let (want, mag) = reference(&bank, &x, &routes, kk, true, None);
        close_dot(
            &format!("f16 dense {leg} {n}x{k}"),
            &gpu.read(&y),
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
        let ct = gpu.upload(&codes, 6, n * k / 2, Dtype::U8);
        let st = gpu.upload(&scales, 6, n * k / 32, Dtype::U8);
        let ctx = tuned(&gpu, |t| {
            t.mxfp4_grouped = leg != "stream";
            t.moe_batch = false;
        });
        let routes = make_routes(t, kk, 6, 62, true);
        let x = f16s(t * kd, 63, 1.0);
        let xt = gpu.tensor(&x, tokens, k, Dtype::F16);
        let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
        let mut y = gpu.zeros(tokens * top_k, n, Dtype::F16);
        moe::matmul_select_quant(&ctx, xt, ct, st, None, rt, &mut y, moe::GroupSeat::RESIDENT)
            .unwrap();
        let (want, mag) = reference(&bank, &x, &routes, kk, true, None);
        close_dot(
            &format!("f16 mxfp4 {leg} {n}x{k}"),
            &gpu.read(&y),
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
        let ct = gpu.upload(&codes, 6, n * k * bits / 8, Dtype::U8);
        let st = gpu.upload(&scales, 6, 2 * n * k / group, Dtype::U8);
        let bt = gpu.upload(&biases, 6, 2 * n * k / group, Dtype::U8);
        let ctx = tuned(&gpu, |t| {
            t.affine_grouped = false;
            t.moe_batch = false;
        });
        let routes = make_routes(t, kk, 6, 72, true);
        let x = f16s(t * kd, 73, 1.0);
        let xt = gpu.tensor(&x, tokens, k, Dtype::F16);
        let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
        let mut y = gpu.zeros(tokens * top_k, n, Dtype::F16);
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
            &gpu.read(&y),
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
            let rt = gpu.upload(&routes, tokens, groups, Dtype::I32);
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
    for k in [256u32, 4096] {
        let (experts, tokens, top_k, n) = (4u32, 4u32, 4u32, 128u32);
        let (codes, scales, bank) = mxfp4_bank(4, 128, k as usize, 77);
        let ct = gpu.upload(&codes, experts, n * k / 2, Dtype::U8);
        let st = gpu.upload(&scales, experts, n * k / 32, Dtype::U8);
        let routes = make_routes(4, 4, 4, 78, false);
        let x = bf16s(4 * k as usize, 79, 1.0);
        let xt = gpu.tensor(&x, tokens, k, Dtype::Bf16);
        let rt = gpu.upload(&routes, tokens, top_k, Dtype::I32);
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
    let g = values(rows * width, 1, 9.0)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let u = values(rows * width, 2, 9.0)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
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

    let routed = values(rows * hidden, 4, 2.0)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
    let shared = values(tokens * hidden, 5, 2.0)
        .into_iter()
        .map(round_bf16)
        .collect::<Vec<_>>();
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
            let s: f32 = (0..top_k)
                .map(|k| routed[(n * top_k + k) * hidden + h])
                .sum();
            round_bf16(s + shared[i])
        })
        .collect();
    common::close_bf16("moe_sum_add", &gpu.read(&out), &want, 1.0);
}
