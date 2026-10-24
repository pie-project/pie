//! The hyper-connection kernels against a host reference: mHC's norm,
//! projection (and the two fused), Sinkhorn gates, fold and trunk collapse,
//! the stream expand, and the gated-residual mix, inject and per-layer
//! embedding gate. Rows are M streams of H: H off the block, off a 16-byte
//! pack (the scalar forms), eight streams, and the models' decode shapes.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, bf16s, close_bf16, round_bf16, sigmoid, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::elemwise::{hc, hc_project};

/// A row shape: `n` rows of `m` streams of `h`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    n: usize,
    m: usize,
    h: usize,
}

const fn shape(n: usize, m: usize, h: usize) -> Shape {
    Shape { n, m, h }
}

/// The small rows every kernel runs on: H past a block, H off a pack, and
/// eight streams.
const SMALL: [Shape; 3] = [shape(5, 4, 320), shape(3, 4, 36), shape(2, 8, 48)];

/// The models' decode shapes.
const MHC: [Shape; 2] = [shape(7, 4, 4096), shape(128, 4, 5120)];
const GATED: [Shape; 2] = [shape(1, 4, 2560), shape(128, 4, 2560)];

/// Relative closeness for f32 outputs whose sums the device orders
/// differently.
fn close_f32(what: &str, have: &[f32], want: &[f32]) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        assert!(
            (h - w).abs() <= 1e-5 * w.abs().max(1.0),
            "{what}[{i}] = {h}, want {w}"
        );
    }
}

/// Rows outside `live` of a launch `width` wide stay zero; the rest are `want`.
fn windowed(want: &[f32], width: usize, live: core::ops::Range<usize>) -> Vec<f32> {
    want.chunks(width)
        .enumerate()
        .flat_map(|(n, row)| {
            if live.contains(&n) {
                row.to_vec()
            } else {
                vec![0.0; width]
            }
        })
        .collect()
}

struct Mixed {
    post: Vec<f32>,
    comb: Vec<f32>,
    x: Vec<f32>,
}

/// mHC's gates of every row, as `hc_gates` takes them.
fn host_gates(
    s: Shape,
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    streams: &[f32],
    eps: f32,
    alpha: f32,
    iters: usize,
) -> Mixed {
    let Shape { n: rows, m, h } = s;
    let mix_hc = 2 * m + m * m;
    let (mut post, mut comb, mut x) = (Vec::new(), Vec::new(), Vec::new());
    for n in 0..rows {
        let row = &mixes[n * mix_hc..(n + 1) * mix_hc];
        let pre: Vec<f32> = (0..m)
            .map(|i| sigmoid(row[i] * scale[0] + base[i]) + eps)
            .collect();
        post.extend((0..m).map(|i| sigmoid(row[m + i] * scale[1] + base[m + i]) * alpha));
        let mut c: Vec<f32> = (0..m * m)
            .map(|t| row[2 * m + t] * scale[2] + base[2 * m + t])
            .collect();
        for r in c.chunks_mut(m) {
            let top = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in r.iter_mut() {
                *v = (*v - top).exp();
                sum += *v;
            }
            for v in r.iter_mut() {
                *v = *v / sum + eps;
            }
        }
        let columns = |c: &mut [f32]| {
            for j in 0..m {
                let sum = (0..m).map(|i| c[i * m + j]).sum::<f32>() + eps;
                for i in 0..m {
                    c[i * m + j] /= sum;
                }
            }
        };
        columns(&mut c);
        for _ in 1..iters {
            for r in c.chunks_mut(m) {
                let sum = r.iter().sum::<f32>() + eps;
                for v in r.iter_mut() {
                    *v /= sum;
                }
            }
            columns(&mut c);
        }
        comb.extend(&c);
        let st = &streams[n * m * h..(n + 1) * m * h];
        x.extend((0..h).map(|k| round_bf16((0..m).map(|i| pre[i] * st[i * h + k]).sum())));
    }
    Mixed { post, comb, x }
}

fn host_fold(s: Shape, x: &[f32], streams: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    let Shape { n: rows, m, h } = s;
    let mut out = vec![0.0; rows * m * h];
    for n in 0..rows {
        for j in 0..m {
            for k in 0..h {
                let mut acc = post[n * m + j] * x[n * h + k];
                for i in 0..m {
                    acc += comb[n * m * m + i * m + j] * streams[n * m * h + i * h + k];
                }
                out[n * m * h + j * h + k] = round_bf16(acc);
            }
        }
    }
    out
}

/// The f32 row norm `hc_rmsnorm_f32` lands.
fn host_normed(s: Shape, streams: &[f32], eps: f32) -> Vec<f32> {
    let wide = s.m * s.h;
    streams
        .chunks(wide)
        .flat_map(|r| {
            let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / wide as f32 + eps).sqrt();
            r.iter().map(move |v| v * inv).collect::<Vec<_>>()
        })
        .collect()
}

/// `normed · fnᵀ` in f64, and the bound f32 sums in any order stay within:
/// a few f32 steps of the products' absolute sum.
fn host_project(normed: &[f32], fn_h: &[f32], wide: usize, outs: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = normed.len() / wide;
    let mut want = Vec::with_capacity(rows * outs);
    let mut tol = Vec::with_capacity(rows * outs);
    for n in 0..rows {
        let x = &normed[n * wide..(n + 1) * wide];
        for o in 0..outs {
            let w = &fn_h[o * wide..(o + 1) * wide];
            let (mut sum, mut mag) = (0f64, 0f64);
            for (a, b) in x.iter().zip(w) {
                let p = f64::from(*a) * f64::from(*b);
                sum += p;
                mag += p.abs();
            }
            want.push(sum as f32);
            tol.push((mag * 1e-6) as f32 + 1e-6);
        }
    }
    (want, tol)
}

fn close_within(what: &str, have: &[f32], want: &[f32], tol: &[f32]) {
    assert_eq!(have.len(), want.len(), "{what}: lengths");
    for (i, ((h, w), t)) in have.iter().zip(want).zip(tol).enumerate() {
        assert!((h - w).abs() <= *t, "{what}[{i}] = {h}, want {w} (±{t})");
    }
}

fn mhc_chain(gpu: &Gpu, s: Shape) {
    let ctx = gpu.ctx();
    let Shape { n, m, h } = s;
    let (rows, wide) = (n as u32, (m * h) as u32);
    let what = |op: &str| format!("{op} {s:?}");
    let (eps, gate_eps, alpha, iters) = (1e-6f32, 1e-6f32, 2.0f32, 20u32);
    let mix_hc = 2 * m + m * m;
    let streams_h = bf16s(n * m * h, 1, 2.0);
    let streams = gpu.tensor(&streams_h, rows, wide, Dtype::Bf16);

    // The stream row normed to f32.
    let mut normed = gpu.zeros(rows, wide, Dtype::F32);
    hc::rmsnorm_f32(&ctx, streams, eps, &mut normed).unwrap();
    let normed_h = gpu.read(&normed);
    close_f32(
        &what("hc_rmsnorm_f32"),
        &normed_h,
        &host_normed(s, &streams_h, eps),
    );

    // The projection to the mix row (and to a trunk collapse row of M).
    // The independent form's row of 2M (no combine) too.
    for rows_out in [mix_hc, 2 * m, m] {
        let fn_h = values(rows_out * m * h, 2, 0.05);
        let hc_fn = gpu.tensor(&fn_h, rows_out as u32, wide, Dtype::F32);
        let mut mixes = gpu.zeros(rows, rows_out as u32, Dtype::F32);
        hc_project::project(&ctx, normed, hc_fn, m as u32, &mut mixes).unwrap();
        let (want, tol) = host_project(&normed_h, &fn_h, m * h, rows_out);
        close_within(&what("hc_project"), &gpu.read(&mixes), &want, &tol);

        // The norm and the projection in one, against the host's norm.
        let mut fused = gpu.zeros(rows, rows_out as u32, Dtype::F32);
        hc_project::norm_project(&ctx, streams, hc_fn, eps, m as u32, &mut fused).unwrap();
        let (want, tol) = host_project(&host_normed(s, &streams_h, eps), &fn_h, m * h, rows_out);
        close_within(&what("hc_norm_project"), &gpu.read(&fused), &want, &tol);
    }

    // The gates, from a mix row of the scale a real projection gives.
    let mixes_h = values(n * mix_hc, 3, 1.5);
    let mixes = gpu.tensor(&mixes_h, rows, mix_hc as u32, Dtype::F32);
    let scale_h = [0.9f32, 1.1, 1.3];
    let scale = gpu.tensor(&scale_h, 1, 3, Dtype::F32);
    let base_h = values(mix_hc, 4, 0.5);
    let base = gpu.tensor(&base_h, 1, mix_hc as u32, Dtype::F32);
    let mut x = gpu.zeros(rows, h as u32, Dtype::Bf16);
    let mut post = gpu.zeros(rows, m as u32, Dtype::F32);
    let mut comb = gpu.zeros(rows, (m * m) as u32, Dtype::F32);
    hc::gates(
        &ctx, mixes, streams, scale, base, m as u32, gate_eps, alpha, iters, true, &mut x,
        &mut post, &mut comb,
    )
    .unwrap();
    let want = host_gates(
        s,
        &mixes_h,
        &scale_h,
        &base_h,
        &streams_h,
        gate_eps,
        alpha,
        iters as usize,
    );
    close_f32(&what("hc_gates post"), &gpu.read(&post), &want.post);
    close_f32(&what("hc_gates comb"), &gpu.read(&comb), &want.comb);
    close_bf16(&what("hc_gates x"), &gpu.read(&x), &want.x, 1.0);

    // The independent form: pre and post logits alone, the post gate under
    // the pre gate's eps, and the identity combine.
    let ind_h = values(n * 2 * m, 5, 1.5);
    let ind = gpu.tensor(&ind_h, rows, (2 * m) as u32, Dtype::F32);
    let mut xi = gpu.zeros(rows, h as u32, Dtype::Bf16);
    let mut posti = gpu.zeros(rows, m as u32, Dtype::F32);
    let mut combi = gpu.zeros(rows, (m * m) as u32, Dtype::F32);
    hc::gates(
        &ctx, ind, streams, scale, base, m as u32, gate_eps, alpha, iters, false, &mut xi,
        &mut posti, &mut combi,
    )
    .unwrap();
    let (mut post_i, mut comb_i, mut x_i) = (Vec::new(), Vec::new(), Vec::new());
    for r in 0..n {
        let row = &ind_h[r * 2 * m..(r + 1) * 2 * m];
        let pre: Vec<f32> = (0..m)
            .map(|i| sigmoid(row[i] * scale_h[0] + base_h[i]) + gate_eps)
            .collect();
        post_i.extend(
            (0..m).map(|i| sigmoid(row[m + i] * scale_h[1] + base_h[m + i]) * alpha + gate_eps),
        );
        comb_i.extend((0..m * m).map(|t| if t / m == t % m { 1.0 } else { 0.0 }));
        let st = &streams_h[r * m * h..(r + 1) * m * h];
        x_i.extend((0..h).map(|k| round_bf16((0..m).map(|i| pre[i] * st[i * h + k]).sum())));
    }
    close_f32(
        &what("hc_gates independent post"),
        &gpu.read(&posti),
        &post_i,
    );
    close_f32(
        &what("hc_gates independent comb"),
        &gpu.read(&combi),
        &comb_i,
    );
    close_bf16(&what("hc_gates independent x"), &gpu.read(&xi), &x_i, 1.0);

    // The fold, into a fresh plane and in place over the streams.
    let layer_out_h = bf16s(n * h, 5, 1.0);
    let layer_out = gpu.tensor(&layer_out_h, rows, h as u32, Dtype::Bf16);
    let want_fold = host_fold(s, &layer_out_h, &streams_h, &want.post, &want.comb);
    let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
    hc::fold(&ctx, layer_out, streams, post, comb, &mut y).unwrap();
    close_bf16(&what("hc_fold"), &gpu.read(&y), &want_fold, 1.0);
    let mut in_place = gpu.tensor(&streams_h, rows, wide, Dtype::Bf16);
    hc::fold(&ctx, layer_out, in_place, post, comb, &mut in_place).unwrap();
    close_bf16(
        &what("hc_fold in place"),
        &gpu.read(&in_place),
        &want_fold,
        1.0,
    );

    // The trunk collapse.
    let collapse_h = values(n * m, 6, 1.5);
    let collapse_mix = gpu.tensor(&collapse_h, rows, m as u32, Dtype::F32);
    let mut head = gpu.zeros(rows, h as u32, Dtype::Bf16);
    hc::collapse(
        &ctx,
        collapse_mix,
        streams,
        scale,
        base,
        m as u32,
        gate_eps,
        &mut head,
    )
    .unwrap();
    let want_head: Vec<f32> = (0..n * h)
        .map(|t| {
            let (r, k) = (t / h, t % h);
            round_bf16(
                (0..m)
                    .map(|i| {
                        (sigmoid(collapse_h[r * m + i] * scale_h[0] + base_h[i]) + gate_eps)
                            * streams_h[r * m * h + i * h + k]
                    })
                    .sum(),
            )
        })
        .collect();
    close_bf16(
        &what("hc_head_postprocess"),
        &gpu.read(&head),
        &want_head,
        1.0,
    );

    // A window over every kernel: the launch's rows past its first and
    // before its last. A row outside the window is never written.
    if n < 3 {
        return;
    }
    let live = 1..n - 1;
    let win = gpu.ints(&[(n - 2) as i32, 1, 0, 0]);
    let mut normed_w = gpu.zeros(rows, wide, Dtype::F32);
    let fn_h = values(mix_hc * m * h, 2, 0.05);
    let hc_fn = gpu.tensor(&fn_h, mix_hc as u32, wide, Dtype::F32);
    let mut mixes_w = gpu.zeros(rows, mix_hc as u32, Dtype::F32);
    let mut fused_w = gpu.zeros(rows, mix_hc as u32, Dtype::F32);
    let mut x = gpu.zeros(rows, h as u32, Dtype::Bf16);
    let mut post = gpu.zeros(rows, m as u32, Dtype::F32);
    let mut comb = gpu.zeros(rows, (m * m) as u32, Dtype::F32);
    let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
    let mut head = gpu.zeros(rows, h as u32, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    hc::rmsnorm_f32(&ctx, streams, eps, &mut normed_w).unwrap();
    hc_project::project(&ctx, normed, hc_fn, m as u32, &mut mixes_w).unwrap();
    hc_project::norm_project(&ctx, streams, hc_fn, eps, m as u32, &mut fused_w).unwrap();
    hc::gates(
        &ctx, mixes, streams, scale, base, m as u32, gate_eps, alpha, iters, true, &mut x,
        &mut post, &mut comb,
    )
    .unwrap();
    hc::fold(&ctx, layer_out, streams, post, comb, &mut y).unwrap();
    hc::collapse(
        &ctx,
        collapse_mix,
        streams,
        scale,
        base,
        m as u32,
        gate_eps,
        &mut head,
    )
    .unwrap();
    ctx.disarm_stage();
    close_f32(
        &what("windowed hc_rmsnorm_f32"),
        &gpu.read(&normed_w),
        &windowed(&normed_h, m * h, live.clone()),
    );
    let (want_mix, tol) = host_project(&normed_h, &fn_h, m * h, mix_hc);
    close_within(
        &what("windowed hc_project"),
        &gpu.read(&mixes_w),
        &windowed(&want_mix, mix_hc, live.clone()),
        &tol,
    );
    close_within(
        &what("windowed hc_norm_project"),
        &gpu.read(&fused_w),
        &windowed(&want_mix, mix_hc, live.clone()),
        &tol,
    );
    close_f32(
        &what("windowed post"),
        &gpu.read(&post),
        &windowed(&want.post, m, live.clone()),
    );
    close_f32(
        &what("windowed comb"),
        &gpu.read(&comb),
        &windowed(&want.comb, m * m, live.clone()),
    );
    close_bf16(
        &what("windowed x"),
        &gpu.read(&x),
        &windowed(&want.x, h, live.clone()),
        1.0,
    );
    close_bf16(
        &what("windowed fold"),
        &gpu.read(&y),
        &windowed(&want_fold, m * h, live.clone()),
        1.0,
    );
    close_bf16(
        &what("windowed collapse"),
        &gpu.read(&head),
        &windowed(&want_head, h, live),
        1.0,
    );
}

fn gated_chain(gpu: &Gpu, s: Shape) {
    let ctx = gpu.ctx();
    let Shape { n, m, h } = s;
    let (rows, wide) = (n as u32, (m * h) as u32);
    let what = |op: &str| format!("{op} {s:?}");
    let wide_t = |v: &[f32]| gpu.tensor(v, rows, wide, Dtype::Bf16);
    let narrow = |v: &[f32]| gpu.tensor(v, rows, h as u32, Dtype::Bf16);

    // Expand, whole and windowed.
    let x_h = bf16s(n * h, 1, 2.0);
    let x = narrow(&x_h);
    let want: Vec<f32> = x_h.chunks(h).flat_map(|r| r.repeat(m)).collect();
    let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
    hc::expand(&ctx, x, m as u32, &mut y).unwrap();
    close_bf16(&what("hc_expand"), &gpu.read(&y), &want, 0.0);
    if n >= 4 {
        let win = gpu.ints(&[2, 2, 0, 0]);
        let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
        ctx.arm_stage(win.ptr);
        hc::expand(&ctx, x, m as u32, &mut y).unwrap();
        ctx.disarm_stage();
        close_bf16(
            &what("hc_expand windowed"),
            &gpu.read(&y),
            &windowed(&want, m * h, 2..4),
            0.0,
        );
    }

    // Mix: the mean over streams of normed·sigmoid(gate).
    let g_h = bf16s(n * m * h, 2, 3.0);
    let v_h = bf16s(n * m * h, 3, 2.0);
    let (g, v) = (wide_t(&g_h), wide_t(&v_h));
    let mut y = gpu.zeros(rows, h as u32, Dtype::Bf16);
    hc::mix(&ctx, g, v, m as u32, &mut y).unwrap();
    let want_mix: Vec<f32> = (0..n * h)
        .map(|t| {
            let (r, k) = (t / h, t % h);
            let at = |st: usize| r * m * h + st * h + k;
            round_bf16(
                (0..m)
                    .map(|st| v_h[at(st)] * sigmoid(g_h[at(st)]))
                    .sum::<f32>()
                    / m as f32,
            )
        })
        .collect();
    close_bf16(&what("hc_mix"), &gpu.read(&y), &want_mix, 1.0);

    // Inject: hyper_s += 2·sigmoid(gate_s/M)·o.
    let o_h = bf16s(n * h, 4, 1.0);
    let gate_h = bf16s(n * m, 5, 4.0);
    let hyper_h = bf16s(n * m * h, 6, 2.0);
    let mut hyper = wide_t(&hyper_h);
    let gates = gpu.tensor(&gate_h, rows, m as u32, Dtype::Bf16);
    let o = narrow(&o_h);
    hc::inject(&ctx, o, gates, m as u32, &mut hyper).unwrap();
    let want_inject: Vec<f32> = (0..n * m * h)
        .map(|t| {
            let (r, st, k) = (t / (m * h), t / h % m, t % h);
            let g = 2.0 * sigmoid(gate_h[r * m + st] / m as f32);
            round_bf16(hyper_h[t] + g * o_h[r * h + k])
        })
        .collect();
    close_bf16(&what("hc_inject"), &gpu.read(&hyper), &want_inject, 1.0);

    // Mix and inject windowed: a row outside the window keeps what it held.
    if n >= 3 {
        let live = 1..n - 1;
        let win = gpu.ints(&[(n - 2) as i32, 1, 0, 0]);
        let mut y = gpu.zeros(rows, h as u32, Dtype::Bf16);
        let mut hyper = wide_t(&hyper_h);
        ctx.arm_stage(win.ptr);
        hc::mix(&ctx, g, v, m as u32, &mut y).unwrap();
        hc::inject(&ctx, o, gates, m as u32, &mut hyper).unwrap();
        ctx.disarm_stage();
        close_bf16(
            &what("hc_mix windowed"),
            &gpu.read(&y),
            &windowed(&want_mix, h, live.clone()),
            1.0,
        );
        let want_w: Vec<f32> = want_inject
            .chunks(m * h)
            .zip(hyper_h.chunks(m * h))
            .enumerate()
            .flat_map(|(r, (new, old))| if live.contains(&r) { new } else { old }.to_vec())
            .collect();
        close_bf16(&what("hc_inject windowed"), &gpu.read(&hyper), &want_w, 1.0);
    }
}

#[test]
fn hc_mhc_chain() {
    let Some(gpu) = Gpu::open() else { return };
    for s in SMALL.into_iter().chain(MHC) {
        mhc_chain(&gpu, s);
    }
}

#[test]
fn hc_expand_mix_inject() {
    let Some(gpu) = Gpu::open() else { return };
    for s in SMALL.into_iter().chain(GATED) {
        gated_chain(&gpu, s);
    }
}

#[test]
fn hc_ple_gate() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let Shape { n, m, h } = SMALL[0];
    let (rows, wide) = (n as u32, (m * h) as u32);
    let wide_t = |v: &[f32]| gpu.tensor(v, rows, wide, Dtype::Bf16);

    // The per-layer-embedding gate, whole and windowed.
    let k_h = bf16s(n * m * h, 7, 1.0);
    let q_h = bf16s(n * m * h, 8, 1.0);
    let val_h = bf16s(n * h, 9, 2.0);
    let want: Vec<f32> = (0..n * m * h)
        .map(|t| {
            let (r, s, i) = (t / (m * h), t / h % m, t % h);
            let at = (r * m + s) * h;
            let dot = (0..h).map(|d| k_h[at + d] * q_h[at + d]).sum::<f32>() / (h as f32).sqrt();
            let damped = dot.abs().max(1e-6).sqrt().copysign(dot);
            round_bf16(sigmoid(damped) * val_h[r * h + i])
        })
        .collect();
    let (key, query, value): (Tensor, Tensor, Tensor) = (
        wide_t(&k_h),
        wide_t(&q_h),
        gpu.tensor(&val_h, rows, h as u32, Dtype::Bf16),
    );
    let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
    hc::ple_gate(&ctx, key, query, value, m as u32, &mut y).unwrap();
    close_bf16("ple_gate", &gpu.read(&y), &want, 1.0);
    let win = gpu.ints(&[3, 2, 0, 0]);
    let mut y = gpu.zeros(rows, wide, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    hc::ple_gate(&ctx, key, query, value, m as u32, &mut y).unwrap();
    ctx.disarm_stage();
    close_bf16(
        "ple_gate windowed",
        &gpu.read(&y),
        &windowed(&want, m * h, 2..5),
        1.0,
    );
}
