//! The hyper-connection kernels against a host reference: mHC's norm,
//! projection, Sinkhorn gates, fold and trunk collapse, the stream expand,
//! and the gated-residual mix, inject and per-layer-embedding gate. Rows are
//! M streams of H, H not a multiple of the block.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::elemwise::hc;

const M: usize = 4;
const H: usize = 320;
const N: usize = 5;

fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

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
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    streams: &[f32],
    eps: f32,
    alpha: f32,
    iters: usize,
) -> Mixed {
    let mix_hc = 2 * M + M * M;
    let (mut post, mut comb, mut x) = (Vec::new(), Vec::new(), Vec::new());
    for n in 0..N {
        let row = &mixes[n * mix_hc..(n + 1) * mix_hc];
        let pre: Vec<f32> = (0..M)
            .map(|i| sigmoid(row[i] * scale[0] + base[i]) + eps)
            .collect();
        post.extend((0..M).map(|i| sigmoid(row[M + i] * scale[1] + base[M + i]) * alpha));
        let mut c: Vec<f32> = (0..M * M)
            .map(|t| row[2 * M + t] * scale[2] + base[2 * M + t])
            .collect();
        for r in c.chunks_mut(M) {
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
            for j in 0..M {
                let sum = (0..M).map(|i| c[i * M + j]).sum::<f32>() + eps;
                for i in 0..M {
                    c[i * M + j] /= sum;
                }
            }
        };
        columns(&mut c);
        for _ in 1..iters {
            for r in c.chunks_mut(M) {
                let sum = r.iter().sum::<f32>() + eps;
                for v in r.iter_mut() {
                    *v /= sum;
                }
            }
            columns(&mut c);
        }
        comb.extend(&c);
        let s = &streams[n * M * H..(n + 1) * M * H];
        x.extend((0..H).map(|h| round_bf16((0..M).map(|i| pre[i] * s[i * H + h]).sum())));
    }
    Mixed { post, comb, x }
}

fn host_fold(x: &[f32], streams: &[f32], post: &[f32], comb: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; N * M * H];
    for n in 0..N {
        for j in 0..M {
            for h in 0..H {
                let mut acc = post[n * M + j] * x[n * H + h];
                for i in 0..M {
                    acc += comb[n * M * M + i * M + j] * streams[n * M * H + i * H + h];
                }
                out[n * M * H + j * H + h] = round_bf16(acc);
            }
        }
    }
    out
}

#[test]
fn hc_mhc_chain() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (eps, gate_eps, alpha, iters) = (1e-6f32, 1e-6f32, 2.0f32, 20u32);
    let mix_hc = 2 * M + M * M;
    let streams_h = bf16s(N * M * H, 1, 2.0);
    let streams = gpu.tensor(&streams_h, N as u32, (M * H) as u32, Dtype::Bf16);

    // The stream row normed to f32.
    let mut normed = gpu.zeros(N as u32, (M * H) as u32, Dtype::F32);
    hc::rmsnorm_f32(&ctx, streams, eps, &mut normed).unwrap();
    let want_normed: Vec<f32> = streams_h
        .chunks(M * H)
        .flat_map(|r| {
            let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f32>() / (M * H) as f32 + eps).sqrt();
            r.iter().map(move |v| v * inv).collect::<Vec<_>>()
        })
        .collect();
    let normed_h = gpu.read(&normed);
    close_f32("hc_rmsnorm_f32", &normed_h, &want_normed);

    // The projection to the mix row (and to a trunk collapse row of M).
    for rows_out in [mix_hc, M] {
        let fn_h = values(rows_out * M * H, 2, 0.05);
        let hc_fn = gpu.tensor(&fn_h, rows_out as u32, (M * H) as u32, Dtype::F32);
        let mut mixes = gpu.zeros(N as u32, rows_out as u32, Dtype::F32);
        hc::project(&ctx, normed, hc_fn, M as u32, &mut mixes).unwrap();
        let want: Vec<f32> = (0..N * rows_out)
            .map(|t| {
                let (n, o) = (t / rows_out, t % rows_out);
                (0..M * H)
                    .map(|d| f64::from(normed_h[n * M * H + d]) * f64::from(fn_h[o * M * H + d]))
                    .sum::<f64>() as f32
            })
            .collect();
        close_abs("hc_project", &gpu.read(&mixes), &want, 1e-4);
    }

    // The gates, from a mix row of the scale a real projection gives.
    let mixes_h = values(N * mix_hc, 3, 1.5);
    let mixes = gpu.tensor(&mixes_h, N as u32, mix_hc as u32, Dtype::F32);
    let scale_h = [0.9f32, 1.1, 1.3];
    let scale = gpu.tensor(&scale_h, 1, 3, Dtype::F32);
    let base_h = values(mix_hc, 4, 0.5);
    let base = gpu.tensor(&base_h, 1, mix_hc as u32, Dtype::F32);
    let mut x = gpu.zeros(N as u32, H as u32, Dtype::Bf16);
    let mut post = gpu.zeros(N as u32, M as u32, Dtype::F32);
    let mut comb = gpu.zeros(N as u32, (M * M) as u32, Dtype::F32);
    hc::gates(
        &ctx, mixes, streams, scale, base, M as u32, gate_eps, alpha, iters, &mut x, &mut post,
        &mut comb,
    )
    .unwrap();
    let want = host_gates(
        &mixes_h,
        &scale_h,
        &base_h,
        &streams_h,
        gate_eps,
        alpha,
        iters as usize,
    );
    close_f32("hc_gates post", &gpu.read(&post), &want.post);
    close_f32("hc_gates comb", &gpu.read(&comb), &want.comb);
    close_bf16("hc_gates x", &gpu.read(&x), &want.x, 1.0);

    // The fold, into a fresh plane and in place over the streams.
    let layer_out_h = bf16s(N * H, 5, 1.0);
    let layer_out = gpu.tensor(&layer_out_h, N as u32, H as u32, Dtype::Bf16);
    let want_fold = host_fold(&layer_out_h, &streams_h, &want.post, &want.comb);
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    hc::fold(&ctx, layer_out, streams, post, comb, &mut y).unwrap();
    close_bf16("hc_fold", &gpu.read(&y), &want_fold, 1.0);
    let mut in_place = gpu.tensor(&streams_h, N as u32, (M * H) as u32, Dtype::Bf16);
    hc::fold(&ctx, layer_out, in_place, post, comb, &mut in_place).unwrap();
    close_bf16("hc_fold in place", &gpu.read(&in_place), &want_fold, 1.0);

    // The trunk collapse.
    let collapse_h = values(N * M, 6, 1.5);
    let collapse_mix = gpu.tensor(&collapse_h, N as u32, M as u32, Dtype::F32);
    let mut head = gpu.zeros(N as u32, H as u32, Dtype::Bf16);
    hc::collapse(
        &ctx,
        collapse_mix,
        streams,
        scale,
        base,
        M as u32,
        gate_eps,
        &mut head,
    )
    .unwrap();
    let want_head: Vec<f32> = (0..N * H)
        .map(|t| {
            let (n, h) = (t / H, t % H);
            round_bf16(
                (0..M)
                    .map(|i| {
                        (sigmoid(collapse_h[n * M + i] * scale_h[0] + base_h[i]) + gate_eps)
                            * streams_h[n * M * H + i * H + h]
                    })
                    .sum(),
            )
        })
        .collect();
    close_bf16("hc_head_postprocess", &gpu.read(&head), &want_head, 1.0);

    // A window over the gates and the fold: rows 1..4 of the launch's 5.
    let win = gpu.ints(&[3, 1, 0, 0]);
    let mut x = gpu.zeros(N as u32, H as u32, Dtype::Bf16);
    let mut post = gpu.zeros(N as u32, M as u32, Dtype::F32);
    let mut comb = gpu.zeros(N as u32, (M * M) as u32, Dtype::F32);
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    hc::gates(
        &ctx, mixes, streams, scale, base, M as u32, gate_eps, alpha, iters, &mut x, &mut post,
        &mut comb,
    )
    .unwrap();
    hc::fold(&ctx, layer_out, streams, post, comb, &mut y).unwrap();
    ctx.disarm_stage();
    close_f32(
        "windowed post",
        &gpu.read(&post),
        &windowed(&want.post, M, 1..4),
    );
    close_bf16(
        "windowed x",
        &gpu.read(&x),
        &windowed(&want.x, H, 1..4),
        1.0,
    );
    close_bf16(
        "windowed fold",
        &gpu.read(&y),
        &windowed(&want_fold, M * H, 1..4),
        1.0,
    );
}

#[test]
fn hc_expand_mix_inject_ple() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let wide = |gpu: &Gpu, v: &[f32]| gpu.tensor(v, N as u32, (M * H) as u32, Dtype::Bf16);
    let narrow = |gpu: &Gpu, v: &[f32]| gpu.tensor(v, N as u32, H as u32, Dtype::Bf16);

    // Expand, whole and windowed.
    let x_h = bf16s(N * H, 1, 2.0);
    let x = narrow(&gpu, &x_h);
    let want: Vec<f32> = x_h.chunks(H).flat_map(|r| r.repeat(M)).collect();
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    hc::expand(&ctx, x, M as u32, &mut y).unwrap();
    close_bf16("hc_expand", &gpu.read(&y), &want, 0.0);
    let win = gpu.ints(&[2, 2, 0, 0]);
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    hc::expand(&ctx, x, M as u32, &mut y).unwrap();
    ctx.disarm_stage();
    close_bf16(
        "hc_expand windowed",
        &gpu.read(&y),
        &windowed(&want, M * H, 2..4),
        0.0,
    );

    // Mix: the mean over streams of normed·sigmoid(gate).
    let g_h = bf16s(N * M * H, 2, 3.0);
    let v_h = bf16s(N * M * H, 3, 2.0);
    let mut y = gpu.zeros(N as u32, H as u32, Dtype::Bf16);
    hc::mix(&ctx, wide(&gpu, &g_h), wide(&gpu, &v_h), M as u32, &mut y).unwrap();
    let want: Vec<f32> = (0..N * H)
        .map(|t| {
            let (n, h) = (t / H, t % H);
            let at = |s: usize| n * M * H + s * H + h;
            round_bf16(
                (0..M)
                    .map(|s| v_h[at(s)] * sigmoid(g_h[at(s)]))
                    .sum::<f32>()
                    / M as f32,
            )
        })
        .collect();
    close_bf16("hc_mix", &gpu.read(&y), &want, 1.0);

    // Inject: hyper_s += 2·sigmoid(gate_s/M)·o.
    let o_h = bf16s(N * H, 4, 1.0);
    let gate_h = bf16s(N * M, 5, 4.0);
    let hyper_h = bf16s(N * M * H, 6, 2.0);
    let mut hyper = wide(&gpu, &hyper_h);
    let gates = gpu.tensor(&gate_h, N as u32, M as u32, Dtype::Bf16);
    hc::inject(&ctx, narrow(&gpu, &o_h), gates, M as u32, &mut hyper).unwrap();
    let want: Vec<f32> = (0..N * M * H)
        .map(|t| {
            let (n, s, h) = (t / (M * H), t / H % M, t % H);
            let g = 2.0 * sigmoid(gate_h[n * M + s] / M as f32);
            round_bf16(hyper_h[t] + g * o_h[n * H + h])
        })
        .collect();
    close_bf16("hc_inject", &gpu.read(&hyper), &want, 1.0);

    // The per-layer-embedding gate, whole and windowed.
    let k_h = bf16s(N * M * H, 7, 1.0);
    let q_h = bf16s(N * M * H, 8, 1.0);
    let val_h = bf16s(N * H, 9, 2.0);
    let want: Vec<f32> = (0..N * M * H)
        .map(|t| {
            let (n, s, i) = (t / (M * H), t / H % M, t % H);
            let at = (n * M + s) * H;
            let dot = (0..H).map(|d| k_h[at + d] * q_h[at + d]).sum::<f32>() / (H as f32).sqrt();
            let damped = dot.abs().max(1e-6).sqrt().copysign(dot);
            round_bf16(sigmoid(damped) * val_h[n * H + i])
        })
        .collect();
    let (key, query, value): (Tensor, Tensor, Tensor) =
        (wide(&gpu, &k_h), wide(&gpu, &q_h), narrow(&gpu, &val_h));
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    hc::ple_gate(&ctx, key, query, value, M as u32, &mut y).unwrap();
    close_bf16("ple_gate", &gpu.read(&y), &want, 1.0);
    let win = gpu.ints(&[3, 2, 0, 0]);
    let mut y = gpu.zeros(N as u32, (M * H) as u32, Dtype::Bf16);
    ctx.arm_stage(win.ptr);
    hc::ple_gate(&ctx, key, query, value, M as u32, &mut y).unwrap();
    ctx.disarm_stage();
    close_bf16(
        "ple_gate windowed",
        &gpu.read(&y),
        &windowed(&want, M * H, 2..5),
        1.0,
    );
}
