//! The grouped, gated and fused row norms, layer norm, the residual blend
//! and the per-column standardize against a host reference: f32
//! accumulation, one bf16 rounding per stored step.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_abs, close_bf16, round_bf16, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::elemwise::{layernorm, norm};

fn bf16s(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    values(n, seed, scale).into_iter().map(round_bf16).collect()
}

fn inv_rms(x: &[f32], eps: f32) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The rows of `got` outside `live` are untouched zeros.
fn untouched(what: &str, got: &[f32], width: usize, live: std::ops::Range<usize>) {
    for (row, have) in got.chunks(width).enumerate() {
        if !live.contains(&row) {
            assert!(
                have.iter().all(|v| *v == 0.0),
                "{what}: row {row} is outside the window and stays untouched"
            );
        }
    }
}

#[test]
fn rmsnorm_grouped_plus_one() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Qwen3.8-Next's grouped head norm (4 x 128), and a ragged group that
    // takes the element path.
    for (rows, group, groups) in [(5u32, 128u32, 4u32), (3, 100, 3)] {
        let width = group * groups;
        let xs = bf16s((rows * width) as usize, 1, 2.0);
        let ws = bf16s(width as usize, 2, 0.5);
        let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let w = gpu.tensor(&ws, 1, width, Dtype::Bf16);
        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        norm::rmsnorm_grouped_plus_one(&ctx, x, w, group, 1e-6, &mut y).unwrap();
        let g = group as usize;
        let want: Vec<f32> = xs
            .chunks(g)
            .enumerate()
            .flat_map(|(b, slice)| {
                let inv = inv_rms(slice, 1e-6);
                let wg = &ws[(b % groups as usize) * g..][..g];
                slice.iter().zip(wg).map(move |(v, w)| v * inv * (w + 1.0))
            })
            .collect();
        close_bf16("rmsnorm_grouped_plus_one", &gpu.read(&y), &want, 1.0);
    }

    // A window over rows 1..3 of a 4-row launch.
    let (rows, group, groups) = (4u32, 64u32, 2u32);
    let width = group * groups;
    let xs = bf16s((rows * width) as usize, 3, 1.0);
    let ws = bf16s(width as usize, 4, 0.5);
    let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let w = gpu.tensor(&ws, 1, width, Dtype::Bf16);
    let mut y = gpu.zeros(rows, width, Dtype::Bf16);
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    norm::rmsnorm_grouped_plus_one(&ctx, x, w, group, 1e-6, &mut y).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&y);
    let (g, wd) = (group as usize, width as usize);
    for row in 1..3 {
        for b in 0..groups as usize {
            let slice = &xs[row * wd + b * g..][..g];
            let inv = inv_rms(slice, 1e-6);
            let want: Vec<f32> = slice
                .iter()
                .zip(&ws[b * g..][..g])
                .map(|(v, w)| v * inv * (w + 1.0))
                .collect();
            close_bf16(
                "windowed grouped",
                &got[row * wd + b * g..][..g],
                &want,
                1.0,
            );
        }
    }
    untouched("windowed grouped", &got, wd, 1..3);
}

#[test]
fn rmsnorm_gated_by() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // KDA's 128-wide heads, a one-wave head and a ragged one.
    for (rows, heads, d) in [(3u32, 4u32, 128u32), (2, 3, 64), (2, 5, 100)] {
        let n = (rows * heads * d) as usize;
        let os = values(n, 5, 1.5);
        let gs = bf16s(n, 6, 3.0);
        let ws = values(d as usize, 7, 1.0);
        let o = gpu.tensor(&os, rows, heads * d, Dtype::F32);
        let g = gpu.tensor(&gs, rows, heads * d, Dtype::Bf16);
        let w = gpu.tensor(&ws, 1, d, Dtype::F32);
        let mut y = gpu.zeros(rows, heads * d, Dtype::Bf16);
        norm::rmsnorm_gated_by(&ctx, o, g, w, heads, 1e-6, &mut y).unwrap();
        let want: Vec<f32> = os
            .chunks(d as usize)
            .zip(gs.chunks(d as usize))
            .flat_map(|(head, gate)| {
                let inv = inv_rms(head, 1e-6);
                head.iter()
                    .zip(gate)
                    .zip(&ws)
                    .map(move |((v, g), w)| v * inv * w * sigmoid(*g))
                    .collect::<Vec<_>>()
            })
            .collect();
        close_bf16("rmsnorm_gated_by", &gpu.read(&y), &want, 1.0);
    }
}

struct Chain {
    t: Vec<f32>,
    y: Vec<f32>,
    scaled: Option<Vec<f32>>,
    out: Option<Vec<f32>>,
}

#[test]
fn rmsnorm_residual_add() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // Packed widths (one chunk, a ragged second chunk, a sub-wave row) and
    // element widths (8 and 16 per thread).
    for (rows, width) in [(4u32, 2048u32), (3, 5120), (2, 104), (3, 1000), (2, 3000)] {
        for (scale, post, plus_one) in [
            (false, false, false),
            (true, false, false),
            (false, true, true),
            (true, true, false),
        ] {
            let n = (rows * width) as usize;
            let wd = width as usize;
            let xs = bf16s(n, 11, 2.0);
            let ys = bf16s(n, 12, 2.0);
            let w0s = bf16s(wd, 13, 1.0);
            let w1s = bf16s(wd, 14, 0.5);
            let s = round_bf16(0.75);
            let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
            let w0 = gpu.tensor(&w0s, 1, width, Dtype::Bf16);
            let w1 = gpu.tensor(&w1s, 1, width, Dtype::Bf16);
            let sv = gpu.tensor(&[s], 1, 1, Dtype::Bf16);
            let mut t = gpu.zeros(rows, width, Dtype::Bf16);
            let mut y = gpu.tensor(&ys, rows, width, Dtype::Bf16);
            let mut scaled = gpu.zeros(rows, width, Dtype::Bf16);
            let mut out = gpu.zeros(rows, width, Dtype::Bf16);
            norm::rmsnorm_residual_add(
                &ctx,
                x,
                w0,
                1e-6,
                &mut t,
                &mut y,
                scale.then_some((sv, &mut scaled)),
                post.then_some(norm::PostNorm {
                    weight: w1,
                    plus_one,
                    eps: 1e-5,
                    out: &mut out,
                }),
            )
            .unwrap();
            let have = Chain {
                t: gpu.read(&t),
                y: gpu.read(&y),
                scaled: scale.then(|| gpu.read(&scaled)),
                out: post.then(|| gpu.read(&out)),
            };
            let what = format!("rmsnorm_residual_add {width} scale={scale} post={post}");

            // Each step against the host step over the device's previous
            // (rounded) step: the norms within an ulp, the add and the
            // scale exact.
            let want_t: Vec<f32> = xs
                .chunks(wd)
                .flat_map(|r| {
                    let inv = inv_rms(r, 1e-6);
                    r.iter().zip(&w0s).map(move |(v, w)| v * inv * w)
                })
                .collect();
            close_bf16(&format!("{what}: t"), &have.t, &want_t, 1.0);
            let want_y: Vec<f32> = ys
                .iter()
                .zip(&have.t)
                .map(|(a, b)| round_bf16(a + b))
                .collect();
            close_bf16(&format!("{what}: y"), &have.y, &want_y, 0.0);
            let last = match &have.scaled {
                Some(got) => {
                    let want: Vec<f32> = have.y.iter().map(|v| round_bf16(v * s)).collect();
                    close_bf16(&format!("{what}: scaled"), got, &want, 0.0);
                    got.clone()
                }
                None => have.y.clone(),
            };
            if let Some(got) = &have.out {
                let want: Vec<f32> = last
                    .chunks(wd)
                    .flat_map(|r| {
                        let inv = inv_rms(r, 1e-5);
                        r.iter()
                            .zip(&w1s)
                            .map(move |(v, w)| v * inv * (w + if plus_one { 1.0 } else { 0.0 }))
                    })
                    .collect();
                close_bf16(&format!("{what}: out"), got, &want, 1.0);
            }
        }
    }

    // A window over rows 2..4 of a 5-row launch, on both paths.
    for width in [512u32, 521] {
        let rows = 5u32;
        let (n, wd) = ((rows * width) as usize, width as usize);
        let xs = bf16s(n, 15, 1.0);
        let w0s = bf16s(wd, 16, 1.0);
        let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let w0 = gpu.tensor(&w0s, 1, width, Dtype::Bf16);
        let mut t = gpu.zeros(rows, width, Dtype::Bf16);
        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        let win = gpu.ints(&[2, 2, 0, 0]);
        ctx.arm_stage(win.ptr);
        norm::rmsnorm_residual_add(&ctx, x, w0, 1e-6, &mut t, &mut y, None, None).unwrap();
        ctx.disarm_stage();
        let got = gpu.read(&t);
        for row in 2..4 {
            let r = &xs[row * wd..][..wd];
            let inv = inv_rms(r, 1e-6);
            let want: Vec<f32> = r.iter().zip(&w0s).map(|(v, w)| v * inv * w).collect();
            close_bf16("windowed chain", &got[row * wd..][..wd], &want, 1.0);
        }
        untouched("windowed chain", &got, wd, 2..4);
        assert_eq!(gpu.read(&y), got, "the stream from zero is the normed row");
    }
}

#[test]
fn layernorm() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // SigLIP's 1152 (packed), an odd width (elements), an offset mean.
    for (rows, width) in [(4u32, 1152u32), (3, 1001)] {
        let (n, wd) = ((rows * width) as usize, width as usize);
        let xs: Vec<f32> = values(n, 21, 2.0)
            .into_iter()
            .map(|v| round_bf16(v + 3.0))
            .collect();
        let ws = bf16s(wd, 22, 1.0);
        let bs = bf16s(wd, 23, 0.5);
        let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let w = gpu.tensor(&ws, 1, width, Dtype::Bf16);
        let b = gpu.tensor(&bs, 1, width, Dtype::Bf16);
        let normed = |r: &[f32]| -> Vec<f32> {
            let mean = r.iter().sum::<f32>() / r.len() as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / r.len() as f32;
            let inv = 1.0 / (var + 1e-6).sqrt();
            r.iter().map(|v| (v - mean) * inv).collect()
        };
        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        layernorm::layernorm_no_scale(&ctx, x, 1e-6, &mut y).unwrap();
        let want: Vec<f32> = xs.chunks(wd).flat_map(normed).collect();
        close_bf16("layernorm_no_scale", &gpu.read(&y), &want, 1.0);

        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        layernorm::layernorm(&ctx, x, w, b, 1e-6, &mut y).unwrap();
        let want: Vec<f32> = xs
            .chunks(wd)
            .flat_map(|r| {
                normed(r)
                    .into_iter()
                    .zip(ws.iter().zip(&bs))
                    .map(|(c, (w, b))| c.mul_add(*w, *b))
                    .collect::<Vec<_>>()
            })
            .collect();
        // The affine output cancels (c·w + b), so its error is set by the
        // operands' scale: an ulp of |c·w| + |b|.
        close_abs("layernorm", &gpu.read(&y), &want, 3e-2);
    }

    // A window over rows 1..2 of a 3-row launch.
    let (rows, width) = (3u32, 256u32);
    let wd = width as usize;
    let xs = bf16s((rows * width) as usize, 24, 1.0);
    let x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let mut y = gpu.zeros(rows, width, Dtype::Bf16);
    let win = gpu.ints(&[1, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    layernorm::layernorm_no_scale(&ctx, x, 1e-6, &mut y).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&y);
    let r = &xs[wd..2 * wd];
    let mean = r.iter().sum::<f32>() / wd as f32;
    let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / wd as f32;
    let want: Vec<f32> = r.iter().map(|v| (v - mean) / (var + 1e-6).sqrt()).collect();
    close_bf16("windowed layernorm", &got[wd..2 * wd], &want, 1.0);
    untouched("windowed layernorm", &got, wd, 1..2);
}

#[test]
fn standardize_and_scalars() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, width) in [(3u32, 1024u32), (2, 1500), (2, 7)] {
        let (n, wd) = ((rows * width) as usize, width as usize);
        let xs = bf16s(n, 31, 2.0);
        let bs = bf16s(wd, 32, 1.0);
        let ss = bf16s(wd, 33, 2.0);
        let mut out = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        let b = gpu.tensor(&bs, 1, width, Dtype::Bf16);
        let s = gpu.tensor(&ss, 1, width, Dtype::Bf16);
        norm::standardize(&ctx, b, s, &mut out).unwrap();
        let want: Vec<f32> = xs
            .chunks(wd)
            .flat_map(|r| {
                r.iter()
                    .zip(bs.iter().zip(&ss))
                    .map(|(x, (b, s))| (x - b) * s)
            })
            .collect();
        close_bf16("standardize", &gpu.read(&out), &want, 0.5);

        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        norm::silu_scaled(&ctx, 1.7, &mut x).unwrap();
        let want: Vec<f32> = xs
            .iter()
            .map(|v| {
                let v = v * 1.7;
                v * sigmoid(v)
            })
            .collect();
        close_bf16("silu_scaled", &gpu.read(&x), &want, 1.0);

        let f = round_bf16(-0.3);
        let fs = gpu.tensor(&[f], 1, 1, Dtype::Bf16);
        let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
        norm::scale(&ctx, fs, &mut x).unwrap();
        let want: Vec<f32> = xs.iter().map(|v| v * f).collect();
        close_bf16("scale", &gpu.read(&x), &want, 0.5);
    }

    // silu_scaled under a window: rows 1..3 of 4.
    let (rows, width) = (4u32, 333u32);
    let wd = width as usize;
    let xs = bf16s((rows * width) as usize, 34, 2.0);
    let mut x = gpu.tensor(&xs, rows, width, Dtype::Bf16);
    let win = gpu.ints(&[2, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    norm::silu_scaled(&ctx, 0.5, &mut x).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&x);
    for (i, (h, v)) in got.iter().zip(&xs).enumerate() {
        if (wd..3 * wd).contains(&i) {
            let u = v * 0.5;
            close_bf16("windowed silu_scaled", &[*h], &[u * sigmoid(u)], 1.0);
        } else {
            assert_eq!(h, v, "element {i} is outside the window and stays");
        }
    }
}

/// The host blend in f64: softmax over the candidates' scores, then their
/// weighted sum.
fn blend(cands: &[&[f32]], nw: &[f32], pw: &[f32], eps: f32) -> Vec<f32> {
    let h = nw.len();
    let scores: Vec<f64> = cands
        .iter()
        .map(|v| {
            let ss: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            let inv = 1.0 / (ss / h as f64 + f64::from(eps)).sqrt();
            v.iter()
                .zip(nw.iter().zip(pw))
                .map(|(x, (a, b))| f64::from(*x) * inv * f64::from(*a) * f64::from(*b))
                .sum()
        })
        .collect();
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
    let sum: f64 = e.iter().sum();
    (0..h)
        .map(|i| {
            cands
                .iter()
                .zip(&e)
                .map(|(v, p)| p / sum * f64::from(v[i]))
                .sum::<f64>() as f32
        })
        .collect()
}

#[test]
fn res_blend() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, hidden, nblocks) in [(3u32, 2048u32, 4usize), (2, 1000, 1), (2, 512, 0)] {
        let (n, hd) = ((rows * hidden) as usize, hidden as usize);
        let prefix_v = bf16s(n, 41, 2.0);
        let nw = bf16s(hd, 42, 1.0);
        let pw = bf16s(hd, 43, 1.0);
        let block_vs: Vec<Vec<f32>> = (0..nblocks).map(|j| bf16s(n, 44 + j as u64, 2.0)).collect();
        let prefix = gpu.tensor(&prefix_v, rows, hidden, Dtype::Bf16);
        let norm_w = gpu.tensor(&nw, 1, hidden, Dtype::Bf16);
        let proj = gpu.tensor(&pw, 1, hidden, Dtype::Bf16);
        let want: Vec<f32> = (0..rows as usize)
            .flat_map(|r| {
                let mut cands: Vec<&[f32]> = block_vs.iter().map(|b| &b[r * hd..][..hd]).collect();
                cands.push(&prefix_v[r * hd..][..hd]);
                blend(&cands, &nw, &pw, 1e-6)
            })
            .collect();

        // Scattered blocks (staged into one stack) ...
        let scattered: Vec<Tensor> = block_vs
            .iter()
            .map(|b| gpu.tensor(b, rows, hidden, Dtype::Bf16))
            .collect();
        let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
        norm::res_blend(&ctx, prefix, &scattered, norm_w, 1e-6, proj, &mut y).unwrap();
        close_abs("res_blend scattered", &gpu.read(&y), &want, 2e-2);

        // ... and blocks already stacked side by side.
        let flat: Vec<f32> = block_vs.concat();
        if nblocks > 0 {
            let stack = gpu.tensor(&flat, rows * nblocks as u32, hidden, Dtype::Bf16);
            let plane = u64::from(rows * hidden) * 2;
            let stacked: Vec<Tensor> = (0..nblocks as u64)
                .map(|j| Tensor::new(stack.ptr + j * plane, rows, hidden, Dtype::Bf16))
                .collect();
            let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
            norm::res_blend(&ctx, prefix, &stacked, norm_w, 1e-6, proj, &mut y).unwrap();
            close_abs("res_blend stacked", &gpu.read(&y), &want, 2e-2);
        }
    }

    // A window over row 1 of a 3-row launch, two stacked blocks.
    let (rows, hidden) = (3u32, 256u32);
    let (n, hd) = ((rows * hidden) as usize, hidden as usize);
    let prefix_v = bf16s(n, 51, 2.0);
    let flat = bf16s(2 * n, 52, 2.0);
    let nw = bf16s(hd, 53, 1.0);
    let pw = bf16s(hd, 54, 1.0);
    let prefix = gpu.tensor(&prefix_v, rows, hidden, Dtype::Bf16);
    let stack = gpu.tensor(&flat, 2 * rows, hidden, Dtype::Bf16);
    let plane = u64::from(rows * hidden) * 2;
    let blocks = [
        Tensor::new(stack.ptr, rows, hidden, Dtype::Bf16),
        Tensor::new(stack.ptr + plane, rows, hidden, Dtype::Bf16),
    ];
    let norm_w = gpu.tensor(&nw, 1, hidden, Dtype::Bf16);
    let proj = gpu.tensor(&pw, 1, hidden, Dtype::Bf16);
    let mut y = gpu.zeros(rows, hidden, Dtype::Bf16);
    let win = gpu.ints(&[1, 1, 0, 0]);
    ctx.arm_stage(win.ptr);
    norm::res_blend(&ctx, prefix, &blocks, norm_w, 1e-6, proj, &mut y).unwrap();
    ctx.disarm_stage();
    let got = gpu.read(&y);
    let cands: [&[f32]; 3] = [
        &flat[hd..2 * hd],
        &flat[n + hd..n + 2 * hd],
        &prefix_v[hd..2 * hd],
    ];
    close_abs(
        "windowed res_blend",
        &got[hd..2 * hd],
        &blend(&cands, &nw, &pw, 1e-6),
        2e-2,
    );
    untouched("windowed res_blend", &got, hd, 1..2);
}

/// Muse Spark's norm and residual gate (`norm::norm_gate`), both norms, on
/// rows of whole eights (the packed walk) and a ragged one, one row a token
/// and two (a routed plane).
#[test]
fn norm_gate() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (eps, temp) = (1e-6f32, 0.7f32);
    for (rows, n, fan, gated) in [
        (3u32, 4096u32, 1u32, true),
        (4, 8192, 2, false),
        (2, 100, 1, true),
    ] {
        let len = (rows * n) as usize;
        let xs = bf16s(len, 11, 2.0);
        let ss = bf16s(n as usize, 12, 1.0);
        let rs = bf16s(len, 13, 1.0);
        let gs = values(n as usize, 14, 2.0);
        let nw = bf16s(n as usize, 15, 0.5);
        let x = gpu.tensor(&xs, rows, n, Dtype::Bf16);
        let scale = gpu.tensor(&ss, 1, n, Dtype::Bf16);
        let res = gpu.tensor(&rs, rows, n, Dtype::Bf16);
        let gate = gpu.tensor(&gs, 1, n, Dtype::F32);
        let mut y = gpu.zeros(rows, n, Dtype::Bf16);
        let z = gpu.zeros(rows, n, Dtype::Bf16);
        let next = norm::Next {
            weight: gpu.tensor(&nw, 1, n, Dtype::Bf16),
            eps,
            plus_one: true,
            y: z,
        };
        let (res_t, gate_t) = if gated {
            (Some(res), Some(gate))
        } else {
            (None, None)
        };
        norm::norm_gate(
            &ctx,
            x,
            Some(scale),
            res_t,
            gate_t,
            eps,
            temp,
            fan,
            Some(next),
            &mut y,
        )
        .unwrap();
        let (mut want_y, mut want_z) = (Vec::new(), Vec::new());
        for r in 0..rows as usize {
            let row = &xs[r * n as usize..(r + 1) * n as usize];
            let u: Vec<f32> = row
                .iter()
                .zip(&ss)
                .map(|(v, s)| round_bf16(v * s))
                .collect();
            let inv = inv_rms(&u, eps);
            let out: Vec<f32> = u
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let o = v * inv;
                    round_bf16(if gated {
                        let w = gs[i] / temp;
                        let beta = sigmoid(w);
                        let alpha = (sigmoid(-w) * (1.0 + beta)).max(1e-3).sqrt();
                        alpha * rs[r * n as usize + i] + beta * round_bf16(o)
                    } else {
                        o
                    })
                })
                .collect();
            let inv2 = inv_rms(&out, eps);
            want_z.extend(
                out.iter()
                    .zip(&nw)
                    .map(|(o, w)| round_bf16(o * inv2 * (w + 1.0))),
            );
            want_y.extend(out);
        }
        close_bf16(&format!("norm_gate {n} y"), &gpu.read(&y), &want_y, 2.0);
        close_bf16(&format!("norm_gate {n} z"), &gpu.read(&z), &want_z, 2.0);
    }
}
