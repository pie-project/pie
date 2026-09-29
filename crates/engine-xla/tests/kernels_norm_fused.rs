//! The norm entries beyond kernels_norm.rs: the Qwen3.5 path's gated and
//! plus-one norms, the scale-free layernorm, the fused residual chains and
//! the depth blend, against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::elemwise::norm;

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

fn rms(chunk: &[f32], eps: f32) -> f32 {
    1.0 / (chunk.iter().map(|v| v * v).sum::<f32>() / chunk.len() as f32 + eps).sqrt()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[test]
fn the_qwen_path_norms_answer_the_host() {
    // rmsnorm_plus_one over the row; rmsnorm_gated over 128-wide value
    // heads of an f32 accumulator, both gate forms; rmsnorm_gated_by.
    let (rows, width, vd) = (3usize, 384usize, 128usize);
    let xs = data(rows * width, 1);
    let ws = data(width, 2);
    let acc: Vec<f32> = data(rows * width, 3).iter().map(|v| v * 3.0 + 0.1).collect();
    let zs = data(rows * width, 4);
    let gw: Vec<f32> = data(vd, 5).iter().map(|v| v + 1.0).collect();
    let mut b = Bench::new();
    let (r, w) = (rows as u32, width as u32);
    let x = b.bf16(r, w, &xs);
    let wt = b.bf16(1, w, &ws);
    let y1 = b.zeros(Dtype::Bf16, r, w);
    let a = b.f32(r, w, &acc);
    let z = b.bf16(r, w, &zs);
    let g = b.f32(1, vd as u32, &gw);
    let y2 = b.zeros(Dtype::Bf16, r, w);
    let y3 = b.zeros(Dtype::Bf16, r, w);
    let y4 = b.zeros(Dtype::Bf16, r, w);
    if !b
        .run(|ctx| {
            norm::rmsnorm_plus_one(ctx, x, wt, 1e-6, y1)?;
            norm::rmsnorm_gated(ctx, a, z, g, vd as u32, 1e-6, false, y2)?;
            norm::rmsnorm_gated(ctx, a, z, g, vd as u32, 1e-6, true, y3)?;
            norm::rmsnorm_gated_by(ctx, a, z, g, (width / vd) as u32, 1e-6, y4)
        })
        .unwrap()
    {
        return;
    }
    let mut w1 = vec![0f32; xs.len()];
    for (row, c) in xs.chunks(width).enumerate() {
        let inv = rms(c, 1e-6);
        for (i, v) in c.iter().enumerate() {
            w1[row * width + i] = round_bf16((1.0 + ws[i]) * (v * inv));
        }
    }
    assert_close(&b.read_f32(y1), &w1, 1e-2, 1e-2);
    let gated = |sig: bool| -> Vec<f32> {
        let mut out = vec![0f32; acc.len()];
        for (run, c) in acc.chunks(vd).enumerate() {
            let inv = rms(c, 1e-6);
            for (i, v) in c.iter().enumerate() {
                let zv = zs[run * vd + i];
                let gate = if sig { sigmoid(zv) } else { zv * sigmoid(zv) };
                out[run * vd + i] = round_bf16(v * inv * gw[i] * gate);
            }
        }
        out
    };
    assert_close(&b.read_f32(y2), &gated(false), 1e-2, 1e-2);
    assert_close(&b.read_f32(y3), &gated(true), 1e-2, 1e-2);
    assert_close(&b.read_f32(y4), &gated(true), 1e-2, 1e-2);
}

#[test]
fn layernorm_without_scale_and_the_fused_residual_norms_answer_the_host() {
    let (rows, width) = (4usize, 200usize);
    let (r, w) = (rows as u32, width as u32);
    let xs = data(rows * width, 6);
    let ys = data(rows * width, 7);
    let ws = data(width, 8);
    let w1s = data(width, 9);
    let mut b = Bench::new();
    let x = b.bf16(r, w, &xs);
    let ln = b.zeros(Dtype::Bf16, r, w);
    // residual_add_rmsnorm
    let y = b.bf16(r, w, &ys);
    let wt = b.bf16(1, w, &ws);
    let out = b.zeros(Dtype::Bf16, r, w);
    // rmsnorm_residual_add with scale and post
    let y2 = b.bf16(r, w, &ys);
    let t = b.zeros(Dtype::Bf16, r, w);
    let s = b.bf16(1, 1, &[0.75]);
    let scaled = b.zeros(Dtype::Bf16, r, w);
    let w1 = b.bf16(1, w, &w1s);
    let post = b.zeros(Dtype::Bf16, r, w);
    // and the bare form
    let y3 = b.bf16(r, w, &ys);
    let t3 = b.zeros(Dtype::Bf16, r, w);
    if !b
        .run(|ctx| {
            norm::layernorm_no_scale(ctx, x, 1e-5, ln)?;
            norm::residual_add_rmsnorm(ctx, x, y, wt, true, 1e-6, out)?;
            norm::rmsnorm_residual_add(
                ctx,
                x,
                wt,
                1e-6,
                t,
                y2,
                Some((s, scaled)),
                Some(norm::PostNorm { weight: w1, plus_one: true, eps: 1e-6, out: post }),
            )?;
            norm::rmsnorm_residual_add(ctx, x, wt, 1e-6, t3, y3, None, None)
        })
        .unwrap()
    {
        return;
    }
    let mut want_ln = vec![0f32; xs.len()];
    let mut want_sum = vec![0f32; xs.len()];
    let mut want_out = vec![0f32; xs.len()];
    let mut want_t = vec![0f32; xs.len()];
    let mut want_y2 = vec![0f32; xs.len()];
    let mut want_scaled = vec![0f32; xs.len()];
    let mut want_post = vec![0f32; xs.len()];
    for row in 0..rows {
        let at = row * width;
        let xr = &xs[at..at + width];
        let yr = &ys[at..at + width];
        let mean = xr.iter().sum::<f32>() / width as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        let sum: Vec<f32> = (0..width).map(|i| round_bf16(yr[i] + xr[i])).collect();
        let is = rms(&sum, 1e-6);
        let ix = rms(xr, 1e-6);
        let tv: Vec<f32> = (0..width).map(|i| round_bf16(xr[i] * ix * ws[i])).collect();
        let folded: Vec<f32> = (0..width).map(|i| round_bf16(yr[i] + tv[i])).collect();
        let sc: Vec<f32> = folded.iter().map(|v| round_bf16(v * 0.75)).collect();
        let ip = rms(&sc, 1e-6);
        for i in 0..width {
            want_ln[at + i] = round_bf16((xr[i] - mean) * inv);
            want_sum[at + i] = sum[i];
            want_out[at + i] = round_bf16(sum[i] * is * (1.0 + ws[i]));
            want_t[at + i] = tv[i];
            want_y2[at + i] = folded[i];
            want_scaled[at + i] = sc[i];
            want_post[at + i] = round_bf16(sc[i] * ip * (1.0 + w1s[i]));
        }
    }
    assert_close(&b.read_f32(ln), &want_ln, 1e-2, 1e-2);
    assert_close(&b.read_f32(y), &want_sum, 0.0, 0.0);
    assert_close(&b.read_f32(out), &want_out, 1e-2, 1e-2);
    assert_close(&b.read_f32(t), &want_t, 1e-2, 1e-2);
    assert_close(&b.read_f32(y2), &want_y2, 1e-2, 1e-2);
    assert_close(&b.read_f32(scaled), &want_scaled, 1e-2, 1e-2);
    assert_close(&b.read_f32(post), &want_post, 2e-2, 1e-2);
    assert_close(&b.read_f32(t3), &want_t, 1e-2, 1e-2);
    assert_close(&b.read_f32(y3), &want_y2, 1e-2, 1e-2);
}

#[test]
fn res_blend_softmaxes_over_blocks_and_prefix() {
    let (rows, hidden, n) = (3usize, 96usize, 3usize);
    let (r, h) = (rows as u32, hidden as u32);
    let prefix = data(rows * hidden, 10);
    let blocks: Vec<Vec<f32>> = (0..n).map(|j| data(rows * hidden, 11 + j as u32)).collect();
    let nw = data(hidden, 20);
    let pw: Vec<f32> = data(hidden, 21).iter().map(|v| v * 4.0).map(round_bf16).collect();
    let mut b = Bench::new();
    let p = b.bf16(r, h, &prefix);
    let bs: Vec<_> = blocks.iter().map(|x| b.bf16(r, h, x)).collect();
    let nt = b.bf16(1, h, &nw);
    let pt = b.bf16(1, h, &pw);
    let y = b.zeros(Dtype::Bf16, r, h);
    if !b.run(|ctx| norm::res_blend(ctx, p, &bs, nt, 1e-6, pt, y)).unwrap() {
        return;
    }
    let mut want = vec![0f32; rows * hidden];
    for row in 0..rows {
        let at = row * hidden;
        let mut cands: Vec<&[f32]> = blocks.iter().map(|x| &x[at..at + hidden]).collect();
        cands.push(&prefix[at..at + hidden]);
        let logits: Vec<f32> = cands
            .iter()
            .map(|c| {
                let inv = rms(c, 1e-6);
                (0..hidden).map(|i| c[i] * inv * nw[i] * pw[i]).sum()
            })
            .collect();
        let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let s: f32 = e.iter().sum();
        for i in 0..hidden {
            want[at + i] = round_bf16(cands.iter().zip(&e).map(|(c, e)| e / s * c[i]).sum());
        }
    }
    assert_close(&b.read_f32(y), &want, 1e-2, 1e-2);
}

#[test]
fn the_remaining_norm_entries_answer_the_host() {
    let (rows, width, head, group) = (3usize, 192usize, 64usize, 48usize);
    let (r, w) = (rows as u32, width as u32);
    let xs = data(rows * width, 30);
    let ws = data(width, 31);
    let hs = data(head, 32);
    let bs = data(width, 33);
    let mut b = Bench::new();
    let x = b.bf16(r, w, &xs);
    let wt = b.bf16(1, w, &ws);
    let ht = b.bf16(1, head as u32, &hs);
    let bt = b.bf16(1, w, &bs);
    let per_head = b.zeros(Dtype::Bf16, r, w);
    let grouped = b.zeros(Dtype::Bf16, r, w);
    let no_scale = b.zeros(Dtype::Bf16, r, w);
    let ln = b.zeros(Dtype::Bf16, r, w);
    let biased = b.bf16(r, w, &xs);
    let std = b.bf16(r, w, &xs);
    let ms = b.bf16(r, w, &xs);
    let ss = b.bf16(r, w, &xs);
    let sc = b.bf16(r, w, &xs);
    let s = b.bf16(1, 1, &[1.5]);
    if !b
        .run(|ctx| {
            norm::rmsnorm_per_head(ctx, x, ht, head as u32, 1e-6, per_head)?;
            norm::rmsnorm_grouped_plus_one(ctx, x, wt, group as u32, 1e-6, grouped)?;
            norm::rmsnorm_no_scale(ctx, x, head as u32, 1e-6, no_scale)?;
            norm::layernorm(ctx, x, wt, bt, 1e-5, ln)?;
            norm::add_bias(ctx, bt, biased)?;
            norm::standardize(ctx, bt, wt, std)?;
            norm::mul_scalar(ctx, 0.3, ms)?;
            norm::silu_scaled(ctx, 1.7, ss)?;
            norm::scale(ctx, s, sc)
        })
        .unwrap()
    {
        return;
    }
    let runs = |axis: usize, g: &dyn Fn(usize) -> f32| -> Vec<f32> {
        let mut out = vec![0f32; xs.len()];
        for (run, c) in xs.chunks(axis).enumerate() {
            let inv = rms(c, 1e-6);
            for (i, v) in c.iter().enumerate() {
                let at = run * axis + i;
                out[at] = round_bf16(g(at) * (v * inv));
            }
        }
        out
    };
    assert_close(&b.read_f32(per_head), &runs(head, &|at| hs[at % head]), 1e-2, 1e-2);
    assert_close(&b.read_f32(grouped), &runs(group, &|at| 1.0 + ws[at % width]), 1e-2, 1e-2);
    assert_close(&b.read_f32(no_scale), &runs(head, &|_| 1.0), 1e-2, 1e-2);
    let mut want_ln = vec![0f32; xs.len()];
    for (row, c) in xs.chunks(width).enumerate() {
        let mean = c.iter().sum::<f32>() / width as f32;
        let var = c.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for (i, v) in c.iter().enumerate() {
            want_ln[row * width + i] = round_bf16((v - mean) * inv * ws[i] + bs[i]);
        }
    }
    assert_close(&b.read_f32(ln), &want_ln, 1e-2, 1e-2);
    let each = |f: &dyn Fn(usize, f32) -> f32| -> Vec<f32> {
        xs.iter().enumerate().map(|(at, &v)| round_bf16(f(at % width, v))).collect()
    };
    assert_close(&b.read_f32(biased), &each(&|c, v| v + bs[c]), 0.0, 0.0);
    assert_close(&b.read_f32(std), &each(&|c, v| (v - bs[c]) * ws[c]), 0.0, 0.0);
    assert_close(&b.read_f32(ms), &each(&|_, v| v * round_bf16(0.3)), 0.0, 0.0);
    assert_close(&b.read_f32(ss), &each(&|_, v| (v * 1.7) * sigmoid(v * 1.7)), 1e-2, 1e-2);
    assert_close(&b.read_f32(sc), &each(&|_, v| v * 1.5), 0.0, 0.0);
}
