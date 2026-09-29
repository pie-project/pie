//! Pointwise activations, binaries, modulation, the timestep and bucket
//! tables, embed/scale/add, gates and clamps against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::elemwise::{act, clip, gate};

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn erf(x: f32) -> f32 {
    // Abramowitz–Stegun 7.1.26 in f64: 1.5e-7, far under bf16.
    let x = f64::from(x);
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    (x.signum() * y) as f32
}

#[test]
fn activations_and_binaries_answer_the_host() {
    let (rows, width) = (5u32, 72u32);
    let xs: Vec<f32> = data((rows * width) as usize, 1).iter().map(|v| round_bf16(v * 4.0)).collect();
    let ys = data((rows * width) as usize, 2);
    let mut b = Bench::new();
    let x = b.bf16(rows, width, &xs);
    let y = b.bf16(rows, width, &ys);
    let outs: Vec<_> = (0..6).map(|_| b.zeros(Dtype::Bf16, rows, width)).collect();
    let xf = b.f32(rows, width, &xs);
    let of = b.zeros(Dtype::F32, rows, width);
    if !b
        .run(|ctx| {
            act::silu(ctx, x, outs[0])?;
            act::tanh(ctx, x, outs[1])?;
            act::gelu_tanh(ctx, x, outs[2])?;
            act::gelu_erf(ctx, x, outs[3])?;
            act::add(ctx, x, y, outs[4])?;
            act::mul(ctx, x, y, outs[5])?;
            act::silu(ctx, xf, of)
        })
        .unwrap()
    {
        return;
    }
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    let fs: [Box<dyn Fn(f32, f32) -> f32>; 6] = [
        Box::new(|v, _| v * sigmoid(v)),
        Box::new(|v, _| v.tanh()),
        Box::new(move |v, _| 0.5 * v * (1.0 + (k * (v + 0.044715 * v * v * v)).tanh())),
        Box::new(|v, _| 0.5 * v * (1.0 + erf(v / std::f32::consts::SQRT_2))),
        Box::new(|v, w| v + w),
        Box::new(|v, w| v * w),
    ];
    for (o, f) in outs.iter().zip(&fs) {
        let want: Vec<f32> = xs.iter().zip(&ys).map(|(&v, &w)| round_bf16(f(v, w))).collect();
        assert_close(&b.read_f32(*o), &want, 1e-2, 1e-2);
    }
    let want: Vec<f32> = xs.iter().map(|&v| v * sigmoid(v)).collect();
    assert_close(&b.read_f32(of), &want, 1e-5, 1e-5);
}

fn norm_rows(x: &[f32], width: usize, layer: bool, head: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    if layer {
        for (r, c) in x.chunks(width).enumerate() {
            let mean = c.iter().sum::<f32>() / width as f32;
            let var = c.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (i, v) in c.iter().enumerate() {
                out[r * width + i] = round_bf16((v - mean) * inv);
            }
        }
    } else {
        for (r, c) in x.chunks(head).enumerate() {
            let inv = 1.0 / (c.iter().map(|v| v * v).sum::<f32>() / head as f32 + eps).sqrt();
            for (i, v) in c.iter().enumerate() {
                out[r * head + i] = round_bf16(v * inv);
            }
        }
    }
    out
}

#[test]
fn modulation_forms_lane_maps_and_fused_norms_answer_the_host() {
    let (rows, width, lanes) = (6usize, 64usize, 2usize);
    let (r, w) = (rows as u32, width as u32);
    let xs = data(rows * width, 3);
    let ms = data(rows * 2 * width, 4);
    let lm = data(lanes * 2 * width, 5);
    let map = [0i32, 0, 1, 1, 1, 0];
    let ys = data(rows * width, 6);
    let mut b = Bench::new();
    let x = b.bf16(r, w, &xs);
    let m = b.bf16(r, 2 * w, &ms);
    let mf = b.f32(lanes as u32, 2 * w, &lm);
    let lane = b.i32(r, 1, &map);
    let o_ss = b.zeros(Dtype::Bf16, r, w);
    let o_sc = b.zeros(Dtype::Bf16, r, w);
    let o_tg = b.zeros(Dtype::Bf16, r, w);
    let g1 = b.f32(lanes as u32, w, &lm[..lanes * width]);
    let y = b.bf16(r, w, &ys);
    let rr = b.bf16(r, w, &xs);
    let normed = b.zeros(Dtype::Bf16, r, w);
    let out = b.zeros(Dtype::Bf16, r, w);
    let r2 = b.bf16(r, w, &xs);
    let normed2 = b.zeros(Dtype::Bf16, r, w);
    let out2 = b.zeros(Dtype::Bf16, r, w);
    if !b
        .run(|ctx| {
            act::modulate(ctx, act::Form::ScaleShift, x, m, None, o_ss)?;
            act::modulate(ctx, act::Form::Scale, x, g1, Some(lane), o_sc)?;
            act::modulate(ctx, act::Form::TanhGate, x, g1, Some(lane), o_tg)?;
            act::gated_residual_add(ctx, rr, g1, y, Some(lane), rr)?;
            act::norm_modulate(
                ctx,
                x,
                act::NormKind::Rmsnorm { head_dim: 32, eps: 1e-6 },
                normed,
                mf,
                Some(lane),
                act::Form::ScaleShift,
                out,
            )?;
            act::gated_residual_norm_modulate(
                ctx,
                r2,
                g1,
                y,
                Some(lane),
                r2,
                act::NormKind::Layernorm { eps: 1e-6 },
                normed2,
                mf,
                act::Form::ScaleShift,
                out2,
            )
        })
        .unwrap()
    {
        return;
    }
    let lane_of = |n: usize| map[n] as usize;
    let mut w_ss = vec![0f32; rows * width];
    let mut w_sc = w_ss.clone();
    let mut w_tg = w_ss.clone();
    let mut w_r = w_ss.clone();
    for n in 0..rows {
        let l = lane_of(n);
        for i in 0..width {
            let v = xs[n * width + i];
            let (s, t) = (ms[n * 2 * width + i], ms[n * 2 * width + width + i]);
            w_ss[n * width + i] = round_bf16(v * (1.0 + s) + t);
            w_sc[n * width + i] = round_bf16(v * (1.0 + lm[l * width + i]));
            w_tg[n * width + i] = round_bf16(lm[l * width + i].tanh() * v);
            w_r[n * width + i] = round_bf16(lm[l * width + i] * ys[n * width + i] + v);
        }
    }
    assert_close(&b.read_f32(o_ss), &w_ss, 1e-2, 1e-2);
    assert_close(&b.read_f32(o_sc), &w_sc, 1e-2, 1e-2);
    assert_close(&b.read_f32(o_tg), &w_tg, 1e-2, 1e-2);
    assert_close(&b.read_f32(rr), &w_r, 1e-2, 1e-2);
    let bend = |nv: &[f32]| -> Vec<f32> {
        (0..rows * width)
            .map(|at| {
                let (n, i) = (at / width, at % width);
                let l = lane_of(n);
                round_bf16(nv[at] * (1.0 + lm[l * 2 * width + i]) + lm[l * 2 * width + width + i])
            })
            .collect()
    };
    let n1 = norm_rows(&xs, width, false, 32, 1e-6);
    assert_close(&b.read_f32(normed), &n1, 1e-2, 1e-2);
    assert_close(&b.read_f32(out), &bend(&b.read_f32(normed)), 1e-2, 1e-2);
    assert_close(&b.read_f32(r2), &w_r, 1e-2, 1e-2);
    let n2 = norm_rows(&b.read_f32(r2), width, true, 0, 1e-6);
    assert_close(&b.read_f32(normed2), &n2, 1e-2, 1e-2);
    assert_close(&b.read_f32(out2), &bend(&b.read_f32(normed2)), 1e-2, 1e-2);
}

fn bucket(d: i32, bi: bool, nb: i32, log_ratio: f32) -> i32 {
    let (mut base, mut nb) = (0, nb);
    let n = if bi {
        nb /= 2;
        if d > 0 {
            base += nb;
        }
        d.abs()
    } else {
        (-d).max(0)
    };
    let exact = nb / 2;
    if n < exact {
        return base + n;
    }
    let large = exact + ((n as f32 / exact as f32).ln() / log_ratio * (nb - exact) as f32) as i32;
    base + large.min(nb - 1)
}

#[test]
fn sinusoid_and_bucket_tables_answer_the_host() {
    let (rows, dim) = (4usize, 33usize);
    let ts = [0.0f32, 0.25, 7.5, 999.0];
    let (heads, max_len, nb) = (3usize, 40u32, 32u32);
    let emb = data(nb as usize * 4, 7);
    let mut b = Bench::new();
    let t = b.f32(rows as u32, 1, &ts);
    let y = b.zeros(Dtype::F32, rows as u32, dim as u32);
    let yf = b.zeros(Dtype::F32, rows as u32, 32);
    let e = b.bf16(nb, 4, &emb);
    let span = 2 * max_len - 1;
    let tb = b.zeros(Dtype::F32, heads as u32, span);
    let tu = b.zeros(Dtype::F32, heads as u32, span);
    if !b
        .run(|ctx| {
            act::sinusoid(ctx, t, dim as u32, 10000.0, false, 1000.0 / 1000.0, y)?;
            act::sinusoid(ctx, t, 32, 10000.0, true, 2.0, yf)?;
            act::relative_bucket_bias(ctx, e, max_len, nb, 128.0, true, tb)?;
            act::relative_bucket_bias(ctx, e, max_len, nb, 128.0, false, tu)
        })
        .unwrap()
    {
        return;
    }
    let table = |dim: usize, flip: bool, scale: f32| -> Vec<f32> {
        let half = dim / 2;
        let mut out = vec![0f32; rows * dim];
        for n in 0..rows {
            for i in 0..half {
                let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
                let a = scale * (ts[n] * f);
                let (s, c) = a.sin_cos();
                out[n * dim + i] = if flip { c } else { s };
                out[n * dim + i + half] = if flip { s } else { c };
            }
        }
        out
    };
    assert_close(&b.read_f32(y), &table(dim, false, 1.0), 2e-3, 1e-3);
    assert_close(&b.read_f32(yf), &table(32, true, 2.0), 2e-3, 1e-3);
    for (bi, got) in [(true, tb), (false, tu)] {
        let nbd = if bi { nb / 2 } else { nb };
        let lr = (128.0f64 / f64::from(nbd / 2)).ln() as f32;
        let mut want = vec![0f32; heads * span as usize];
        for h in 0..heads {
            for c in 0..span as usize {
                let d = c as i32 - (max_len as i32 - 1);
                want[h * span as usize + c] = emb[bucket(d, bi, nb as i32, lr) as usize * 4 + h];
            }
        }
        assert_close(&b.read_f32(got), &want, 0.0, 0.0);
    }
}

#[test]
fn embed_scale_add_and_its_select_form_answer_the_host() {
    let (rows, hidden, vocab, layers) = (4usize, 48usize, 10u32, 3usize);
    let table = data(vocab as usize * hidden, 8);
    let ids = [3i32, -1, 9, 12];
    let ys = data(rows * hidden, 9);
    let stacked = data(rows * layers * hidden, 10);
    let (a, bs) = (1.7f32, 0.3f32);
    let (r, h) = (rows as u32, hidden as u32);
    let mut b = Bench::new();
    let it = b.i32(r, 1, &ids);
    let tt = b.bf16(vocab, h, &table);
    let e = b.zeros(Dtype::Bf16, r, h);
    let es = b.zeros(Dtype::Bf16, r, h);
    let y = b.bf16(r, h, &ys);
    let yss = b.zeros(Dtype::Bf16, r, h);
    let e2 = b.zeros(Dtype::Bf16, r, h);
    let es2 = b.zeros(Dtype::Bf16, r, h);
    let st = b.bf16(r, (layers * hidden) as u32, &stacked);
    let yo = b.zeros(Dtype::Bf16, r, h);
    let ys2 = b.zeros(Dtype::Bf16, r, h);
    if !b
        .run(|ctx| {
            act::embed_scale_add(ctx, it, tt, vocab, e, a, es, y, bs, yss)?;
            act::embed_scale_add_select(ctx, it, tt, vocab, e2, a, es2, st, 1, h, yo, bs, ys2)
        })
        .unwrap()
    {
        return;
    }
    let (ar, br) = (round_bf16(a), round_bf16(bs));
    let mut we = vec![0f32; rows * hidden];
    let mut wes = we.clone();
    let mut wy = we.clone();
    let mut wys = we.clone();
    let mut wyo = we.clone();
    let mut wys2 = we.clone();
    for n in 0..rows {
        let id = if ids[n] >= 0 && (ids[n] as u32) < vocab { ids[n] as usize } else { 0 };
        for k in 0..hidden {
            let at = n * hidden + k;
            let ev = table[id * hidden + k];
            let esv = round_bf16(ev * ar);
            let yv = round_bf16(ys[at] + esv);
            let yo_v = round_bf16(stacked[n * layers * hidden + hidden + k] + esv);
            we[at] = ev;
            wes[at] = esv;
            wy[at] = yv;
            wys[at] = round_bf16(yv * br);
            wyo[at] = yo_v;
            wys2[at] = round_bf16(yo_v * br);
        }
    }
    assert_close(&b.read_f32(e), &we, 0.0, 0.0);
    assert_close(&b.read_f32(es), &wes, 0.0, 0.0);
    assert_close(&b.read_f32(y), &wy, 0.0, 0.0);
    assert_close(&b.read_f32(yss), &wys, 0.0, 0.0);
    assert_close(&b.read_f32(e2), &we, 0.0, 0.0);
    assert_close(&b.read_f32(es2), &wes, 0.0, 0.0);
    assert_close(&b.read_f32(yo), &wyo, 0.0, 0.0);
    assert_close(&b.read_f32(ys2), &wys2, 0.0, 0.0);
}

#[test]
fn gates_and_clamps_answer_the_host() {
    let (rows, heads, hd) = (5usize, 3usize, 24usize);
    let width = heads * hd;
    let (r, w) = (rows as u32, width as u32);
    let xs: Vec<f32> = data(rows * width, 11).iter().map(|v| round_bf16(v * 3.0)).collect();
    let gs: Vec<f32> = data(rows * width, 12).iter().map(|v| round_bf16(v * 5.0)).collect();
    let gh: Vec<f32> = data((rows + 2) * heads, 13).iter().map(|v| round_bf16(v * 5.0)).collect();
    let mut b = Bench::new();
    let x1 = b.bf16(r, w, &xs);
    let g = b.bf16(r, w, &gs);
    let x2 = b.bf16(r, w, &xs);
    let ghd = b.bf16((rows + 2) as u32, heads as u32, &gh);
    let x3 = b.bf16(r, w, &xs);
    let x4 = b.bf16(r, w, &xs);
    let lo = b.bf16(1, 1, &[-0.5]);
    let hi = b.bf16(1, 1, &[1.25]);
    if !b
        .run(|ctx| {
            gate::sigmoid_mul(ctx, g, x1)?;
            gate::sigmoid_mul_heads(ctx, ghd, hd as u32, 0.5, x2)?;
            clip::clamp(ctx, -1.3, 0.7001, x3)?;
            clip::clamp_learned(ctx, lo, hi, x4)
        })
        .unwrap()
    {
        return;
    }
    let w1: Vec<f32> = xs.iter().zip(&gs).map(|(v, g)| round_bf16(v * sigmoid(*g))).collect();
    let w2: Vec<f32> = xs
        .iter()
        .enumerate()
        .map(|(at, v)| {
            let (n, i) = (at / width, at % width);
            round_bf16(v * (0.5 * sigmoid(gh[n * heads + i / hd])))
        })
        .collect();
    let (l, h) = (round_bf16(-1.3), round_bf16(0.7001));
    let w3: Vec<f32> = xs.iter().map(|v| v.max(l).min(h)).collect();
    let w4: Vec<f32> = xs.iter().map(|v| v.max(-0.5).min(1.25)).collect();
    assert_close(&b.read_f32(x1), &w1, 1e-2, 1e-2);
    assert_close(&b.read_f32(x2), &w2, 1e-2, 1e-2);
    assert_close(&b.read_f32(x3), &w3, 0.0, 0.0);
    assert_close(&b.read_f32(x4), &w4, 0.0, 0.0);
}
