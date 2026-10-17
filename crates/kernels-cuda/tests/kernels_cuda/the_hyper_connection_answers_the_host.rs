#![cfg(feature = "cuda")]

//! DeepSeek-V4's hyper-connection launchers (four 4096-wide streams)
//! against the host: the projection and its normed form on each walk, the
//! gates' Sinkhorn walk alone, with its norm and as the whole pre-mix, the
//! fold, the per-head norm and the stream norm.

use crate::common::attn_ref::{near, rmsnorm, ten, tuned, wide};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, F32};
use kernels_cuda::elemwise::{hc, norm};
use kernels_cuda::tuning::Tuning;
use std::ops::Range;

const M: usize = 4;
const H: usize = 4096;
const MIX: usize = 2 * M + M * M;

fn forms(tile: bool, mma: bool) -> Tuning {
    tuned(|t| (t.hc_tile, t.hc_mma) = (tile, mma))
}

/// A context under `tuning` with `(live, base)` staged where given; the
/// rows of `rows` past `base` it holds.
type Window = Option<(u32, u32)>;
fn ctx(
    gpu: &mut Gpu,
    tuning: Tuning,
    window: Window,
    rows: usize,
) -> (kernels_cuda::Ctx, Range<usize>) {
    let ctx = gpu.ctx().with_tuning(tuning);
    let (live, base) = window.unwrap_or((rows as u32, 0));
    if window.is_some() {
        ctx.arm_stage(gpu.up(&[live, base, 0, 0]));
    }
    (ctx, base as usize..(base + live) as usize)
}

/// `n` draws scaled by `k` as bf16: values and bits.
fn bf16s(lcg: &mut Lcg, n: usize, k: f32) -> (Vec<f32>, Vec<u16>) {
    let bits: Vec<u16> = (0..n).map(|_| to_bf16(lcg.unit() * k)).collect();
    (wide(&bits), bits)
}

/// Each row's mixes `x . w`.
fn dots(x: &[f32], w: &[f32], width: usize) -> Vec<f32> {
    let f = |a: &f32, b: &f32| f64::from(*a) * f64::from(*b);
    let dot = |r: &[f32], o: &[f32]| r.iter().zip(o).map(|(a, b)| f(a, b)).sum::<f64>() as f32;
    let rows = x.chunks(width);
    rows.flat_map(|r| w.chunks(width).map(move |o| dot(r, o)))
        .collect()
}

/// The mixes of `rows` stream rows (`prenorm`: the bf16 rows RMS-normed
/// first), on the rows a window stages and nowhere else.
fn project(rows: usize, mix: usize, window: Window, prenorm: bool, tuning: Tuning) {
    let (fan, planes) = (M * H, rows + window.map_or(0, |w| w.1 as usize));
    let mut lcg = Lcg::seeded(0x7113 + (rows * mix) as u64);
    let (x, bits) = bf16s(&mut lcg, planes * fan, 3.0);
    let w: Vec<f32> = (0..mix * fan).map(|_| lcg.unit() * 0.01).collect();
    let normed = if prenorm {
        rmsnorm(&x, fan, None, 1e-6)
    } else {
        x.clone()
    };
    let mut gpu = Gpu::open();
    let x_at = if prenorm { gpu.up(&bits) } else { gpu.up(&x) };
    let (w_at, m_at) = (gpu.up(&w), gpu.zeros(planes * mix * 4));
    let (ctx, live) = ctx(&mut gpu, tuning, window, rows);
    let (w_t, m) = (ten(w_at, mix, fan, F32), &mut ten(m_at, rows, mix, F32));
    let fired = match prenorm {
        true => hc::prenorm_project(&ctx, ten(x_at, rows, fan, Bf16), 1e-6, w_t, 4, m),
        false => hc::project(&ctx, ten(x_at, rows, fan, F32), w_t, 4, m).map(|()| true),
    };
    assert!(
        fired.expect("the projection fires"),
        "the tile takes the rows"
    );
    gpu.sync();
    let (got, mut want) = (gpu.down::<f32>(m_at, planes * mix), dots(&normed, &w, fan));
    for (i, v) in want.iter_mut().enumerate() {
        *v *= f32::from(u8::from(live.contains(&(i / mix))));
    }
    let what = format!("{rows} rows of {mix}, prenorm {prenorm}, {window:?}");
    near(&got, &want, 2e-3, 1.0, &what);
}

#[test]
fn the_projections_answer_the_dot() {
    let (split, tile, mma) = (forms(false, false), forms(true, false), forms(true, true));
    // The split walk, the tile past 128 rows, on a window; the normed form's tile and tensor cores.
    project(5, 4, None, false, split);
    project(40, 24, Some((29, 6)), false, tile);
    project(300, 24, None, false, tile);
    project(1, 24, None, true, tile);
    project(70, 24, Some((50, 3)), true, tile);
    project(300, 24, None, true, mma);
    project(200, 4, Some((150, 6)), true, mma);
}

/// One row's gates from its mixes: pre, post and the Sinkhorn combine.
fn gates_of(row: &[f32], base: &[f32]) -> [Vec<f32>; 3] {
    let (eps, s) = (1e-6f32, [0.7, 1.3, 0.9]);
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let pre = (0..M).map(|i| sig(row[i] * s[0] + base[i]) + eps).collect();
    let post = (0..M)
        .map(|i| sig(row[M + i] * s[1] + base[M + i]) * 2.0)
        .collect();
    let mut c: Vec<f32> = (0..M * M)
        .map(|i| row[2 * M + i] * s[2] + base[2 * M + i])
        .collect();
    for r in c.chunks_mut(M) {
        let mx = r.iter().copied().fold(f32::MIN, f32::max);
        let z: f32 = r.iter().map(|v| (v - mx).exp()).sum();
        r.iter_mut().for_each(|v| *v = (*v - mx).exp() / z + eps);
    }
    for it in 0..20 {
        for r in c.chunks_mut(M).filter(|_| it > 0) {
            let z = r.iter().sum::<f32>() + eps;
            r.iter_mut().for_each(|v| *v /= z);
        }
        for j in 0..M {
            let z = (0..M).map(|i| c[i * M + j]).sum::<f32>() + eps;
            (0..M).for_each(|i| c[i * M + j] /= z);
        }
    }
    [pre, post, c]
}

/// `hc::gates` (the pre-mixed row itself), `hc::gates_norm` (that row
/// RMS-normed) or `hc::pre` (the mixes projected first, then both).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gates {
    Alone,
    Normed,
    Pre,
}

fn gates(form: Gates, rows: usize, window: Window, tuning: Tuning) {
    let planes = rows + window.map_or(0, |w| w.1 as usize);
    let mut lcg = Lcg::seeded(0x9a7e + rows as u64);
    let (s, s_bits) = bf16s(&mut lcg, planes * M * H, 2.0);
    let mut mixes: Vec<f32> = (0..planes * MIX).map(|_| lcg.unit() * 3.0).collect();
    let plane: Vec<f32> = (0..MIX * M * H).map(|_| lcg.unit() * 0.02).collect();
    let bias: Vec<f32> = (0..MIX).map(|_| lcg.unit()).collect();
    let w_bits: Vec<u16> = (0..H).map(|_| to_bf16(1.0 + lcg.unit() * 0.5)).collect();
    let mut gpu = Gpu::open();
    let (s_at, mix_at, plane_at) = (gpu.up(&s_bits), gpu.up(&mixes), gpu.up(&plane));
    let (sc_at, b_at, w_at) = (gpu.up(&[0.7f32, 1.3, 0.9]), gpu.up(&bias), gpu.up(&w_bits));
    let (y_at, post_at) = (gpu.zeros(planes * H * 2), gpu.zeros(planes * M * 4));
    let comb_at = gpu.zeros(planes * M * M * 4);
    let (ctx, live) = ctx(&mut gpu, tuning, window, rows);
    let (st, mut mt) = (ten(s_at, rows, M * H, Bf16), ten(mix_at, rows, MIX, F32));
    let (w_t, sc, b) = (
        ten(w_at, H, 1, Bf16),
        ten(sc_at, 1, 3, F32),
        ten(b_at, 1, MIX, F32),
    );
    let gate = hc::Gate {
        scale: sc,
        base: b,
        gate_eps: 1e-6,
        alpha: 2.0,
        sinkhorn: 20,
    };
    let (y, post) = (
        &mut ten(y_at, rows, H, Bf16),
        &mut ten(post_at, rows, M, F32),
    );
    let (comb, fn_t) = (
        &mut ten(comb_at, rows, M * M, F32),
        ten(plane_at, MIX, M * H, F32),
    );
    let fired = match form {
        Gates::Alone => {
            let fired = hc::gates(&ctx, mt, st, sc, b, 4, 1e-6, 2.0, 20, true, y, post, comb);
            fired.map(|()| true)
        }
        Gates::Normed => hc::gates_norm(&ctx, mt, st, 4, gate, w_t, 1e-6, post, comb, y),
        Gates::Pre => hc::pre(
            &ctx, st, 1e-6, fn_t, 4, &mut mt, gate, w_t, 1e-6, None, post, comb, y,
        ),
    };
    assert!(
        fired.expect("the gates fire"),
        "{form:?}: the kernel takes 4 streams of 4096"
    );
    gpu.sync();
    let got_y = wide(&gpu.down::<u16>(y_at, planes * H));
    let got_post: Vec<f32> = gpu.down(post_at, planes * M);
    let got_comb: Vec<f32> = gpu.down(comb_at, planes * M * M);
    let what = format!("{form:?}, {rows} rows, {window:?}");
    for r in live {
        let s = &s[r * M * H..][..M * H];
        if form == Gates::Pre {
            let normed = rmsnorm(s, M * H, None, 1e-6);
            mixes[r * MIX..][..MIX].copy_from_slice(&dots(&normed, &plane, M * H));
        }
        let [pre, p, c] = gates_of(&mixes[r * MIX..][..MIX], &bias);
        let mixed = |d: usize| (0..M).map(|i| pre[i] * s[i * H + d]).sum();
        let mut x: Vec<f32> = (0..H).map(mixed).collect();
        if form != Gates::Alone {
            x = rmsnorm(&x, H, Some(&wide(&w_bits)), 1e-6);
        }
        near(&got_post[r * M..][..M], &p, 1e-3, 1.0, &what);
        near(&got_comb[r * M * M..][..M * M], &c, 1e-3, 1.0, &what);
        near(&got_y[r * H..][..H], &x, 2e-2, 0.05, &what);
    }
}

#[test]
fn the_gates_answer_the_host_walk() {
    let (split, tile) = (forms(false, false), forms(true, false));
    // Each form at a decode's rows, a prefill's and on a window; the gates' pre-mix in both forms.
    gates(Gates::Alone, 7, None, split);
    gates(Gates::Alone, 24, Some((17, 5)), tile);
    gates(Gates::Normed, 19, Some((17, 5)), tile);
    gates(Gates::Pre, 1, None, tile);
    gates(Gates::Pre, 300, None, tile);
    gates(Gates::Pre, 40, Some((29, 6)), tile);
}

/// The fold (`post_j x + sum_i comb_ij s_i`) and the scale-free per-head
/// norm on each form (the tile's 16-byte runs, the tensor cores' block a
/// row) and on a window; the stream norm in f32.
#[test]
fn the_fold_and_the_norms_answer_the_host() {
    let windows = [(37, None), (24, Some((17, 5))), (300, Some((170, 9)))];
    for ((rows, window), tuning) in
        windows
            .into_iter()
            .zip([forms(false, false), forms(true, false), forms(true, true)])
    {
        let planes = rows + window.map_or(0, |w: (u32, u32)| w.1 as usize);
        let mut lcg = Lcg::seeded(0xf01d);
        let (x, x_bits) = bf16s(&mut lcg, planes * H, 1.0);
        let (s, s_bits) = bf16s(&mut lcg, planes * M * H, 1.0);
        let (q, q_bits) = bf16s(&mut lcg, planes * H, 4.0);
        let post: Vec<f32> = (0..planes * M).map(|_| lcg.unit()).collect();
        let comb: Vec<f32> = (0..planes * M * M).map(|_| lcg.unit()).collect();
        let mut gpu = Gpu::open();
        let (x_at, s_at, q_at) = (gpu.up(&x_bits), gpu.up(&s_bits), gpu.up(&q_bits));
        let (p_at, c_at) = (gpu.up(&post), gpu.up(&comb));
        let (f_at, n_at) = (gpu.zeros(planes * M * H * 2), gpu.zeros(planes * H * 2));
        let (ctx, live) = ctx(&mut gpu, tuning, window, rows);
        let (xt, st) = (ten(x_at, rows, H, Bf16), ten(s_at, rows, M * H, Bf16));
        let (pt, ct) = (ten(p_at, rows, M, F32), ten(c_at, rows, M * M, F32));
        let (f, n) = (
            &mut ten(f_at, rows, M * H, Bf16),
            &mut ten(n_at, rows, H, Bf16),
        );
        hc::fold(&ctx, xt, st, pt, ct, f).expect("the fold fires");
        norm::rmsnorm_no_scale(&ctx, ten(q_at, rows, H, Bf16), 512, 1e-6, n)
            .expect("the norm fires");
        gpu.sync();
        let fold = wide(&gpu.down::<u16>(f_at, planes * M * H));
        let normed = wide(&gpu.down::<u16>(n_at, planes * H));
        let want_n = rmsnorm(&q, 512, None, 1e-6);
        let what = format!("{tuning:?}");
        for r in live.clone() {
            let comb = &comb[r * M * M..][..M * M];
            let s = &s[r * M * H..][..M * H];
            let mixed = |e: usize| {
                (0..M)
                    .map(|i| comb[i * M + e / H] * s[i * H + e % H])
                    .sum::<f32>()
            };
            let fold_at = |e: usize| post[r * M + e / H] * x[r * H + e % H] + mixed(e);
            let want: Vec<f32> = (0..M * H).map(fold_at).collect();
            near(&fold[r * M * H..][..M * H], &want, 2e-2, 0.05, &what);
            near(
                &normed[r * H..][..H],
                &want_n[r * H..][..H],
                2e-2,
                1.0,
                &what,
            );
        }
        assert!(
            fold[..live.start * M * H].iter().all(|&v| v == 0.0),
            "a row before the window moved"
        );
    }
    let (rows, mut lcg, mut gpu) = (5usize, Lcg::seeded(0x40f), Gpu::open());
    let (x, x_bits) = bf16s(&mut lcg, rows * H * M, 3.0);
    let (x_at, f_at) = (gpu.up(&x_bits), gpu.zeros(rows * M * H * 4));
    let (xt, f) = (
        ten(x_at, rows, M * H, Bf16),
        &mut ten(f_at, rows, M * H, F32),
    );
    hc::rmsnorm_f32(&gpu.ctx(), xt, 1e-6, f).expect("the stream norm fires");
    gpu.sync();
    let want = rmsnorm(&x, M * H, None, 1e-6);
    near(
        &gpu.down::<f32>(f_at, rows * M * H),
        &want,
        1e-4,
        1e-3,
        "the stream norm",
    );
}
