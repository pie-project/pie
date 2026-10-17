#![cfg(feature = "cuda")]

//! DeepSeek-V4's hyper-connection launchers on the forms every card runs
//! (sm_100's tiles and one-pass pre-mix are its parity's), at four
//! 4096-wide streams, against the host: the projection's split walk, the
//! gates' Sinkhorn walk, the fold, the per-head norm and the stream norm.

use crate::common::attn_ref::{dot, landed, near, rmsnorm, ten, tuned, wide};
use crate::common::{Gpu, Lcg, to_bf16};
use dtype::Dtype::{Bf16, F32};
use kernels_cuda::elemwise::{hc, norm};

const M: usize = 4;
const H: usize = 4096;
const MIX: usize = 2 * M + M * M;

/// `n` draws scaled by `k` as bf16: values and bits.
fn bf16s(lcg: &mut Lcg, n: usize, k: f32) -> (Vec<f32>, Vec<u16>) {
    let bits: Vec<u16> = (0..n).map(|_| to_bf16(lcg.unit() * k)).collect();
    (wide(&bits), bits)
}

/// One row's gates from its mixes: pre, post and the Sinkhorn combine.
fn gates_of(row: &[f32], base: &[f32]) -> [Vec<f32>; 3] {
    let (eps, s) = (1e-6f32, [0.7, 1.3, 0.9]);
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    let pre = (0..M).map(|i| sig(row[i] * s[0] + base[i]) + eps).collect();
    let post = (0..M).map(|i| sig(row[M + i] * s[1] + base[M + i]) * 2.0);
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
    [pre, post.collect(), c]
}

/// The projection (`hc::project`) to a head's 4 mix columns and a layer's
/// 24; the gates (pre, post, the Sinkhorn combine and the pre-mixed row).
#[test]
fn the_projection_and_the_gates_answer_the_host() {
    let (rows, fan) = (7usize, M * H);
    let mut lcg = Lcg::seeded(0x4c0);
    let (x, _) = bf16s(&mut lcg, rows * fan, 1.0);
    let w: Vec<f32> = (0..MIX * fan).map(|_| lcg.unit() * 0.01).collect();
    let (s, s_bits) = bf16s(&mut lcg, rows * fan, 1.0);
    let mixes: Vec<f32> = (0..rows * MIX).map(|_| lcg.unit() * 3.0).collect();
    let bias: Vec<f32> = (0..MIX).map(|_| lcg.unit()).collect();
    let mut gpu = Gpu::open();
    let (x_at, w_at) = (gpu.up(&x), gpu.up(&w));
    for mix in [4, MIX] {
        let m_at = gpu.zeros(rows * mix * 4);
        let (x_t, w_t, m) = (
            ten(x_at, rows, fan, F32),
            ten(w_at, mix, fan, F32),
            &mut ten(m_at, rows, mix, F32),
        );
        hc::project(&gpu.ctx(), x_t, w_t, 4, m).expect("the projection fires");
        gpu.sync();
        let want: Vec<f32> = x
            .chunks(fan)
            .flat_map(|r| w.chunks(fan).take(mix).map(|o| dot(r, o)))
            .collect();
        near(
            &gpu.down::<f32>(m_at, rows * mix),
            &want,
            2e-3,
            1.0,
            &format!("{mix} columns"),
        );
    }
    let (s_at, mix_at, b_at) = (gpu.up(&s_bits), gpu.up(&mixes), gpu.up(&bias));
    let (sc_at, y_at) = (gpu.up(&[0.7f32, 1.3, 0.9]), gpu.zeros(rows * H * 2));
    let (post_at, comb_at) = (gpu.zeros(rows * M * 4), gpu.zeros(rows * M * M * 4));
    let (st, mt) = (ten(s_at, rows, fan, Bf16), ten(mix_at, rows, MIX, F32));
    let (sc, b, y) = (
        ten(sc_at, 1, 3, F32),
        ten(b_at, 1, MIX, F32),
        &mut ten(y_at, rows, H, Bf16),
    );
    let (post, comb) = (
        &mut ten(post_at, rows, M, F32),
        &mut ten(comb_at, rows, M * M, F32),
    );
    let fired = hc::gates(
        &gpu.ctx(),
        mt,
        st,
        sc,
        b,
        4,
        1e-6,
        2.0,
        20,
        true,
        y,
        post,
        comb,
    );
    fired.expect("the gates fire");
    gpu.sync();
    let (got_post, got_comb): (Vec<f32>, Vec<f32>) =
        (gpu.down(post_at, rows * M), gpu.down(comb_at, rows * M * M));
    let mut want_y = Vec::new();
    for r in 0..rows {
        let [pre, p, c] = gates_of(&mixes[r * MIX..][..MIX], &bias);
        let s = &s[r * fan..][..fan];
        want_y.extend((0..H).map(|d| (0..M).map(|i| pre[i] * s[i * H + d]).sum::<f32>()));
        near(&got_post[r * M..][..M], &p, 1e-3, 1.0, "the post gates");
        near(
            &got_comb[r * M * M..][..M * M],
            &c,
            1e-3,
            1.0,
            "the combine",
        );
    }
    landed(&gpu, y_at, &want_y, 2e-2, 0.05, "the pre-mixed row");
}

/// The fold (`post_j x + sum_i comb_ij s_i`) and the scale-free per-head
/// norm, on a staged window's rows and nowhere else; the stream norm in f32.
#[test]
fn the_fold_and_the_norms_answer_the_host() {
    let (rows, live, base) = (24usize, 17usize, 5usize);
    let mut lcg = Lcg::seeded(0xf01d);
    let (x, x_bits) = bf16s(&mut lcg, rows * H, 1.0);
    let (s, s_bits) = bf16s(&mut lcg, rows * M * H, 1.0);
    let (q, q_bits) = bf16s(&mut lcg, rows * H, 4.0);
    let post: Vec<f32> = (0..rows * M).map(|_| lcg.unit()).collect();
    let comb: Vec<f32> = (0..rows * M * M).map(|_| lcg.unit()).collect();
    let mut gpu = Gpu::open();
    let (x_at, s_at, q_at) = (gpu.up(&x_bits), gpu.up(&s_bits), gpu.up(&q_bits));
    let (p_at, c_at) = (gpu.up(&post), gpu.up(&comb));
    let (f_at, n_at) = (gpu.zeros(rows * M * H * 2), gpu.zeros(rows * H * 2));
    let ctx = gpu.ctx();
    ctx.arm_stage(gpu.up(&[live as u32, base as u32, 0, 0]));
    let (xt, st) = (ten(x_at, live, H, Bf16), ten(s_at, live, M * H, Bf16));
    let (pt, ct) = (ten(p_at, live, M, F32), ten(c_at, live, M * M, F32));
    let (f, n) = (
        &mut ten(f_at, live, M * H, Bf16),
        &mut ten(n_at, live, H, Bf16),
    );
    hc::fold(&ctx, xt, st, pt, ct, f).expect("the fold fires");
    norm::rmsnorm_no_scale(&ctx, ten(q_at, live, H, Bf16), 512, 1e-6, n).expect("the norm fires");
    let (mut fold, mut normed) = (vec![0f32; rows * M * H], vec![0f32; rows * H]);
    let want_n = rmsnorm(&q, 512, None, 1e-6);
    for r in base..base + live {
        let (comb, s) = (&comb[r * M * M..][..M * M], &s[r * M * H..][..M * H]);
        let mixed = |e: usize| {
            (0..M)
                .map(|i| comb[i * M + e / H] * s[i * H + e % H])
                .sum::<f32>()
        };
        let fold_at = |e: usize| post[r * M + e / H] * x[r * H + e % H] + mixed(e);
        fold[r * M * H..][..M * H]
            .iter_mut()
            .enumerate()
            .for_each(|(e, v)| *v = fold_at(e));
        normed[r * H..][..H].copy_from_slice(&want_n[r * H..][..H]);
    }
    landed(&gpu, f_at, &fold, 2e-2, 0.05, "the fold");
    landed(&gpu, n_at, &normed, 2e-2, 1.0, "the per-head norm");
    let (x_at, f_at) = (gpu.up(&s_bits[..5 * M * H]), gpu.zeros(5 * M * H * 4));
    let (xt, f) = (ten(x_at, 5, M * H, Bf16), &mut ten(f_at, 5, M * H, F32));
    hc::rmsnorm_f32(&gpu.ctx(), xt, 1e-6, f).expect("the stream norm fires");
    gpu.sync();
    let want = rmsnorm(&s[..5 * M * H], M * H, None, 1e-6);
    near(
        &gpu.down::<f32>(f_at, 5 * M * H),
        &want,
        1e-4,
        1e-3,
        "the stream norm",
    );
}

/// B200's one-pass stream norm and projection on the tensor cores
/// (`hc_tile`, `hc_mma`), a prefill's rows folding a thread a row.
#[test]
fn the_tensor_core_prenorm_projection_answers_the_host() {
    let (rows, fan, eps) = (1024usize, M * H, 1e-6f32);
    let mut lcg = Lcg::seeded(0x9e7);
    let (x, x_bits) = bf16s(&mut lcg, rows * fan, 3.0);
    let w: Vec<f32> = (0..MIX * fan).map(|_| lcg.unit() * 0.01).collect();
    let mut gpu = Gpu::open();
    let (x_at, w_at, m_at) = (gpu.up(&x_bits), gpu.up(&w), gpu.zeros(rows * MIX * 4));
    let ctx = gpu
        .ctx()
        .with_tuning(tuned(|t| (t.hc_tile, t.hc_mma) = (true, true)));
    let (x_t, w_t) = (ten(x_at, rows, fan, Bf16), ten(w_at, MIX, fan, F32));
    let mixes = &mut ten(m_at, rows, MIX, F32);
    let landed = hc::prenorm_project(&ctx, x_t, eps, w_t, M as u32, mixes);
    assert!(
        landed.expect("the projection fires"),
        "a 16384-wide row is the tile's"
    );
    gpu.sync();
    let want: Vec<f32> = x
        .chunks(fan)
        .flat_map(|r| {
            let s = (r.iter().map(|v| v * v).sum::<f32>() / fan as f32 + eps)
                .sqrt()
                .recip();
            w.chunks(fan).map(move |o| dot(r, o) * s)
        })
        .collect();
    let got = gpu.down::<f32>(m_at, rows * MIX);
    near(&got, &want, 2e-3, 1.0, "the tensor-core prenorm projection");
}
