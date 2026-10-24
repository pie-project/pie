#![cfg(feature = "cuda")]

//! DeepSeek-V4's hyper-connection launchers on the forms every card runs
//! (sm_100's tiles and one-pass pre-mix are its parity's), at four
//! 4096-wide streams, against the host: the projection's split walk, the
//! gates' Sinkhorn walk, the fold, the per-head norm and the stream norm.

use crate::common::attn_ref::{dot, landed, near, rmsnorm, ten, tuned, wide};
use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype::{Bf16, F32};
use kernels_cuda::elemwise::{hc, hc_project, norm};

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

/// The projection (`hc_project::project`) to a head's 4 mix columns and a layer's
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
        hc_project::project(&gpu.ctx(), x_t, w_t, 4, m).expect("the projection fires");
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
    let landed = hc_project::prenorm_project(&ctx, x_t, eps, w_t, M as u32, mixes);
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

/// `hc::pre` (the prenorm projection, then the gates and the norm) at a
/// decode's rows over four 4096-wide streams: its launches' rate.
#[test]
#[ignore = "a rate, not an answer"]
fn the_pre_rate() {
    let fan = M * H;
    let mut lcg = Lcg::seeded(0x5e7);
    let mut gpu = Gpu::open();
    for rows in [1usize, 8, 64] {
        let (_, s_bits) = bf16s(&mut lcg, rows * fan, 1.0);
        let w: Vec<f32> = (0..MIX * fan).map(|_| lcg.unit() * 0.01).collect();
        let bias: Vec<f32> = (0..MIX).map(|_| lcg.unit()).collect();
        let (_, nw) = bf16s(&mut lcg, H, 1.0);
        let (s_at, w_at, b_at, nw_at) = (gpu.up(&s_bits), gpu.up(&w), gpu.up(&bias), gpu.up(&nw));
        let sc_at = gpu.up(&[0.7f32, 1.3, 0.9]);
        let (mix_at, y_at) = (gpu.zeros(rows * MIX * 4), gpu.zeros(rows * H * 2));
        let (post_at, comb_at) = (gpu.zeros(rows * M * 4), gpu.zeros(rows * M * M * 4));
        let ctx = gpu.ctx();
        let fire = || {
            let gate = hc::Gate {
                scale: ten(sc_at, 1, 3, F32),
                base: ten(b_at, 1, MIX, F32),
                gate_eps: 1e-6,
                alpha: 2.0,
                sinkhorn: 20,
            };
            let fired = hc::pre(
                &ctx,
                ten(s_at, rows, fan, Bf16),
                1e-6,
                ten(w_at, MIX, fan, F32),
                M as u32,
                &mut ten(mix_at, rows, MIX, F32),
                gate,
                ten(nw_at, 1, H, Bf16),
                1e-6,
                None,
                &mut ten(post_at, rows, M, F32),
                &mut ten(comb_at, rows, M * M, F32),
                &mut ten(y_at, rows, H, Bf16),
            );
            assert!(fired.expect("the pre fires"), "the pre takes these planes");
        };
        fire();
        gpu.sync();
        let start = std::time::Instant::now();
        for _ in 0..200 {
            fire();
        }
        gpu.sync();
        eprintln!(
            "hc pre rows {rows}: {:.1} us a call",
            start.elapsed().as_secs_f64() * 1e6 / 200.0
        );
    }
}

/// One seam's inputs: a sublayer's output and the streams it folds into
/// with their gates, then the next sublayer's mix and norm weights, on a
/// plane of `rows` rows.
struct SeamIn {
    rows: usize,
    at: [u64; 9],
}

/// What a seam lands, read back: the folded streams, the post and combine
/// gates, the normed row and (held) its e4m3 codes and atoms.
struct SeamOut {
    folded: Vec<u16>,
    post: Vec<f32>,
    comb: Vec<f32>,
    y: Vec<u16>,
    codes: Vec<u8>,
    atoms: Vec<u8>,
}

/// The atom bytes of a `rows`-row 1x128 e4m3 operand `H` wide.
fn held_atoms(rows: usize) -> usize {
    rows.div_ceil(128) * (H / 128).div_ceil(4) * 512
}

impl SeamIn {
    fn draw(gpu: &mut Gpu, lcg: &mut Lcg, rows: usize) -> Self {
        let (_, x) = bf16s(lcg, rows * H, 1.0);
        let (_, s) = bf16s(lcg, rows * M * H, 1.0);
        // Gates of every f32 bit, as the gates kernel lands them: a gate
        // drawn bf16 times a bf16 stream is exact in f32, where a fused
        // multiply-add and a rounded product then a sum agree.
        let post: Vec<f32> = (0..rows * M)
            .map(|_| lcg.unit().abs() * 2.0 / 3.0)
            .collect();
        let comb: Vec<f32> = (0..rows * M * M).map(|_| lcg.unit().abs() / 3.0).collect();
        let w: Vec<f32> = (0..MIX * M * H).map(|_| lcg.unit() * 0.01).collect();
        let bias: Vec<f32> = (0..MIX).map(|_| lcg.unit()).collect();
        let (_, nw) = bf16s(lcg, H, 1.0);
        let at = [
            gpu.up(&x),
            gpu.up(&s),
            gpu.up(&post),
            gpu.up(&comb),
            gpu.up(&w),
            gpu.up(&[0.7f32, 1.3, 0.9]),
            gpu.up(&bias),
            gpu.up(&nw),
            0,
        ];
        Self { rows, at }
    }

    /// The seam on `ctx` over a fresh, zeroed output plane: in one launch
    /// (`mega`, which must take it) or as the fold then `hc::pre`.
    fn run(
        &self,
        gpu: &mut Gpu,
        ctx: &kernels_cuda::jit::Ctx,
        live: usize,
        mega: bool,
        held: bool,
    ) -> SeamOut {
        let rows = self.rows;
        let [x_at, s_at, p_at, c_at, w_at, sc_at, b_at, nw_at, _] = self.at;
        let f_at = gpu.zeros(rows * M * H * 2);
        let (po_at, co_at) = (gpu.zeros(rows * M * 4), gpu.zeros(rows * M * M * 4));
        let y_at = gpu.zeros(rows * H * 2);
        let (codes_at, atoms_at) = (gpu.zeros(rows * H), gpu.zeros(held_atoms(rows)));
        let mix_at = gpu.zeros(rows * MIX * 4);
        let gate = hc::Gate {
            scale: ten(sc_at, 1, 3, F32),
            base: ten(b_at, 1, MIX, F32),
            gate_eps: 1e-6,
            alpha: 2.0,
            sinkhorn: 20,
        };
        let held = held.then_some([codes_at, atoms_at]);
        let (x, s) = (ten(x_at, live, H, Bf16), ten(s_at, live, M * H, Bf16));
        let (p, c) = (ten(p_at, live, M, F32), ten(c_at, live, M * M, F32));
        let folded = &mut ten(f_at, live, M * H, Bf16);
        let (post, comb) = (
            &mut ten(po_at, live, M, F32),
            &mut ten(co_at, live, M * M, F32),
        );
        let y = &mut ten(y_at, live, H, Bf16);
        // The norm weight `[H]` as a load hands it: H rows of one column.
        let (plane, weight) = (ten(w_at, MIX, M * H, F32), ten(nw_at, H, 1, Bf16));
        if mega {
            let seam = hc::Seam {
                x,
                streams: s,
                post_mix: p,
                comb_mix: c,
                mix_eps: 1e-6,
                hc_fn: plane,
                stream_count: M as u32,
                gate,
                weight,
                norm_eps: 1e-6,
            };
            let fired = hc::fold_pre(ctx, seam, folded, held, post, comb, y);
            assert!(
                fired.expect("the seam fires"),
                "the seam takes these planes"
            );
        } else {
            hc::fold(ctx, x, s, p, c, folded).expect("the fold fires");
            let fired = hc::pre(
                ctx,
                *folded,
                1e-6,
                plane,
                M as u32,
                &mut ten(mix_at, live, MIX, F32),
                gate,
                weight,
                1e-6,
                held,
                post,
                comb,
                y,
            );
            assert!(fired.expect("the pre fires"), "the pre takes these planes");
        }
        gpu.sync();
        SeamOut {
            folded: gpu.down(f_at, rows * M * H),
            post: gpu.down(po_at, rows * M),
            comb: gpu.down(co_at, rows * M * M),
            y: gpu.down(y_at, rows * H),
            codes: gpu.down(codes_at, rows * H),
            atoms: gpu.down(atoms_at, held_atoms(rows)),
        }
    }
}

/// B200's hyper-connection decisions, the seam in one launch at any rows
/// (`mega`) or never.
fn seam_tuning(mega: bool) -> kernels_cuda::tuning::Tuning {
    tuned(|t| {
        t.hc_tile = true;
        t.hc_mma = true;
        t.hc_mega_until = if mega { 65_536 } else { 0 };
    })
}

/// The worst of `got` against `want` relative to `want` (at least 1).
fn worst(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
        .fold(0.0, f32::max)
}

/// Holds the seam's one launch to the fold and `hc::pre` it replaces, over
/// the plane rows `base .. base + live`: the folded streams bit for bit,
/// the post gates and the combine to 1e-5, the normed row apart in at most
/// a thousandth of its values and those within two bf16 steps (or 2^-18
/// where the pre-mix cancels), nothing past the window, and (held, the
/// seam's and the pre's alike) the e4m3 operand exactly what quantizing
/// its own normed row lands.
fn seam_matches(gpu: &mut Gpu, input: &SeamIn, live: usize, base: usize, held: bool, what: &str) {
    let stage = (base > 0 || live < input.rows).then(|| gpu.up(&[live as u32, base as u32, 0, 0]));
    let staged = |mega: bool| {
        let ctx = gpu.ctx().with_tuning(seam_tuning(mega));
        if let Some(at) = stage {
            ctx.arm_stage(at);
        }
        ctx
    };
    let (unfused, fused) = (staged(false), staged(true));
    let want = input.run(gpu, &unfused, live, false, held);
    let got = input.run(gpu, &fused, live, true, held);
    assert!(got.folded == want.folded, "{what}: the folded streams");
    let (post, comb) = (worst(&got.post, &want.post), worst(&got.comb, &want.comb));
    assert!(post <= 1e-5, "{what}: the post gates off by {post}");
    assert!(comb <= 1e-5, "{what}: the combine off by {comb}");
    // A normed value is the pre-mixed value (rounded to bf16) times the
    // norm and the weight, rounded again: a mix a few f32 ulps apart moves
    // a pre-mixed value by at most a bf16 step where it rounds the other
    // way, so a normed value by at most two of its steps; where the pre-mix
    // cancels to a value far under its terms, the mixes' f32 rounding is
    // what is left, at most 2^-18 of a normed row of unit scale.
    let (mut apart, mut most, mut flat) = (0usize, 0f32, 0f32);
    for (g, w) in got.y.iter().zip(&want.y) {
        let (g, w) = (from_bf16(*g), from_bf16(*w));
        if g != w {
            apart += 1;
            let step = 2f32.powi(g.abs().max(w.abs()).log2().floor() as i32 - 7);
            let d = (g - w).abs();
            if d <= 2.0 * step {
                most = most.max(d / step);
            } else {
                flat = flat.max(d);
            }
        }
    }
    let n = live * H;
    eprintln!(
        "{what}: {apart} of {n} normed values apart (at most {most} steps, else {flat:e}), post {post:e}, combine {comb:e}"
    );
    assert!(
        flat <= 2f32.powi(-18),
        "{what}: a cancelled normed value {flat} apart"
    );
    assert!(
        apart * 1000 <= n,
        "{what}: {apart} of {n} normed values apart"
    );
    for (r, row) in got.y.chunks(H).enumerate() {
        let inside = (base..base + live).contains(&r);
        if !inside {
            assert!(
                row.iter().all(|v| *v == 0),
                "{what}: row {r} past the window landed"
            );
        }
    }
    for (out, path) in [(&got, "the seam"), (&want, "the pre")] {
        if !held {
            break;
        }
        for r in base..base + live {
            let y: Vec<f32> = out.y[r * H..][..H].iter().map(|v| from_bf16(*v)).collect();
            for (b, block) in y.chunks(128).enumerate() {
                let (codes, e) = crate::common::linear_ref::quant_e4m3(block);
                assert_eq!(
                    &out.codes[r * H + b * 128..][..128],
                    &codes[..],
                    "{what}: {path}'s row {r} block {b} codes"
                );
                let at = kernels_cuda::linear::gemm_sm100::atom_offset(r, b, H / 128);
                assert_eq!(out.atoms[at], e, "{what}: {path}'s row {r} block {b} scale");
            }
        }
    }
}

/// The seam (`hc::fold_pre`, `elemwise/hc_sm100.cu`) against the fold and
/// `hc::pre` at DeepSeek-V4-Flash's four 4096-wide streams: a decode's rows,
/// a block of 128 and past it, a staged window, the normed row held.
#[test]
fn the_seam_answers_the_fold_and_the_pre_it_replaces() {
    if !kernels_cuda::linear::gemm_sm100::supported() {
        return;
    }
    let mut lcg = Lcg::seeded(0x5ea1);
    let mut gpu = Gpu::open();
    for (rows, held) in [
        (1usize, false),
        (8, true),
        (64, false),
        (200, true),
        (512, false),
    ] {
        let input = SeamIn::draw(&mut gpu, &mut lcg, rows);
        seam_matches(&mut gpu, &input, rows, 0, held, &format!("{rows} rows"));
    }
    let input = SeamIn::draw(&mut gpu, &mut lcg, 300);
    seam_matches(&mut gpu, &input, 141, 37, true, "a window of 141 at 37");
}

/// Two launches answer alike bit for bit (the splits summed in a fixed
/// order, whichever block takes them), and a graph replays the seam on its
/// counters as the eager launch left them: each replay over a window
/// rewritten between them lands what an eager launch over that window does.
#[test]
fn the_seam_replays_in_a_graph() {
    if !kernels_cuda::linear::gemm_sm100::supported() {
        return;
    }
    let mut lcg = Lcg::seeded(0x9a4);
    let mut gpu = Gpu::open();
    let rows = 160usize;
    let input = SeamIn::draw(&mut gpu, &mut lcg, rows);
    let win = gpu.up(&[rows as u32, 0u32, 0, 0]);
    let ctx = gpu.ctx().with_tuning(seam_tuning(true));
    ctx.arm_stage(win);
    let first = input.run(&mut gpu, &ctx, rows, true, true);
    let again = input.run(&mut gpu, &ctx, rows, true, true);
    assert!(
        first.folded == again.folded && first.y == again.y && first.codes == again.codes,
        "a second launch lands the first's rows"
    );
    assert!(
        first.post == again.post && first.comb == again.comb,
        "a second launch lands the first's gates"
    );
    // The graph: its outputs on fixed planes, the window read at replay.
    let [x_at, s_at, p_at, c_at, w_at, sc_at, b_at, nw_at, _] = input.at;
    let f_at = gpu.zeros(rows * M * H * 2);
    let (po_at, co_at) = (gpu.zeros(rows * M * 4), gpu.zeros(rows * M * M * 4));
    let y_at = gpu.zeros(rows * H * 2);
    let fire = || {
        let seam = hc::Seam {
            x: ten(x_at, rows, H, Bf16),
            streams: ten(s_at, rows, M * H, Bf16),
            post_mix: ten(p_at, rows, M, F32),
            comb_mix: ten(c_at, rows, M * M, F32),
            mix_eps: 1e-6,
            hc_fn: ten(w_at, MIX, M * H, F32),
            stream_count: M as u32,
            gate: hc::Gate {
                scale: ten(sc_at, 1, 3, F32),
                base: ten(b_at, 1, MIX, F32),
                gate_eps: 1e-6,
                alpha: 2.0,
                sinkhorn: 20,
            },
            weight: ten(nw_at, 1, H, Bf16),
            norm_eps: 1e-6,
        };
        let fired = hc::fold_pre(
            &ctx,
            seam,
            &mut ten(f_at, rows, M * H, Bf16),
            None,
            &mut ten(po_at, rows, M, F32),
            &mut ten(co_at, rows, M * M, F32),
            &mut ten(y_at, rows, H, Bf16),
        );
        assert!(
            fired.expect("the seam fires"),
            "the seam takes these planes"
        );
    };
    // Warmed eagerly: the slabs a capture sees are the ones it replays.
    fire();
    gpu.sync();
    let exec = crate::common::captured(&ctx, fire);
    for (live, base) in [(rows, 0usize), (37, 5), (129, 31), (rows, 0)] {
        let window = [live as u32, base as u32, 0, 0];
        let staged = gpu.up(&window);
        let eager_ctx = gpu.ctx().with_tuning(seam_tuning(true));
        eager_ctx.arm_stage(staged);
        let want = input.run(&mut gpu, &eager_ctx, live, true, false);
        crate::common::host::upload_to(win, &window);
        for _ in 0..3 {
            crate::common::replay(&ctx, exec);
        }
        gpu.sync();
        let what = format!("a replay over {live} rows at {base}");
        let at = |r: usize| (base..base + live).contains(&r);
        let folded: Vec<u16> = gpu.down(f_at, rows * M * H);
        let y: Vec<u16> = gpu.down(y_at, rows * H);
        let post: Vec<f32> = gpu.down(po_at, rows * M);
        let comb: Vec<f32> = gpu.down(co_at, rows * M * M);
        for r in (0..rows).filter(|r| at(*r)) {
            assert!(
                folded[r * M * H..][..M * H] == want.folded[r * M * H..][..M * H],
                "{what}: row {r}'s folded streams"
            );
            assert!(
                y[r * H..][..H] == want.y[r * H..][..H],
                "{what}: row {r}'s normed row"
            );
            assert!(
                post[r * M..][..M] == want.post[r * M..][..M],
                "{what}: row {r}'s post gates"
            );
            assert!(
                comb[r * M * M..][..M * M] == want.comb[r * M * M..][..M * M],
                "{what}: row {r}'s combine"
            );
        }
    }
    crate::common::forget(exec);
}

/// The seam's rate against the launches it replaces, as B200's profile
/// runs them (programmatic dependents, the held e4m3 operand): the fold
/// then `hc::pre` under 128 rows, past them the fold, the one-pass
/// projection, the gates and the norm-and-quantize (`Run::hc_pre`). Each
/// is timed in a graph of calls over enough copies of the planes that
/// every call reads its weights and streams from memory, not L2.
#[test]
#[ignore = "a rate, not an answer"]
fn the_seam_rate() {
    use kernels_cuda::linear::act_quant;
    if !kernels_cuda::linear::gemm_sm100::supported() {
        return;
    }
    let mut lcg = Lcg::seeded(0x7a7e);
    let mut gpu = Gpu::open();
    let tuning = |mega: bool| {
        let mut t = seam_tuning(mega);
        t.pdl = true;
        t.norm_emit = true;
        t
    };
    // `PIE_HC_SEAM_ROWS` (comma-separated) narrows the rows measured.
    let rows_of: Vec<usize> = std::env::var("PIE_HC_SEAM_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|r| r.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]);
    for rows in rows_of {
        let per = 24 * M * H * 4 + rows * (H * 2 * 3 + M * H * 2 * 2);
        let sets = (320usize << 20).div_ceil(per).clamp(2, 64);
        let inputs: Vec<SeamIn> = (0..sets)
            .map(|_| SeamIn::draw(&mut gpu, &mut lcg, rows))
            .collect();
        let outs: Vec<[u64; 8]> = (0..sets)
            .map(|_| {
                [
                    gpu.zeros(rows * M * H * 2),
                    gpu.zeros(rows * M * 4),
                    gpu.zeros(rows * M * M * 4),
                    gpu.zeros(rows * H * 2),
                    gpu.zeros(rows * H),
                    gpu.zeros(held_atoms(rows)),
                    gpu.zeros(rows * MIX * 4),
                    gpu.zeros(rows * H * 2),
                ]
            })
            .collect();
        let ctxs = [
            gpu.ctx().with_tuning(tuning(false)),
            gpu.ctx().with_tuning(tuning(true)),
        ];
        let mut execs = Vec::new();
        for (ctx, mega) in ctxs.iter().zip([false, true]) {
            let call = |i: usize| {
                let [x_at, s_at, p_at, c_at, w_at, sc_at, b_at, nw_at, _] = inputs[i].at;
                let [f_at, po_at, co_at, y_at, codes_at, atoms_at, mix_at, pre_at] = outs[i];
                let gate = hc::Gate {
                    scale: ten(sc_at, 1, 3, F32),
                    base: ten(b_at, 1, MIX, F32),
                    gate_eps: 1e-6,
                    alpha: 2.0,
                    sinkhorn: 20,
                };
                let (x, s) = (ten(x_at, rows, H, Bf16), ten(s_at, rows, M * H, Bf16));
                let (p, c) = (ten(p_at, rows, M, F32), ten(c_at, rows, M * M, F32));
                let folded = &mut ten(f_at, rows, M * H, Bf16);
                let (post, comb) = (
                    &mut ten(po_at, rows, M, F32),
                    &mut ten(co_at, rows, M * M, F32),
                );
                let y = &mut ten(y_at, rows, H, Bf16);
                let (plane, weight) = (ten(w_at, MIX, M * H, F32), ten(nw_at, 1, H, Bf16));
                let held = [codes_at, atoms_at];
                if mega {
                    let seam = hc::Seam {
                        x,
                        streams: s,
                        post_mix: p,
                        comb_mix: c,
                        mix_eps: 1e-6,
                        hc_fn: plane,
                        stream_count: M as u32,
                        gate,
                        weight,
                        norm_eps: 1e-6,
                    };
                    let fired = hc::fold_pre(ctx, seam, folded, Some(held), post, comb, y);
                    assert!(fired.expect("the seam fires"));
                    return;
                }
                hc::fold(ctx, x, s, p, c, folded).expect("the fold fires");
                let mixes = &mut ten(mix_at, rows, MIX, F32);
                if rows < hc::PRE_WIDE_FROM as usize {
                    let fired = hc::pre(
                        ctx,
                        *folded,
                        1e-6,
                        plane,
                        M as u32,
                        mixes,
                        gate,
                        weight,
                        1e-6,
                        Some(held),
                        post,
                        comb,
                        y,
                    );
                    assert!(fired.expect("the pre fires"));
                    return;
                }
                let fired = hc_project::prenorm_project(ctx, *folded, 1e-6, plane, M as u32, mixes);
                assert!(fired.expect("the projection fires"));
                let pre_x = &mut ten(pre_at, rows, H, Bf16);
                hc::gates(
                    ctx, *mixes, *folded, gate.scale, gate.base, M as u32, 1e-6, 2.0, 20, true,
                    pre_x, post, comb,
                )
                .expect("the gates fire");
                let fired = act_quant::rmsnorm_act_quant_e4m3(ctx, *pre_x, weight, 1e-6, y, held);
                assert!(fired.expect("the norm fires"));
            };
            for i in 0..sets {
                call(i);
            }
            gpu.sync();
            let exec = crate::common::captured(ctx, || (0..sets).for_each(call));
            crate::common::replay(ctx, exec);
            gpu.sync();
            execs.push(exec);
        }
        // The card is shared: the two graphs' trials alternate, each one
        // replay short enough to fit a time slice, and each keeps its best.
        let reps = 1;
        let mut times = [f64::MAX; 2];
        for _ in 0..60 {
            for k in 0..2 {
                let start = std::time::Instant::now();
                for _ in 0..reps {
                    crate::common::replay(&ctxs[k], execs[k]);
                }
                gpu.sync();
                let per = start.elapsed().as_secs_f64() * 1e6 / (reps * sets) as f64;
                times[k] = times[k].min(per);
            }
        }
        execs.into_iter().for_each(crate::common::forget);
        eprintln!(
            "hc seam rows {rows:4}: parts {:7.2} us seam {:7.2} us  x{:.2}",
            times[0],
            times[1],
            times[0] / times[1]
        );
    }
}
