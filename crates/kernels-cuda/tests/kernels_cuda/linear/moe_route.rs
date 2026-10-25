//! The MoE routers: each row's best experts, the lowest on a tie, and their
//! weights, against the host's serial walk; the softmax router's warp form
//! lands the block form's bits. The fused gate (the router's product and its
//! top-k in one tcgen05 launch) lands the routes of the product and the
//! router apart, but where two scores sit within a rounding of each other.

use crate::common::linear_ref::{Bench, pick, t, tab};
use crate::common::{captured, forget, from_bf16, replay, to_bf16};
use dtype::Dtype::{self, Bf16, F32, I32};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::{gemm, moe};
use kernels_cuda::tensor::Tensor;
use kernels_cuda::tuning::Tuning;

#[test]
fn the_ranked_routers_pick_the_best_experts() {
    // (DeepSeek-V4's sqrt-softplus or the sigmoid router, rows, experts,
    // picks): at the expert counts the families route over (Kimi-K3's 896).
    #[rustfmt::skip]
    let cases = [(true, 9, 256, 6), (false, 9, 256, 8), (false, 6, 896, 16), (false, 3, 20, 6)];
    for (softplus, rows, ex, k) in cases {
        let (mut b, scaling) = (Bench::new(1), 1.5f32);
        // Evenly spaced logits, a few planted ties.
        let even = |i: usize| -4.0 + 0.03 * ((i % ex * 97 + i / ex * 31) % ex) as f32;
        let tie = |i: usize| (i / ex).is_multiple_of(3) && i % ex % 50 == 7;
        let logits = tab(rows * ex, |i| to_bf16(pick(tie(i), 2.5, even(i))));
        let bias = tab(ex, |e| pick(e % 5 == 0, 0.0f32, -0.001));
        let (l, bt) = (t(b.up(&logits), rows, ex, Bf16), t(b.up(&bias), 1, ex, F32));
        let mut routes = t(b.zeros(rows * k * 4), rows, k, I32);
        let mut weights = t(b.zeros(rows * k * 4), rows, k, F32);
        let (ctx, e, k32, out) = (b.ctx(), ex as u32, k as u32, (&mut routes, &mut weights));
        #[rustfmt::skip]
        let fired = match softplus {
            true => moe::topk_sqrt_softplus(&ctx, l, bt, e, k32, true, scaling, out.0, out.1),
            false => moe::topk_sigmoid(&ctx, l, Some(bt), e, k32, true, scaling, out.0, out.1),
        };
        fired.expect("the router fires");
        b.sync();
        let got_r: Vec<i32> = b.down(routes.ptr, rows * k);
        let got_w: Vec<f32> = b.down(weights.ptr, rows * k);
        let soft = |x: f32| pick(x > 20.0, x, x.exp().ln_1p()).max(0.0).sqrt();
        let score = |x: f32| pick(softplus, soft(x), 1.0 / (1.0 + (-x).exp()));
        for r in 0..rows {
            let orig = tab(ex, |e| score(from_bf16(logits[r * ex + e])));
            let mut biased = tab(ex, |e| orig[e] + bias[e]);
            let picks = tab(k, |_| {
                let best = (0..ex).fold(0, |m, e| pick(biased[e] > biased[m], e, m));
                biased[best] = f32::MIN;
                best
            });
            // Unbiased scores renormalized and scaled.
            let sum: f32 = picks.iter().map(|&e| orig[e]).sum();
            let same = got_r[r * k..][..k] == tab(k, |j| picks[j] as i32);
            assert!(same, "{ex} experts, row {r}: the picks");
            for (j, &got) in got_w[r * k..][..k].iter().enumerate() {
                let want = orig[picks[j]] * scaling / sum;
                let near = (got - want).abs() <= 1e-5 * want.abs().max(1.0);
                assert!(near, "row {r} pick {j}: weight {got} against {want}");
            }
        }
    }
}

#[test]
fn the_softmax_routers_pick_the_best_experts() {
    // (rows, experts, picks, scaled, live rows): Qwen3.6's 256 of 8, Gemma
    // 4's 128 of 8 under its per-expert scale, a small and an odd router,
    // under a live window.
    #[rustfmt::skip]
    let cases = [(9, 256, 8, false, 0), (5, 128, 8, true, 0), (4, 20, 6, false, 0), (6, 400, 16, true, 0),
        (8, 256, 8, false, 5), (3, 512, 4, false, 0)];
    for (rows, ex, k, scaled, live) in cases {
        let mut b = Bench::new(3);
        let even = |i: usize| -4.0 + 0.02 * ((i % ex * 97 + i / ex * 31) % ex) as f32;
        let tie = |i: usize| (i / ex).is_multiple_of(3) && i % ex % 50 == 7;
        let logits = tab(rows * ex, |i| to_bf16(pick(tie(i), 2.5, even(i))));
        let gain = tab(ex, |e| to_bf16(0.5 + (e % 7) as f32 / 8.0));
        let (l, g) = (
            t(b.up(&logits), rows, ex, Bf16),
            t(b.up(&gain), 1, ex, Bf16),
        );
        let mut routes = t(b.up(&vec![-7i32; rows * k]), rows, k, I32);
        let mut weights = t(b.zeros(rows * k * 4), rows, k, F32);
        let ctx = b.ctx();
        if live > 0 {
            ctx.arm_stage(b.up(&[live as u32, 0, 0, 0]));
        }
        let (e, k32) = (ex as u32, k as u32);
        let fired = match scaled {
            true => moe::topk_softmax_scaled(&ctx, l, g, e, k32, &mut routes, &mut weights),
            false => moe::topk_softmax(&ctx, l, e, k32, &mut routes, &mut weights),
        };
        fired.expect("the router fires");
        b.sync();
        let got_r: Vec<i32> = b.down(routes.ptr, rows * k);
        let got_w: Vec<f32> = b.down(weights.ptr, rows * k);
        for r in 0..rows {
            if live > 0 && r >= live {
                assert!(
                    got_r[r * k..][..k].iter().all(|&e| e == -7),
                    "row {r} past the window"
                );
                continue;
            }
            let orig = tab(ex, |e| from_bf16(logits[r * ex + e]));
            let mut left = orig.clone();
            let picks = tab(k, |_| {
                let best = (0..ex).fold(0, |m, e| pick(left[e] > left[m], e, m));
                left[best] = f32::MIN;
                best
            });
            let exps = tab(k, |j| (orig[picks[j]] - orig[picks[0]]).exp());
            let sum: f32 = exps.iter().sum();
            let same = got_r[r * k..][..k] == tab(k, |j| picks[j] as i32);
            assert!(
                same,
                "{ex} experts, row {r}: the picks {:?}",
                &got_r[r * k..][..k]
            );
            for (j, &got) in got_w[r * k..][..k].iter().enumerate() {
                let s = pick(scaled, from_bf16(gain[picks[j]]), 1.0);
                let want = exps[j] / sum * s;
                let near = (got - want).abs() <= 1e-5 * want.abs().max(1.0);
                assert!(
                    near,
                    "{ex} experts row {r} pick {j}: weight {got} against {want}"
                );
            }
        }
    }
}

/// A fused gate's scoring, as the router it stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scored {
    /// `topk_softmax` (Qwen3's).
    Softmax,
    /// `topk_softmax_scaled`, a bf16 gain an expert.
    Gained,
    /// `topk_sigmoid` under a correction bias.
    Sigmoid,
    /// `topk_sigmoid` without one.
    Bare,
    /// `topk_sqrt_softplus` (DeepSeek-V4's).
    Softplus,
}

/// The routed scaling the ranked cases weight under (V4-Flash's).
const SCALING: f32 = 1.5;

fn gate_of(
    s: Scored,
    experts: u32,
    top_k: u32,
    logits: Dtype,
    bias: Tensor,
    gain: Tensor,
) -> moe::Gate {
    let score = match s {
        Scored::Softmax => moe::GateScore::Softmax { gain: None },
        Scored::Gained => moe::GateScore::Softmax { gain: Some(gain) },
        Scored::Sigmoid | Scored::Bare => moe::GateScore::Sigmoid {
            bias: (s == Scored::Sigmoid).then_some(bias),
            renormalize: true,
            scaling: SCALING,
        },
        Scored::Softplus => moe::GateScore::SqrtSoftplus {
            bias,
            renormalize: true,
            scaling: SCALING,
        },
    };
    moe::Gate {
        experts,
        top_k,
        logits,
        score,
    }
}

/// The path the engine runs today: the router's product (`matmul`, or
/// `matmul_f32` for f32 logits), then the router over its logits.
fn unfused(
    ctx: &Ctx,
    x: Tensor,
    w: Tensor,
    logits: &mut Tensor,
    gate: moe::Gate,
    r: &mut Tensor,
    wt: &mut Tensor,
) {
    match logits.dtype {
        F32 => gemm::matmul_f32(ctx, x, w, logits),
        _ => gemm::matmul(ctx, x, w, logits),
    }
    .expect("the router's product fires");
    let (l, e, k) = (*logits, gate.experts, gate.top_k);
    match gate.score {
        moe::GateScore::Softmax { gain: None } => moe::topk_softmax(ctx, l, e, k, r, wt),
        moe::GateScore::Softmax { gain: Some(g) } => {
            moe::topk_softmax_scaled(ctx, l, g, e, k, r, wt)
        }
        moe::GateScore::Sigmoid {
            bias,
            renormalize,
            scaling,
        } => moe::topk_sigmoid(ctx, l, bias, e, k, renormalize, scaling, r, wt),
        moe::GateScore::SqrtSoftplus {
            bias,
            renormalize,
            scaling,
        } => moe::topk_sqrt_softplus(ctx, l, bias, e, k, renormalize, scaling, r, wt),
    }
    .expect("the router fires");
}

/// B200's profile as the unfused path runs it (programmatic dependents, the
/// few-rows GEMM tactic, the split's launch charge), the gate taking every
/// row count.
fn sm100() -> Tuning {
    Tuning {
        pdl: true,
        gemm_rows_m: 8,
        gemm_split_launch_ns: 2000,
        gemv_narrow: true,
        gemm_bias_epilogue: true,
        gemm_tc5_split_tickets: 8,
        rows_vec: true,
        gate_tc5_until: 65_536,
        ..crate::common::SM100
    }
}

/// The score a row's expert ranks by, from its logit.
fn ranked(s: Scored, x: f64) -> f64 {
    match s {
        Scored::Softmax | Scored::Gained => x,
        Scored::Sigmoid | Scored::Bare => 1.0 / (1.0 + (-x).exp()),
        Scored::Softplus => (if x > 20.0 { x } else { x.exp().ln_1p() }).max(0.0).sqrt(),
    }
}

/// One bf16 step at `x`'s magnitude.
fn bf16_ulp(x: f64) -> f64 {
    let x = x.abs().max(f64::from(f32::MIN_POSITIVE));
    2f64.powi(x.log2().floor() as i32 - 7)
}

#[test]
fn the_fused_gate_lands_the_unfused_routes() {
    if !kernels_cuda::linear::gemm_sm100::supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // (scoring, rows, hidden, experts, picks, logits, live window): V4-Flash
    // (4096 by 256, 6 picks) across a decode's rows, V4-Pro (7168 by 384),
    // Qwen3-30B-A3B (2048 by 128 of 8) and Coder-480B (6144 by 160: a part
    // accumulator), the sigmoid router at f32 logits and bf16, a gained
    // softmax, a fire under a live window.
    #[rustfmt::skip]
    let cases = [
        (Scored::Softplus, 1, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 7, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 64, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 130, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 256, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 300, 4096, 256, 6, Bf16, None),
        (Scored::Softplus, 9, 7168, 384, 6, Bf16, None),
        (Scored::Softmax, 33, 2048, 128, 8, Bf16, None),
        (Scored::Softmax, 5, 6144, 160, 8, Bf16, None),
        (Scored::Gained, 17, 2048, 128, 8, Bf16, None),
        (Scored::Sigmoid, 20, 4096, 256, 8, F32, None),
        (Scored::Sigmoid, 12, 2880, 512, 32, Bf16, None),
        (Scored::Bare, 40, 4096, 256, 8, Bf16, None),
        (Scored::Softplus, 48, 4096, 256, 6, Bf16, Some((30u32, 9u32))),
    ];
    let (mut same_rows, mut tied_rows, mut worst) = (0usize, 0usize, 0f64);
    for (s, rows, hidden, ex, k, ldt, live) in cases {
        let mut b = Bench::new(11 + rows as u64);
        let (x_at, xv) = b.bf16(rows * hidden);
        // A router whose logits spread about one (V4's run 1-3).
        let wraw = tab(ex * hidden, |_| to_bf16(b.lcg.unit() * 0.04));
        let wv = tab(ex * hidden, |i| f64::from(from_bf16(wraw[i])));
        let bias = tab(ex, |_| b.lcg.unit() * 0.05);
        let gain = tab(ex, |e| to_bf16(0.5 + (e % 7) as f32 / 8.0));
        let w_at = b.up(&wraw);
        let (x, w) = (t(x_at, rows, hidden, Bf16), t(w_at, ex, hidden, Bf16));
        let (bt, gt) = (t(b.up(&bias), 1, ex, F32), t(b.up(&gain), 1, ex, Bf16));
        let lbytes = pick(ldt == F32, 4, 2);
        let mut logits = t(b.zeros(rows * ex * lbytes), rows, ex, ldt);
        let lanes = |b: &mut Bench| {
            (
                t(b.up(&vec![-7i32; rows * k]), rows, k, I32),
                t(b.up(&vec![-7f32; rows * k]), rows, k, F32),
            )
        };
        let ((mut r0, mut w0), (mut r1, mut w1)) = (lanes(&mut b), lanes(&mut b));
        let ctx = b.ctx().with_tuning(sm100());
        if let Some((n, first)) = live {
            ctx.arm_stage(b.up(&[n, first, 0, 0]));
        }
        let g = gate_of(s, ex as u32, k as u32, ldt, bt, gt);
        unfused(&ctx, x, w, &mut logits, g, &mut r0, &mut w0);
        let took = moe::gate(&ctx, x, w, g, &mut r1, &mut w1).expect("the gate fires");
        assert!(took, "{s:?} {rows}x{hidden}x{ex}: the gate takes the fire");
        b.sync();
        let (want_r, got_r): (Vec<i32>, Vec<i32>) =
            (b.down(r0.ptr, rows * k), b.down(r1.ptr, rows * k));
        let (want_w, got_w): (Vec<f32>, Vec<f32>) =
            (b.down(w0.ptr, rows * k), b.down(w1.ptr, rows * k));
        for row in 0..rows {
            let at = row * k..(row + 1) * k;
            if let Some((n, first)) = live {
                let inside = (first as usize..(first + n) as usize).contains(&row);
                if !inside {
                    assert!(
                        got_r[at.clone()].iter().all(|&e| e == -7),
                        "row {row} past the window"
                    );
                    continue;
                }
            }
            if want_r[at.clone()] == got_r[at.clone()] {
                same_rows += 1;
                for j in at {
                    let (got, want) = (f64::from(got_w[j]), f64::from(want_w[j]));
                    // A logit a bf16 step apart moves its weight by about
                    // that step; an f32 one by its accumulation's rounding.
                    let tol = pick(ldt == F32, 1e-5, 1.0 / 64.0) * want.abs() + 1e-6;
                    worst = worst.max((got - want).abs() / want.abs().max(1e-6));
                    assert!(
                        (got - want).abs() <= tol,
                        "{s:?} {rows}x{hidden}x{ex} row {row}: weight {got} against {want}"
                    );
                }
                continue;
            }
            // The picks differ: only between experts whose scores the
            // logits' rounding can reorder. A logit is the exact product
            // up to a rounding of its dtype (and the f32 sums' order).
            tied_rows += 1;
            let logit = |e: usize| {
                (0..hidden)
                    .map(|i| xv[row * hidden + i] * wv[e * hidden + i])
                    .sum::<f64>()
            };
            let mass = |e: usize| {
                (0..hidden)
                    .map(|i| (xv[row * hidden + i] * wv[e * hidden + i]).abs())
                    .sum::<f64>()
            };
            let score = |e: usize| {
                let (l, u) = (
                    logit(e),
                    pick(ldt == F32, 0.0, bf16_ulp(logit(e))) + 4e-7 * mass(e),
                );
                let bias = pick(
                    matches!(s, Scored::Sigmoid | Scored::Softplus),
                    f64::from(bias[e]),
                    0.0,
                );
                (
                    ranked(s, l) + bias,
                    (ranked(s, l + u) - ranked(s, l - u)).abs(),
                )
            };
            for j in at {
                let (a, c) = (want_r[j], got_r[j]);
                if a == c {
                    continue;
                }
                let ((sa, da), (sc, dc)) = (score(a as usize), score(c as usize));
                assert!(
                    (sa - sc).abs() <= 2.0 * (da + dc),
                    "{s:?} {rows}x{hidden}x{ex} row {row} pick {}: expert {c} for {a}, scores {sc} and {sa} \
                     apart past their rounding ({dc}, {da})",
                    j % k
                );
            }
        }
    }
    eprintln!(
        "the fused gate: {same_rows} rows the same routes (weights within {worst:.2e} relative), \
         {tied_rows} reordered within a rounding"
    );
}

#[test]
fn a_replayed_fused_gate_lands_what_it_lands_eagerly() {
    if !kernels_cuda::linear::gemm_sm100::supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // V4-Flash's router at a decode's 24 rows: a graph of the gate replayed
    // over new rows lands the eager gate's bits for them, each time.
    let (rows, hidden, ex, k) = (24usize, 4096usize, 256usize, 6usize);
    let mut b = Bench::new(9);
    let wraw = tab(ex * hidden, |_| to_bf16(b.lcg.unit() * 0.04));
    let w = t(b.up(&wraw), ex, hidden, Bf16);
    let bias = tab(ex, |_| b.lcg.unit() * 0.05);
    let bt = t(b.up(&bias), 1, ex, F32);
    let x = t(b.zeros(rows * hidden * 2), rows, hidden, Bf16);
    let lanes = |b: &mut Bench| {
        (
            t(b.zeros(rows * k * 4), rows, k, I32),
            t(b.zeros(rows * k * 4), rows, k, F32),
        )
    };
    let ((mut r0, mut w0), (mut r1, mut w1)) = (lanes(&mut b), lanes(&mut b));
    let ctx = b.ctx().with_tuning(sm100());
    let g = gate_of(Scored::Softplus, ex as u32, k as u32, Bf16, bt, bt);
    // Eagerly first: a capture fires what is already loaded.
    assert!(moe::gate(&ctx, x, w, g, &mut r1, &mut w1).expect("the gate fires"));
    b.sync();
    let exec = captured(&ctx, || {
        assert!(moe::gate(&ctx, x, w, g, &mut r1, &mut w1).expect("the gate fires"));
    });
    for _ in 0..3 {
        let (_, xs) = b.lcg.row(rows * hidden);
        crate::common::host::upload_to(x.ptr, &tab(rows * hidden, |i| to_bf16(xs[i])));
        assert!(moe::gate(&ctx, x, w, g, &mut r0, &mut w0).expect("the gate fires"));
        replay(&ctx, exec);
        b.sync();
        let (want_r, got_r): (Vec<i32>, Vec<i32>) =
            (b.down(r0.ptr, rows * k), b.down(r1.ptr, rows * k));
        let (want_w, got_w): (Vec<u32>, Vec<u32>) =
            (b.down(w0.ptr, rows * k), b.down(w1.ptr, rows * k));
        assert_eq!(got_r, want_r, "the replay's routes");
        assert_eq!(got_w, want_w, "the replay's weights, bit for bit");
    }
    forget(exec);
}

#[test]
fn the_fused_gate_declines_what_it_does_not_take() {
    // (scoring, rows, experts, picks, x's dtype, tuning): past B200's 256
    // rows, a softmax of more than 16 picks, more than 512 experts, f32
    // activations, and a profile that never fuses.
    let b200 = crate::common::SM100;
    #[rustfmt::skip]
    let cases = [
        (Scored::Softplus, 300, 256, 6, Bf16, b200),
        (Scored::Softmax, 4, 256, 20, Bf16, b200),
        (Scored::Sigmoid, 4, 640, 8, Bf16, b200),
        (Scored::Softplus, 4, 256, 6, F32, b200),
        (Scored::Softplus, 1, 256, 6, Bf16, Tuning::DEFAULT),
    ];
    let hidden = 512;
    for (s, rows, ex, k, xdt, tuning) in cases {
        let mut b = Bench::new(2);
        let x = t(b.zeros(rows * hidden * 4), rows, hidden, xdt);
        let w = t(b.zeros(ex * hidden * 2), ex, hidden, Bf16);
        let (bt, gt) = (
            t(b.zeros(ex * 4), 1, ex, F32),
            t(b.zeros(ex * 2), 1, ex, Bf16),
        );
        let mut r = t(b.up(&vec![-7i32; rows * k]), rows, k, I32);
        let mut wt = t(b.zeros(rows * k * 4), rows, k, F32);
        let ctx = b.ctx().with_tuning(tuning);
        let g = gate_of(s, ex as u32, k as u32, Bf16, bt, gt);
        let took = moe::gate(&ctx, x, w, g, &mut r, &mut wt).expect("the gate answers");
        assert!(
            !took,
            "{s:?} {rows} rows of {ex} experts to {k}: the gate declines"
        );
        b.sync();
        let left: Vec<i32> = b.down(r.ptr, rows * k);
        assert!(
            left.iter().all(|&e| e == -7),
            "a declined gate writes nothing"
        );
    }
}
