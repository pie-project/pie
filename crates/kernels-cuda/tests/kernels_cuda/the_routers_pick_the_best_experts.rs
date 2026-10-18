//! The MoE routers: each row's best experts, the lowest on a tie, and their
//! weights, against the host's serial walk; the softmax router's warp form
//! lands the block form's bits.

use crate::common::linear_ref::{Bench, pick, t, tab};
use crate::common::{from_bf16, to_bf16};
use dtype::Dtype::{Bf16, F32, I32};
use kernels_cuda::linear::moe;

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
        let tie = |i: usize| i / ex % 3 == 0 && i % ex % 50 == 7;
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
