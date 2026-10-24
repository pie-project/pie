//! Speculative sampling's acceptance rule (Leviathan et al. 2023, Chen et al.
//! 2023): what a verify keeps of a draft when the target samples rather than
//! takes its argmax, so the text is distributed as the target's own.
//!
//! A round verifies proposals `x_0 .. x_{k-1}` against `k + 1` target rows:
//! row `j` is the target's distribution `p_j` for the position proposal `j`
//! claims, and row `k` the one past the last proposal. Walking the proposals
//! in order, proposal `j`, drawn from the draft's `q_j`, is kept with
//! probability `min(1, p_j(x_j) / q_j(x_j))`; the first one refused ends the
//! round with a draw from the residual `max(0, p_j - q_j)`, normalized, and a
//! round that keeps every proposal ends with a draw from row `k`. Either way
//! the round emits `kept + 1` tokens, each distributed as the target would
//! have sampled it after the ones before it.
//!
//! A drafter that only names its tokens (a block head's argmax, a prompt
//! lookup) drafted from a point mass: `q_j` is all on `x_j`. Then the rule
//! keeps `x_j` with probability `p_j(x_j)` and on a refusal draws from `p_j`
//! with `x_j` struck out — not the argmax rule, which is right only at
//! temperature zero, and not the full rule fed a guessed `q`, which is wrong
//! for any guess but the point mass. The point mass is exact for any drafter
//! that is deterministic given its context; one that samples must hand over
//! the distribution it sampled from.
//!
//! The distributions are the ones the target *samples* from — after
//! temperature, penalties, masks and truncation — not raw logits: the rule
//! preserves whichever distribution it is given. `verify_point_rows` is the
//! point-mass rule over what a device can land without reading back a
//! vocabulary row: each row's chance of its proposal and a residual draw.

/// A source of uniform draws in `[0, 1)`.
pub trait Uniform {
    fn uniform(&mut self) -> f64;
}

/// SplitMix64 (Steele, Lea and Flood 2014): a small, seedable generator, so a
/// loop's acceptance draws are reproducible from its request's seed.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

impl Uniform for SplitMix64 {
    fn uniform(&mut self) -> f64 {
        // The top 53 bits: every double in [0, 1) a multiple of 2^-53.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// What a verify kept: the first `kept` proposals, then `next`, the
/// target's token where the round stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub kept: usize,
    pub next: u32,
}

/// The round's verdict for proposals drawn from `draft` (row `j` is the
/// distribution proposal `j` was drawn from), under `target`'s `k + 1` rows.
///
/// # Panics
/// If `target` has no row past the proposals, or `draft` fewer rows than
/// there are proposals.
pub fn verify_sampled<D: AsRef<[f32]>, T: AsRef<[f32]>>(
    proposals: &[u32],
    draft: &[D],
    target: &[T],
    uniform: &mut impl Uniform,
) -> Verdict {
    assert!(
        target.len() > proposals.len(),
        "a verify needs a row past its proposals"
    );
    assert!(
        draft.len() >= proposals.len(),
        "every proposal needs its draft row"
    );
    for (j, &x) in proposals.iter().enumerate() {
        let (p, q) = (target[j].as_ref(), draft[j].as_ref());
        let (px, qx) = (chance(p, x), chance(q, x));
        // A proposal its own draft gave no mass was not drawn from it; treat
        // it as the point mass it must have been.
        let keep = if qx > 0.0 { (px / qx).min(1.0) } else { px };
        if uniform.uniform() < keep {
            continue;
        }
        let residual = |v: usize| {
            let qv = if qx > 0.0 {
                q.get(v).copied().map_or(0.0, f64::from)
            } else {
                f64::from(u8::from(v == x as usize))
            };
            (f64::from(p[v]) - qv).max(0.0)
        };
        return Verdict {
            kept: j,
            next: draw(p.len(), residual, p, uniform),
        };
    }
    let last = target[proposals.len()].as_ref();
    Verdict {
        kept: proposals.len(),
        next: draw(last.len(), |v| f64::from(last[v]), last, uniform),
    }
}

/// The round's verdict for proposals a drafter named without a distribution
/// (a point mass on each), under `target`'s `k + 1` rows.
///
/// # Panics
/// If `target` has no row past the proposals.
pub fn verify_point<T: AsRef<[f32]>>(
    proposals: &[u32],
    target: &[T],
    uniform: &mut impl Uniform,
) -> Verdict {
    assert!(
        target.len() > proposals.len(),
        "a verify needs a row past its proposals"
    );
    for (j, &x) in proposals.iter().enumerate() {
        let p = target[j].as_ref();
        if uniform.uniform() < chance(p, x) {
            continue;
        }
        let struck = |v: usize| {
            if v == x as usize {
                0.0
            } else {
                f64::from(p[v])
            }
        };
        return Verdict {
            kept: j,
            next: draw(p.len(), struck, p, uniform),
        };
    }
    let last = target[proposals.len()].as_ref();
    Verdict {
        kept: proposals.len(),
        next: draw(last.len(), |v| f64::from(last[v]), last, uniform),
    }
}

/// [`verify_point`] over what a verify fire can land per row instead of the
/// rows themselves: `chances[j]`, the target's probability of proposal `j`;
/// `residuals[j]`, a draw from row `j` with proposal `j` struck out; and
/// `bonus`, a draw from the row past the last proposal. The draws must be
/// independent of each other and of `uniform`, as a Gumbel-max draw per row
/// under its own noise is.
///
/// # Panics
/// If there are fewer chances or residual draws than proposals.
pub fn verify_point_rows(
    chances: &[f32],
    residuals: &[u32],
    bonus: u32,
    uniform: &mut impl Uniform,
) -> Verdict {
    let k = chances.len();
    assert!(
        residuals.len() >= k,
        "every proposal needs its residual draw"
    );
    for (j, &c) in chances.iter().enumerate() {
        if uniform.uniform() >= f64::from(c) {
            return Verdict {
                kept: j,
                next: residuals[j],
            };
        }
    }
    Verdict {
        kept: k,
        next: bonus,
    }
}

/// `row[token]`, or nothing for a token past the row.
fn chance(row: &[f32], token: u32) -> f64 {
    row.get(token as usize).copied().map_or(0.0, f64::from)
}

/// One draw from the unnormalized weights `weight(0..len)`; if they hold no
/// mass (a refusal floating-point rounding let through where `p` and `q`
/// agree), from `fallback`, which then is the target row itself.
fn draw(
    len: usize,
    weight: impl Fn(usize) -> f64,
    fallback: &[f32],
    uniform: &mut impl Uniform,
) -> u32 {
    let total: f64 = (0..len).map(&weight).sum();
    if total.is_nan() || total <= 0.0 {
        let total: f64 = fallback.iter().map(|&v| f64::from(v.max(0.0))).sum();
        return inverse(
            fallback.len(),
            |v| f64::from(fallback[v].max(0.0)),
            total,
            uniform,
        );
    }
    inverse(len, weight, total, uniform)
}

/// The inverse of the weights' running sum at one uniform draw; rounding
/// past the end lands on the last token with any mass.
fn inverse(
    len: usize,
    weight: impl Fn(usize) -> f64,
    total: f64,
    uniform: &mut impl Uniform,
) -> u32 {
    let target = uniform.uniform() * total;
    let mut sum = 0.0;
    let mut last = 0;
    for v in 0..len {
        let w = weight(v);
        if w <= 0.0 {
            continue;
        }
        sum += w;
        last = v;
        if target < sum {
            return v as u32;
        }
    }
    last as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trials per statistical check: a frequency's standard error is at most
    /// 0.0011, so the tolerance below is about five of them.
    const TRIALS: usize = 200_000;
    const TOLERANCE: f64 = 0.006;
    const VOCAB: usize = 5;

    fn normalized(weights: [f64; VOCAB]) -> Vec<f32> {
        let total: f64 = weights.iter().sum();
        weights.iter().map(|w| (w / total) as f32).collect()
    }

    /// A target and a draft that disagree, each conditioned on the token
    /// before: row `prev` of each is the distribution of what follows `prev`.
    fn target(prev: u32) -> Vec<f32> {
        let p = prev as f64;
        normalized([1.0 + p, 3.0, 0.5 + 0.5 * p, 2.0, 0.25])
    }

    fn draft(prev: u32) -> Vec<f32> {
        let p = prev as f64;
        normalized([2.0, 1.0 + 0.5 * p, 3.0, 0.1, 1.0 + p])
    }

    fn argmax(row: &[f32]) -> u32 {
        (0..row.len())
            .max_by(|&a, &b| row[a].total_cmp(&row[b]))
            .unwrap() as u32
    }

    fn sample(row: &[f32], uniform: &mut impl Uniform) -> u32 {
        draw(row.len(), |v| f64::from(row[v]), row, uniform)
    }

    /// The first `n` tokens after `start` a round loop emits, drafting `k`
    /// proposals a round, each drawn from the draft (`sampled`) or its
    /// argmax, and verifying them by the matching rule.
    fn generate(
        n: usize,
        k: usize,
        start: u32,
        uniform: &mut SplitMix64,
        sampled: bool,
    ) -> Vec<u32> {
        let mut text = vec![start];
        while text.len() <= n {
            let mut proposals = Vec::new();
            let mut drafts = Vec::new();
            let mut prev = *text.last().unwrap();
            for _ in 0..k {
                let q = draft(prev);
                let x = if sampled {
                    sample(&q, uniform)
                } else {
                    argmax(&q)
                };
                proposals.push(x);
                drafts.push(q);
                prev = x;
            }
            let mut rows = vec![target(*text.last().unwrap())];
            rows.extend(proposals.iter().map(|&x| target(x)));
            let verdict = if sampled {
                verify_sampled(&proposals, &drafts, &rows, uniform)
            } else {
                verify_point(&proposals, &rows, uniform)
            };
            text.extend_from_slice(&proposals[..verdict.kept]);
            text.push(verdict.next);
        }
        text[1..=n].to_vec()
    }

    /// The largest gap between the joint frequency of the first two tokens
    /// and the target's own `p(a) p(b | a)`.
    fn worst_gap_from_the_target(texts: impl Iterator<Item = Vec<u32>>) -> f64 {
        let mut counts = [0usize; VOCAB * VOCAB];
        let mut total = 0;
        for text in texts {
            counts[text[0] as usize * VOCAB + text[1] as usize] += 1;
            total += 1;
        }
        let first = target(0);
        let mut worst: f64 = 0.0;
        for a in 0..VOCAB {
            let second = target(a as u32);
            for b in 0..VOCAB {
                let want = f64::from(first[a]) * f64::from(second[b]);
                let got = counts[a * VOCAB + b] as f64 / total as f64;
                worst = worst.max((want - got).abs());
            }
        }
        worst
    }

    #[test]
    fn a_sampled_draft_emits_text_distributed_as_the_target() {
        let mut uniform = SplitMix64::new(0x5eed);
        let gap =
            worst_gap_from_the_target((0..TRIALS).map(|_| generate(2, 3, 0, &mut uniform, true)));
        assert!(gap < TOLERANCE, "{gap}");
    }

    #[test]
    fn a_greedy_draft_emits_text_distributed_as_the_target() {
        let mut uniform = SplitMix64::new(0xd1ce);
        let gap =
            worst_gap_from_the_target((0..TRIALS).map(|_| generate(2, 3, 0, &mut uniform, false)));
        assert!(gap < TOLERANCE, "{gap}");
    }

    #[test]
    fn a_greedy_draft_verified_by_the_argmax_rule_is_not() {
        // The control: the argmax rule every greedy loop verifies by keeps a
        // proposal only where it is the target's argmax and stops on the
        // argmax, which writes greedy text; the check above must see that.
        let gap = worst_gap_from_the_target((0..TRIALS).map(|_| {
            let mut text = Vec::new();
            let mut prev = 0;
            while text.len() < 2 {
                let x = argmax(&draft(prev));
                let truth = argmax(&target(prev));
                text.push(if truth == x { x } else { truth });
                prev = *text.last().unwrap();
            }
            text
        }));
        assert!(gap > 10.0 * TOLERANCE, "{gap}");
    }

    #[test]
    fn a_sampled_proposal_is_kept_as_often_as_the_two_distributions_overlap() {
        // Chen et al.: the first proposal survives with probability
        // sum over v of min(p(v), q(v)).
        let (p, q) = (target(0), draft(0));
        let overlap: f64 = p.iter().zip(&q).map(|(a, b)| f64::from(a.min(*b))).sum();
        let mut uniform = SplitMix64::new(7);
        let kept = (0..TRIALS)
            .filter(|_| {
                let x = sample(&q, &mut uniform);
                verify_sampled(&[x], &[&q], &[&p, &p], &mut uniform).kept == 1
            })
            .count();
        let rate = kept as f64 / TRIALS as f64;
        assert!((rate - overlap).abs() < TOLERANCE, "{rate} vs {overlap}");
    }

    #[test]
    fn the_device_landed_rows_decide_as_the_point_rule_does() {
        // Land what a verify fire would: each row's chance of its proposal,
        // a draw with the proposal struck out, and a draw past the last.
        let mut uniform = SplitMix64::new(0xfeed);
        let gap = worst_gap_from_the_target((0..TRIALS).map(|_| {
            let mut text = Vec::new();
            let mut prev = 0;
            while text.len() < 2 {
                let mut proposals = Vec::new();
                let mut p = prev;
                for _ in 0..2 {
                    p = argmax(&draft(p));
                    proposals.push(p);
                }
                let mut rows = vec![target(prev)];
                rows.extend(proposals.iter().map(|&x| target(x)));
                let chances: Vec<f32> = proposals
                    .iter()
                    .zip(&rows)
                    .map(|(&x, r)| r[x as usize])
                    .collect();
                let residuals: Vec<u32> = proposals
                    .iter()
                    .zip(&rows)
                    .map(|(&x, r)| {
                        let mut struck = r.clone();
                        struck[x as usize] = 0.0;
                        sample(&struck, &mut uniform)
                    })
                    .collect();
                let bonus = sample(&rows[2], &mut uniform);
                let verdict = verify_point_rows(&chances, &residuals, bonus, &mut uniform);
                text.extend_from_slice(&proposals[..verdict.kept]);
                text.push(verdict.next);
                prev = *text.last().unwrap();
            }
            text[..2].to_vec()
        }));
        assert!(gap < TOLERANCE, "{gap}");
    }

    #[test]
    fn a_draft_that_matches_the_target_is_always_kept() {
        let p = target(1);
        let mut uniform = SplitMix64::new(3);
        for _ in 0..1000 {
            let x = sample(&p, &mut uniform);
            let verdict = verify_sampled(&[x, x], &[&p, &p], &[&p, &p, &p], &mut uniform);
            assert_eq!(verdict.kept, 2);
        }
    }

    #[test]
    fn a_proposal_the_target_rules_out_is_never_kept_nor_drawn() {
        let p = vec![0.5, 0.0, 0.5];
        let mut uniform = SplitMix64::new(11);
        for _ in 0..1000 {
            let point = verify_point(&[1], &[&p, &p], &mut uniform);
            assert_eq!(point.kept, 0);
            assert_ne!(point.next, 1);
            let q = [0.1f32, 0.8, 0.1];
            let sampled = verify_sampled(&[1], &[&q], &[&p, &p], &mut uniform);
            assert_eq!(sampled.kept, 0);
            assert_ne!(sampled.next, 1);
        }
    }

    #[test]
    fn a_target_sure_of_the_proposal_keeps_it_and_draws_past_it() {
        let sure = vec![0.0, 0.0, 1.0];
        let mut uniform = SplitMix64::new(5);
        let verdict = verify_point(&[2], &[&sure, &sure], &mut uniform);
        assert_eq!(verdict, Verdict { kept: 1, next: 2 });
    }

    #[test]
    fn the_generator_is_reproducible_from_its_seed() {
        let (mut a, mut b) = (SplitMix64::new(42), SplitMix64::new(42));
        for _ in 0..16 {
            let u = a.uniform();
            assert!((0.0..1.0).contains(&u));
            assert_eq!(u.to_bits(), b.uniform().to_bits());
        }
    }
}
