//! How far `--verify` lets a beam's score stand from its hypothesis rescored
//! alone.
//!
//! The beam's step and the rescoring fire compute the same log-probability
//! from logits the model stores in its readout format (`significand` bits,
//! the implicit one included: 8 for bf16, 24 for f32) and the guest reads
//! widened. The two fires run different kernels (the beam's rows together,
//! the rescore one row), whose f32 sums of the same logit round to the same
//! stored value or to its neighbour: they differ by at most one rounding
//! step, `step(x) = 2^(floor(log2 |x|) - (significand - 1))`.
//!
//! A step's log-probability is `x_t - LSE(x)`, `t` the chosen token. When
//! every logit `x_i` moves by `d_i`, `|d_i| <= step(x_i)`, it moves by
//! `d_t - sum_i p_i d_i = (1 - p_t) d_t - sum_{i != t} p_i d_i` to first
//! order, so by at most
//!
//! `(1 - p_t) step(x_t) + sum_{i != t} p_i step(x_i)`.
//!
//! The rescoring fire reads its `TOP` likeliest logits and their
//! probabilities, the largest |x| over the whole row, and the chosen
//! token's logit and probability; the terms the top logits do not hold are
//! bounded by their remaining mass times the step at that largest |x|. A
//! logit's step is taken at its magnitude grown by one step, so a neighbour
//! across a power of two is covered. Each step adds `F32_SLACK` for the
//! f32 arithmetic of the two log-softmaxes over the row, which run in
//! different orders. A beam's bound is the sum over its steps.
//!
//! The bound covers the readout's rounding only: two fires whose hidden
//! states differ by more than their own rounding (another beam's cells
//! read, a wrong parent's ancestry) move a step's log-probability by far
//! more, and `--verify` refuses them.

/// The logits the rescoring fire reads with their probabilities.
pub const TOP: u32 = 8;

/// A step's f32 arithmetic: two log-softmaxes over ~1.5e5 terms in
/// different orders differ by a few f32 ulps of the normaliser.
pub const F32_SLACK: f64 = 1e-5;

/// The rounding step of a float of `significand` bits at magnitude `x`;
/// the smallest normal f32's step at zero.
#[must_use]
pub fn step(significand: u32, x: f64) -> f64 {
    let x = x.abs().max(f64::from(f32::MIN_POSITIVE));
    2f64.powi(x.log2().floor() as i32 - (significand as i32 - 1))
}

/// `step` at a magnitude grown by one step: the larger of the two
/// neighbours' steps a value near a power of two may round to.
fn step_grown(significand: u32, x: f64) -> f64 {
    let x = x.abs();
    step(significand, x + step(significand, x))
}

/// What the rescoring fire read at one step of a hypothesis.
#[derive(Clone, Debug, Default)]
pub struct Rescored {
    /// The chosen token, its logit and its log-probability.
    pub token: u32,
    pub logit: f32,
    pub logprob: f32,
    /// The likeliest tokens, their logits and log-probabilities.
    pub top_ids: Vec<u32>,
    pub top_logits: Vec<f32>,
    pub top_logprobs: Vec<f32>,
    /// The largest |logit| over the row.
    pub max_abs: f32,
}

/// The bound on one step's log-probability between two fires that round
/// the same logits independently.
#[must_use]
pub fn step_bound(significand: u32, r: &Rescored) -> f64 {
    let p_t = f64::from(r.logprob).exp().min(1.0);
    let mut bound = (1.0 - p_t) * step_grown(significand, f64::from(r.logit));
    let mut held = 0.0;
    let mut chosen_held = false;
    for ((&id, &x), &lp) in r.top_ids.iter().zip(&r.top_logits).zip(&r.top_logprobs) {
        let p = f64::from(lp).exp();
        held += p;
        if id == r.token {
            chosen_held = true;
            continue;
        }
        bound += p * step_grown(significand, f64::from(x));
    }
    // The terms past the top ones; the chosen token's own term is not one.
    let rest = (1.0 - held - if chosen_held { 0.0 } else { p_t }).max(0.0);
    bound += rest * step_grown(significand, f64::from(r.max_abs));
    bound + F32_SLACK
}

/// The bound on a hypothesis' summed score: its steps' bounds.
#[must_use]
pub fn score_bound(significand: u32, steps: &[Rescored]) -> f64 {
    steps.iter().map(|r| step_bound(significand, r)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bf16_step_at_20_is_an_eighth() {
        assert_eq!(step(8, 20.0), 0.125);
        assert_eq!(step(8, -20.0), 0.125);
        assert_eq!(step(8, 16.0), 0.125);
        assert_eq!(step(8, 15.99), 0.0625);
        assert_eq!(step(8, 1.0), 2f64.powi(-7));
    }

    #[test]
    fn an_f32_step_is_tiny() {
        assert_eq!(step(24, 20.0), 2f64.powi(-19));
        assert!(step(24, 20.0) < 2e-6);
    }

    fn rescored(logit: f32, logprob: f32) -> Rescored {
        // A row whose chosen token holds `exp(logprob)`, the rest on two
        // others at -20 and 30 and the largest |x| 30.
        let p = logprob.exp();
        Rescored {
            token: 7,
            logit,
            logprob,
            top_ids: vec![7, 1, 2],
            top_logits: vec![logit, -20.0, 30.0],
            top_logprobs: vec![logprob, ((1.0 - p) / 2.0).ln(), ((1.0 - p) / 2.0).ln()],
            max_abs: 30.0,
        }
    }

    #[test]
    fn a_sure_token_bounds_nothing_but_the_slack() {
        let bound = step_bound(8, &rescored(20.0, 0.0));
        assert!((bound - F32_SLACK).abs() < 1e-9, "{bound}");
    }

    #[test]
    fn an_unsure_token_bounds_its_steps_by_mass() {
        // p_t = 1/2: half of the chosen step, a quarter each of -20's and
        // 30's (0.125 each), and the slack.
        let bound = step_bound(8, &rescored(20.0, 0.5f32.ln()));
        let want = 0.5 * 0.125 + 0.25 * 0.125 + 0.25 * 0.125 + F32_SLACK;
        assert!((bound - want).abs() < 1e-6, "{bound} vs {want}");
        // In f32 the same step is bounded by ~1e-5 alone.
        assert!(step_bound(24, &rescored(20.0, 0.5f32.ln())) < 2e-5);
    }

    /// A hypothesis compared against another beam's score: the bound,
    /// a few bf16 steps over six tokens, does not cover the nats between
    /// two beams' scores.
    #[test]
    fn another_beams_score_is_past_the_bound() {
        let steps: Vec<Rescored> = (0..6).map(|_| rescored(20.0, 0.9f32.ln())).collect();
        let bound = score_bound(8, &steps);
        let (mine, another) = (-0.1757f64, -2.3007f64);
        assert!((mine - another).abs() > bound, "{bound}");
        assert!(bound < 0.2, "{bound}");
    }
}
