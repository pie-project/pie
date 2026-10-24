//! How many rows of a drafted block a round verifies, from the draft's
//! confidence at each position: Splash 1.4.0's verify budget
//! (`model::VerifyBudget`), for one lane.
//!
//! Once the proposals before it are accepted, the target accepts proposal `j`
//! with probability `p_j`, where `logit(p_j) = slope * logit(c_j) +
//! intercept[j]` and `c_j` is the draft's confidence at `j`. A lane's survival
//! at `k`, the product of `p_j` up to it, is what verifying one more row adds
//! to the tokens a round emits, so a round of `w` rows (the anchor and `w - 1`
//! proposals) emits `1 + sum of survivals below w` on average. Acceptance is
//! bimodal (some rounds keep the whole block, many keep nothing), so no one
//! width serves every round.
//!
//! Splash verifies the width that emits the most tokens per unit of the
//! round's own cost. A lane wants the most tokens per second over all its
//! rounds, which a round serves by verifying one more row while that row's
//! survival is worth more than the row's price at the lane's own rate, the
//! tokens it emitted per nanosecond of its latest rounds: the width
//! maximizing `expected - rate * cost`. The round's own ratio stops a round
//! that drafted well too early, and runs one that drafted badly too long.
//! Before any round, the round's own ratio stands in.
//!
//! Each round labels the proposals it verified, up to the first the target
//! rejected, and the model is refit on the latest labels by damped Newton
//! steps under a Gaussian prior, which also stands before any label. A lane
//! lives one request, a few hundred labels, so it refits more often than
//! Splash's engine-wide model does. Every `EXPLORE`-th round verifies every
//! row, so the labels and the prices keep reaching the far end of the block.

use std::collections::VecDeque;

/// A proposal the target accepted or rejected, at `position`, whose
/// confidence had logit `feature`.
#[derive(Clone, Copy, Debug)]
struct Label {
    position: usize,
    feature: f64,
    accepted: bool,
}

pub struct Budget {
    /// `[slope, intercept_0, intercept_1, ...]`.
    theta: Vec<f64>,
    prior: Vec<f64>,
    labels: VecDeque<Label>,
    rounds: u64,
    /// The latest rounds' tokens and nanoseconds, each weighed `DECAY` times
    /// the next's.
    tokens: f64,
    time: f64,
}

impl Budget {
    /// Every this many rounds, the next verifies every row.
    pub const EXPLORE: u64 = 32;
    /// Rounds between refits.
    pub const REFIT: u64 = 16;
    /// The latest labels a refit reads.
    pub const LABELS: usize = 4096;
    const ITERATIONS: usize = 16;
    /// About the last thirty rounds set the lane's rate.
    const DECAY: f64 = 0.97;
    /// The prior's precision on the slope and on each intercept: about twenty
    /// labels' worth each, as Splash's.
    const SLOPE_PRECISION: f64 = 50.0;
    const INTERCEPT_PRECISION: f64 = 4.0;
    /// Splash's fit of its measured targets' cycles (Qwen3.6-35B-A3B among
    /// them); a position past these takes the last.
    const PRIOR_SLOPE: f64 = 0.7;
    const PRIOR_INTERCEPTS: [f64; 7] = [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45];

    /// A budget for a block of `positions` proposals.
    pub fn new(positions: usize) -> Self {
        let mut prior = vec![Self::PRIOR_SLOPE];
        prior.extend(
            (0..positions.max(1))
                .map(|j| Self::PRIOR_INTERCEPTS[j.min(Self::PRIOR_INTERCEPTS.len() - 1)]),
        );
        Budget {
            theta: prior.clone(),
            prior,
            labels: VecDeque::new(),
            rounds: 0,
            tokens: 0.0,
            time: 0.0,
        }
    }

    fn positions(&self) -> usize {
        self.theta.len() - 1
    }

    fn precision(parameter: usize) -> f64 {
        if parameter == 0 {
            Self::SLOPE_PRECISION
        } else {
            Self::INTERCEPT_PRECISION
        }
    }

    /// `None` for a confidence that is not a probability (a draft row that
    /// is not finite), which labels nothing.
    fn feature(confidence: f32) -> Option<f64> {
        let c = f64::from(confidence);
        if !(0.0..=1.0).contains(&c) {
            return None;
        }
        let c = c.clamp(1e-6, 1.0 - 1e-6);
        Some((c / (1.0 - c)).ln())
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// The probability the target accepts the proposal at `position`, the
    /// ones before it accepted. An unreadable confidence reads as an even one.
    pub fn accept(&self, position: usize, confidence: f32) -> f64 {
        let j = position.min(self.positions() - 1);
        let feature = Self::feature(confidence).unwrap_or(0.0);
        Self::sigmoid(self.theta[0] * feature + self.theta[1 + j])
    }

    /// The tokens a round of `width` rows emits on average: the anchor's,
    /// and each proposal's survival.
    pub fn expected(&self, confidence: &[f32], width: u32) -> f64 {
        1.0 + self.survival(confidence, width).iter().sum::<f64>()
    }

    /// The chance a round of `width` rows keeps each of its proposals and
    /// every one before it: what a lane bids for the runtime's verify rows.
    pub fn survival(&self, confidence: &[f32], width: u32) -> Vec<f64> {
        let mut survival = 1.0;
        confidence
            .iter()
            .take(width.saturating_sub(1) as usize)
            .enumerate()
            .map(|(j, &c)| {
                survival *= self.accept(j, c);
                survival
            })
            .collect()
    }

    /// Whether this round verifies every row.
    pub fn explores(&self) -> bool {
        self.rounds % Self::EXPLORE == Self::EXPLORE - 1
    }

    /// `[slope, intercepts...]`, as last fit.
    pub fn model(&self) -> &[f64] {
        &self.theta
    }

    /// The tokens a nanosecond the lane's latest rounds emitted; `None`
    /// before any.
    pub fn rate(&self) -> Option<f64> {
        (self.time > 0.0).then(|| self.tokens / self.time)
    }

    /// The width, from 1 to `widest`, worth the most at the lane's rate,
    /// `cost` being the price of a round of that many rows. A round with no
    /// price yet, or an exploring one, verifies every row.
    pub fn width(&self, confidence: &[f32], widest: u32, cost: impl Fn(u32) -> Option<f64>) -> u32 {
        if self.explores() {
            return widest;
        }
        let rate = self.rate();
        let mut best = (widest, f64::NEG_INFINITY);
        for width in 1..=widest {
            let Some(price) = cost(width).filter(|p| *p > 0.0) else {
                return widest;
            };
            let expected = self.expected(confidence, width);
            let worth = match rate {
                Some(rate) => expected - rate * price,
                None => expected / price,
            };
            // Strictly better: a tie keeps the narrower width.
            if worth > best.1 {
                best = (width, worth);
            }
        }
        best.0
    }

    /// A completed round: its confidences, the rows it verified, the
    /// proposals the target accepted before its first rejection, and the
    /// nanoseconds its fires took.
    pub fn observe(&mut self, confidence: &[f32], width: u32, kept: u32, ns: f64) {
        if ns.is_finite() && ns > 0.0 {
            self.tokens = Self::DECAY * self.tokens + f64::from(kept + 1);
            self.time = Self::DECAY * self.time + ns;
        }
        let verified = (width.saturating_sub(1) as usize).min(confidence.len());
        let kept = kept as usize;
        // Proposal j is checked against the target's row j: those past the
        // first rejection, or past the verify, say nothing.
        for (position, &c) in confidence.iter().enumerate().take(verified.min(kept + 1)) {
            let Some(feature) = Self::feature(c) else {
                continue;
            };
            if self.labels.len() == Self::LABELS {
                self.labels.pop_front();
            }
            self.labels.push_back(Label {
                position: position.min(self.positions() - 1),
                feature,
                accepted: position < kept,
            });
        }
        self.rounds += 1;
        if self.rounds.is_multiple_of(Self::REFIT) && !self.labels.is_empty() {
            self.theta = self.fit();
        }
    }

    /// The negative log posterior, up to a constant.
    fn objective(&self, theta: &[f64]) -> f64 {
        let mut value = 0.0;
        for label in &self.labels {
            let z = theta[0] * label.feature + theta[1 + label.position];
            // log(1 + e^-margin), stably.
            let margin = if label.accepted { z } else { -z };
            value += if margin > 0.0 {
                (-margin).exp().ln_1p()
            } else {
                -margin + margin.exp().ln_1p()
            };
        }
        for (i, (t, p)) in theta.iter().zip(&self.prior).enumerate() {
            value += 0.5 * Self::precision(i) * (t - p) * (t - p);
        }
        value
    }

    /// The maximum a posteriori model of the labels, by damped Newton steps
    /// from the current one.
    fn fit(&self) -> Vec<f64> {
        let n = self.theta.len();
        let mut theta = self.theta.clone();
        let mut current = self.objective(&theta);
        for _ in 0..Self::ITERATIONS {
            let mut gradient = vec![0.0; n];
            let mut hessian = vec![vec![0.0; n]; n];
            for label in &self.labels {
                let b = 1 + label.position;
                let p = Self::sigmoid(theta[0] * label.feature + theta[b]);
                let residual = p - if label.accepted { 1.0 } else { 0.0 };
                let weight = p * (1.0 - p);
                gradient[0] += residual * label.feature;
                gradient[b] += residual;
                hessian[0][0] += weight * label.feature * label.feature;
                hessian[0][b] += weight * label.feature;
                hessian[b][0] += weight * label.feature;
                hessian[b][b] += weight;
            }
            for i in 0..n {
                gradient[i] += Self::precision(i) * (theta[i] - self.prior[i]);
                hessian[i][i] += Self::precision(i);
            }
            let step = solve(hessian, gradient);
            // Damped: the Newton step, halved until the objective does not grow.
            let mut scale = 1.0;
            let mut moved = None;
            for _ in 0..30 {
                let next: Vec<f64> = theta
                    .iter()
                    .zip(&step)
                    .map(|(t, s)| t - scale * s)
                    .collect();
                let value = self.objective(&next);
                if value <= current {
                    moved = Some((next, value));
                    break;
                }
                scale *= 0.5;
            }
            let Some((next, value)) = moved else {
                break;
            };
            let largest = next
                .iter()
                .zip(&theta)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            theta = next;
            current = value;
            if largest < 1e-9 {
                break;
            }
        }
        theta
    }
}

/// `x` with `h x = g`, `h` positive definite: Gaussian elimination with
/// partial pivoting.
fn solve(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Vec<f64> {
    let n = g.len();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&a, &b| h[a][column].abs().total_cmp(&h[b][column].abs()))
            .unwrap_or(column);
        h.swap(column, pivot);
        g.swap(column, pivot);
        for row in column + 1..n {
            let factor = h[row][column] / h[column][column];
            for k in column..n {
                h[row][k] -= factor * h[column][k];
            }
            g[row] -= factor * g[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = g[row] - (row + 1..n).map(|k| h[row][k] * x[k]).sum::<f64>();
        x[row] = sum / h[row][row];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round of `rows` rows prices as Pie's M4 Pro rounds of a 35B-A3B do:
    /// a draft fire, and a verify that adds about 0.85 ms a row.
    fn cost(rows: u32) -> Option<f64> {
        Some(8.0 + 12.4 + 0.85 * f64::from(rows))
    }

    fn fresh() -> Budget {
        Budget::new(7)
    }

    #[test]
    fn a_confident_draft_verifies_the_whole_block() {
        assert_eq!(fresh().width(&[0.99; 7], 8, cost), 8);
    }

    #[test]
    fn an_unsure_draft_verifies_a_few_rows() {
        let width = fresh().width(&[0.15; 7], 8, cost);
        assert!(width <= 3, "{width}");
    }

    #[test]
    fn the_width_stops_where_the_draft_loses_its_confidence() {
        let confidence = [0.98, 0.97, 0.96, 0.05, 0.05, 0.05, 0.05];
        assert_eq!(fresh().width(&confidence, 8, cost), 4);
    }

    #[test]
    fn dearer_rows_narrow_the_verify() {
        let confidence = [0.8; 7];
        let cheap = fresh().width(&confidence, 8, cost);
        let dear = fresh().width(&confidence, 8, |rows| Some(8.0 + 4.0 * f64::from(rows)));
        assert!(dear < cheap, "{dear} vs {cheap}");
    }

    #[test]
    fn a_lane_that_runs_fast_verifies_only_rows_worth_its_rate() {
        // The same middling draft, in a lane that emits 1 token a round of
        // 21 ms and in one that emits 5 a round of 27 ms.
        let confidence = [0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4];
        let (mut slow, mut fast) = (fresh(), fresh());
        for _ in 0..30 {
            slow.observe(&[], 1, 0, 21.0);
            fast.observe(&[], 8, 4, 27.0);
        }
        let (narrow, wide) = (
            fast.width(&confidence, 8, cost),
            slow.width(&confidence, 8, cost),
        );
        assert!(narrow < wide, "{narrow} vs {wide}");
    }

    #[test]
    fn every_thirty_second_round_verifies_every_row() {
        let mut budget = fresh();
        let mut explored = Vec::new();
        for round in 0..96u64 {
            if budget.width(&[0.05; 7], 8, cost) == 8 {
                explored.push(round);
            }
            budget.observe(&[0.05; 7], 2, 0, 21.0);
        }
        assert_eq!(explored, [31, 63, 95]);
    }

    #[test]
    fn an_unpriced_width_verifies_every_row() {
        assert_eq!(
            fresh().width(&[0.05; 7], 8, |rows| (rows == 8).then_some(1.0)),
            8
        );
    }

    #[test]
    fn labels_stop_at_the_first_rejection_and_the_verify() {
        let mut budget = fresh();
        budget.observe(&[0.9; 7], 8, 2, 27.0);
        let accepted: Vec<bool> = budget.labels.iter().map(|l| l.accepted).collect();
        assert_eq!(accepted, [true, true, false]);
        budget.observe(&[0.9; 7], 3, 2, 23.0);
        assert_eq!(
            budget.labels.len(),
            5,
            "a round of three rows labels two proposals"
        );
        budget.observe(&[f32::NAN; 7], 8, 7, 27.0);
        assert_eq!(
            budget.labels.len(),
            5,
            "an unreadable confidence labels nothing"
        );
    }

    #[test]
    fn the_fit_learns_a_draft_that_is_wrong_when_it_is_sure() {
        let mut budget = fresh();
        let (before, wide) = (budget.accept(0, 0.95), budget.width(&[0.95; 7], 8, cost));
        for _ in 0..64 {
            budget.observe(&[0.95; 7], 8, 0, 27.0);
        }
        let (after, narrow) = (budget.accept(0, 0.95), budget.width(&[0.95; 7], 8, cost));
        assert!(after < before - 0.2, "{before} -> {after}");
        assert!(narrow < wide, "{wide} -> {narrow}");
    }

    #[test]
    fn the_fit_recovers_the_model_its_labels_were_drawn_from() {
        // A small LCG, so the test needs no crate.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let truth = [1.2, 0.5, 0.0, -0.5, -0.5, -0.5, -0.5, -0.5];
        let mut budget = fresh();
        for _ in 0..1024 {
            let confidence: Vec<f32> = (0..7)
                .map(|_| Budget::sigmoid(4.0 * uniform() - 2.0) as f32)
                .collect();
            let mut kept = 0;
            for (j, &c) in confidence.iter().enumerate() {
                let z = truth[0] * Budget::feature(c).unwrap() + truth[1 + j];
                if uniform() < Budget::sigmoid(z) {
                    kept += 1;
                } else {
                    break;
                }
            }
            budget.observe(&confidence, 8, kept, 27.0);
        }
        assert!(
            (budget.theta[0] - truth[0]).abs() < 0.15,
            "{:?}",
            budget.theta
        );
        assert!(
            (budget.theta[1] - truth[1]).abs() < 0.3,
            "{:?}",
            budget.theta
        );
    }
}
