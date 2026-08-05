//! The measurement loop: one resident model, many rounds, load restarted each
//! time.
//!
//! The shape of this module is the conclusion of `.wiki/plan/config-optimize.md`
//! §5. An earlier design booted a process per candidate, which works and is even
//! fast — 1.03 s for a 1.4 GiB model — right up until the model is the size that
//! makes tuning worth doing, at which point a candidate costs minutes and the
//! whole approach has a model-size ceiling.
//!
//! So the expensive thing happens once. What restarts per round is the load,
//! which is milliseconds, and the knobs move in between through
//! `scheduler::reconfigure`.
//!
//! That ordering is also what makes the reconfigure legal: it refuses while any
//! guest is live, because `model.frame-size()` is cached for the life of a
//! program and a value already handed out cannot be recalled. A round that ends
//! by draining its lanes leaves exactly the state the next round needs.

pub mod fleet;

use anyhow::{Context, Result};

/// The batching knobs one round holds fixed.
///
/// Only the three that `scheduler::reconfigure` can move. The memory-lattice
/// knobs (`kv_page_size`, `max_forward_tokens`, `max_forward_requests`) are
/// fixed at boot and belong to the driver's own sweep — see the plan's §6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Knobs {
    pub frame_size: usize,
    pub submit_depth: usize,
    pub dispatch_depth: usize,
}

impl Knobs {
    /// Steps the driver's upload staging pool sees. The bound this feeds is
    /// `SchedulerConfig::validate`'s, and it is checked there rather than here
    /// so the constant has one home.
    pub fn steps_in_flight(&self) -> usize {
        self.frame_size * self.dispatch_depth
    }
}

impl std::fmt::Display for Knobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k={} submit={} dispatch={}",
            self.frame_size, self.submit_depth, self.dispatch_depth
        )
    }
}

/// What one candidate measured, across its repeats.
pub struct Round {
    pub knobs: Knobs,
    /// Median throughput across repeats. Median rather than mean because one
    /// slow fleet — a host hiccup, a stray process — should not move the
    /// estimate, and with a handful of repeats it easily would.
    pub throughput_tok_s: f64,
    /// Spread as a fraction of the median. This is the number that decides
    /// whether a difference between two candidates is real, so it is carried
    /// rather than discarded: at the serving level the noise floor measured
    /// ~6% on an L40S while the gap between k=2 and k=4 was ~2.4%, and a
    /// ranking that ignores that is reporting noise.
    pub throughput_rel_sigma: f64,
    /// Median of the per-repeat p95s.
    pub lane_p95_us: u128,
    /// Spread of the per-repeat p95s, as a fraction of the median. Carried for
    /// the same reason as the throughput spread: it is what decides whether a
    /// latency difference is real.
    pub lane_p95_rel_sigma: f64,
    /// Lanes that returned nothing, summed over repeats. Non-zero means this
    /// is not a measurement, and the caller must not rank it against one that
    /// is.
    pub failed_lanes: usize,
    pub repeats: usize,
}

impl Round {
    /// A round only ranks if every lane came back. A configuration that fails
    /// half its lanes can post an excellent tokens-per-second, because the
    /// tokens it did produce came out of a fleet that was half the size.
    pub fn is_measurement(&self) -> bool {
        self.failed_lanes == 0
    }

    /// Is this candidate better than `other`, on `metric`, by more than the two
    /// of them can explain by noise?
    ///
    /// The same rule the driver's own sweep uses after `fe8d85040`: combine the
    /// two candidates' spreads in quadrature and require the gap to clear it.
    /// Anything closer than that is a coin flip that will land the other way on
    /// the next run, and reporting it as a win is how a sweep produces
    /// confident garbage.
    ///
    /// `metric` is not cosmetic. Ranking a `--for latency` sweep on throughput
    /// was the state of this code before: the command asked for one thing and
    /// ordered its answers by another.
    pub fn beats(&self, other: &Round, metric: Metric) -> bool {
        if !self.is_measurement() || !other.is_measurement() {
            return false;
        }
        let (mine, theirs) = (metric.value(self), metric.value(other));
        let gap = if metric.higher_is_better() {
            (mine - theirs) / theirs.max(f64::EPSILON)
        } else {
            (theirs - mine) / theirs.max(f64::EPSILON)
        };
        let noise = (metric.sigma(self).powi(2) + metric.sigma(other).powi(2)).sqrt();
        gap > noise
    }
}

/// What a sweep ranks its candidates by.
///
/// One per objective, because the objective already names a serving shape and
/// the quantity that shape is judged on follows from it. There is no ranking
/// that serves both: latency and throughput pull opposite ways, which is why
/// `memory_profile` has two names in the first place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Aggregate tokens per second over the fleet. Higher wins.
    Throughput,
    /// 95th-percentile lane latency. Lower wins.
    LaneP95,
}

impl Metric {
    pub fn value(self, round: &Round) -> f64 {
        match self {
            Self::Throughput => round.throughput_tok_s,
            Self::LaneP95 => round.lane_p95_us as f64,
        }
    }

    pub fn sigma(self, round: &Round) -> f64 {
        match self {
            Self::Throughput => round.throughput_rel_sigma,
            Self::LaneP95 => round.lane_p95_rel_sigma,
        }
    }

    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Throughput)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Throughput => "throughput",
            Self::LaneP95 => "p95 lane latency",
        }
    }
}

/// Median of a sample set, and the spread as a fraction of it.
fn median_and_rel_sigma(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    if samples.len() < 2 || median <= 0.0 {
        return (median, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    (median, variance.sqrt() / median)
}

/// Run the load once and throw the result away.
///
/// **Not optional, and not test scaffolding.** Measured on an L40S: the first
/// round of a sweep came in at 844 tok/s and the identical configuration
/// measured 1228 tok/s when it ran again at the end — 45% apart, with the
/// candidates in between judged against whichever position they happened to
/// occupy. A sweep without this ranks round order.
///
/// It is a separate call rather than something [`measure`] does silently,
/// because a warmup folded into every round would pay the cost N times and
/// hide it. The first round is the one that is different; the rest are not.
pub async fn warmup(addr: &str, program: &str, inputs: &[String]) -> Result<()> {
    // Until two consecutive fleets agree, not a fixed count. One round was not
    // enough: with a single warmup the first MEASURED candidate absorbed what
    // was left and came back at +/-18.2% spread where every later candidate sat
    // near 2%. That candidate is the baseline everything else is ranked
    // against, so its inflated spread made `Round::beats` nearly unsatisfiable
    // and the sweep reported "nothing is faster" from a noisy reference rather
    // than from the machine.
    let mut previous: Option<f64> = None;
    for round in 0..MAX_WARMUP_ROUNDS {
        let run = fleet::run(addr, program, inputs).await;
        if run.failed_lanes() > 0 {
            anyhow::bail!(
                "{} of {} lanes failed during warmup; the fleet cannot run here at all",
                run.failed_lanes(),
                inputs.len()
            );
        }
        let rate = run.throughput_tok_s();
        if let Some(previous) = previous
            && (rate - previous).abs() / previous.max(f64::EPSILON) < WARMUP_SETTLED
        {
            return Ok(());
        }
        previous = Some(rate);
        let _ = round;
    }
    // Not an error. A machine that never settles still gets measured; what it
    // does not get is a claim that the measurement is tight, and the spread on
    // every round will say so.
    Ok(())
}

/// Two consecutive warmup fleets this close means the machine has settled.
const WARMUP_SETTLED: f64 = 0.05;
/// Give up warming and measure anyway. A busy host may never settle, and
/// refusing to measure it is worse than measuring it with an honest spread.
const MAX_WARMUP_ROUNDS: usize = 5;

/// Set the knobs, run the load, report.
///
/// Every lane connects fresh and retires before this returns, so the next call
/// finds the engine idle and its `reconfigure` succeeds. A caller that leaves
/// guests running between rounds gets a refusal rather than a silently
/// mis-attributed measurement, which is the intended failure.
pub async fn measure(
    addr: &str,
    program: &str,
    inputs: &[String],
    knobs: Knobs,
    repeats: usize,
) -> Result<Round> {
    // One candidate, so interleaving is a no-op and this is [`sweep_all`] with
    // a shorter list. Delegating rather than repeating the loop keeps ONE
    // definition of what a `Round` means: two aggregations that must agree
    // about medians, spreads and failed lanes is two chances to disagree.
    Ok(sweep_all(addr, program, inputs, &[knobs], repeats, |_, _| {})
        .await?
        .pop()
        .expect("one candidate in, one round out"))
}

/// Measure every candidate, interleaved.
///
/// One fleet per candidate per pass, cycling, rather than all of a candidate's
/// fleets back to back. Batched repeats share whatever state the machine is in
/// for those few seconds, so they measure the WITHIN-BURST variation and report
/// it as the uncertainty — which made `Round::beats` over-confident by the
/// difference. Measured: a candidate's own reported spread was 1.2-2.7% while
/// the same knobs re-measured at the end of the sweep landed 3.4% away.
///
/// Interleaving spreads each candidate's fleets across the whole run, so a slow
/// stretch of machine time lands on every candidate instead of on whichever one
/// happened to occupy it — and the spread it reports is the one that matters.
pub async fn sweep_all(
    addr: &str,
    program: &str,
    inputs: &[String],
    candidates: &[Knobs],
    repeats: usize,
    mut on_pass: impl FnMut(usize, usize),
) -> Result<Vec<Round>> {
    let repeats = repeats.max(1);
    let mut throughputs: Vec<Vec<f64>> = vec![Vec::with_capacity(repeats); candidates.len()];
    let mut p95s: Vec<Vec<f64>> = vec![Vec::with_capacity(repeats); candidates.len()];
    let mut failed = vec![0usize; candidates.len()];

    for pass in 0..repeats {
        for (index, knobs) in candidates.iter().enumerate() {
            pie_engine::scheduler::reconfigure(
                knobs.frame_size,
                knobs.submit_depth,
                knobs.dispatch_depth,
            )
            .map_err(anyhow::Error::from)
            .with_context(|| format!("apply {knobs}"))?;
            let run = fleet::run(addr, program, inputs).await;
            throughputs[index].push(run.throughput_tok_s());
            p95s[index].push(run.lane_percentile_us(95) as f64);
            failed[index] += run.failed_lanes();
        }
        on_pass(pass + 1, repeats);
    }

    Ok(candidates
        .iter()
        .enumerate()
        .map(|(index, knobs)| {
            let (throughput_tok_s, throughput_rel_sigma) = median_and_rel_sigma(&throughputs[index]);
            let (lane_p95, lane_p95_rel_sigma) = median_and_rel_sigma(&p95s[index]);
            Round {
                knobs: *knobs,
                throughput_tok_s,
                throughput_rel_sigma,
                lane_p95_us: lane_p95 as u128,
                lane_p95_rel_sigma,
                failed_lanes: failed[index],
                repeats,
            }
        })
        .collect())
}

/// Every knob combination worth measuring, given the driver's staging bound.
///
/// Enumerated rather than searched, and the plan's §8 argues why: the feasible
/// set is small, each point is seconds, and enumeration has no surrogate model
/// to misfit or evaluation order to depend on. `steps_in_flight < staging_depth`
/// is what makes it small — at a staging depth of 13 there are only a few dozen
/// combinations, most of which the bound removes.
pub fn candidates(staging_depth: usize) -> Vec<Knobs> {
    let mut groups: Vec<Vec<Knobs>> = Vec::new();
    for frame_size in [1usize, 2, 3, 4] {
        let mut group = Vec::new();
        for dispatch_depth in 1usize..=4 {
            // `submit_depth` does not enter the bound, so any value answers it.
            let shape = Knobs {
                frame_size,
                submit_depth: 2,
                dispatch_depth,
            };
            if shape.steps_in_flight() >= staging_depth {
                continue;
            }
            // `frame_submit_depth` must be at least 2 -- one frame runs while
            // the rest stay queued, and 1 leaves nothing queued. Its own field
            // doc has the argument, and `SchedulerConfig::validate` enforces it.
            for submit_depth in 2usize..=6 {
                group.push(Knobs {
                    frame_size,
                    submit_depth,
                    dispatch_depth,
                });
            }
        }
        groups.push(group);
    }

    // Round-robin across frame sizes rather than lexicographic order, because
    // `--budget` truncates this list and a lexicographic one spends the whole
    // budget in a single corner. Measured: a budget of six explored k=1 five
    // times and nothing else, so the report ranked one axis and called it a
    // sweep. Interleaved, the same six touch every k.
    let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = Vec::new();
    for index in 0..longest {
        for group in &groups {
            if let Some(knobs) = group.get(index) {
                out.push(*knobs);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_staging_bound_removes_candidates_rather_than_the_engine_rejecting_them_later() {
        // Generating a candidate the engine will refuse wastes a round and
        // reports it as a failure, which is indistinguishable from a real one.
        let candidates = candidates(13);
        assert!(!candidates.is_empty());
        for knobs in &candidates {
            assert!(
                knobs.steps_in_flight() < 13,
                "{knobs} would block on the staging pool"
            );
            assert!(knobs.submit_depth >= 2, "{knobs} leaves nothing queued");
        }
    }

    #[test]
    fn a_budget_sees_every_frame_size_before_it_sees_a_second_of_any() {
        // A lexicographic list spent a budget of six entirely on k=1. The
        // report then ranked one axis and presented it as a sweep.
        let candidates = candidates(13);
        let first_four: Vec<usize> = candidates.iter().take(4).map(|k| k.frame_size).collect();
        let mut distinct = first_four.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            first_four.len(),
            "the first four candidates repeat a frame size: {first_four:?}"
        );
    }

    #[test]
    fn a_narrower_staging_pool_narrows_the_search() {
        // The bound is the driver's, so a driver with a different pool gets a
        // different space rather than the same space and a runtime error.
        let wide = candidates(13);
        let narrow = candidates(5);
        assert!(narrow.len() < wide.len());
        assert!(narrow.iter().all(|k| k.steps_in_flight() < 5));
    }

    fn round(knobs: Knobs, tok_s: f64, rel_sigma: f64, failed: usize) -> Round {
        Round {
            knobs,
            throughput_tok_s: tok_s,
            throughput_rel_sigma: rel_sigma,
            lane_p95_us: 1_000,
            lane_p95_rel_sigma: rel_sigma,
            failed_lanes: failed,
            repeats: 3,
        }
    }

    const BASE: Knobs = Knobs { frame_size: 2, submit_depth: 3, dispatch_depth: 2 };

    #[test]
    fn a_gap_smaller_than_the_noise_is_not_a_win() {
        // The measured case: 6% round-to-round noise against a 2.4% gap between
        // k=2 and k=4. Calling that a win produces a ranking that lands the
        // other way on the next run.
        let a = round(BASE, 1296.0, 0.06, 0);
        let b = round(BASE, 1265.0, 0.06, 0);
        assert!(!a.beats(&b, Metric::Throughput), "2.4% gap under 8.5% combined noise");

        // A 3.4x collapse is not ambiguous, and the rule must not be so
        // conservative that it misses one.
        let slow = round(BASE, 375.0, 0.06, 0);
        assert!(b.beats(&slow, Metric::Throughput), "k=1 collapse must register");
    }

    #[test]
    fn quieter_measurements_resolve_smaller_gaps() {
        // The point of repeats: the same 2.4% gap becomes a real result once
        // the spread comes down.
        let a = round(BASE, 1296.0, 0.005, 0);
        let b = round(BASE, 1265.0, 0.005, 0);
        assert!(a.beats(&b, Metric::Throughput));
    }

    #[test]
    fn a_failed_round_never_beats_anything() {
        // Fewer lanes finishing raises tokens per second, so a broken round
        // can post the best number. It must not be allowed to rank at all.
        let broken = round(BASE, 4000.0, 0.001, 2);
        let good = round(BASE, 1265.0, 0.01, 0);
        assert!(!broken.beats(&good, Metric::Throughput));
        assert!(!good.beats(&broken, Metric::Throughput));
    }

    #[test]
    fn the_median_ignores_one_bad_fleet() {
        // A host hiccup in one repeat should move the spread, not the estimate.
        let (median, sigma) = median_and_rel_sigma(&[1200.0, 1210.0, 400.0]);
        assert_eq!(median, 1200.0);
        assert!(sigma > 0.3, "the outlier has to show up somewhere: {sigma}");
    }

    #[test]
    fn a_round_with_a_dead_lane_does_not_rank() {
        assert!(round(BASE, 100.0, 0.01, 0).is_measurement());
        assert!(!round(BASE, 400.0, 0.01, 1).is_measurement());
    }
}
