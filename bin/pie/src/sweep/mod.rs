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

    /// Is this candidate faster than `other` by more than the two of them can
    /// explain by noise?
    ///
    /// The same rule the driver's own sweep uses after `fe8d85040`: combine the
    /// two candidates' spreads in quadrature and require the gap to clear it.
    /// Anything closer than that is a coin flip that will land the other way on
    /// the next run, and reporting it as a win is how a sweep produces
    /// confident garbage.
    pub fn beats(&self, other: &Round) -> bool {
        if !self.is_measurement() || !other.is_measurement() {
            return false;
        }
        let gap = (self.throughput_tok_s - other.throughput_tok_s)
            / other.throughput_tok_s.max(f64::EPSILON);
        let noise = (self.throughput_rel_sigma.powi(2) + other.throughput_rel_sigma.powi(2)).sqrt();
        gap > noise
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
    let run = fleet::run(addr, program, inputs).await;
    if run.failed_lanes() > 0 {
        anyhow::bail!(
            "{} of {} lanes failed during warmup; the fleet cannot run here at all",
            run.failed_lanes(),
            inputs.len()
        );
    }
    Ok(())
}

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
    // Once per candidate, not once per repeat: the knobs are what the repeats
    // hold fixed, and re-applying them between identical fleets would only add
    // a way for the sweep to fail.
    pie_engine::scheduler::reconfigure(knobs.frame_size, knobs.submit_depth, knobs.dispatch_depth)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("apply {knobs}"))?;

    let repeats = repeats.max(1);
    let mut throughputs = Vec::with_capacity(repeats);
    let mut p95s = Vec::with_capacity(repeats);
    let mut failed_lanes = 0;
    for _ in 0..repeats {
        let run = fleet::run(addr, program, inputs).await;
        throughputs.push(run.throughput_tok_s());
        p95s.push(run.lane_percentile_us(95) as f64);
        failed_lanes += run.failed_lanes();
    }

    let (throughput_tok_s, throughput_rel_sigma) = median_and_rel_sigma(&throughputs);
    let (lane_p95, _) = median_and_rel_sigma(&p95s);
    Ok(Round {
        knobs,
        throughput_tok_s,
        throughput_rel_sigma,
        lane_p95_us: lane_p95 as u128,
        failed_lanes,
        repeats,
    })
}

/// Every knob combination worth measuring, given the driver's staging bound.
///
/// Enumerated rather than searched, and the plan's §8 argues why: the feasible
/// set is small, each point is seconds, and enumeration has no surrogate model
/// to misfit or evaluation order to depend on. `steps_in_flight < staging_depth`
/// is what makes it small — at a staging depth of 13 there are only a few dozen
/// combinations, most of which the bound removes.
pub fn candidates(staging_depth: usize) -> Vec<Knobs> {
    let mut out = Vec::new();
    for frame_size in [1usize, 2, 3, 4] {
        for dispatch_depth in 1usize..=4 {
            if frame_size * dispatch_depth >= staging_depth {
                continue;
            }
            // `frame_submit_depth` must be at least 2 -- one frame runs while
            // the rest stay queued, and 1 leaves nothing queued. Its own field
            // doc has the argument, and `SchedulerConfig::validate` enforces it.
            for submit_depth in 2usize..=6 {
                out.push(Knobs {
                    frame_size,
                    submit_depth,
                    dispatch_depth,
                });
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
            lane_p95_us: 0,
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
        assert!(!a.beats(&b), "2.4% gap under 8.5% combined noise");

        // A 3.4x collapse is not ambiguous, and the rule must not be so
        // conservative that it misses one.
        let slow = round(BASE, 375.0, 0.06, 0);
        assert!(b.beats(&slow), "k=1 collapse must register");
    }

    #[test]
    fn quieter_measurements_resolve_smaller_gaps() {
        // The point of repeats: the same 2.4% gap becomes a real result once
        // the spread comes down.
        let a = round(BASE, 1296.0, 0.005, 0);
        let b = round(BASE, 1265.0, 0.005, 0);
        assert!(a.beats(&b));
    }

    #[test]
    fn a_failed_round_never_beats_anything() {
        // Fewer lanes finishing raises tokens per second, so a broken round
        // can post the best number. It must not be allowed to rank at all.
        let broken = round(BASE, 4000.0, 0.001, 2);
        let good = round(BASE, 1265.0, 0.01, 0);
        assert!(!broken.beats(&good));
        assert!(!good.beats(&broken));
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
