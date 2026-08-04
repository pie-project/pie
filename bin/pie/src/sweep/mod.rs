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

/// What one round measured.
pub struct Round {
    pub knobs: Knobs,
    pub throughput_tok_s: f64,
    pub lane_p95_us: u128,
    /// Lanes that returned nothing. Non-zero means this round is not a
    /// measurement, and the caller must not rank it against one that is.
    pub failed_lanes: usize,
}

impl Round {
    /// A round only ranks if every lane came back. A configuration that fails
    /// half its lanes can post an excellent tokens-per-second, because the
    /// tokens it did produce came out of a fleet that was half the size.
    pub fn is_measurement(&self) -> bool {
        self.failed_lanes == 0
    }
}

/// Set the knobs, run the load, report.
///
/// Every lane connects fresh and retires before this returns, so the next call
/// finds the engine idle and its `reconfigure` succeeds. A caller that leaves
/// guests running between rounds gets a refusal rather than a silently
/// mis-attributed measurement, which is the intended failure.
pub async fn measure(addr: &str, program: &str, inputs: &[String], knobs: Knobs) -> Result<Round> {
    pie_engine::scheduler::reconfigure(knobs.frame_size, knobs.submit_depth, knobs.dispatch_depth)
        .map_err(anyhow::Error::from)
        .with_context(|| format!("apply {knobs}"))?;

    let run = fleet::run(addr, program, inputs).await;
    Ok(Round {
        knobs,
        throughput_tok_s: run.throughput_tok_s(),
        lane_p95_us: run.lane_percentile_us(95),
        failed_lanes: run.failed_lanes(),
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

    #[test]
    fn a_round_with_a_dead_lane_does_not_rank() {
        // The trap this guards: fewer lanes finishing can RAISE tokens per
        // second, so a broken configuration can look like the winner.
        let good = Round {
            knobs: Knobs { frame_size: 2, submit_depth: 3, dispatch_depth: 2 },
            throughput_tok_s: 100.0,
            lane_p95_us: 10,
            failed_lanes: 0,
        };
        let broken = Round {
            failed_lanes: 1,
            throughput_tok_s: 400.0,
            ..Round {
                knobs: good.knobs,
                throughput_tok_s: 0.0,
                lane_p95_us: 0,
                failed_lanes: 0,
            }
        };
        assert!(good.is_measurement());
        assert!(!broken.is_measurement());
    }
}
