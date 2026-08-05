//! Synthetic load: N lanes driven through the real client edge.
//!
//! Promoted out of `tests/cuda_contention.rs` rather than written fresh. That
//! matters twice over: it is the load generator with the most measurement
//! mileage on it, and `cuda_contention` now consumes this module, so the
//! measurement path stays covered by a test that already exists instead of
//! drifting away from a copy.
//!
//! Load goes over the websocket edge on purpose. The sweep exists to tell an
//! operator what their deployment will do, and a deployment is reached through
//! that edge; measuring an in-process shortcut would measure something nobody
//! runs.

use std::time::{Duration, Instant};

use pie_client::client::Client;

/// One round's worth of load, and what it cost.
pub struct FleetRun {
    /// Per lane: the tokens it returned, or `None` if it produced none — which
    /// is how a lane that errored arrives here.
    pub outputs: Vec<Option<Vec<i64>>>,
    /// Per lane: wall time from launch to return.
    pub lane_latencies: Vec<Duration>,
    /// Wall time for the whole fleet, which is what throughput divides by.
    pub elapsed: Duration,
}

impl FleetRun {
    pub fn total_tokens(&self) -> usize {
        self.outputs
            .iter()
            .filter_map(Option::as_ref)
            .map(Vec::len)
            .sum()
    }

    /// Lanes that returned nothing. A round with any of these is not a
    /// measurement — it is a failure that happens to have a duration.
    pub fn failed_lanes(&self) -> usize {
        self.outputs.iter().filter(|o| o.is_none()).count()
    }

    pub fn throughput_tok_s(&self) -> f64 {
        self.total_tokens() as f64 / self.elapsed.as_secs_f64().max(1e-9)
    }

    /// Lane latency percentile in microseconds. Nearest-rank; `p` is 0..=100.
    pub fn lane_percentile_us(&self, p: usize) -> u128 {
        if self.lane_latencies.is_empty() {
            return 0;
        }
        let mut samples: Vec<_> = self.lane_latencies.iter().map(Duration::as_micros).collect();
        samples.sort_unstable();
        let index = ((samples.len() - 1) * p).div_ceil(100);
        samples[index]
    }
}

/// Run one lane to completion: connect, launch, wait, take the tokens.
///
/// A fresh connection per lane, and therefore a fresh guest. That is not
/// incidental — `scheduler::reconfigure` refuses while any guest is live,
/// because `model.frame-size()` is cached for the life of a program. Rounds
/// that reused guests could not change the knobs they exist to compare.
async fn run_one(addr: &str, program: &str, input: &str) -> Option<Vec<i64>> {
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-sweep")
        .await
        .ok()?;
    client.authenticate("pie-sweep", &None).await.ok()?;
    let mut process = client
        .launch_process(program.to_string(), input.to_string(), true)
        .await
        .ok()?;
    parse_tokens(&process.wait_for_return().await.ok()?)
}

/// Pull the token array out of an inferlet's JSON return.
fn parse_tokens(json: &str) -> Option<Vec<i64>> {
    let lb = json.rfind('[')?;
    let rb = json[lb..].find(']')? + lb;
    let tokens: Vec<i64> = json[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if tokens.is_empty() { None } else { Some(tokens) }
}

/// Launch every lane at once and wait for all of them.
///
/// Concurrent rather than sequential because the quantity under measurement is
/// what the engine does with a fleet: batch composition, frame overlap and
/// admission all only exist when lanes contend.
pub async fn run(addr: &str, program: &str, inputs: &[String]) -> FleetRun {
    let started = Instant::now();
    let mut lanes = Vec::with_capacity(inputs.len());
    for input in inputs {
        let addr = addr.to_string();
        let program = program.to_string();
        let input = input.clone();
        lanes.push(tokio::spawn(async move {
            let lane_started = Instant::now();
            (
                run_one(&addr, &program, &input).await,
                lane_started.elapsed(),
            )
        }));
    }
    let mut outputs = Vec::with_capacity(lanes.len());
    let mut lane_latencies = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let (output, latency) = lane.await.unwrap_or((None, Duration::ZERO));
        outputs.push(output);
        lane_latencies.push(latency);
    }
    FleetRun {
        outputs,
        lane_latencies,
        elapsed: started.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_of(latencies_us: &[u64], tokens: &[usize], elapsed_us: u64) -> FleetRun {
        FleetRun {
            outputs: tokens
                .iter()
                .map(|&n| if n == 0 { None } else { Some(vec![1i64; n]) })
                .collect(),
            lane_latencies: latencies_us
                .iter()
                .map(|&us| Duration::from_micros(us))
                .collect(),
            elapsed: Duration::from_micros(elapsed_us),
        }
    }

    #[test]
    fn throughput_counts_tokens_over_fleet_wall_time() {
        // Not the sum of lane times: the lanes overlap, and what an operator
        // gets is what came out per second of wall clock.
        let run = run_of(&[1_000, 2_000], &[10, 30], 2_000_000);
        assert_eq!(run.total_tokens(), 40);
        assert!((run.throughput_tok_s() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn a_lane_that_returned_nothing_is_counted_rather_than_averaged_away() {
        // A round with failures still produces a throughput number, and that
        // number looks fine. The count is what stops it being reported as one.
        let run = run_of(&[1_000, 1_000], &[10, 0], 1_000_000);
        assert_eq!(run.failed_lanes(), 1);
        assert_eq!(run.total_tokens(), 10);
    }

    #[test]
    fn percentiles_are_nearest_rank_over_sorted_lanes() {
        let run = run_of(&[500, 100, 400, 200, 300], &[1; 5], 1_000);
        assert_eq!(run.lane_percentile_us(0), 100);
        assert_eq!(run.lane_percentile_us(50), 300);
        assert_eq!(run.lane_percentile_us(100), 500);
    }

    #[test]
    fn tokens_come_out_of_the_trailing_array() {
        assert_eq!(parse_tokens("{\"t\": [1, 2, 3]}"), Some(vec![1, 2, 3]));
        assert_eq!(parse_tokens("no array here"), None);
        assert_eq!(parse_tokens("[]"), None);
    }
}
