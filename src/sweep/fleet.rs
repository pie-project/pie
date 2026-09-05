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

use client::client::Client;

/// One round's worth of load, and what it cost.
pub struct FleetRun {
    /// Per lane: the tokens it returned, or `None` if it produced none — which
    /// is how a lane that errored arrives here.
    pub outputs: Vec<Option<Vec<i64>>>,
    /// Per lane: wall time from launch to return.
    pub lane_latencies: Vec<Duration>,
    /// Wall time for the whole fleet, which is what throughput divides by.
    pub elapsed: Duration,
    /// Why the failed lanes failed, in lane order, deduplicated by the caller.
    ///
    /// A count alone sends the reader to the wrong place. `pie config tune`
    /// reported "64 of 64 lanes failed during warmup; is it in `pie inferlet
    /// list`?" for a program that WAS in the list, because every step of a
    /// lane discarded its error with `.ok()?` and only the count survived.
    pub failures: Vec<String>,
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
        let mut samples: Vec<_> = self
            .lane_latencies
            .iter()
            .map(Duration::as_micros)
            .collect();
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
async fn run_one(addr: &str, program: &str, input: &str) -> Result<Vec<i64>, String> {
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-sweep")
        .await
        .map_err(|e| format!("connect ws://{addr}/v1/ws: {e}"))?;
    client
        .authenticate("pie-sweep", &None)
        .await
        .map_err(|e| format!("authenticate: {e}"))?;
    let mut process = client
        .launch_process(program.to_string(), input.to_string(), true)
        .await
        .map_err(|e| format!("launch {program}: {e}"))?;
    let returned = process
        .wait_for_return()
        .await
        .map_err(|e| format!("{program} returned an error: {e}"))?;
    parse_tokens(&returned).ok_or_else(|| {
        // The guest ran and answered; its answer just has no token array. Show
        // what it said instead of calling it a missing program.
        let head: String = returned.chars().take(200).collect();
        format!("{program} returned no tokens: {head}")
    })
}

/// Pull the token array out of an inferlet's JSON return.
fn parse_tokens(json: &str) -> Option<Vec<i64>> {
    let lb = json.rfind('[')?;
    let rb = json[lb..].find(']')? + lb;
    let tokens: Vec<i64> = json[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}

/// Launch every lane at once and wait for all of them.
///
/// Concurrent rather than sequential because the quantity under measurement is
/// what the runtime does with a fleet: batch composition, frame overlap and
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
    let mut failures = Vec::new();
    for lane in lanes {
        let (result, latency) = lane
            .await
            .unwrap_or_else(|e| (Err(format!("lane task: {e}")), Duration::ZERO));
        match result {
            Ok(tokens) => outputs.push(Some(tokens)),
            Err(reason) => {
                outputs.push(None);
                failures.push(reason);
            }
        }
        lane_latencies.push(latency);
    }
    FleetRun {
        outputs,
        lane_latencies,
        elapsed: started.elapsed(),
        failures,
    }
}
