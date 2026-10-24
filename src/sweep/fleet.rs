use std::time::{Duration, Instant};

use client::client::Client;

pub struct FleetRun {
    pub outputs: Vec<Option<Vec<i64>>>,
    pub lane_latencies: Vec<Duration>,
    pub elapsed: Duration,
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

    pub fn failed_lanes(&self) -> usize {
        self.outputs.iter().filter(|o| o.is_none()).count()
    }

    pub fn throughput_tok_s(&self) -> f64 {
        self.total_tokens() as f64 / self.elapsed.as_secs_f64().max(1e-9)
    }

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

async fn run_one(addr: &str, program: &str, input: &str) -> Result<Vec<i64>, String> {
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-sweep")
        .await
        .map_err(|e| format!("connect ws://{addr}/v1/ws: {e}"))?;
    client
        .authenticate("pie-sweep", None)
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
        let head: String = returned.chars().take(200).collect();
        format!("{program} returned no tokens: {head}")
    })
}

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
