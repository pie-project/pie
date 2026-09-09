pub mod fleet;

use anyhow::{Context, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Knobs {
    pub frame_size: usize,
    pub dispatch_depth: usize,
}

impl Knobs {
    pub const MAX_FRAME_SIZE: usize = worker::config::Runahead::STEPS_MAX as usize;
    pub const MAX_DISPATCH_DEPTH: usize = worker::config::Runahead::MAX_FRAMES as usize;

    pub fn steps_in_flight(&self) -> usize {
        self.frame_size * self.dispatch_depth
    }

    pub fn staging_depth(&self) -> usize {
        worker::config::Runahead::of(self.dispatch_depth.min(255) as u8).staging_depth()
    }

    pub fn submit_depth(&self) -> usize {
        worker::config::Runahead::of(self.dispatch_depth.min(255) as u8).submit_depth()
    }

    pub fn admissible(&self) -> bool {
        self.frame_size >= 1
            && self.frame_size <= Self::MAX_FRAME_SIZE
            && self.dispatch_depth >= 1
            && self.dispatch_depth <= Self::MAX_DISPATCH_DEPTH
    }
}

impl std::fmt::Display for Knobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "k={} dispatch={} (submit={})",
            self.frame_size,
            self.dispatch_depth,
            self.submit_depth()
        )
    }
}

pub struct Round {
    pub knobs: Knobs,
    pub throughput_tok_s: f64,
    pub throughput_rel_sigma: f64,
    pub lane_p95_us: u128,
    pub lane_p95_rel_sigma: f64,
    pub failed_lanes: usize,
    pub repeats: usize,
}

impl Round {
    pub fn is_measurement(&self) -> bool {
        self.failed_lanes == 0
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Throughput,
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

fn distinct_reasons(failures: &[String]) -> String {
    const SHOWN: usize = 3;
    let mut counts: Vec<(String, usize)> = Vec::new();
    for reason in failures {
        match counts.iter_mut().find(|(seen, _)| seen == reason) {
            Some((_, n)) => *n += 1,
            None => counts.push((reason.clone(), 1)),
        }
    }
    counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    let mut out: Vec<String> = counts
        .iter()
        .take(SHOWN)
        .map(|(reason, n)| format!("  {n}x {reason}"))
        .collect();
    if counts.len() > SHOWN {
        out.push(format!("  ... and {} more distinct", counts.len() - SHOWN));
    }
    out.join("\n")
}

pub async fn warmup(addr: &str, program: &str, inputs: &[String]) -> Result<()> {
    let mut previous: Option<f64> = None;
    for round in 0..MAX_WARMUP_ROUNDS {
        let run = fleet::run(addr, program, inputs).await;
        if run.failed_lanes() > 0 {
            anyhow::bail!(
                "{} of {} lanes failed during warmup; the fleet cannot run here at all.\n{}",
                run.failed_lanes(),
                inputs.len(),
                distinct_reasons(&run.failures)
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
    Ok(())
}

const WARMUP_SETTLED: f64 = 0.05;
const MAX_WARMUP_ROUNDS: usize = 5;

pub async fn measure(
    addr: &str,
    program: &str,
    inputs: &[String],
    knobs: Knobs,
    repeats: usize,
) -> Result<Round> {
    Ok(
        sweep_all(addr, program, inputs, &[knobs], repeats, |_, _| {})
            .await?
            .pop()
            .expect("one candidate in, one round out"),
    )
}

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
            runtime::scheduler::reconfigure(knobs.frame_size, knobs.dispatch_depth)
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
            let (throughput_tok_s, throughput_rel_sigma) =
                median_and_rel_sigma(&throughputs[index]);
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

pub fn candidates() -> Vec<Knobs> {
    let mut groups: Vec<Vec<Knobs>> = Vec::new();
    const _: () = assert!(4 <= Knobs::MAX_FRAME_SIZE && 4 <= Knobs::MAX_DISPATCH_DEPTH);
    for frame_size in [1usize, 2, 3, 4] {
        let mut group = Vec::new();
        for dispatch_depth in 1usize..=4 {
            group.push(Knobs {
                frame_size,
                dispatch_depth,
            });
        }
        groups.push(group);
    }

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
    fn sweep_every_case() {
        every_candidate_is_one_the_runtime_admits();
        a_failed_round_never_beats_anything();
    }

    fn every_candidate_is_one_the_runtime_admits() {
        let candidates = candidates();
        assert!(!candidates.is_empty());
        for knobs in &candidates {
            assert!(knobs.admissible(), "{knobs} is a shape the runtime refuses");
            assert!(knobs.submit_depth() >= 2, "{knobs} leaves nothing queued");
        }
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

    const BASE: Knobs = Knobs {
        frame_size: 2,
        dispatch_depth: 2,
    };

    fn a_failed_round_never_beats_anything() {
        let broken = round(BASE, 4000.0, 0.001, 2);
        let good = round(BASE, 1265.0, 0.01, 0);
        assert!(!broken.beats(&good, Metric::Throughput));
        assert!(!good.beats(&broken, Metric::Throughput));
    }
}
