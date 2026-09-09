use anyhow::{Context, Result, anyhow};

use super::typed_by_schema;
use crate::sweep::{self, Knobs};
use crate::ui::{Align, Mark, Palette, Row, Table};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Objective {
    Latency,
    Throughput,
}

impl Objective {
    fn as_profile(self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::Throughput => "throughput",
        }
    }

    pub fn metric(self) -> sweep::Metric {
        match self {
            Self::Latency => sweep::Metric::LaneP95,
            Self::Throughput => sweep::Metric::Throughput,
        }
    }

    pub fn workload(self) -> Workload {
        match self {
            Self::Latency => Workload {
                fleet: 4,
                tokens: 256,
                repeats: 5,
            },
            Self::Throughput => Workload {
                fleet: 64,
                tokens: 256,
                repeats: 3,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Workload {
    pub fleet: usize,
    pub tokens: usize,
    pub repeats: usize,
}

#[derive(clap::Args, Debug)]
pub struct TuneArgs {
    #[arg(long = "for", value_name = "SHAPE")]
    pub objective: Option<Objective>,

    #[arg(long)]
    pub program: String,

    #[arg(long)]
    pub fleet: Option<usize>,

    #[arg(long)]
    pub repeats: Option<usize>,

    #[arg(long)]
    pub tokens: Option<usize>,

    #[arg(long)]
    pub budget: Option<usize>,

    #[arg(long)]
    pub write: bool,
}

impl TuneArgs {
    pub fn workload(&self, objective: Objective) -> Workload {
        let base = objective.workload();
        Workload {
            fleet: self.fleet.unwrap_or(base.fleet),
            tokens: self.tokens.unwrap_or(base.tokens),
            repeats: self.repeats.unwrap_or(base.repeats),
        }
    }
}

pub fn resolve_objective(flag: Option<Objective>) -> Result<Objective> {
    flag.ok_or_else(|| {
        anyhow!(
            "no objective to optimise toward — latency and throughput pull \
             opposite ways. Pass `--for latency` or `--for throughput`."
        )
    })
}

pub fn plan(baseline: Knobs, budget: Option<usize>) -> (Vec<Knobs>, usize) {
    let mut all = vec![baseline];
    all.extend(sweep::candidates().into_iter().filter(|k| *k != baseline));
    match budget {
        Some(n) if n < all.len() => {
            let skipped = all.len() - n;
            all.truncate(n.max(1));
            (all, skipped)
        }
        _ => (all, 0),
    }
}

pub fn winner<'a>(
    rounds: &'a [sweep::Round],
    baseline: &Knobs,
    metric: sweep::Metric,
) -> Option<&'a sweep::Round> {
    let base = rounds.iter().find(|r| r.knobs == *baseline)?;
    rounds
        .iter()
        .filter(|r| r.knobs != *baseline && r.beats(base, metric))
        .max_by(|a, b| {
            let (a, b) = (metric.value(a), metric.value(b));
            let ordering = a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
            if metric.higher_is_better() {
                ordering
            } else {
                ordering.reverse()
            }
        })
}

#[derive(serde::Serialize)]
pub struct TuneReport {
    ranked_by: &'static str,
    ranked_by_throughput: bool,
    candidates: Vec<Candidate>,
    winner: Option<KnobSet>,
    gain_percent: Option<f64>,
    wrote: bool,
    not_measured: usize,
    fleet: usize,
}

#[derive(serde::Serialize, Clone, Copy, PartialEq)]
struct KnobSet {
    frame_size: usize,
    dispatch_depth: usize,
    submit_depth: usize,
}

impl From<Knobs> for KnobSet {
    fn from(k: Knobs) -> Self {
        Self {
            frame_size: k.frame_size,
            dispatch_depth: k.dispatch_depth,
            submit_depth: k.submit_depth(),
        }
    }
}

#[derive(serde::Serialize)]
struct Candidate {
    #[serde(flatten)]
    knobs: KnobSet,
    throughput_tok_s: f64,
    lane_p95_ms: f64,
    rel_sigma: f64,
    current: bool,
    winner: bool,
}

pub fn build_report(
    rounds: &[sweep::Round],
    baseline: &Knobs,
    best: Option<&sweep::Round>,
    metric: sweep::Metric,
    skipped: usize,
    wrote: bool,
    fleet: usize,
) -> TuneReport {
    let base = rounds.iter().find(|r| r.knobs == *baseline);
    let gain = match (best, base) {
        (Some(best), Some(base)) => {
            let (mine, theirs) = (metric.value(best), metric.value(base));
            Some(if metric.higher_is_better() {
                (mine - theirs) / theirs * 100.0
            } else {
                (theirs - mine) / theirs * 100.0
            })
        }
        _ => None,
    };
    TuneReport {
        ranked_by: metric.label(),
        ranked_by_throughput: matches!(metric, sweep::Metric::Throughput),
        candidates: rounds
            .iter()
            .map(|round| Candidate {
                knobs: round.knobs.into(),
                throughput_tok_s: round.throughput_tok_s,
                lane_p95_ms: round.lane_p95_us as f64 / 1_000.0,
                rel_sigma: metric.sigma(round),
                current: round.knobs == *baseline,
                winner: Some(round.knobs) == best.map(|b| b.knobs),
            })
            .collect(),
        fleet,
        winner: best.map(|b| b.knobs.into()),
        gain_percent: gain,
        wrote,
        not_measured: skipped,
    }
}

impl crate::ui::Report for TuneReport {
    fn render(&self, palette: &Palette) {
        println!(
            "Measured {} candidate(s), ranked by {}:",
            self.candidates.len(),
            self.ranked_by
        );
        let mut table = Table::new(
            [
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Right,
                Align::Left,
            ],
            5,
        );
        for candidate in &self.candidates {
            let (ranked, other) = if self.ranked_by_throughput {
                (
                    format!("{:.0} tok/s", candidate.throughput_tok_s),
                    format!("p95 {:.0} ms", candidate.lane_p95_ms),
                )
            } else {
                (
                    format!("p95 {:.0} ms", candidate.lane_p95_ms),
                    format!("{:.0} tok/s", candidate.throughput_tok_s),
                )
            };
            table.push(Row::new(
                if candidate.winner {
                    Mark::Chosen
                } else {
                    Mark::Plain
                },
                [
                    format!("k={}", candidate.knobs.frame_size),
                    format!("dispatch={}", candidate.knobs.dispatch_depth),
                    format!("submit={}", candidate.knobs.submit_depth),
                    ranked,
                    format!("+/-{:.1}%", candidate.rel_sigma * 100.0),
                    if candidate.current {
                        format!("{other}  (current)")
                    } else {
                        other
                    },
                ],
            ));
        }
        table.print(palette);
        println!();

        match (self.winner, self.gain_percent) {
            (Some(winner), Some(gain)) => {
                let ranked = |c: &Candidate| {
                    if self.ranked_by_throughput {
                        c.throughput_tok_s
                    } else {
                        c.lane_p95_ms
                    }
                };
                let from = self.candidates.iter().find(|c| c.current).map(ranked);
                let to = self.candidates.iter().find(|c| c.winner).map(ranked);
                let change = match (from, to) {
                    (Some(from), Some(to)) => format!(" ({from:.0} -> {to:.0})"),
                    _ => String::new(),
                };
                println!(
                    "  k={} dispatch={} (submit={}) beats the current config by \
                     {gain:.1}% on {}{change}.",
                    winner.frame_size, winner.dispatch_depth, winner.submit_depth, self.ranked_by
                );
                if self.wrote {
                    println!("  Written to the config.");
                } else {
                    println!("  Run again with --write to apply it.");
                }
            }
            _ => println!(
                "  Nothing measured better than the current config on {} by more than \
                 the measurement noise. Nothing to change.",
                self.ranked_by
            ),
        }

        if self.not_measured > 0 {
            println!();
            println!(
                "  {} candidate(s) not measured (--budget). The report ranks only what ran.",
                self.not_measured
            );
        }
        println!();
        println!("  Not searched: kv_page_size, max_forward_tokens and max_forward_requests are");
        println!("  fixed at boot; state them in the config rather than expecting this sweep");
        println!("  to move them.");
        println!();
        println!(
            "  Measured at {} lanes. A geometry that wins at one fleet width can lose at",
            self.fleet
        );
        println!("  another -- shallower dispatch wins where there is little to overlap and");
        println!("  loses where there is a lot. Sweep at the width you serve (`--fleet`).");
    }
}

pub fn lane_inputs(fleet: usize, tokens: usize) -> Vec<String> {
    (0..fleet).map(|_| tokens.to_string()).collect()
}

pub fn apply(content: &str, knobs: Knobs) -> Result<String> {
    let mut content = content.to_string();
    for (key, value) in [
        ("runtime.frame_size", knobs.frame_size),
        ("runtime.frame_dispatch_depth", knobs.dispatch_depth),
    ] {
        let (updated, _) = typed_by_schema(&content, key, &value.to_string())
            .with_context(|| format!("write {key}"))?;
        content = updated;
    }
    Ok(content)
}

pub async fn run(global: &bootstrap::GlobalArgs, args: TuneArgs) -> Result<crate::ui::Answer> {
    let (cfg_path, origin) = bootstrap::cli_config_path(global);
    let content = std::fs::read_to_string(&cfg_path).with_context(|| {
        format!(
            "no config file at {} ({}); `pie config init` writes one",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        )
    })?;

    let objective = resolve_objective(args.objective)?;

    let (controller, gateway, worker) = crate::derive::derive_standalone(&content)?;
    let baseline = Knobs {
        frame_size: worker.runtime.frame_size as usize,
        dispatch_depth: worker.runtime.frame_dispatch_depth as usize,
    };
    let (plan, skipped) = plan(baseline, args.budget);
    let workload = args.workload(objective);
    let metric = objective.metric();

    println!(
        "Optimizing for {} on this machine: {} candidate(s), ranked by {}.",
        objective.as_profile(),
        plan.len(),
        metric.label()
    );
    println!(
        "  Load: {} lanes x {} tokens, {} pass(es) -- the shape `{}` implies.",
        workload.fleet,
        workload.tokens,
        workload.repeats,
        objective.as_profile()
    );
    println!("  This holds the whole device. Do not run it against a machine that is serving.");
    println!();

    let inputs = lane_inputs(workload.fleet, workload.tokens);
    let rounds = async {
        let pie = crate::compose::run_standalone(controller, gateway, worker)
            .await
            .context("boot the engine (is something already serving on this port?)")?;
        let addr = pie.listen_addr.to_string();

        sweep::warmup(&addr, &args.program, &inputs)
            .await
            .with_context(|| {
                format!(
                    "warmup with {}: is it in `pie inferlet list`?",
                    args.program
                )
            })?;

        let rounds = sweep::sweep_all(
            &addr,
            &args.program,
            &inputs,
            &plan,
            workload.repeats,
            |pass, total| println!("  pass {pass}/{total} over {} candidates", plan.len()),
        )
        .await?;
        anyhow::Ok(rounds)
    }
    .await?;
    println!();

    let best = winner(&rounds, &baseline, metric);
    let mut wrote = false;
    if let Some(best) = best
        && args.write
    {
        let content = apply(&content, best.knobs)?;
        std::fs::write(&cfg_path, &content).map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
        wrote = true;
    }

    Ok(crate::ui::Answer::report(build_report(
        &rounds,
        &baseline,
        best,
        metric,
        skipped,
        wrote,
        workload.fleet,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_every_case() {
        the_winner_is_written_through_the_schema();
        a_combination_the_engine_would_refuse_is_never_written();
    }

    fn the_winner_is_written_through_the_schema() {
        let content = crate::ops::config::default_config_for_test();
        let updated = apply(
            &content,
            Knobs {
                frame_size: 3,
                dispatch_depth: 2,
            },
        )
        .expect("a valid combination applies");
        let parsed: toml::Value = toml::from_str(&updated).unwrap();
        let runtime = parsed.get("runtime").and_then(|r| r.as_table()).unwrap();
        assert_eq!(runtime["frame_size"].as_integer(), Some(3));
        assert_eq!(runtime["frame_dispatch_depth"].as_integer(), Some(2));
        assert!(runtime.get("frame_submit_depth").is_none());
        assert!(!updated.contains("frame_size = \"3\""));
    }

    fn a_combination_the_engine_would_refuse_is_never_written() {
        let content = crate::ops::config::default_config_for_test();
        let error = apply(
            &content,
            Knobs {
                frame_size: 5,
                dispatch_depth: 4,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("frame"), "got: {error}");
    }
}
