//! `pie config tune` — measure this machine instead of remembering someone
//! else's.
//!
//! One boot, many rounds. The three rules the command surface enforces:
//!
//!   * **An objective has to be stated.** `latency` and `throughput` pull
//!     opposite ways, so there is no unqualified "optimal" to search for. A
//!     config still saying `auto` names no objective and the flag is required.
//!   * **Reporting is the default.** `--write` applies. The inverse of the usual
//!     `--dry-run` arrangement, deliberately: this harness is young, and an
//!     unverified optimiser that writes by default puts confident wrong numbers
//!     into an operator's config.
//!   * **Nothing is truncated silently.** `--budget` bounds candidates, and what
//!     it cut is printed. So is every axis this command does not yet touch.

use anyhow::{Context, Result, anyhow};

use super::typed_by_schema;
use crate::sweep::{self, Knobs};
use crate::ui::{Align, Mark, Palette, Row, Table};

/// The serving shape being optimised for.
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

    /// What this objective ranks by.
    ///
    /// Before this existed, `--for latency` measured and ranked THROUGHPUT: the
    /// command asked for one thing and ordered its answers by another. The
    /// objective already names a serving shape, and the quantity that shape is
    /// judged on follows from it.
    pub fn metric(self) -> sweep::Metric {
        match self {
            Self::Latency => sweep::Metric::LaneP95,
            Self::Throughput => sweep::Metric::Throughput,
        }
    }

    /// The load this objective is measured under.
    ///
    /// Also derived rather than defaulted, because an arbitrary load measures an
    /// arbitrary thing. The first version of this command shipped 8 lanes of 48
    /// tokens picked by nothing, and that load turned out to be dominated by
    /// process bootstrap rather than by batching — k=1 measured 4x slower there
    /// and identical at a longer fleet.
    ///
    /// `latency` is a low-concurrency regime by definition, so its fleet is
    /// small. That costs sample count: four lanes over five passes is twenty
    /// lane latencies, and a p95 over twenty samples is the second-worst of
    /// them. `--repeats` is the lever if that is too coarse; there is no
    /// arrangement that makes a low-concurrency measurement dense.
    ///
    /// **256 tokens is not arbitrary, and shorter is not a smaller version of
    /// the same measurement.** At 48 tokens over 8 lanes the fast candidates
    /// finished in about a quarter second and measured at 10-12% spread while
    /// the slow ones, running four times longer, sat near 2%. That is timing
    /// noise on a short interval and no number of repeats removes it: at the
    /// short setting the sweep reported a candidate beating the baseline by
    /// 20.4% against a 15.5% noise floor -- clearing its own significance test
    /// with a false positive -- and at the long one nothing in that region
    /// beats the baseline at all.
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

/// The shape of the synthetic load one objective is measured under.
#[derive(Debug, Clone, Copy)]
pub struct Workload {
    pub fleet: usize,
    pub tokens: usize,
    pub repeats: usize,
}

/// What one `pie config tune` run was asked to do.
#[derive(clap::Args, Debug)]
pub struct TuneArgs {
    /// Which serving shape to optimize for. Required: latency and
    /// throughput pull opposite ways, and a guess would wear a
    /// measurement's authority.
    #[arg(long = "for", value_name = "SHAPE")]
    pub objective: Option<Objective>,

    /// The inferlet to drive the load with, e.g. `generate@0.1.0`.
    /// `pie inferlet list` shows what is available.
    ///
    /// It has to accept a BARE TOKEN COUNT as its input — the sweep launches
    /// every lane with `"256"`, not with a JSON object (see `lane_inputs`).
    /// An inferlet whose input is a struct fails every lane with a parse
    /// error, which the warmup now prints verbatim.
    #[arg(long)]
    pub program: String,

    /// Lanes per fleet. Defaults to the shape `--for` implies: 4 for latency,
    /// 64 for throughput.
    //
    // These four are overrides for the objective's own workload. Absent means
    // "use the shape the objective implies", which is the answer for almost
    // everyone.
    #[arg(long)]
    pub fleet: Option<usize>,

    /// Fleets per candidate. Defaults to the shape `--for` implies. Below 3 the
    /// spread means little, and the spread is what decides whether a difference
    /// is real.
    #[arg(long)]
    pub repeats: Option<usize>,

    /// Tokens each lane decodes. Defaults to 256. Shorter fleets are measurably
    /// noisier, and the noise decides what counts as a win — at 48 tokens the
    /// sweep reported a 20% improvement that a longer fleet showed was nothing.
    #[arg(long)]
    pub tokens: Option<usize>,

    /// Stop after this many candidates. Counts candidates, not minutes: an
    /// operator cannot predict how many fleets fit in a wall-clock budget.
    #[arg(long)]
    pub budget: Option<usize>,

    /// Apply the winner to the config file.
    #[arg(long)]
    pub write: bool,
}

impl TuneArgs {
    /// The load actually run: the objective's shape, with any explicit override
    /// applied over it.
    pub fn workload(&self, objective: Objective) -> Workload {
        let base = objective.workload();
        Workload {
            fleet: self.fleet.unwrap_or(base.fleet),
            tokens: self.tokens.unwrap_or(base.tokens),
            repeats: self.repeats.unwrap_or(base.repeats),
        }
    }
}

/// Resolve the objective from the flag, refusing when none is stated.
///
/// This used to fall back to `[engine] memory_profile` -- the key tune itself
/// wrote to remember the last `--for`. The key retired when it was measured
/// to have no other reader anywhere (no planner input exists on
/// `DeviceBoot`), so the objective is stated per run: what to optimise
/// toward is a property of THIS measurement, not of the deployment.
pub fn resolve_objective(flag: Option<Objective>) -> Result<Objective> {
    flag.ok_or_else(|| {
        anyhow!(
            "no objective to optimise toward — latency and throughput pull \
             opposite ways. Pass `--for latency` or `--for throughput`."
        )
    })
}

/// The candidates to measure, in order, and how many the budget cut.
///
/// The baseline goes first so it is measured while the machine is in the same
/// state as the candidates it will be compared against — a baseline taken last
/// is a baseline taken on a different machine.
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

/// Rank measured rounds against the baseline.
///
/// A candidate only wins if it clears both spreads in quadrature
/// (`Round::beats`). Everything closer is a coin flip, and a sweep that reports
/// coin flips as findings is worse than no sweep.
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

/// What the sweep measured, whether anything won, and what was not looked at.
///
/// Every number the ranking used is carried, including the spreads: at the
/// serving level the noise floor measured ~6% on an L40S while the gap between
/// two candidates was ~2.4%, so a report of the ranking without the spread is a
/// report of noise. A script reading this needs the same thing the table shows.
#[derive(serde::Serialize)]
pub struct TuneReport {
    /// What the candidates were ranked by, as `Metric::label` words it.
    ranked_by: &'static str,
    ranked_by_throughput: bool,
    candidates: Vec<Candidate>,
    /// Set only when something beat the current config by more than the
    /// measurement noise.
    winner: Option<KnobSet>,
    gain_percent: Option<f64>,
    /// Whether the winner was applied. `--write` applies; reporting is default.
    wrote: bool,
    /// Candidates `--budget` cut. Never silently: the report ranks only what
    /// ran, and says how much did not.
    not_measured: usize,
    /// Lanes the sweep actually ran. Carried into the report because the
    /// ranking is only valid near this width -- see `print`.
    fleet: usize,
}

#[derive(serde::Serialize, Clone, Copy, PartialEq)]
struct KnobSet {
    frame_size: usize,
    dispatch_depth: usize,
    /// Derived from `dispatch_depth`, reported because it is what a guest
    /// feels — see `engine::runahead::Runahead::submit_depth`.
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
    /// Spread of the RANKED quantity, as a fraction of its median.
    rel_sigma: f64,
    /// The knobs the config currently asks for.
    current: bool,
    winner: bool,
}

/// Bundle what was measured into the report `present` will render.
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
            // The RANKED quantity, with the spread that decided the ranking.
            // Printing the throughput spread beside a p95 ordering was the same
            // mismatch this command had between `--for` and its metric.
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
                // The two numbers the percentage is a ratio of. A bare "12.3%
                // better" cannot be sanity-checked against the table above it,
                // and this line is the one an operator copies into a decision.
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
        // The width caveat, printed whether or not a winner was found, because
        // it bounds the answer either way.
        //
        // Measured, and the reason this line exists: at the throughput
        // objective's own default of 64 lanes this sweep ranked
        // `k=3 submit=4 dispatch=1` 18.5% above the current config. The same
        // knobs at 256 lanes came back level (15,209 vs 15,338) with a 36.8%
        // spread, and on the real bench shapes at 256-way concurrency they
        // were 12.8% and 16.4% SLOWER. A shallow dispatch queue wins where
        // there is little to overlap and loses where there is a lot; 64 lanes
        // cannot see that, and `--write` would have applied it.
        println!();
        println!(
            "  Measured at {} lanes. A geometry that wins at one fleet width can lose at",
            self.fleet
        );
        println!("  another -- shallower dispatch wins where there is little to overlap and");
        println!("  loses where there is a lot. Sweep at the width you serve (`--fleet`).");
    }
}

/// One lane's input. A bare integer makes the `generate` family decode that
/// many tokens; anything else is passed through as the program's own input.
pub fn lane_inputs(fleet: usize, tokens: usize) -> Vec<String> {
    (0..fleet).map(|_| tokens.to_string()).collect()
}

/// Write the winning knobs into a config document.
///
/// Through `typed_by_schema`, the same path `pie config set` uses, so a value
/// the schema would refuse is refused here too rather than written and
/// discovered at the next boot. It also means the joint staging bound is
/// checked on the way in: `RuntimeConfig::validate` runs on each candidate
/// document, and it sees both factors at once.
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

/// Sweep the frame knobs against one boot of the derived config.
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
    // The CLI's `#[tokio::main]` owns the one runtime (see main.rs's "Model A"
    // note), so this borrows it rather than building a second one -- nesting
    // runtimes panics outright.
    let rounds = async {
        // ONE boot for the whole sweep. Everything after it is cheap, which is
        // the whole reason the knobs were made swappable. This is a different
        // boot from stage one's on purpose -- it reads the profile that one
        // wrote, and plans the arena it will actually serve from.
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

        // Interleaved: one fleet per candidate per pass. Batched repeats would
        // report the within-burst spread as the uncertainty, and that spread is
        // what decides which differences count.
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

    // Decided once and then rendered. Computing it again inside `report` would
    // be a second answer to the same question, and the one thing that must not
    // differ is what was written from what the operator is told was written.
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
    fn the_winner_is_written_through_the_schema() {
        // The apply path is only reached when a candidate wins, which on a
        // healthy machine is rare -- so it is tested here rather than left to
        // be exercised for the first time on someone's config.
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
        // AND NOT A THIRD KEY: `runtime.frame_submit_depth` is derived from
        // the dispatch depth, so writing it would be writing a knob
        // `RuntimeConfig` does not have.
        assert!(runtime.get("frame_submit_depth").is_none());
        // Typed, not stringified: `pie config set` had this exact bug.
        assert!(!updated.contains("frame_size = \"3\""));
    }

    #[test]
    fn a_combination_the_engine_would_refuse_is_never_written() {
        // `frame_size = 5` is past `Runahead::STEPS_MAX`: the frame scheduler
        // was built and measured around four waves per frame, and the staging
        // formula's `k` is that number. Writing it would produce a config that
        // parses and that the runtime then refuses at boot.
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
