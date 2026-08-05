//! `pie config optimize` — measure this machine instead of remembering someone
//! else's.
//!
//! One boot, many rounds. The design and the arguments behind it are in
//! `.wiki/plan/config-optimize.md`; what matters here is the three rules the
//! command surface enforces:
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

use anyhow::{Context, Result, anyhow, bail};

use super::typed_by_schema;
use crate::sweep::{self, Knobs};
use crate::ui::{Align, Mark, Palette, Row, Stream, Table};


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
    /// process startup rather than by batching — k=1 measured 4x slower there
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

pub struct Args {
    pub objective: Option<Objective>,
    pub program: String,
    /// Overrides for the objective's own workload. Absent means "use the shape
    /// the objective implies", which is the answer for almost everyone.
    pub fleet: Option<usize>,
    pub repeats: Option<usize>,
    pub tokens: Option<usize>,
    pub budget: Option<usize>,
    pub write: bool,
}

impl Args {
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

/// Refuse to sweep on a boot that is also calibrating the memory planner.
///
/// The two tuning stages measure different things and must not be stacked. A
/// calibration boot makes `plan_cuda_memory` abandon its score and build the
/// LARGEST forward shape in the feasible region — deliberately, so the driver's
/// own ladder has room to sweep downward, and deliberately accepting the
/// starved KV pool that leaves. Frame knobs measured against that arena are
/// measured against a machine nobody serves from, and `--write` would then put
/// them in the config as if they described this one.
///
/// An error rather than a warning: the numbers would look completely ordinary.
pub fn refuse_stacked_calibration(configured: bool) -> Result<()> {
    if configured {
        bail!(
            "`[driver] calibrate_planner` is on, so this boot would build the \
             planner's calibration arena — the largest forward shape it can fit, \
             with the small KV pool that implies — and the sweep would measure \
             the frame knobs against that instead of against the arena you serve \
             from. Let the calibration boot finish, unset the flag, then run this."
        );
    }
    Ok(())
}

/// Resolve the objective from the flag and the config, refusing when neither
/// states one.
///
/// `--for` is not a separate axis from `memory_profile`; it SETS it. Letting the
/// two disagree would mean measuring for one shape and serving with another,
/// with nothing to report it.
pub fn resolve_objective(flag: Option<Objective>, configured: &str) -> Result<Objective> {
    match (flag, configured) {
        (Some(objective), _) => Ok(objective),
        (None, "latency") => Ok(Objective::Latency),
        (None, "throughput") => Ok(Objective::Throughput),
        (None, _) => bail!(
            "`[driver] memory_profile` is \"auto\", which names no objective to \
             optimise toward — latency and throughput pull opposite ways. Pass \
             `--for latency` or `--for throughput`; it sets the profile as well \
             as choosing the load."
        ),
    }
}

/// The candidates to measure, in order, and how many the budget cut.
///
/// The baseline goes first so it is measured while the machine is in the same
/// state as the candidates it will be compared against — a baseline taken last
/// is a baseline taken on a different machine.
pub fn plan(baseline: Knobs, budget: Option<usize>) -> (Vec<Knobs>, usize) {
    let staging = pie_worker::config::UPLOAD_STAGING_DEPTH as usize;
    let mut all = vec![baseline];
    all.extend(
        sweep::candidates(staging)
            .into_iter()
            .filter(|k| *k != baseline),
    );
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

/// Print what was measured, whether anything won, and what was not looked at.
pub fn report(
    rounds: &[sweep::Round],
    baseline: &Knobs,
    best: Option<&sweep::Round>,
    metric: sweep::Metric,
    skipped: usize,
    wrote: bool,
) {
    let palette = Palette::for_stream(Stream::Stdout);
    let base = rounds.iter().find(|r| r.knobs == *baseline);

    println!("Measured {} candidate(s), ranked by {}:", rounds.len(), metric.label());
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
    for round in rounds {
        let mark = if Some(round.knobs) == best.map(|b| b.knobs) {
            Mark::Chosen
        } else {
            Mark::Plain
        };
        table.push(Row::new(
            mark,
            [
                format!("k={}", round.knobs.frame_size),
                format!("submit={}", round.knobs.submit_depth),
                format!("dispatch={}", round.knobs.dispatch_depth),
                // The RANKED quantity, with the spread that decided the
                // ranking. Printing the throughput spread beside a p95 ordering
                // was the same mismatch this command had between `--for` and
                // its metric.
                match metric {
                    sweep::Metric::Throughput => format!("{:.0} tok/s", round.throughput_tok_s),
                    sweep::Metric::LaneP95 => {
                        format!("p95 {:.0} ms", round.lane_p95_us as f64 / 1_000.0)
                    }
                },
                format!("+/-{:.1}%", metric.sigma(round) * 100.0),
                {
                    let other = match metric {
                        sweep::Metric::Throughput => {
                            format!("p95 {:.0} ms", round.lane_p95_us as f64 / 1_000.0)
                        }
                        sweep::Metric::LaneP95 => format!("{:.0} tok/s", round.throughput_tok_s),
                    };
                    if round.knobs == *baseline {
                        format!("{other}  (current)")
                    } else {
                        other
                    }
                },
            ],
        ));
    }
    table.print(&palette);
    println!();

    match (best, base) {
        (Some(best), Some(base)) => {
            let (mine, theirs) = (metric.value(best), metric.value(base));
            let gain = if metric.higher_is_better() {
                (mine - theirs) / theirs * 100.0
            } else {
                (theirs - mine) / theirs * 100.0
            };
            println!(
                "  {} beats the current config by {gain:.1}% on {} ({:.0} -> {:.0}).",
                best.knobs,
                metric.label(),
                theirs,
                mine
            );
            if wrote {
                println!("  Written to the config.");
            } else {
                println!("  Run again with --write to apply it.");
            }
        }
        _ => println!(
            "  Nothing measured better than the current config on {} by more than \
             the measurement noise. Nothing to change.",
            metric.label()
        ),
    }

    if skipped > 0 {
        println!();
        println!("  {skipped} candidate(s) not measured (--budget). The report ranks only what ran.");
    }
    println!();
    println!(
        "  Not searched: kv_page_size, max_forward_tokens and max_forward_requests are"
    );
    println!(
        "  fixed at boot and belong to the driver's own sweep (`[driver] calibrate_planner`)."
    );
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
/// checked on the way in: `SchedulerConfig::validate` runs on each candidate
/// document, and it sees both factors at once.
pub fn apply(content: &str, knobs: Knobs) -> Result<String> {
    let mut content = content.to_string();
    for (key, value) in [
        ("runtime.frame_size", knobs.frame_size),
        ("runtime.frame_submit_depth", knobs.submit_depth),
        ("runtime.frame_dispatch_depth", knobs.dispatch_depth),
    ] {
        let (updated, _) = typed_by_schema(&content, key, &value.to_string())
            .with_context(|| format!("write {key}"))?;
        content = updated;
    }
    Ok(content)
}

/// Boot once, sweep, report.
pub async fn run(global: &startup::GlobalArgs, args: Args) -> Result<()> {
    let (cfg_path, origin) = startup::cli_config_path(global);
    let content = std::fs::read_to_string(&cfg_path).with_context(|| {
        format!(
            "no config file at {} ({}); `pie config init` writes one",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        )
    })?;

    let file: toml::Value =
        toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?;
    let configured_profile =
        pie_worker::config_schema::lookup(&file, "driver.memory_profile")
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| "auto".to_string());
    refuse_stacked_calibration(
        pie_worker::config_schema::lookup(&file, "driver.calibrate_planner")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    )?;
    let objective = resolve_objective(args.objective, &configured_profile)?;

    // The objective is a config value, not just a flag: measuring for one shape
    // and serving with another is a mismatch nothing downstream would report.
    let content = if configured_profile != objective.as_profile() {
        let (updated, _) =
            typed_by_schema(&content, "driver.memory_profile", objective.as_profile())?;
        updated
    } else {
        content
    };

    let (controller, gateway, worker) = crate::derive::derive_standalone(&content)?;
    let baseline = Knobs {
        frame_size: worker.model.scheduler.frame_size as usize,
        submit_depth: worker.model.scheduler.frame_submit_depth as usize,
        dispatch_depth: worker.model.scheduler.frame_dispatch_depth as usize,
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
        // ONE boot. Everything after it is cheap, which is the whole reason
        // the knobs were made swappable.
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

    report(&rounds, &baseline, best, metric, skipped, wrote);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: Knobs = Knobs {
        frame_size: 2,
        submit_depth: 3,
        dispatch_depth: 2,
    };

    fn round(knobs: Knobs, tok_s: f64, sigma: f64) -> sweep::Round {
        sweep::Round {
            knobs,
            throughput_tok_s: tok_s,
            throughput_rel_sigma: sigma,
            lane_p95_us: 1_000,
            lane_p95_rel_sigma: sigma,
            failed_lanes: 0,
            repeats: 3,
        }
    }

    #[test]
    fn an_auto_profile_is_refused_rather_than_guessed() {
        // Guessing here would measure one shape and serve another, and nothing
        // downstream would report the mismatch.
        let error = resolve_objective(None, "auto").unwrap_err().to_string();
        assert!(error.contains("--for"), "got: {error}");
        assert_eq!(
            resolve_objective(None, "throughput").unwrap(),
            Objective::Throughput
        );
        // The flag wins over the file, because it also rewrites it.
        assert_eq!(
            resolve_objective(Some(Objective::Latency), "throughput").unwrap(),
            Objective::Latency
        );
    }

    #[test]
    fn a_calibration_boot_is_not_also_a_sweep() {
        // Stacked, the sweep would rank frame knobs against the arena the
        // planner builds to be MEASURED -- largest forward shape, smallest KV
        // pool -- and `--write` would record that as this machine's answer.
        // The rounds would look entirely normal, which is why this is an error
        // and not a warning.
        let error = refuse_stacked_calibration(true).unwrap_err().to_string();
        assert!(error.contains("calibrate_planner"), "got: {error}");
        assert!(
            error.contains("unset"),
            "the error has to say how to get out of it: {error}"
        );
        assert!(refuse_stacked_calibration(false).is_ok());
    }

    #[test]
    fn the_baseline_is_measured_first_and_only_once() {
        // First: so it sees the same machine state as what it is compared to.
        // Once: an extra copy would be ranked against itself.
        let (plan, skipped) = plan(BASE, None);
        assert_eq!(plan[0], BASE);
        assert_eq!(plan.iter().filter(|k| **k == BASE).count(), 1);
        assert_eq!(skipped, 0);
    }

    #[test]
    fn a_budget_truncates_and_says_how_much() {
        // Silent truncation reads as "covered everything".
        let (full, _) = plan(BASE, None);
        let (capped, skipped) = plan(BASE, Some(5));
        assert_eq!(capped.len(), 5);
        assert_eq!(skipped, full.len() - 5);
        assert_eq!(capped[0], BASE, "the baseline survives any budget");
    }

    #[test]
    fn every_planned_candidate_respects_the_staging_bound() {
        let staging = pie_worker::config::UPLOAD_STAGING_DEPTH as usize;
        let (plan, _) = plan(BASE, None);
        for knobs in plan {
            assert!(knobs.steps_in_flight() < staging, "{knobs}");
        }
    }

    #[test]
    fn a_win_has_to_clear_the_noise() {
        let fast = Knobs {
            frame_size: 4,
            ..BASE
        };
        // 2.5% apart, 8.5% combined noise: not a result.
        let rounds = vec![round(BASE, 1200.0, 0.06), round(fast, 1230.0, 0.06)];
        assert!(winner(&rounds, &BASE, sweep::Metric::Throughput).is_none());

        // Same gap, quiet measurements: a result.
        let rounds = vec![round(BASE, 1200.0, 0.005), round(fast, 1230.0, 0.005)];
        assert_eq!(
            winner(&rounds, &BASE, sweep::Metric::Throughput).map(|r| r.knobs),
            Some(fast)
        );
    }

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
                submit_depth: 4,
                dispatch_depth: 2,
            },
        )
        .expect("a valid combination applies");
        let parsed: toml::Value = toml::from_str(&updated).unwrap();
        let runtime = parsed.get("runtime").and_then(|r| r.as_table()).unwrap();
        assert_eq!(runtime["frame_size"].as_integer(), Some(3));
        assert_eq!(runtime["frame_submit_depth"].as_integer(), Some(4));
        assert_eq!(runtime["frame_dispatch_depth"].as_integer(), Some(2));
        // Typed, not stringified: `pie config set` had this exact bug.
        assert!(!updated.contains("frame_size = \"3\""));
    }

    #[test]
    fn a_combination_the_engine_would_refuse_is_never_written() {
        // 4 * 4 = 16 exceeds the driver's staging pool. Writing it would produce
        // a config that parses and then blocks every submit with no diagnostic
        // -- the failure `UPLOAD_STAGING_DEPTH` exists to make loud.
        let content = crate::ops::config::default_config_for_test();
        let error = apply(
            &content,
            Knobs {
                frame_size: 4,
                submit_depth: 3,
                dispatch_depth: 4,
            },
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("frame"), "got: {error}");
    }

    #[test]
    fn latency_ranks_by_latency_and_throughput_by_throughput() {
        // The bug this replaces: `--for latency` measured and ranked
        // THROUGHPUT, so the command asked for one thing and ordered its
        // answers by another.
        assert_eq!(Objective::Latency.metric(), sweep::Metric::LaneP95);
        assert_eq!(Objective::Throughput.metric(), sweep::Metric::Throughput);
        assert!(!sweep::Metric::LaneP95.higher_is_better());
        assert!(sweep::Metric::Throughput.higher_is_better());
    }

    #[test]
    fn a_lower_p95_wins_a_latency_sweep() {
        // Direction matters as much as the quantity: ranked as if higher were
        // better, a latency sweep would pick the slowest candidate.
        let quicker = Knobs { frame_size: 1, ..BASE };
        let mut base = round(BASE, 1000.0, 0.01);
        base.lane_p95_us = 400_000;
        base.lane_p95_rel_sigma = 0.01;
        let mut challenger = round(quicker, 500.0, 0.01);
        challenger.lane_p95_us = 200_000;
        challenger.lane_p95_rel_sigma = 0.01;

        // Half the latency, and half the throughput -- so the two objectives
        // must disagree about it, which is the whole reason there are two.
        assert_eq!(
            winner(&[base, challenger], &BASE, sweep::Metric::LaneP95).map(|r| r.knobs),
            Some(quicker)
        );
        let mut base = round(BASE, 1000.0, 0.01);
        base.lane_p95_us = 400_000;
        base.lane_p95_rel_sigma = 0.01;
        let mut challenger = round(quicker, 500.0, 0.01);
        challenger.lane_p95_us = 200_000;
        challenger.lane_p95_rel_sigma = 0.01;
        assert!(winner(&[base, challenger], &BASE, sweep::Metric::Throughput).is_none());
    }

    #[test]
    fn the_workload_follows_the_objective_unless_overridden() {
        // An arbitrary load measures an arbitrary thing: the first version of
        // this command shipped 8 lanes of 48 tokens picked by nothing, and that
        // load was dominated by process startup rather than by batching.
        let latency = Objective::Latency.workload();
        let throughput = Objective::Throughput.workload();
        assert!(
            latency.fleet < throughput.fleet,
            "latency is the low-concurrency regime by definition"
        );
        assert!(
            latency.repeats > throughput.repeats,
            "a small fleet yields few lane samples, so it needs more passes"
        );

        let args = Args {
            objective: None,
            program: String::new(),
            fleet: Some(7),
            repeats: None,
            tokens: None,
            budget: None,
            write: false,
        };
        let resolved = args.workload(Objective::Latency);
        assert_eq!(resolved.fleet, 7, "an explicit flag wins");
        assert_eq!(resolved.tokens, latency.tokens, "the rest stays derived");
        assert_eq!(resolved.repeats, latency.repeats);
    }

    #[test]
    fn without_a_baseline_round_nothing_wins() {
        // Ranking against an absent baseline would make the fastest candidate
        // look like an improvement over nothing.
        let other = Knobs {
            frame_size: 1,
            ..BASE
        };
        let rounds = vec![round(other, 9_000.0, 0.001)];
        assert!(winner(&rounds, &BASE, sweep::Metric::Throughput).is_none());
    }
}
