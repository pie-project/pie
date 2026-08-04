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

/// Lanes per fleet. Enough to co-batch — a single lane never exercises the
/// frame knobs, because there is nothing for it to overlap with — and enough
/// that a fleet lasts long enough to time. See [`DEFAULT_TOKENS`].
const DEFAULT_FLEET: usize = 16;
/// Fleets per candidate. Three is the smallest count that gives a median an
/// outlier cannot move and a spread that means anything.
const DEFAULT_REPEATS: usize = 3;
/// Tokens each lane decodes.
///
/// **The measurement's precision is set here, not by the repeat count.** At 48
/// tokens over 8 lanes the fast configurations finished in about a quarter of a
/// second and came back at 10-12% spread, while the slow ones — running four
/// times longer — sat near 2%. That is timing noise on a short interval, and no
/// number of repeats fixes it. At 256 over 16 the same configurations measure
/// 1-6%.
///
/// It is not a cosmetic difference. At the short setting a candidate was
/// reported as beating the baseline by 20.4% against a 15.5% noise floor; at
/// the long one nothing in that region beats the baseline at all. The sweep was
/// producing a false positive and clearing its own significance test while
/// doing it.
const DEFAULT_TOKENS: usize = 256;

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
}

pub struct Args {
    pub objective: Option<Objective>,
    pub program: String,
    pub fleet: usize,
    pub repeats: usize,
    pub tokens: usize,
    pub budget: Option<usize>,
    pub write: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            objective: None,
            program: String::new(),
            fleet: DEFAULT_FLEET,
            repeats: DEFAULT_REPEATS,
            tokens: DEFAULT_TOKENS,
            budget: None,
            write: false,
        }
    }
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
pub fn winner<'a>(rounds: &'a [sweep::Round], baseline: &Knobs) -> Option<&'a sweep::Round> {
    let base = rounds.iter().find(|r| r.knobs == *baseline)?;
    rounds
        .iter()
        .filter(|r| r.knobs != *baseline && r.beats(base))
        .max_by(|a, b| {
            a.throughput_tok_s
                .partial_cmp(&b.throughput_tok_s)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Print what was measured, whether anything won, and what was not looked at.
pub fn report(rounds: &[sweep::Round], baseline: &Knobs, skipped: usize, wrote: bool) {
    let palette = Palette::for_stream(Stream::Stdout);
    let best = winner(rounds, baseline);
    let base = rounds.iter().find(|r| r.knobs == *baseline);

    println!("Measured {} candidate(s):", rounds.len());
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
                format!("{:.0} tok/s", round.throughput_tok_s),
                format!("+/-{:.1}%", round.throughput_rel_sigma * 100.0),
                if round.knobs == *baseline {
                    format!("p95 {:.0} ms  (current)", round.lane_p95_us as f64 / 1_000.0)
                } else {
                    format!("p95 {:.0} ms", round.lane_p95_us as f64 / 1_000.0)
                },
            ],
        ));
    }
    table.print(&palette);
    println!();

    match (best, base) {
        (Some(best), Some(base)) => {
            let gain =
                (best.throughput_tok_s - base.throughput_tok_s) / base.throughput_tok_s * 100.0;
            println!(
                "  {} beats the current config by {gain:.1}% ({:.0} -> {:.0} tok/s).",
                best.knobs, base.throughput_tok_s, best.throughput_tok_s
            );
            if wrote {
                println!("  Written to the config.");
            } else {
                println!("  Run again with --write to apply it.");
            }
        }
        _ => println!(
            "  Nothing measured faster than the current config by more than the \
             measurement noise. Nothing to change."
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

    let configured_profile = pie_worker::config_schema::lookup(
        &toml::from_str(&content).map_err(|e| anyhow!("parse {cfg_path:?}: {e}"))?,
        "driver.memory_profile",
    )
    .and_then(|v| v.as_str().map(str::to_string))
    .unwrap_or_else(|| "auto".to_string());
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

    println!(
        "Optimizing for {} on this machine: {} candidate(s), {} fleet(s) of {} lanes each.",
        objective.as_profile(),
        plan.len(),
        args.repeats,
        args.fleet
    );
    println!("  This holds the whole device. Do not run it against a machine that is serving.");
    println!();

    let inputs = lane_inputs(args.fleet, args.tokens);
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

        let mut rounds = Vec::with_capacity(plan.len());
        for (index, knobs) in plan.iter().enumerate() {
            let round = sweep::measure(&addr, &args.program, &inputs, *knobs, args.repeats)
                .await
                .with_context(|| format!("candidate {} of {}", index + 1, plan.len()))?;
            println!(
                "  [{}/{}] {knobs} → {:.0} tok/s ±{:.1}%",
                index + 1,
                plan.len(),
                round.throughput_tok_s,
                round.throughput_rel_sigma * 100.0
            );
            rounds.push(round);
        }
        anyhow::Ok(rounds)
    }
    .await?;
    println!();

    let mut wrote = false;
    if let Some(best) = winner(&rounds, &baseline)
        && args.write
    {
        let mut content = content;
        for (key, value) in [
            ("runtime.frame_size", best.knobs.frame_size),
            ("runtime.frame_submit_depth", best.knobs.submit_depth),
            ("runtime.frame_dispatch_depth", best.knobs.dispatch_depth),
        ] {
            let (updated, _) = typed_by_schema(&content, key, &value.to_string())?;
            content = updated;
        }
        std::fs::write(&cfg_path, &content)
            .map_err(|e| anyhow!("write {cfg_path:?}: {e}"))?;
        wrote = true;
    }

    report(&rounds, &baseline, skipped, wrote);
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
        assert!(winner(&rounds, &BASE).is_none());

        // Same gap, quiet measurements: a result.
        let rounds = vec![round(BASE, 1200.0, 0.005), round(fast, 1230.0, 0.005)];
        assert_eq!(winner(&rounds, &BASE).map(|r| r.knobs), Some(fast));
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
        assert!(winner(&rounds, &BASE).is_none());
    }
}
