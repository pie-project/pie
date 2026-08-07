//! `pie config tune` — measure this machine instead of remembering someone
//! else's.
//!
//! `optimize` until the rename; `pie model build` builds an artifact and
//! this one tunes a machine, and one verb could not carry both.
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

/// What one `pie config tune` run was asked to do.
#[derive(clap::Args, Debug)]
pub struct TuneArgs {
    /// Which serving shape to optimize for. Sets `[driver] memory_profile` as
    /// well as choosing the load. Required when the profile is `auto`.
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

    /// Skip stage one, the memory-planner calibration.
    ///
    /// Stage one costs a boot and measures the forward step on this device; its
    /// result is cached, so a machine that has already been measured does not
    /// need it again. Nothing else changes -- the sweep then runs against
    /// whatever shape the planner picks, which is that cached measurement if it
    /// still applies and the analytic score otherwise.
    #[arg(long)]
    pub skip_planner: bool,

    /// Internal: run stage one only, then exit.
    ///
    /// `tune` re-execs itself with this to get stage one its own PROCESS, not
    /// just its own boot — see `calibrate_planner`. Hidden because it is a
    /// step of this command, not a mode an operator picks.
    #[arg(long, hide = true)]
    pub calibrate_only: bool,
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
    submit_depth: usize,
    dispatch_depth: usize,
}

impl From<Knobs> for KnobSet {
    fn from(k: Knobs) -> Self {
        Self {
            frame_size: k.frame_size,
            submit_depth: k.submit_depth,
            dispatch_depth: k.dispatch_depth,
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
                    format!("submit={}", candidate.knobs.submit_depth),
                    format!("dispatch={}", candidate.knobs.dispatch_depth),
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
                    "  k={} submit={} dispatch={} beats the current config by {gain:.1}% on {}{change}.",
                    winner.frame_size, winner.submit_depth, winner.dispatch_depth, self.ranked_by
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
        println!("  fixed at boot, and belong to stage 1 above rather than to this sweep.");
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

/// When the planner profile was last written, or `None` if it is not there.
///
/// Modification time and length together: a calibration that replaces an entry
/// with one the same size still moves the mtime, and a filesystem with coarse
/// mtime still changes the length when the numbers differ. Either alone has a
/// case where a real write looks like no write.
fn written_at(profile: &std::path::Path) -> Option<(std::time::SystemTime, u64)> {
    let meta = std::fs::metadata(profile).ok()?;
    Some((meta.modified().ok()?, meta.len()))
}

/// Measure the memory planner on this device, in its own boot.
///
/// **This is why `calibrate_planner` is not a config key.** It is one run of a
/// measurement, and the only thing that ever needs it is right here: the flag
/// is set on a config this process derived in memory, the boot that reads it
/// ends when this function returns, and nothing about it is written down. There
/// is no state for an operator to leave behind, because there is no state.
///
/// It has to be its own boot rather than a stage of the sweep's. A calibrating
/// planner abandons its score and builds the largest forward shape that fits —
/// the arena the ladder needs room to sweep inside — and accepts the starved KV
/// pool that leaves. That is the right arena to measure the forward step in and
/// the wrong one to measure anything else in, so it goes away before stage two
/// opens a fresh one from the profile it just wrote.
///
/// **The cost is a second weight load, and that is not free.** The premise the
/// frame-knob sweep is built on is that boots are expensive and knob changes
/// are not — a TB-scale model spends minutes loading — so a command that boots
/// twice spends those minutes twice. There is no way around it while the arena
/// is decided at boot: the two stages need different ones. What there is, is
/// `--skip-planner`, which is the right flag for every run after the first on a
/// machine whose forward step has not changed.
async fn calibrate_planner(content: &str) -> Result<()> {
    let (controller, gateway, mut worker) = crate::derive::derive_standalone(content)?;
    worker.server.calibrate_planner = true;
    // Bind an ephemeral port. The sweep boot that follows wants the configured
    // one, and a calibration boot serves nothing that needs a stable address.
    worker.server.port = 0;

    let pie = crate::compose::run_standalone(controller, gateway, worker)
        .await
        .context("boot the engine to calibrate the planner")?;
    // Calibration runs inside the driver's own startup, so by the time the boot
    // returns the sweep is done and the profile is written. Nothing to drive.
    pie.shutdown().await;
    Ok(())
}

/// Run stage one in a CHILD PROCESS and wait for it.
///
/// The engine allows one runtime per process — `bootstrap` latches on the way
/// up and the admission pools behind it are `OnceLock`s whose own comment says
/// "reset is by process exit only". Two boots in one process therefore do not
/// merely fail, they fail three layers deep: the runtime flag first, then the
/// program-manager and linker singletons, then the execution-slot capacity,
/// each one a panic rather than an error. Stage two was unreachable on every
/// machine for exactly this reason.
///
/// A child process is the contract the engine states, and it costs nothing
/// this command was not already paying: stage one is a whole boot, so one more
/// fork-and-exec is noise beside the weight load inside it.
///
/// The child reads the SAME config text this process derived — objective
/// applied and all — through a temporary file and `PIE_CONFIG`, so the
/// measurement is taken against the config the sweep is about to serve from
/// rather than whatever is on disk.
async fn calibrate_planner_in_child(content: &str, program: &str) -> Result<()> {
    let dir = std::env::temp_dir().join(format!("pie-tune-{}", std::process::id()));
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    let cfg = dir.join("config.toml");
    std::fs::write(&cfg, content).with_context(|| format!("write {}", cfg.display()))?;

    let exe = std::env::current_exe().context("locate this executable")?;
    let mut command = std::process::Command::new(exe);
    // stdio is inherited, so the child's boot and calibration lines land in
    // this command's output where the operator is already reading.
    command
        .env("PIE_CONFIG", &cfg)
        .arg("config")
        .arg("tune")
        .arg("--calibrate-only")
        .arg("--program")
        .arg(program);

    let status = tokio::task::spawn_blocking(move || command.status())
        .await
        .context("join the calibration child")?
        .context("spawn the calibration child")?;

    if !status.success() {
        // The config is left behind ON FAILURE only: it is the derived one,
        // not the file on disk, so it is the thing worth looking at when the
        // calibration boot refuses it.
        anyhow::bail!(
            "the calibration boot failed ({status}); its output is above, and \
             the config it was given is at {}",
            cfg.display()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

/// Calibrate the planner, then sweep the frame knobs inside the arena that
/// produced.
pub async fn run(global: &startup::GlobalArgs, args: TuneArgs) -> Result<crate::ui::Answer> {
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

    // The child this command re-execs for stage one. Placed after the
    // objective is folded into `content` so the measurement is taken against
    // the config the sweep will serve from, and before anything that boots.
    if args.calibrate_only {
        calibrate_planner(&content).await?;
        return Ok(crate::ui::Answer::quiet());
    }

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

    // Stage one, and it has to come first: it decides the forward shape, and
    // the frame knobs are measured INSIDE whatever arena that shape produces.
    // Sweeping first would rank candidates against an arena the next boot
    // replaces.
    if args.skip_planner {
        println!("Stage 1/2  planner calibration skipped (--skip-planner)");
        println!("  The sweep runs against whatever shape the planner picks for this boot --");
        println!("  a cached measurement if one applies, the analytic score otherwise.");
    } else {
        println!("Stage 1/2  calibrating the memory planner (one boot, serves nothing)");
        // Said before it is paid, not after. The two stages need two different
        // arenas and therefore two boots, which on a large model is the weight
        // load twice -- and the whole reason the frame-knob sweep reuses one
        // boot is that weight loads are the expensive part.
        println!("  This boots the model a second time; `--skip-planner` reuses an earlier one.");
        let profile = pie_worker::state::planner_profile_path();
        // Observed, not assumed. The driver REFUSES to calibrate for
        // `tensor_parallel_size > 1` and for recurrent-state models, saying so
        // on stderr and booting normally -- so a successful boot is not a
        // measurement, and claiming "written to ..." after one would be this
        // command telling an operator it did something it did not.
        let before = written_at(&profile);
        calibrate_planner_in_child(&content, &args.program).await?;
        if written_at(&profile) == before {
            println!("  no measurement was recorded -- the driver declined to calibrate");
            println!("  (it refuses for tensor_parallel_size > 1 and recurrent-state models;");
            println!("   its reason is on stderr above). Stage 2 runs against the scored shape.");
        } else {
            println!("  written to {}", crate::ui::short_path(&profile));
        }
    }
    println!();
    println!("Stage 2/2  sweeping the frame knobs inside that arena");

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
    fn a_calibration_request_cannot_be_written_down() {
        // The whole reason calibration became a stage rather than a key. A
        // config that asks for it is refused at parse, so there is no state an
        // operator can leave behind -- and the sweep can never be ranked
        // against the arena calibration builds, which is sized to be measured
        // rather than to serve.
        let asked = "\
[model]
name = \"Qwen/Qwen3-0.6B\"
hf_repo = \"Qwen/Qwen3-0.6B\"

[driver]
type = \"cuda_native\"
device = [\"cuda:0\"]
calibrate_planner = true
";
        let error = pie_worker::Config::parse(asked)
            .expect_err("a config cannot request a measurement")
            .to_string();
        assert!(
            error.contains("calibrate_planner"),
            "the refusal has to name the key: {error}"
        );
        // And it is not a settable key either, so `pie config set` cannot
        // produce the document above in the first place.
        assert!(
            !pie_worker::config_schema::fields(pie_worker::config::DriverKind::CudaNative)
                .iter()
                .any(|f| f.key.ends_with("calibrate_planner")),
            "a measurement is not a setting"
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

        let args = TuneArgs {
            objective: None,
            program: String::new(),
            fleet: Some(7),
            repeats: None,
            tokens: None,
            budget: None,
            write: false,
            skip_planner: false,
            calibrate_only: false,
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
