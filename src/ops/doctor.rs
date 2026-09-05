//! `pie doctor` — will pie run here, and with this config?
//!
//! One command because there was no way to tell the old three apart by name.
//! `doctor` reported the platform, `check` parsed the config, and `smoke`
//! reported whether an engine was compiled in -- which `doctor` already did.
//! Each answered part of one question, and none of them answered it.
//!
//! The gap that mattered was the config: `doctor` passed on a machine whose
//! config would not parse, so the next thing an operator did was watch `serve`
//! die. A readiness check that does not read the config is not one.
//!
//! Exit codes:
//!   * 0 — pie can boot here. Warnings are allowed: a missing GPU is a fact
//!     about the machine, not a broken installation.
//!   * 1 — it cannot. Reserved for what actually stops a boot: an unparseable
//!     config, or a config asking for an engine this binary does not have.

use std::path::Path;
use std::process::Command;

use anyhow::Result;

use crate::ui::{Mark, Palette};

/// One section as the checks PRODUCE it: `(label, [(name, detail, status)])`.
/// [`Section`] is the same thing as the JSON report serializes it.
type CollectedSection = (&'static str, Vec<(String, String, Status)>);

/// Everything `doctor` checked, and the verdict that follows from it.
///
/// Collected first and rendered second, so the table and the JSON cannot drift
/// into disagreeing about `ready` -- which is the one thing a readiness probe
/// reads. [`crate::ui::Report`] is what holds every command to it.
#[derive(serde::Serialize)]
pub struct DoctorReport {
    ready: bool,
    passed: usize,
    warnings: usize,
    blocking: usize,
    sections: Vec<Section>,
}

#[derive(serde::Serialize)]
struct Section {
    section: &'static str,
    checks: Vec<Check>,
}

#[derive(serde::Serialize)]
struct Check {
    check: String,
    detail: String,
    status: &'static str,
}

impl Status {
    fn word(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "blocking",
        }
    }

    /// The glyph a status carries. Keyed off [`Status::word`] rather than the
    /// enum, because the word is what crossed into the report and is the
    /// serialized contract a script reads.
    fn mark(word: &str) -> Mark {
        match word {
            "pass" => Mark::Did,
            "warn" => Mark::Warn,
            _ => Mark::Blocked,
        }
    }
}

impl crate::ui::Report for DoctorReport {
    fn render(&self, palette: &Palette) {
        println!("Pie standalone — environment doctor");
        for section in &self.sections {
            println!("\n{}", palette.bold(format!("[{}]", section.section)));
            let mut table =
                crate::ui::Table::new([crate::ui::Align::Left, crate::ui::Align::Left], 1);
            for check in &section.checks {
                table.push(crate::ui::Row::new(
                    Status::mark(check.status),
                    [check.check.clone(), check.detail.clone()],
                ));
            }
            table.print(palette);
        }

        println!();
        let plural = if self.warnings == 1 { "" } else { "s" };
        // The same three glyphs the tables use, from the same vocabulary. These
        // three lines spelled them out as literals, which is how `✓` came to
        // mean one thing here and another in `pie model list`.
        let (mark, line) = if !self.ready {
            (
                Mark::Blocked,
                format!(
                    "pie cannot boot here ({} blocking, {} warning{plural}).",
                    self.blocking, self.warnings
                ),
            )
        } else if self.warnings > 0 {
            (
                Mark::Warn,
                format!(
                    "Ready, with warnings ({} passed, {} warning{plural}).",
                    self.passed, self.warnings
                ),
            )
        } else {
            (Mark::Did, format!("Ready ({} checks).", self.passed))
        };
        println!("{} {line}", mark.render(palette));
    }
}

/// `pie doctor` entry point. Exits non-zero when pie cannot boot here.
pub fn run(global: &bootstrap::GlobalArgs) -> Result<crate::ui::Answer> {
    let mut warnings = 0usize;
    let mut passes = 0usize;
    let mut failures = 0usize;

    let mut sections: Vec<CollectedSection> = Vec::new();

    // The config decides whether an NVIDIA probe is even the right question,
    // so it is read here rather than inside `check_config` alone -- the section
    // order below is unchanged, only where the path is computed.
    let (path, origin) = bootstrap::cli_config_path(global);

    sections.push(("system", vec![check_platform(), check_py_runtime()]));
    sections.push(("gpus", check_gpus(configured_engine(&path).as_deref())));
    sections.push((
        "engines",
        worker::backend::flavor::compiled_embedded()
            .iter()
            .map(|(name, on)| {
                if *on {
                    (name.to_string(), "compiled in".to_string(), Status::Pass)
                } else {
                    // An engine you did not build is not a fault until the
                    // config asks for it -- which the config section checks.
                    (name.to_string(), absent_because(name), Status::Warn)
                }
            })
            .collect(),
    ));
    // Last, because its verdict depends on everything above: whether a config
    // is servable is a question about this binary and this machine, not about
    // the file alone.
    sections.push(("config", check_config(&path, origin)));
    sections.push(("tuning", check_tuning(&path)));

    for (_, checks) in &sections {
        for (_, _, status) in checks {
            match status {
                Status::Pass => passes += 1,
                Status::Warn => warnings += 1,
                Status::Fail => failures += 1,
            }
        }
    }
    let ready = failures == 0;

    let report = DoctorReport {
        ready,
        passed: passes,
        warnings,
        blocking: failures,
        sections: sections
            .into_iter()
            .map(|(section, checks)| Section {
                section,
                checks: checks
                    .into_iter()
                    .map(|(check, detail, status)| Check {
                        check,
                        detail,
                        status: status.word(),
                    })
                    .collect(),
            })
            .collect(),
    };

    // The exit code IS the answer -- `pie doctor && pie serve` should be a
    // thing an operator can write.
    let answer = crate::ui::Answer::report(report);
    Ok(if ready {
        answer
    } else {
        answer.with_code(std::process::ExitCode::FAILURE)
    })
}

/// Parse the config and say whether this binary could serve it.
///
/// Absent is not a failure: `Origin::Default` says an absent file is normal
/// and the engine falls back to its own defaults. Named explicitly and absent
/// IS a failure, because the engine treats that as fatal -- the same split
/// `pie config show` makes.
fn check_config(path: &Path, origin: bootstrap::Origin) -> Vec<(String, String, Status)> {
    if !path.exists() {
        return if origin == bootstrap::Origin::Default {
            vec![(
                "config".into(),
                format!(
                    "none at {} — running on defaults",
                    crate::ui::short_path(path)
                ),
                Status::Warn,
            )]
        } else {
            vec![(
                "config".into(),
                format!(
                    "{} does not exist ({})",
                    crate::ui::short_path(path),
                    origin.describe()
                ),
                Status::Fail,
            )]
        };
    }

    let combined = match crate::derive::read_config_file(path) {
        Ok(c) => c,
        Err(e) => {
            return vec![(
                "config".into(),
                format!("{}: {e}", crate::ui::short_path(path)),
                Status::Fail,
            )];
        }
    };
    let worker = match crate::derive::derive_standalone(&combined) {
        Ok((_controller, _gateway, worker)) => worker,
        // `{:#}` so the chain reaches the line and column, which is the whole
        // value of being told the config is bad.
        Err(e) => {
            return vec![(
                "config".into(),
                format!("{}: {e:#}", crate::ui::short_path(path)),
                Status::Fail,
            )];
        }
    };

    let mut out = vec![(
        "config".into(),
        format!("{} parses", crate::ui::short_path(path)),
        Status::Pass,
    )];
    // A parsing config that names a model pie cannot find is the next thing
    // that stops a boot, and the check `pie check` could never make: the
    // artifact store is on this disk, not in the file. `weights::resolve` is
    // the same call the worker makes, so a pass here means the worker's will
    // pass too.
    //
    // With the flavor this binary hosts, because that is what the worker
    // resolves with: a model imported for two shells is two files in one
    // directory, and "can pie find it" is only answerable for one engine at
    // a time. A config naming an engine this build lacks has no flavor, and
    // the lookup then reports the ambiguity rather than a pick.
    let flavor = worker::backend::flavor::resolve(worker.model.engine.kind, &worker.model.name);
    let want = worker::weights::Want {
        backend: flavor.as_ref().ok().map(|flavor| flavor.as_str()),
        sku: worker.model.sku.as_deref(),
    };
    match worker::weights::resolve(&worker.model.model, want) {
        Ok(resolved) => out.push((
            "weights".into(),
            match resolved {
                worker::weights::Model::Artifact(path) => {
                    format!("artifact {}", crate::ui::short_path(&path))
                }
                worker::weights::Model::Snapshot(path) => format!(
                    "raw snapshot {} — `pie model import` makes an artifact",
                    crate::ui::short_path(&path)
                ),
            },
            Status::Pass,
        )),
        Err(error) => out.push(("weights".into(), format!("{error}"), Status::Fail)),
    }
    // The check the old `pie check` could not make and `pie smoke` made in
    // isolation: the config names an engine, and this binary either has it or
    // does not.
    let kind = worker.model.engine.kind.as_str();
    let compiled = worker::backend::flavor::compiled_embedded()
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, on)| *on)
        .unwrap_or(false);
    out.push(if compiled {
        (
            "model".into(),
            format!("{} on {}", worker.model.name, kind),
            Status::Pass,
        )
    } else {
        (
            "model".into(),
            format!(
                "{} asks for the {kind} engine: {}",
                worker.model.name,
                absent_because(kind)
            ),
            Status::Fail,
        )
    });
    out
}

/// The engine types this build knows how to host, for the message below.
///
/// Not `flavor::compiled_summary()`, which lists what this binary HAS: the
/// point of naming an unknown type is to say which spellings exist at all,
/// and on a binary with no feature on that summary is empty.
const KNOWN_ENGINES: &str = "cuda, metal, vulkan, wgpu";

/// Why an engine flavor this binary does not have is missing.
///
/// THREE answers, because there are three ways to not have one and only two
/// of them are a build choice. A feature was off (rebuild with it); the
/// feature cannot apply here, because Metal's device half is Apple-only at
/// the crate level and telling a Linux operator to enable a flag they may
/// already have on is advice that cannot work; or the config named a
/// spelling this build does not know. It is the distinction
/// `worker::backend::flavor` draws between `missing_feature_msg` and
/// `non_apple_msg`, kept in the same words here.
fn absent_because(name: &str) -> String {
    match name {
        "cuda_native" => "not compiled — build with `--features cuda`".to_string(),
        "metal" if cfg!(target_vendor = "apple") => {
            "not compiled — build with `--features metal`".to_string()
        }
        "metal" => "metal engines run on Apple hardware only".to_string(),
        // One answer, not Metal's two: the Vulkan shell has no target half,
        // so leaving the feature off is the only way to be without it.
        "vulkan" => "not compiled — build with `--features vulkan`".to_string(),
        // One answer here too: wgpu picks its backend at run time, so there is
        // no target half and the feature is the whole of it.
        "wgpu" => "not compiled — build with `--features wgpu`".to_string(),
        other => format!("unknown engine type `{other}`; this build knows: {KNOWN_ENGINES}"),
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Status {
    Pass,
    /// True of the machine, not wrong with the installation. Never blocks.
    Warn,
    /// Would stop a boot.
    Fail,
}

fn check_platform() -> (String, String, Status) {
    let info = format!(
        "{} {} ({})",
        std::env::consts::OS,
        std::env::consts::FAMILY,
        std::env::consts::ARCH,
    );
    ("Platform".to_string(), info, Status::Pass)
}

/// Whether Python inferlets can run.
///
/// A warning rather than a failure: the answer belongs with the other "will
/// this work here" answers, and the fix happens by itself on the next `serve`.
fn check_py_runtime() -> (String, String, Status) {
    let dir = crate::local::py_runtime::runtime_dir();
    if crate::local::py_runtime::is_installed() {
        (
            "python".to_string(),
            format!("runtime at {}", crate::ui::short_path(&dir)),
            Status::Pass,
        )
    } else {
        (
            "python".to_string(),
            "runtime not installed — `pie serve` fetches it on the way up \
             (Rust inferlets do not need it)"
                .to_string(),
            Status::Warn,
        )
    }
}

/// The engine type this config names, without parsing the whole document.
///
/// The same `schema::lookup` `check_tuning` reads keys with, and for the same
/// reason: a config too broken to parse still has to be looked at, and the
/// real verdict on it is `check_config`'s.
fn configured_engine(config_path: &Path) -> Option<String> {
    let file: toml::Value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())?;
    worker::config::schema::lookup(&file, "engine.type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

/// Whether `nvidia-smi` answers a question this deployment is asking.
///
/// A probe for a vendor the config does not name is not a check: running it
/// unconditionally tells a Metal config on a Mac "no NVIDIA GPUs detected" --
/// a true sentence about a card nothing here wants, filed as a warning against
/// an installation that is fine.
fn nvidia_probe_applies(named_engine: Option<&str>) -> bool {
    match named_engine {
        // `cuda` as well as `cuda_native`: `pie config set` accepts both
        // spellings for `engine.type` (see `ops::config::engine_kind`).
        Some(kind) => kind == "cuda_native" || kind == "cuda",
        // No config, or one that names no engine: fall back to what the binary
        // carries, which is the same question one step earlier.
        None => worker::backend::flavor::compiled_embedded()
            .iter()
            .any(|(name, on)| *name == "cuda_native" && *on),
    }
}

fn check_gpus(named_engine: Option<&str>) -> Vec<(String, String, Status)> {
    if !nvidia_probe_applies(named_engine) {
        return vec![(
            "GPU".into(),
            match named_engine {
                Some(kind) => format!("not probed — this config names the {kind} engine"),
                None => "not probed — this binary carries no CUDA engine".to_string(),
            },
            Status::Pass,
        )];
    }
    // nvidia-smi is the cheapest "GPU visible" probe — no link to
    // libnvidia-ml needed.
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.is_empty() {
                vec![("GPU".into(), "no NVIDIA GPUs detected".into(), Status::Warn)]
            } else {
                lines
                    .into_iter()
                    .map(|line| {
                        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                        let idx = parts.first().copied().unwrap_or("?");
                        let rest = parts[1..].join(", ");
                        (format!("GPU {idx}"), rest, Status::Pass)
                    })
                    .collect()
            }
        }
        Ok(_) | Err(_) => vec![(
            "GPU".into(),
            "nvidia-smi not available (CPU-only? non-NVIDIA? or driver missing)".into(),
            Status::Warn,
        )],
    }
}

/// Has this machine been measured, or is it running on defaults someone else
/// measured?
///
/// `pie config tune` measures the batching knobs on this machine; the
/// forward-shape keys are stated by the operator or left to the engine's own
/// defaults. A machine where neither has happened runs on numbers measured
/// somewhere else. That is a perfectly serviceable state, but not one an
/// operator should have to infer from the absence of keys in a file.
///
/// Warnings, never failures. An unmeasured machine serves.
fn check_tuning(config_path: &std::path::Path) -> Vec<(String, String, Status)> {
    let file: toml::Value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())
        .unwrap_or_else(|| toml::Value::Table(Default::default()));
    let set = |key: &str| worker::config::schema::lookup(&file, key).map(|v| v.to_string());

    let mut checks = Vec::new();

    match (
        set("engine.max_forward_tokens"),
        set("engine.max_forward_requests"),
    ) {
        (Some(tokens), Some(requests)) => checks.push((
            "forward shape".to_string(),
            format!("pinned at {tokens} tokens x {requests} requests"),
            Status::Pass,
        )),
        // One without the other is worth saying out loud: the axes share one
        // memory budget, so pinning half of a lattice point leaves the planner
        // choosing the other half around it.
        (Some(tokens), None) => checks.push((
            "forward shape".to_string(),
            format!("max_forward_tokens pinned at {tokens}, decode width still derived"),
            Status::Warn,
        )),
        (None, Some(requests)) => checks.push((
            "forward shape".to_string(),
            format!("max_forward_requests pinned at {requests}, token budget still derived"),
            Status::Warn,
        )),
        (None, None) => checks.push((
            "forward shape".to_string(),
            "derived from the engine's own defaults; state them to pin this machine's shape"
                .to_string(),
            Status::Warn,
        )),
    }

    let frame_knobs = [
        "runtime.frame_size",
        "runtime.frame_submit_depth",
        "runtime.frame_dispatch_depth",
    ];
    let pinned: Vec<&str> = frame_knobs
        .iter()
        .copied()
        .filter(|k| set(k).is_some())
        .collect();
    checks.push(if pinned.is_empty() {
        (
            "batching".to_string(),
            "defaults, measured on other hardware (`pie config tune --for ...`)".to_string(),
            Status::Warn,
        )
    } else {
        (
            "batching".to_string(),
            format!("{} of 3 knobs set in this config", pinned.len()),
            Status::Pass,
        )
    });

    checks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning_of(config: &str) -> Vec<(String, String, Status)> {
        let path = std::env::temp_dir().join(format!(
            "pie-doctor-tuning-{}-{:?}.toml",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, config).unwrap();
        let checks = check_tuning(&path);
        let _ = std::fs::remove_file(&path);
        checks
    }

    #[test]
    fn the_unmeasured_machine_still_serves() {
        // This section describes the machine rather than faulting the config,
        // and an unmeasured machine is a perfectly serviceable one. Nothing
        // here may block a boot.
        for config in ["", "[engine]\nkv_page_size = 32\n"] {
            let checks = tuning_of(config);
            assert!(
                !checks.iter().any(|(_, _, status)| *status == Status::Fail),
                "nothing here blocks a boot: {checks:?}"
            );
        }
    }
}
