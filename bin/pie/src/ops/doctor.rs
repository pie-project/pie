//! `pie doctor` — will pie run here, and with this config?
//!
//! One command because there was no way to tell the old three apart by name.
//! `doctor` reported the platform, `check` parsed the config, and `smoke`
//! reported whether a driver was compiled in -- which `doctor` already did.
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
//!     config, or a config asking for a driver this binary was not built with.

use std::path::Path;
use std::process::Command;

use anyhow::Result;

use crate::ui::{Mark, Palette};

/// Everything `doctor` checked, and the verdict that follows from it.
///
/// Collected first and rendered second, so the table and the JSON cannot drift
/// into disagreeing about `ready` -- which is the one thing a readiness probe
/// reads. That discipline used to be this command's alone, held by a comment;
/// it is [`crate::ui::Report`] now and every command has it.
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
pub fn run(global: &startup::GlobalArgs) -> Result<crate::ui::Answer> {
    let mut warnings = 0usize;
    let mut passes = 0usize;
    let mut failures = 0usize;

    let mut sections: Vec<(&'static str, Vec<(String, String, Status)>)> = Vec::new();

    sections.push(("system", vec![check_platform(), check_py_runtime()]));
    sections.push(("gpus", check_gpus()));
    sections.push((
        "drivers",
        pie_worker::driver_ffi::compiled_embedded()
            .iter()
            .map(|(name, on)| {
                if *on {
                    (name.to_string(), "compiled in".to_string(), Status::Pass)
                } else {
                    // A driver you did not build is not a fault until the
                    // config asks for it -- which the config section checks.
                    // The cargo feature, which is NOT the driver name with
                    // its underscore swapped: `cuda_native` builds from
                    // `driver-cuda`, and the derived spelling named a feature
                    // that does not exist.
                    let feature = match *name {
                        "cuda_native" => "driver-cuda",
                        "metal" => "driver-metal",
                        "dummy" => "driver-dummy",
                        other => other,
                    };
                    (
                        name.to_string(),
                        format!("not compiled (build with --features {feature})"),
                        Status::Warn,
                    )
                }
            })
            .collect(),
    ));
    // Last, because its verdict depends on everything above: whether a config
    // is servable is a question about this binary and this machine, not about
    // the file alone.
    let (path, origin) = startup::cli_config_path(global);
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
fn check_config(path: &Path, origin: startup::Origin) -> Vec<(String, String, Status)> {
    if !path.exists() {
        return if origin == startup::Origin::Default {
            vec![(
                "config".into(),
                format!("none at {} — running on defaults", crate::ui::short_path(path)),
                Status::Warn,
            )]
        } else {
            vec![(
                "config".into(),
                format!("{} does not exist ({})", crate::ui::short_path(path), origin.describe()),
                Status::Fail,
            )]
        };
    }

    let combined = match crate::derive::read_config_file(path) {
        Ok(c) => c,
        Err(e) => return vec![("config".into(), format!("{}: {e}", crate::ui::short_path(path)), Status::Fail)],
    };
    let worker = match crate::derive::derive_standalone(&combined) {
        Ok((_controller, _gateway, worker)) => worker,
        // `{:#}` so the chain reaches the line and column, which is the whole
        // value of being told the config is bad.
        Err(e) => {
            return vec![("config".into(), format!("{}: {e:#}", crate::ui::short_path(path)), Status::Fail)];
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
    match pie_worker::weights::resolve(&worker.model.model) {
        Ok(resolved) => out.push((
            "weights".into(),
            match resolved {
                pie_worker::weights::Model::Artifact(path) => {
                    format!("artifact {}", crate::ui::short_path(&path))
                }
                pie_worker::weights::Model::Snapshot(path) => format!(
                    "raw snapshot {} — `pie model import` makes an artifact",
                    crate::ui::short_path(&path)
                ),
            },
            Status::Pass,
        )),
        Err(error) => out.push(("weights".into(), format!("{error}"), Status::Fail)),
    }
    // The check the old `pie check` could not make and `pie smoke` made in
    // isolation: the config names a driver, and this binary either has it or
    // does not.
    let kind = worker.model.driver.kind.as_str();
    let compiled = pie_worker::driver_ffi::compiled_embedded()
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
                "{} asks for the {kind} driver, which this binary was not built with",
                worker.model.name
            ),
            Status::Fail,
        )
    });
    out
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
/// A warning rather than a failure, and it used to be `pie runtime status` --
/// a top-level command for a question, next to `pie runtime install` for
/// something `serve` already does on the way up. Both are gone: the answer
/// belongs with the other "will this work here" answers, and the fix happens
/// by itself.
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

fn check_gpus() -> Vec<(String, String, Status)> {
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
/// The plan's §7 makes this mandatory rather than nice. `pie config tune`
/// writes both of the things below: its first stage measures the forward shape
/// and its second the batching knobs. A machine where it has not run gets the
/// analytic planner's judgement, which is a model of the machine rather than
/// the machine.
/// That is a perfectly serviceable state -- it is what every deployment has had
/// until now -- but it is not one an operator should have to infer from the
/// absence of keys in a file.
///
/// Warnings, never failures. An unmeasured machine serves.
///
/// This briefly grew a blocking check for a `calibrate_planner` left on in the
/// config. The check is gone because what it guarded is gone: calibration is a
/// stage of `pie config tune` now, set on a config derived in memory, and
/// the key is `#[serde(skip)]` — a file cannot carry it, so it cannot be left
/// on. Guarding a state that no longer exists is how a checklist grows a step
/// nobody can explain.
fn check_tuning(config_path: &std::path::Path) -> Vec<(String, String, Status)> {
    let file: toml::Value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())
        .unwrap_or_else(|| toml::Value::Table(Default::default()));
    let set = |key: &str| {
        pie_worker::config_schema::lookup(&file, key).map(|v| v.to_string())
    };

    let mut checks = Vec::new();

    match (set("driver.max_forward_tokens"), set("driver.max_forward_requests")) {
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
            "derived by the planner's analytic score (`pie config tune --for ...`)".to_string(),
            Status::Warn,
        )),
    }

    let frame_knobs = ["runtime.frame_size", "runtime.frame_submit_depth", "runtime.frame_dispatch_depth"];
    let pinned: Vec<&str> = frame_knobs.iter().copied().filter(|k| set(k).is_some()).collect();
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

    // The driver's own measurement, keyed by (device, model, tp, kv format) --
    // so its mere presence is not proof it applies HERE. Saying "measured on
    // some machine" would be worse than saying nothing, hence the wording.
    let profile_cache = pie_worker::state::planner_profile_path();
    checks.push(if profile_cache.is_file() {
        (
            "planner profile".to_string(),
            format!(
                "{} exists; the driver checks its key at boot",
                crate::ui::short_path(&profile_cache)
            ),
            Status::Pass,
        )
    } else {
        (
            "planner profile".to_string(),
            "none; the forward step has never been timed here (`pie config tune` measures it)"
                .to_string(),
            Status::Warn,
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
        // and an unmeasured machine is a perfectly serviceable one -- it is
        // what every deployment had until now. Nothing here may block a boot.
        for config in ["", "[driver]\nmemory_profile = \"latency\"\n"] {
            let checks = tuning_of(config);
            assert!(
                !checks.iter().any(|(_, _, status)| *status == Status::Fail),
                "nothing here blocks a boot: {checks:?}"
            );
        }
    }
}
