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

/// `pie doctor` entry point. Returns whether pie can boot.
pub fn doctor(global: &startup::GlobalArgs) -> Result<bool> {
    let mut warnings = 0usize;
    let mut passes = 0usize;
    let mut failures = 0usize;

    println!("Pie standalone — environment doctor\n");

    // ── System ────────────────────────────────────────────────────────────
    println!("[system]");
    let (key, value, status) = check_platform();
    print_check(&key, &value, status);
    tally(status, &mut passes, &mut warnings);

    // ── GPUs ──────────────────────────────────────────────────────────────
    println!("\n[gpus]");
    for (key, value, status) in check_gpus() {
        print_check(&key, &value, status);
        tally(status, &mut passes, &mut warnings);
    }

    // ── Embedded drivers ──────────────────────────────────────────────────
    println!("\n[embedded drivers]");
    for (name, on) in pie_worker::driver_ffi::compiled_embedded() {
        let (val, st) = if on {
            ("compiled in".to_string(), Status::Pass)
        } else {
            // A driver you did not build is not a fault until the config asks
            // for it -- which the section below is what checks.
            ("not compiled (build with --features driver-{})".replace("{}", &name.replace('_', "-")), Status::Warn)
        };
        print_check(name, &val, st);
        tally(st, &mut passes, &mut warnings);
    }

    // ── Config ────────────────────────────────────────────────────────────
    // Last, because its verdict depends on everything above: whether the
    // config is servable is a question about this binary and this machine, not
    // about the file alone.
    println!("\n[config]");
    let (path, origin) = startup::cli_config_path(global);
    for (key, value, status) in check_config(&path, origin) {
        print_check(&key, &value, status);
        match status {
            Status::Pass => passes += 1,
            Status::Warn => warnings += 1,
            Status::Fail => failures += 1,
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!();
    if failures > 0 {
        println!("✗ pie cannot boot here ({failures} blocking, {warnings} warnings).");
        return Ok(false);
    }
    if warnings > 0 {
        println!("! Ready, with warnings ({passes} passed, {warnings} warnings).");
    } else {
        println!("✓ Ready ({passes} checks).");
    }
    Ok(true)
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
                format!("none at {} — running on defaults", path.display()),
                Status::Warn,
            )]
        } else {
            vec![(
                "config".into(),
                format!("{} does not exist ({})", path.display(), origin.describe()),
                Status::Fail,
            )]
        };
    }

    let combined = match crate::derive::read_config_file(path) {
        Ok(c) => c,
        Err(e) => return vec![("config".into(), format!("{}: {e}", path.display()), Status::Fail)],
    };
    let worker = match crate::derive::derive_standalone(&combined) {
        Ok((_controller, _gateway, worker)) => worker,
        // `{:#}` so the chain reaches the line and column, which is the whole
        // value of being told the config is bad.
        Err(e) => {
            return vec![("config".into(), format!("{}: {e:#}", path.display()), Status::Fail)];
        }
    };

    let mut out = vec![(
        "config".into(),
        format!("{} parses", path.display()),
        Status::Pass,
    )];
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

#[derive(Copy, Clone, Eq, PartialEq)]
enum Status {
    Pass,
    /// True of the machine, not wrong with the installation. Never blocks.
    Warn,
    /// Would stop a boot.
    Fail,
}

fn print_check(key: &str, value: &str, status: Status) {
    let glyph = match status {
        Status::Pass => "✓",
        Status::Warn => "!",
        Status::Fail => "✗",
    };
    println!("  {glyph} {:<20} {}", key, value);
}

/// Only ever called for checks that cannot fail; the config section counts
/// its own, because it is the only one that can block.
fn tally(s: Status, passes: &mut usize, warnings: &mut usize) {
    match s {
        Status::Pass => *passes += 1,
        Status::Warn | Status::Fail => *warnings += 1,
    }
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
