//! `pie doctor` — overall environment readiness check.
//!
//! Mirrors `pie/src/pie_cli/commands/doctor.py` in spirit: a quick
//! scan that surfaces the user's platform, GPUs, compiled-in drivers,
//! and per-subprocess-driver venv resolution status. Falls through to
//! the per-driver checks (`pie driver <type> doctor`) for deep
//! diagnostics.
//!
//! Exit codes:
//!   * 0  — no critical failures (warnings allowed)
//!   * 1  — critical failure (e.g. nothing compiled in, all venvs missing)

use std::process::Command;

use anyhow::Result;

use crate::python_resolve::{self, DriversConfig};
use crate::subprocess_driver::SubprocessFlavor;

/// `pie doctor` entry point.
pub fn doctor() -> Result<()> {
    let mut critical = false;
    let mut warnings = 0usize;
    let mut passes = 0usize;

    println!("Pie standalone — environment doctor\n");

    // ── System ────────────────────────────────────────────────────────────
    println!("[system]");
    let (key, value, status) = check_platform();
    print_check(&key, &value, status);
    tally(status, &mut passes, &mut warnings, &mut critical);

    // ── GPUs ──────────────────────────────────────────────────────────────
    println!("\n[gpus]");
    for (key, value, status) in check_gpus() {
        print_check(&key, &value, status);
        tally(status, &mut passes, &mut warnings, &mut critical);
    }

    // ── Embedded drivers ──────────────────────────────────────────────────
    println!("\n[embedded drivers]");
    for (name, on) in crate::driver_ffi::compiled_embedded() {
        let (val, st) = if on {
            ("compiled in".to_string(), Status::Pass)
        } else {
            ("not compiled".to_string(), Status::Warn)
        };
        print_check(name, &val, st);
        tally(st, &mut passes, &mut warnings, &mut critical);
    }

    // ── Subprocess drivers ────────────────────────────────────────────────
    println!("\n[subprocess drivers]");
    let global = DriversConfig::load().unwrap_or_default();
    let empty = toml::Table::new();
    for flavor in [
        SubprocessFlavor::Dev,
        SubprocessFlavor::Vllm,
        SubprocessFlavor::Sglang,
    ] {
        let key = flavor.as_str();
        match python_resolve::resolve_python(flavor, &empty, Some(&global)) {
            Ok(resolved) => {
                if !resolved.path.exists() {
                    print_check(
                        key,
                        &format!("{} (DOES NOT EXIST)", resolved.path.display()),
                        Status::Warn,
                    );
                    tally(Status::Warn, &mut passes, &mut warnings, &mut critical);
                    continue;
                }
                // Try a fast import probe; non-zero exit = wheel missing.
                let module = flavor.module_name();
                let import = Command::new(&resolved.path)
                    .args(["-c", &format!("import {module}")])
                    .output();
                let (val, st) = match import {
                    Ok(out) if out.status.success() => (
                        format!("{} (import {} OK)", resolved.path.display(), module),
                        Status::Pass,
                    ),
                    Ok(out) => (
                        format!(
                            "{} (import {} FAILED: {})",
                            resolved.path.display(),
                            module,
                            String::from_utf8_lossy(&out.stderr)
                                .trim()
                                .lines()
                                .next()
                                .unwrap_or(""),
                        ),
                        Status::Warn,
                    ),
                    Err(e) => (
                        format!("{} (cannot exec: {})", resolved.path.display(), e),
                        Status::Warn,
                    ),
                };
                print_check(key, &val, st);
                tally(st, &mut passes, &mut warnings, &mut critical);
            }
            Err(_) => {
                // resolve_python returns a multi-line error; collapse to
                // the headline + steer to the install hint.
                print_check(
                    key,
                    &format!("not configured (run `pie driver {key} install`)"),
                    Status::Warn,
                );
                tally(Status::Warn, &mut passes, &mut warnings, &mut critical);
            }
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!();
    if critical {
        println!(
            "✗ Critical issues found ({passes} passed, {warnings} warnings). \
             See above for details."
        );
        std::process::exit(1);
    } else if warnings > 0 {
        println!(
            "! Ready with warnings ({passes} passed, {warnings} warnings). \
             Run `pie driver <type> doctor` for deeper per-driver diagnostics."
        );
    } else {
        println!("✓ All checks passed ({passes} checks).");
    }
    Ok(())
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Status {
    Pass,
    Warn,
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

fn tally(s: Status, passes: &mut usize, warnings: &mut usize, critical: &mut bool) {
    match s {
        Status::Pass => *passes += 1,
        Status::Warn => *warnings += 1,
        Status::Fail => *critical = true,
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
