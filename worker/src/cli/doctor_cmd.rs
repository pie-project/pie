//! `pie doctor` — overall environment readiness check.
//!
//! Mirrors `pie/src/pie_cli/commands/doctor.py` in spirit: a quick
//! scan that surfaces the user's platform, GPUs, compiled-in drivers,
//! and embedded-driver availability.
//!
//! Exit codes:
//!   * 0  — no critical failures (warnings allowed)

use std::process::Command;

use anyhow::Result;

/// `pie doctor` entry point.
pub fn doctor() -> Result<()> {
    let mut warnings = 0usize;
    let mut passes = 0usize;

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
    for (name, on) in crate::driver_ffi::compiled_embedded() {
        let (val, st) = if on {
            ("compiled in".to_string(), Status::Pass)
        } else {
            ("not compiled".to_string(), Status::Warn)
        };
        print_check(name, &val, st);
        tally(st, &mut passes, &mut warnings);
    }

    // ── Summary ───────────────────────────────────────────────────────────
    println!();
    if warnings > 0 {
        println!(
            "! Ready with warnings ({passes} passed, {warnings} warnings). \
             Run `pie driver <type> doctor` for deeper embedded-driver diagnostics."
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
}

fn print_check(key: &str, value: &str, status: Status) {
    let glyph = match status {
        Status::Pass => "✓",
        Status::Warn => "!",
    };
    println!("  {glyph} {:<20} {}", key, value);
}

fn tally(s: Status, passes: &mut usize, warnings: &mut usize) {
    match s {
        Status::Pass => *passes += 1,
        Status::Warn => *warnings += 1,
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
