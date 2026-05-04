//! `pie doctor` — environment health check.
//!
//! Mirrors `pie/src/pie_cli/commands/doctor.py`, adapted for the
//! standalone binary (no PyTorch/Python deps to probe; instead we
//! report the embedded driver flavor + GPU visibility via the system
//! tools that are already on disk).

use std::process::Command;

use anyhow::Result;

use crate::driver_ffi;

pub fn run() -> Result<()> {
    let mut had_warning = false;

    println!("[System]");
    println!("  Platform        {}", platform());
    let gpus = detect_gpus();
    if gpus.is_empty() {
        println!("  GPU             ! none detected");
        had_warning = true;
    } else {
        for (i, name) in gpus.iter().enumerate() {
            println!("  GPU {i}           ✓ {name}");
        }
    }

    println!("\n[Driver]");
    println!("  Embedded flavor ✓ {} (statically linked)", driver_ffi::FLAVOR);
    let build_dir = option_env!("PIE_DRIVER_BUILD_DIR").unwrap_or("<in-process>");
    println!("  Build dir       {}", build_dir);

    println!("\n[Paths]");
    let pie_home = pie::path::get_pie_home();
    println!("  PIE_HOME        {}", pie_home.display());
    let cfg_path = crate::paths::default_config_path();
    let cfg_status = if cfg_path.exists() { "✓ present" } else { "! missing — run `pie config init`" };
    if !cfg_path.exists() {
        had_warning = true;
    }
    println!("  config.toml     {} ({cfg_status})", cfg_path.display());

    println!();
    if had_warning {
        println!("⚠  Ready with warnings.");
    } else {
        println!("✓  All checks passed.");
    }
    Ok(())
}

fn platform() -> String {
    format!(
        "{} {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
    )
}

/// Probe `nvidia-smi` for GPU names. Returns an empty list if the
/// binary is missing or fails. Best-effort; missing GPU info isn't
/// a fatal `pie doctor` failure.
fn detect_gpus() -> Vec<String> {
    let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}
