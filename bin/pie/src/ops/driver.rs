//! `pie driver <type> ...` — diagnostics for embedded drivers.
//!
//! CLI shape:
//!
//! ```text
//! pie driver list                    [-c <serve-toml>]
//!
//! pie driver <embedded-type> doctor   (cuda_native | metal | dummy)
//! ```
//!
//! Embedded types are `cuda_native` / `metal` / `dummy`.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Subcommand;

/// `pie driver` subcommand tree.
#[derive(Subcommand, Debug)]
pub enum DriverCmd {
    /// List known driver types and which the config asks for.
    List,

    /// `pie driver cuda-native <action>` — embedded CUDA driver.
    #[command(name = "cuda-native")]
    CudaNative {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
    /// `pie driver metal <action>` — embedded Apple Silicon Metal driver.
    Metal {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
    /// `pie driver dummy <action>` — Rust dummy driver.
    Dummy {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
}

#[derive(Subcommand, Debug)]
pub enum EmbeddedCmd {
    /// Diagnose the embedded driver (feature-gate, GPU visibility).
    Doctor,
}

/// Top-level dispatcher.
pub fn run(cmd: DriverCmd, global: &startup::GlobalArgs) -> Result<()> {
    match cmd {
        DriverCmd::List => list(global),

        DriverCmd::CudaNative { action } => run_embedded("cuda_native", action),
        DriverCmd::Metal { action } => run_embedded("metal", action),
        DriverCmd::Dummy { action } => run_embedded("dummy", action),
    }
}

// -----------------------------------------------------------------------------
// list
// -----------------------------------------------------------------------------

fn list(global: &startup::GlobalArgs) -> Result<()> {
    println!("Embedded drivers (compiled into this binary by feature):");
    for (name, on) in pie_worker::driver_ffi::compiled_embedded() {
        println!(
            "  {:<12} {}",
            name,
            if on {
                "(compiled in)"
            } else {
                "(not compiled)"
            },
        );
    }

    // Read whichever config the engine would boot from instead of asking for
    // one. Absent is not an error here -- the compiled-in list above is still
    // the answer to "what can this binary drive?", and the config only adds
    // "and what is it asked to".
    let (path, _) = startup::cli_config_path(global);
    if path.exists() {
        let cfg = crate::derive::load_worker_config(&path)?;
        println!();
        println!("[model] in {}:", path.display());
        let m = &cfg.model;
        println!(
            "  {:<24}  type = {:?}, devices = {:?}",
            m.name, m.driver.kind, m.driver.device,
        );
    }
    Ok(())
}

fn run_embedded(name: &str, action: EmbeddedCmd) -> Result<()> {
    match action {
        EmbeddedCmd::Doctor => doctor_embedded(name),
    }
}

fn doctor_embedded(name: &str) -> Result<()> {
    println!("[{}]", name);
    let compiled = pie_worker::driver_ffi::compiled_embedded()
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, on)| *on)
        .unwrap_or(false);
    println!(
        "  availability: {}",
        if compiled {
            "compiled in"
        } else {
            "NOT compiled in"
        }
    );
    if !compiled {
        println!(
            "  rebuild with `cargo install pie-worker --features driver-{}` \
             (or `--features driver-cuda,driver-metal` to keep both).",
            name.replace('_', "-"),
        );
    }
    if name == "cuda_native" {
        // nvidia-smi is the cheapest "GPU visible" probe; no link to
        // libnvidia-ml needed.
        match capture(
            &PathBuf::from("nvidia-smi"),
            &["--query-gpu=name,driver_version", "--format=csv,noheader"],
        ) {
            Ok(out) => {
                println!("  nvidia-smi:");
                for line in out.lines() {
                    println!("    {}", line);
                }
            }
            Err(e) => println!("  nvidia-smi: skipped ({e})"),
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

/// Capture stdout of a command; treat non-zero exit as an error
/// with stderr in the message.
fn capture(program: &std::path::Path, args: &[&str]) -> Result<String> {
    let out = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("spawning {:?}", program))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        bail!(
            "{:?} exited with {}: {}",
            program,
            out.status,
            stderr.trim(),
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}
