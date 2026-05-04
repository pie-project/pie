//! `pie new` / `pie build` — thin shell-out to Python `bakery`.
//!
//! Inferlet authoring (project scaffolding, `componentize-py` runs,
//! cargo-component invocations, etc.) is genuinely Python today —
//! the `bakery` package owns the templates and build pipelines.
//! Reimplementing that surface in Rust would double the line count
//! and create a second source of truth for templates that drift.
//!
//! Instead, we forward to `python3 -m bakery <subcommand>` if it's
//! installed, and print a clear "install bakery" hint otherwise.
//! Identical UX to the Python `pie_cli/commands/{new,build}.py`
//! wrappers — those are also thin call-throughs.

use std::path::PathBuf;
use std::process::Command;

use anyhow::{Result, anyhow, bail};
use clap::Args;

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Name of the inferlet project.
    pub name: String,
    /// Create a TypeScript project instead of Rust.
    #[arg(short = 't', long)]
    pub ts: bool,
    /// Output directory.
    #[arg(short = 'o', long)]
    pub output: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Project directory or source file to build.
    pub input_path: PathBuf,
    /// Output `.wasm` file path.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
    /// Enable debug build (JS/Python: include source maps).
    #[arg(long)]
    pub debug: bool,
}

pub fn run_new(args: NewArgs) -> Result<()> {
    let mut argv: Vec<String> = vec!["create".to_string(), args.name];
    if args.ts {
        argv.push("--ts".to_string());
    }
    if let Some(out) = args.output {
        argv.push("--output".to_string());
        argv.push(out.display().to_string());
    }
    forward_to_bakery(&argv)
}

pub fn run_build(args: BuildArgs) -> Result<()> {
    let mut argv: Vec<String> = vec![
        "build".to_string(),
        args.input_path.display().to_string(),
        "--output".to_string(),
        args.output.display().to_string(),
    ];
    if args.debug {
        argv.push("--debug".to_string());
    }
    forward_to_bakery(&argv)
}

fn forward_to_bakery(argv: &[String]) -> Result<()> {
    ensure_bakery_available()?;

    // `python3 -m bakery` is the canonical invocation — works whether
    // `bakery` was pip-installed globally, in a venv, or via uv.
    let status = Command::new("python3")
        .arg("-m")
        .arg("bakery")
        .args(argv)
        .status()
        .map_err(|e| anyhow!("could not run `python3 -m bakery`: {e}"))?;

    if !status.success() {
        // Bakery printed its own diagnostics; mirror its exit code so
        // CI scripts see the right rc.
        let code = status.code().unwrap_or(1);
        bail!("bakery exited with code {code}");
    }
    Ok(())
}

/// Probe `python3 -c "import bakery"` once before forwarding. The
/// preflight catches both "python3 not on PATH" and "python3 found
/// but bakery module missing", and lets us print the install hint in
/// each case — the raw `ModuleNotFoundError` python emits otherwise
/// is opaque about *which* package to install.
fn ensure_bakery_available() -> Result<()> {
    let probe = Command::new("python3")
        .args(["-c", "import bakery"])
        .output();
    match probe {
        Ok(o) if o.status.success() => Ok(()),
        Ok(_) | Err(_) => bail!(
            "Inferlet authoring (`pie new` / `pie build`) requires the \
             Python `bakery` package, which is not importable from `python3`.\n\n\
             Install with:\n\
             \n    pip install pie-bakery\n\n\
             (or `uv pip install pie-bakery` inside a venv).\n\n\
             If python3 is missing entirely, install Python 3.10+ first."
        ),
    }
}
