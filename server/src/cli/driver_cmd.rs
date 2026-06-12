//! `pie driver <type> ...` — diagnostics + persisted config for the
//! per-driver Python venv.
//!
//! Three CLI shapes:
//!
//! ```text
//! pie driver list                    [-c <serve-toml>]
//!
//! pie driver <subprocess-type> install [<path>] [--run]
//! pie driver <subprocess-type> doctor
//! pie driver <subprocess-type> set venv|python <path>
//! pie driver <subprocess-type> unset venv|python
//! pie driver <subprocess-type> show
//! pie driver <subprocess-type> exec -- <cmd...>
//!
//! pie driver <embedded-type> doctor   (cuda_native | portable | dummy)
//! ```
//!
//! Subprocess types are `dev` / `vllm` / `sglang`. Their commands all
//! flow through [`crate::python_resolve`] for venv resolution, and
//! `set` / `unset` edit `~/.pie/drivers.toml` in place.
//!
//! Embedded types are `portable` / `cuda_native` / `dummy`. Only
//! `doctor` is meaningful — the venv knobs are no-ops here, since
//! these drivers run as static libs in this binary.

use std::ffi::OsString;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

#[cfg(unix)]
use anyhow::anyhow;
use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use crate::python_resolve::{self, DriversConfig, PythonBlock};
use crate::subprocess_driver::SubprocessFlavor;

/// `pie driver` subcommand tree.
#[derive(Subcommand, Debug)]
pub enum DriverCmd {
    /// List known driver types and which appear in the loaded config.
    List(ListArgs),

    /// `pie driver dev <action>` — reference Python driver.
    Dev {
        #[command(subcommand)]
        action: PerDriverCmd,
    },
    /// `pie driver vllm <action>` — vLLM-backed driver.
    Vllm {
        #[command(subcommand)]
        action: PerDriverCmd,
    },
    /// `pie driver sglang <action>` — SGLang-backed driver.
    Sglang {
        #[command(subcommand)]
        action: PerDriverCmd,
    },
    /// `pie driver portable <action>` — embedded ggml driver.
    Portable {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
    /// `pie driver cuda-native <action>` — embedded CUDA driver.
    #[command(name = "cuda-native")]
    CudaNative {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
    /// `pie driver dummy <action>` — Rust dummy driver.
    Dummy {
        #[command(subcommand)]
        action: EmbeddedCmd,
    },
}

#[derive(Args, Debug)]
pub struct ListArgs {
    /// Path to a serve config TOML. If provided, prints which driver
    /// types each `[[model]]` uses.
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum PerDriverCmd {
    /// Print install recipe (uv venv + uv pip install). With `--run`,
    /// executes the commands.
    Install {
        /// Where to create the venv. Defaults to `~/.pie/venvs/<driver>`.
        path: Option<PathBuf>,
        /// Run the commands instead of just printing them.
        #[arg(long)]
        run: bool,
    },
    /// Resolve the venv, run preflight imports.
    Doctor,
    /// Persist a knob into `~/.pie/drivers.toml`.
    Set {
        #[command(subcommand)]
        field: SetField,
    },
    /// Remove a persisted knob from `~/.pie/drivers.toml`.
    Unset {
        #[command(subcommand)]
        field: UnsetField,
    },
    /// Print the resolved interpreter + which precedence step matched.
    Show,
    /// `pie driver <type> exec -- <cmd...>` — run a command under the
    /// resolved interpreter. Useful for debugging (`pip list`, etc.).
    Exec {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<OsString>,
    },
}

#[derive(Subcommand, Debug)]
pub enum SetField {
    /// `pie driver <type> set venv <path>` — `<path>/bin/python` is
    /// the resolved interpreter.
    Venv { path: PathBuf },
    /// `pie driver <type> set python <path>` — `<path>` is the
    /// interpreter directly.
    Python { path: PathBuf },
}

#[derive(Subcommand, Debug)]
pub enum UnsetField {
    /// Remove the persisted `venv` knob.
    Venv,
    /// Remove the persisted `python` knob.
    Python,
}

#[derive(Subcommand, Debug)]
pub enum EmbeddedCmd {
    /// Diagnose the embedded driver (feature-gate, GPU visibility).
    Doctor,
}

/// Top-level dispatcher.
pub fn run(cmd: DriverCmd) -> Result<()> {
    match cmd {
        DriverCmd::List(args) => list(args),

        DriverCmd::Dev { action } => run_subprocess(SubprocessFlavor::Dev, action),
        DriverCmd::Vllm { action } => run_subprocess(SubprocessFlavor::Vllm, action),
        DriverCmd::Sglang { action } => run_subprocess(SubprocessFlavor::Sglang, action),

        DriverCmd::Portable { action } => run_embedded("portable", action),
        DriverCmd::CudaNative { action } => run_embedded("cuda_native", action),
        DriverCmd::Dummy { action } => run_embedded("dummy", action),
    }
}

// -----------------------------------------------------------------------------
// list
// -----------------------------------------------------------------------------

fn list(args: ListArgs) -> Result<()> {
    println!("Subprocess drivers (Python wheels):");
    for f in [
        SubprocessFlavor::Dev,
        SubprocessFlavor::Vllm,
        SubprocessFlavor::Sglang,
    ] {
        println!("  {:<8}  python -m {}", f.as_str(), f.module_name());
    }
    println!();
    println!("Embedded drivers (compiled into this binary by feature):");
    for (name, on) in crate::driver_ffi::compiled_embedded() {
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

    if let Some(path) = args.config {
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("reading config {path:?}"))?;
        let cfg: crate::config::Config =
            toml::from_str(&text).with_context(|| format!("parsing {path:?}"))?;
        println!();
        println!("[[model]] entries in {}:", path.display());
        for m in &cfg.models {
            println!(
                "  {:<24}  type = {:?}, devices = {:?}",
                m.name, m.driver.kind, m.driver.device,
            );
        }
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// subprocess actions
// -----------------------------------------------------------------------------

fn run_subprocess(flavor: SubprocessFlavor, action: PerDriverCmd) -> Result<()> {
    match action {
        PerDriverCmd::Install { path, run } => install(flavor, path, run),
        PerDriverCmd::Doctor => doctor_subprocess(flavor),
        PerDriverCmd::Set { field } => set(flavor, field),
        PerDriverCmd::Unset { field } => unset(flavor, field),
        PerDriverCmd::Show => show(flavor),
        PerDriverCmd::Exec { args } => exec(flavor, args),
    }
}

fn install(flavor: SubprocessFlavor, path: Option<PathBuf>, run: bool) -> Result<()> {
    #[cfg(windows)]
    if matches!(
        flavor,
        SubprocessFlavor::Dev | SubprocessFlavor::Vllm | SubprocessFlavor::Sglang
    ) {
        bail!(
            "{} driver is not supported on Windows. Please use the portable driver on Windows, or use WSL/Linux for dev, vllm, or sglang drivers.",
            flavor.as_str()
        );
    }

    let venv = path.unwrap_or_else(|| crate::paths::pie_home().join("venvs").join(flavor.as_str()));
    let python_bin = venv.join("bin").join("python");

    let wheel_extras = match flavor {
        SubprocessFlavor::Dev => "pie-driver-dev[cu128]",
        SubprocessFlavor::Vllm => "pie-driver-vllm",
        SubprocessFlavor::Sglang => "pie-driver-sglang",
    };

    // The recipe.
    let create_cmd = vec![
        OsString::from("uv"),
        OsString::from("venv"),
        venv.clone().into_os_string(),
        OsString::from("--python"),
        OsString::from("3.12"),
    ];
    let install_cmd = vec![
        OsString::from("uv"),
        OsString::from("pip"),
        OsString::from("install"),
        OsString::from("--python"),
        python_bin.clone().into_os_string(),
        OsString::from(wheel_extras),
    ];

    if !run {
        // Print mode — copy-pasteable shell.
        println!("# Recipe for {} driver:", flavor.as_str());
        println!("{}", display_cmd(&create_cmd));
        println!("{}", display_cmd(&install_cmd));
        println!();
        println!(
            "Run with `--run` to execute these. To register the venv as the default \
             for this driver:\n  pie driver {} set venv {}",
            flavor.as_str(),
            venv.display(),
        );
        return Ok(());
    }

    // Run mode.
    println!("$ {}", display_cmd(&create_cmd));
    let status = Command::new(&create_cmd[0])
        .args(&create_cmd[1..])
        .status()
        .with_context(|| format!("spawning {:?} (is `uv` installed?)", create_cmd[0]))?;
    if !status.success() {
        bail!("uv venv exited with {status}");
    }
    println!("$ {}", display_cmd(&install_cmd));
    let status = Command::new(&install_cmd[0])
        .args(&install_cmd[1..])
        .status()
        .with_context(|| format!("spawning {:?}", install_cmd[0]))?;
    if !status.success() {
        bail!("uv pip install exited with {status}");
    }
    println!();
    println!(
        "✓ Installed. Register as the default for this driver with:\n  \
         pie driver {} set venv {}",
        flavor.as_str(),
        venv.display(),
    );
    Ok(())
}

fn doctor_subprocess(flavor: SubprocessFlavor) -> Result<()> {
    let global = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;
    let empty = toml::Table::new();
    let resolved = match python_resolve::resolve_python(flavor, &empty, Some(&global)) {
        Ok(r) => r,
        Err(e) => {
            println!("[{}]", flavor.as_str());
            println!("  python: NOT RESOLVED");
            println!("  reason: {e}");
            return Err(e);
        }
    };

    println!("[{}]", flavor.as_str());
    println!("  python: {}", resolved.path.display());
    println!("  source: {}", resolved.source);

    // 1. Interpreter is callable + reports its version.
    let version = capture(&resolved.path, &["--version"]).context("python --version")?;
    println!("  python --version: {}", version.trim());

    // 2. Driver wheel imports.
    let module = flavor.module_name();
    match capture(
        &resolved.path,
        &["-c", &format!("import {module}; print({module}.__file__)")],
    ) {
        Ok(out) => println!("  import {module}: {}", out.trim()),
        Err(e) => println!("  import {module}: FAIL ({e})"),
    }

    // 3. Optional torch.cuda visibility — informational, never fails
    // the doctor.
    match capture(
        &resolved.path,
        &[
            "-c",
            "import torch; print(f'torch {torch.__version__}, cuda available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}')",
        ],
    ) {
        Ok(out) => println!("  torch.cuda: {}", out.trim()),
        Err(e) => println!("  torch.cuda: skipped ({e})"),
    }

    Ok(())
}

fn set(flavor: SubprocessFlavor, field: SetField) -> Result<()> {
    let mut cfg = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;
    let entry = cfg
        .drivers
        .entry(flavor.as_str().to_string())
        .or_insert_with(PythonBlock::default);
    match field {
        SetField::Venv { path } => {
            entry.venv = Some(path.display().to_string());
            entry.python = None;
        }
        SetField::Python { path } => {
            entry.python = Some(path.display().to_string());
            entry.venv = None;
        }
    }
    save_drivers_config(&cfg)?;
    println!(
        "Set [driver.{}] in {}",
        flavor.as_str(),
        python_resolve::drivers_config_path().display(),
    );
    show(flavor)
}

fn unset(flavor: SubprocessFlavor, field: UnsetField) -> Result<()> {
    let mut cfg = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;
    if let Some(entry) = cfg.drivers.get_mut(flavor.as_str()) {
        match field {
            UnsetField::Venv => entry.venv = None,
            UnsetField::Python => entry.python = None,
        }
        // Drop the empty entry so the file stays tidy.
        if entry.venv.is_none() && entry.python.is_none() {
            cfg.drivers.remove(flavor.as_str());
        }
    }
    save_drivers_config(&cfg)?;
    println!(
        "Removed from [driver.{}] in {}",
        flavor.as_str(),
        python_resolve::drivers_config_path().display(),
    );
    Ok(())
}

fn show(flavor: SubprocessFlavor) -> Result<()> {
    let global = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;
    let empty = toml::Table::new();
    match python_resolve::resolve_python(flavor, &empty, Some(&global)) {
        Ok(r) => {
            println!("driver: {}", flavor.as_str());
            println!("python: {}", r.path.display());
            println!("source: {}", r.source);
            Ok(())
        }
        Err(e) => {
            // `show` doesn't fail noisily — the error message is the
            // interesting payload.
            println!("driver: {}", flavor.as_str());
            println!("python: NOT RESOLVED");
            println!();
            println!("{e}");
            Err(e)
        }
    }
}

fn exec(flavor: SubprocessFlavor, args: Vec<OsString>) -> Result<()> {
    if args.is_empty() {
        bail!(
            "no command to exec. Usage: pie driver {} exec -- <cmd...>",
            flavor.as_str()
        );
    }
    let global = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;
    let empty = toml::Table::new();
    let resolved = python_resolve::resolve_python(flavor, &empty, Some(&global))?;

    // First arg may be a python sub-tool name — rewrite known ones to
    // their interpreter-relative paths so users can `pie driver vllm
    // exec -- python -V` and have it resolve to the venv's python.
    let (program, rest) = match args[0].to_str() {
        Some("python") | Some("python3") => (
            resolved.path.into_os_string(),
            args.into_iter().skip(1).collect::<Vec<_>>(),
        ),
        _ => (
            args[0].clone(),
            args.into_iter().skip(1).collect::<Vec<_>>(),
        ),
    };

    #[cfg(unix)]
    {
        let err = Command::new(&program).args(&rest).exec();
        Err(anyhow!("execvp({program:?}) failed: {err}"))
    }

    #[cfg(windows)]
    {
        let status = Command::new(&program)
            .args(&rest)
            .status()
            .with_context(|| format!("spawn {program:?}"))?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

// -----------------------------------------------------------------------------
// embedded actions
// -----------------------------------------------------------------------------

fn run_embedded(name: &str, action: EmbeddedCmd) -> Result<()> {
    match action {
        EmbeddedCmd::Doctor => doctor_embedded(name),
    }
}

fn doctor_embedded(name: &str) -> Result<()> {
    println!("[{}]", name);
    let compiled = match name {
        "portable" => cfg!(feature = "driver-portable"),
        "cuda_native" => cfg!(feature = "driver-cuda"),
        "dummy" => true,
        _ => false,
    };
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
            "  rebuild with `cargo install pie-server --features driver-{}` \
             (or `--features driver-portable,driver-cuda` to keep both).",
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

/// Capture stdout of a subprocess; treat non-zero exit as an error
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

fn save_drivers_config(cfg: &DriversConfig) -> Result<()> {
    let path = python_resolve::drivers_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {parent:?}"))?;
    }
    // Render manually; serializing through serde would carry the
    // `#[serde(default)]` defaults and produce a noisier file.
    let mut out = String::new();
    if cfg.python.venv.is_some() || cfg.python.python.is_some() {
        out.push_str("[python]\n");
        if let Some(v) = &cfg.python.venv {
            out.push_str(&format!("venv = \"{v}\"\n"));
        }
        if let Some(p) = &cfg.python.python {
            out.push_str(&format!("python = \"{p}\"\n"));
        }
        out.push('\n');
    }
    for (k, block) in &cfg.drivers {
        out.push_str(&format!("[driver.{k}]\n"));
        if let Some(v) = &block.venv {
            out.push_str(&format!("venv = \"{v}\"\n"));
        }
        if let Some(p) = &block.python {
            out.push_str(&format!("python = \"{p}\"\n"));
        }
        out.push('\n');
    }
    std::fs::write(&path, out).with_context(|| format!("writing {path:?}"))?;
    Ok(())
}

/// Render an argv slice as a copy-pasteable shell command. Quotes the
/// args naïvely — fine for paths without exotic chars.
fn display_cmd(argv: &[OsString]) -> String {
    argv.iter()
        .map(|a| {
            let s = a.to_string_lossy();
            if s.contains(' ') || s.contains('"') {
                format!("\"{}\"", s.replace('"', "\\\""))
            } else {
                s.into_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
