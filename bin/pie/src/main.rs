//! `pie` — the standalone composition root (Seam 4). A multi-call CLI that either
//! boots the full engine in-proc (`local`/`serve`, composing the controller +
//! gateway + worker libs over loopback) or runs a one-shot operational command
//! (`model`/`doctor`/...). The only crate that depends on all three role libs.
//!
//! Process model (Model A): `#[tokio::main]` owns the one runtime; every
//! subcommand runs on it. `local`/`serve` use the full daemon `bootstrap::init`
//! + `run_until_signal`; one-shot ops use the light `bootstrap::init_cli`.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pie_bin::{compose, derive, ops};
/// Top-level `pie` invocation. The shared global flags (`--config`,
/// `--log-level`, `--metrics-addr`) are flattened from `bootstrap`.
#[derive(Parser, Debug)]
#[command(
    name = "pie",
    version,
    about = "Pie — Programmable Inference Engine (standalone)",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(flatten)]
    global: bootstrap::GlobalArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Boot the full engine in-proc on loopback (developer / local use).
    Local,

    /// Boot the full engine in-proc on the configured client address.
    Serve,

    /// Manage HuggingFace-cached models (list / pull / remove).
    Model {
        #[command(subcommand)]
        cmd: ops::model::ModelCmd,
    },

    /// Manage the embedded Python-WASM runtime (install / status).
    Runtime {
        #[command(subcommand)]
        cmd: RuntimeCmd,
    },

    /// Manage authorized users for the auth backend (add / remove / list).
    Auth {
        #[command(subcommand)]
        cmd: ops::auth::AuthCmd,
    },

    /// Manage configuration (init / show / set).
    Config {
        #[command(subcommand)]
        cmd: ops::config::ConfigCmd,
    },

    /// Inspect inferlets from the registry.
    Inferlet {
        #[command(subcommand)]
        cmd: ops::inferlet::InferletCmd,
    },

    /// Create a new inferlet project (forwards to the Bakery tooling).
    New(ops::bakery::NewArgs),

    /// Build an inferlet to a `.wasm` component (forwards to the Bakery tooling).
    Build(ops::bakery::BuildArgs),

    /// Manage per-driver venvs + diagnostics (`pie driver <type> ...`).
    Driver {
        #[command(subcommand)]
        cmd: ops::driver::DriverCmd,
    },

    /// Validate a standalone config TOML without booting the engine.
    Check {
        /// Path to the config TOML.
        config: PathBuf,
        /// Print the fully parsed config after validation.
        #[arg(long)]
        debug: bool,
    },

    /// FFI smoke test — invoke a driver's `--help` and report its exit code.
    Smoke {
        /// Accepted for backwards compatibility; no-op.
        #[arg(long)]
        rpc: bool,
        /// Which compiled driver flavor to invoke (`cuda`/`metal`/`dummy`).
        #[arg(long)]
        flavor: Option<String>,
    },

    /// Environment readiness check (platform, GPUs, compiled drivers).
    Doctor,
}

/// `pie runtime` — provision the embedded Python-WASM runtime. The download IO
/// lives here (R3), never in the worker daemon.
#[derive(Subcommand, Debug)]
enum RuntimeCmd {
    /// Download + install the runtime into the local cache (idempotent).
    Install,
    /// Report whether the runtime is installed and where.
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    match cli.command {
        Command::Local => serve(cli.global, compose::Mode::Local).await,
        Command::Serve => serve(cli.global, compose::Mode::Serve).await,

        Command::Model { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            // `model::run` is synchronous + blocking (HF download); keep it off
            // the async reactor.
            tokio::task::spawn_blocking(move || ops::model::run(cmd)).await??;
            Ok(ExitCode::SUCCESS)
        }
        Command::Runtime { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            match cmd {
                RuntimeCmd::Install => {
                    let dir =
                        tokio::task::spawn_blocking(|| ops::py_runtime::ensure_installed(false))
                            .await??;
                    println!("runtime installed at {}", dir.display());
                }
                RuntimeCmd::Status => {
                    let dir = ops::py_runtime::runtime_dir();
                    if ops::py_runtime::is_installed() {
                        println!("runtime installed at {}", dir.display());
                    } else {
                        println!("runtime not installed (expected at {})", dir.display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Doctor => {
            bootstrap::init_cli(&cli.global)?;
            ops::doctor::doctor()?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Auth { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            ops::auth::run(cmd)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Config { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            ops::config::run(cmd)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Inferlet { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            ops::inferlet::run(cmd).await?;
            Ok(ExitCode::SUCCESS)
        }
        Command::New(args) => {
            bootstrap::init_cli(&cli.global)?;
            ops::bakery::run_new(args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Build(args) => {
            bootstrap::init_cli(&cli.global)?;
            ops::bakery::run_build(args)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Driver { cmd } => {
            bootstrap::init_cli(&cli.global)?;
            ops::driver::run(cmd)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Check { config, debug } => {
            bootstrap::init_cli(&cli.global)?;
            ops::diag::check(&config, debug)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Smoke { rpc, flavor } => {
            bootstrap::init_cli(&cli.global)?;
            ops::diag::smoke(rpc, flavor.as_deref())?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// The `local`/`serve` path: full daemon `init` → derive the three typed role
/// Configs from the standalone TOML → boot the in-proc cluster (golf's compose)
/// → run until SIGINT/SIGTERM, then drain. One boot path, parameterized by mode.
async fn serve(global: bootstrap::GlobalArgs, mode: compose::Mode) -> anyhow::Result<ExitCode> {
    let ctx = bootstrap::init(
        bootstrap::BootSpec::pie().version(env!("CARGO_PKG_VERSION")),
        global,
    )?;
    // Provision the embedded Python-WASM runtime before booting — the worker
    // daemon never downloads (R3), so the standalone root does it. Best-effort:
    // a present runtime is a no-op; a failure is logged, not fatal here.
    tokio::task::spawn_blocking(ops::py_runtime::ensure_installed_best_effort)
        .await
        .ok();
    let (controller, gateway, worker) = derive::derive_standalone(ctx.config_str())?;
    let handle = compose::run_standalone(controller, gateway, worker, mode).await?;
    tracing::info!(
        listen = %handle.listen_addr,
        worker = %handle.worker_addr,
        ?mode,
        "pie standalone serving",
    );
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
