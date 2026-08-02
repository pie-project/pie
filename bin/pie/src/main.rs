//! `pie` — the standalone composition root (Seam 4). A multi-call CLI that either
//! boots the full engine in-proc (`local`/`serve`, composing the controller +
//! gateway + worker libs over loopback) or runs a one-shot operational command
//! (`model`/`doctor`/...). The only crate that depends on all three role libs.
//!
//! Process model (Model A): `#[tokio::main]` owns the one runtime; every
//! subcommand runs on it. `local`/`serve` use the full daemon `startup::init`
//! + `run_until_signal`; one-shot ops use the light `startup::init_cli`.

use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pie_bin::{compose, derive, ops};
/// Top-level `pie` invocation. The shared global flags (`--config`,
/// `--log-level`, `--metrics-addr`) are flattened from `startup`.
#[derive(Parser, Debug)]
#[command(
    name = "pie",
    version,
    about = "Pie — Programmable Inference Engine (standalone)",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(flatten)]
    global: startup::GlobalArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Boot the engine. Binds `server.host`, which is loopback by default.
    Serve,

    /// Manage HuggingFace-cached models (list / download / remove / optimize).
    Model {
        #[command(subcommand)]
        cmd: ops::model::ModelCmd,
    },

    /// Manage the embedded Python-WASM runtime (install / status).
    Runtime {
        #[command(subcommand)]
        cmd: RuntimeCmd,
    },

    /// What pie has written under `$PIE_HOME` (list / clear).
    Cache {
        #[command(subcommand)]
        cmd: ops::cache::CacheCmd,
    },

    /// Manage configuration (list / show / set / unset / init).
    Config {
        #[command(subcommand)]
        cmd: ops::config::ConfigCmd,
    },

    /// The programs pie runs (list / info / download / remove).
    Inferlet {
        #[command(subcommand)]
        cmd: ops::inferlet::InferletCmd,
    },

    /// Will pie run here, and with this config? Exits non-zero if not.
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
        Command::Serve => serve(cli.global).await,

        Command::Model { cmd } => {
            startup::init_cli(&cli.global)?;
            // `model::run` is synchronous + blocking (HF download); keep it off
            // the async reactor.
            tokio::task::spawn_blocking(move || ops::model::run(cmd)).await??;
            Ok(ExitCode::SUCCESS)
        }
        Command::Runtime { cmd } => {
            startup::init_cli(&cli.global)?;
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
            startup::init_cli(&cli.global)?;
            // The exit code IS the answer -- `pie doctor && pie serve` should
            // be a thing an operator can write.
            Ok(match ops::doctor::doctor(&cli.global)? {
                true => ExitCode::SUCCESS,
                false => ExitCode::FAILURE,
            })
        }
        Command::Cache { cmd } => {
            startup::init_cli(&cli.global)?;
            ops::cache::run(cmd)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Config { cmd } => {
            startup::init_cli(&cli.global)?;
            ops::config::run(cmd, &cli.global)?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Inferlet { cmd } => {
            startup::init_cli(&cli.global)?;
            ops::inferlet::run(cmd, &cli.global).await?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// The `serve` path: full daemon `init` → derive the three typed role Configs
/// from the standalone TOML → boot the in-proc cluster (golf's compose) → run
/// until SIGINT/SIGTERM, then drain.
async fn serve(global: startup::GlobalArgs) -> anyhow::Result<ExitCode> {
    let ctx = startup::init(
        startup::BootSpec::pie().version(env!("CARGO_PKG_VERSION")),
        global,
    )?;
    // Provision the embedded Python-WASM runtime before booting — the worker
    // daemon never downloads (R3), so the standalone root does it. Best-effort:
    // a present runtime is a no-op; a failure is logged, not fatal here.
    tokio::task::spawn_blocking(ops::py_runtime::ensure_installed_best_effort)
        .await
        .ok();
    let (controller, gateway, worker) = derive::derive_standalone(ctx.config_str())?;
    let handle = compose::run_standalone(controller, gateway, worker).await?;
    tracing::info!(
        listen = %handle.listen_addr,
        worker = %handle.worker_addr,
        "pie standalone serving",
    );
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
