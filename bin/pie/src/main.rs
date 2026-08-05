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
use pie_bin::{compose, derive, ops, ui};
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

    /// Run one inferlet on a one-shot engine, print what it produces, exit.
    ///
    /// Arguments for the inferlet go after `--`:
    /// `pie run chat-completion -- --prompt "The capital of France is"`.
    Run(ops::run::RunArgs),

    /// The models pie serves (list / info / import / remove / optimize).
    Model {
        #[command(subcommand)]
        cmd: ops::model::ModelCmd,
    },

    /// What pie has written under `$PIE_HOME` (list / clear).
    Cache {
        #[command(subcommand)]
        cmd: ops::cache::CacheCmd,
    },

    /// Manage configuration (list / show / set / unset / edit / init).
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
    Doctor {
        /// Emit one JSON document instead of the report.
        #[arg(long)]
        json: bool,
    },
}

/// Die quietly when a reader goes away, the way every other CLI does.
///
/// Rust masks SIGPIPE at startup, so a `println!` into a closed pipe returns
/// EPIPE, and `println!` panics on a write error. `pie config list | head` was
/// therefore printing a panic and exiting non-zero -- for doing exactly what
/// `head` asks of it. Restoring the default disposition turns that back into
/// the signal it is.
#[cfg(unix)]
fn die_quietly_on_closed_pipe() {
    // SAFETY: setting a signal disposition to SIG_DFL before any threads that
    // could observe a different one. This is the documented workaround for
    // rust-lang/rust#62569.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn die_quietly_on_closed_pipe() {}

/// Render a failure the way the rest of the CLI renders everything else.
///
/// anyhow's default is `Error: <context>` followed by a `Caused by:` list,
/// which puts the least specific line first and in the one position a reader
/// actually looks. `pie config set worker.server.port abc` led with "setting
/// worker.server.port = \"abc\"" -- a restatement of the command -- and buried
/// "invalid type: string" under a heading.
///
/// Same glyph vocabulary as everything else: `✗` is what blocks.
fn report(error: &anyhow::Error) {
    let palette = ui::Palette::for_stream(ui::Stream::Stderr);
    eprintln!("{} {error}", ui::Mark::Blocked.render(&palette));
    for cause in error.chain().skip(1) {
        // Indented and dim: the chain is why, and the first line is what.
        // Line by line, because a cause can be several -- a TOML parse error
        // carries its own snippet, and letting those start at column 0 put the
        // detail outside the block it belongs to.
        for line in cause.to_string().lines() {
            eprintln!("  {}{line}{}", palette.dim(), palette.reset());
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    die_quietly_on_closed_pipe();
    match run().await {
        Ok(code) => code,
        Err(error) => {
            report(&error);
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve => serve(cli.global).await,

        Command::Run(args) => {
            startup::init_cli(&cli.global)?;
            // The inferlet's own exit status, not this command's: `pie run`
            // succeeded at running something that failed.
            ops::run::run(&cli.global, args).await
        }

        Command::Model { cmd } => {
            startup::init_cli(&cli.global)?;
            // `model::run` is synchronous + blocking (HF download); keep it off
            // the async reactor.
            tokio::task::spawn_blocking(move || ops::model::run(cmd)).await??;
            Ok(ExitCode::SUCCESS)
        }
        Command::Doctor { json } => {
            startup::init_cli(&cli.global)?;
            // The exit code IS the answer -- `pie doctor && pie serve` should
            // be a thing an operator can write.
            Ok(match ops::doctor::doctor(&cli.global, json)? {
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
            // Not `spawn_blocking`: `config init` used to download the
            // Python-WASM runtime through `reqwest::blocking`, which builds a
            // tokio runtime and drops it inside this one. Writing a config file
            // no longer reaches the network, so there is nothing to isolate.
            ops::config::run(cmd, &cli.global).await?;
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
