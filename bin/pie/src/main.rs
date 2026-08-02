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
use pie_bin::{compose, derive, local, ops, ui};
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

    /// Emit one JSON document instead of the human rendering. Works on every
    /// command.
    //
    // Global for that last reason. It was a per-subcommand flag on five of them
    // and absent from the rest, which is not a policy so much as the order
    // things were written in -- `pie cache list` was scriptable and `pie config
    // show` was not, for no reason either could state.
    #[arg(long, global = true)]
    json: bool,

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

    /// The models pie serves (list / info / import / build / remove).
    Model {
        #[command(subcommand)]
        cmd: ops::model::ModelCmd,
    },

    /// What pie has written under `$PIE_HOME` (list / clear).
    Cache {
        #[command(subcommand)]
        cmd: ops::cache::CacheCmd,
    },

    /// Manage configuration (list / show / set / unset / edit / init / tune).
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
            eprintln!("  {}", palette.dim(line));
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

    // `serve` is the one command that is not an op: it takes the full daemon
    // `init` (banner, config, metrics) rather than the light CLI one, and it
    // ends on a signal rather than with an answer.
    if let Command::Serve = cli.command {
        return serve(cli.global).await;
    }

    // Once, for every op. It was written out at the head of all six arms, which
    // is the kind of repetition that stays correct only until somebody adds a
    // seventh arm.
    startup::init_cli(&cli.global)?;

    // What each op decides for itself is whether it blocks and what it answers
    // with. The threading policy used to be made arm by arm -- `model` got a
    // `spawn_blocking`, `cache` walked weight-sized directory trees on the
    // reactor, and `config` carried a comment explaining why it was no longer
    // `spawn_blocking` -- so it is stated here instead, in one place where the
    // three answers can be compared.
    let answer = match cli.command {
        // Handled above; `Command` has no other arm that skips `init_cli`.
        Command::Serve => unreachable!("serve returns before the op dispatch"),

        // Blocking: HF downloads and multi-gigabyte checkpoint rewrites.
        Command::Model { cmd } => tokio::task::spawn_blocking(move || ops::model::run(cmd)).await??,

        // Blocking: `cache list` walks the whole of `$PIE_HOME`, and `doctor`
        // shells out to `nvidia-smi`.
        Command::Cache { cmd } => tokio::task::spawn_blocking(move || ops::cache::run(cmd)).await??,
        Command::Doctor => {
            let global = cli.global.clone();
            tokio::task::spawn_blocking(move || ops::doctor::run(&global)).await??
        }

        // Async: these reach the engine or the registry over the network.
        Command::Run(args) => ops::run::run(&cli.global, args).await?,
        Command::Config { cmd } => ops::config::run(cmd, &cli.global).await?,
        Command::Inferlet { cmd } => ops::inferlet::run(cmd, &cli.global).await?,
    };

    let code = answer.code();
    ui::present(answer, cli.json)?;
    Ok(code)
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
    tokio::task::spawn_blocking(local::py_runtime::ensure_installed_best_effort)
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
