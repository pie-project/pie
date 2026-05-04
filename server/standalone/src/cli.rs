//! `pie` CLI: clap-based subcommand dispatcher mirroring `pie_cli`.
//!
//! Subcommand layout is identical to the Python `pie_cli`:
//!
//! ```text
//! pie serve   [--config --host --port --no-auth --verbose --no-snapshot]
//! pie run     <inferlet> [--config --port --path --manifest -- args...]
//! pie config  init|show|set
//! pie auth    add|remove|list
//! pie model   list|download|remove
//! pie doctor
//!
//! Diagnostics (standalone-specific):
//!   pie check   <toml>     Validate a config TOML.
//!   pie smoke   [--rpc]    FFI / RpcServer smoke test.
//! ```
//!
//! Authoring (`pie new`, `pie build`) lives in [`crate::cli::bakery_bridge`]
//! once M7 lands — for now those subcommands print "install bakery"
//! and exit non-zero so users discover the gap.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod auth_cmd;
mod bakery_cmd;
mod config_cmd;
mod diag_cmd;
mod doctor_cmd;
mod model_cmd;
mod monitor;
mod run_cmd;
mod serve_cmd;

/// Top-level `pie` invocation.
#[derive(Parser, Debug)]
#[command(
    name = "pie",
    version,
    about = "Pie: Programmable Inference Engine (standalone)",
    disable_help_subcommand = true,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the engine.
    Serve(serve_cmd::ServeArgs),

    /// Run a single inferlet against a one-shot engine instance.
    Run(run_cmd::RunArgs),

    /// Manage configuration (init / show / set).
    Config {
        #[command(subcommand)]
        cmd: config_cmd::ConfigCmd,
    },

    /// Manage authorized users for the auth backend.
    Auth {
        #[command(subcommand)]
        cmd: auth_cmd::AuthCmd,
    },

    /// Manage HuggingFace-cached models (list / download / remove).
    Model {
        #[command(subcommand)]
        cmd: model_cmd::ModelCmd,
    },

    /// Create a new inferlet project (Rust by default, or TypeScript
    /// with `--ts`). Forwards to `python3 -m bakery create`.
    New(bakery_cmd::NewArgs),

    /// Build an inferlet to a `.wasm` component. Auto-detects Rust /
    /// Python / JS / TS. Forwards to `python3 -m bakery build`.
    Build(bakery_cmd::BuildArgs),

    /// Environment + dependency health check.
    Doctor,

    /// Validate a TOML config without booting the engine.
    Check {
        /// Path to the config TOML.
        config: PathBuf,
    },

    /// FFI / RpcServer smoke test for diagnostics.
    Smoke {
        /// Run the RpcServer smoke instead of the FFI smoke.
        #[arg(long)]
        rpc: bool,
    },
}

/// Parse + dispatch. Top-level entry from `main.rs`.
pub fn dispatch() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve_cmd::run(args),
        Command::Run(args) => run_cmd::run(args),
        Command::Config { cmd } => config_cmd::run(cmd),
        Command::Auth { cmd } => auth_cmd::run(cmd),
        Command::Model { cmd } => model_cmd::run(cmd),
        Command::New(args) => bakery_cmd::run_new(args),
        Command::Build(args) => bakery_cmd::run_build(args),
        Command::Doctor => doctor_cmd::run(),
        Command::Check { config } => diag_cmd::check(&config),
        Command::Smoke { rpc } => diag_cmd::smoke(rpc),
    }
}
