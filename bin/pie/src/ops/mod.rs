//! The command tree, one module per `pie` subcommand.
//!
//! Each runs after the light `startup::init_cli` (tracing + paths, no daemon
//! banner/config/metrics) on the shared `#[tokio::main]` runtime.
//!
//! The shape here mirrors the shape a user types: a module per top-level
//! command, and a namespace's subcommands underneath it (`model/import`,
//! `config/tune`) rather than beside it. What a command *uses* — the artifact
//! store, the HF cache, the embedded runtime — is not a command and lives in
//! [`crate::local`].

pub mod cache;
pub mod config;
pub mod doctor;
pub mod inferlet;
pub mod model;
pub mod run;
