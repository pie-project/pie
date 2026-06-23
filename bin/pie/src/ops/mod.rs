//! One-shot operational subcommands (P5b). Each runs after the light
//! `bootstrap::init_cli` (tracing + paths, no daemon banner/config/metrics) on
//! the shared `#[tokio::main]` runtime. The R3 weight/runtime *download* IO
//! (`hf`, `py_runtime`) lives here — the worker lib links no provisioning code.

pub mod auth;
pub mod bakery;
pub mod config;
pub mod diag;
pub mod doctor;
pub mod driver;
pub mod hf;
pub mod inferlet;
pub mod model;
pub mod py_runtime;
