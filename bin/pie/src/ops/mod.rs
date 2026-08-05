//! One-shot operational subcommands (P5b). Each runs after the light
//! `startup::init_cli` (tracing + paths, no daemon banner/config/metrics) on
//! the shared `#[tokio::main]` runtime. The R3 weight/runtime *download* IO
//! (`hf`, `py_runtime`) lives here — the worker lib links no provisioning code.

pub mod cache;
pub mod config;
pub mod convert;
pub mod doctor;
pub mod hf;
pub mod inferlet;
pub mod model;
pub mod optimize;
pub mod py_runtime;
pub mod run;
pub mod store;
