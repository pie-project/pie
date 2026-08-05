//! `pie-worker` library — engine boot path + supporting modules.
//!
//! The `pie` CLI binary (`src/main.rs`) is one consumer; the upcoming
//! `pie-server` pyo3 wheel under `sdk/python-server/` is another.
//! Both link against this lib, so the engine boot logic
//! ([`serve::start_engine`]) has a single source of truth.
//!
//! Modules are `pub` so external callers (the pyo3 wheel) can reach
//! the surface they need — `serve::start_engine`, `config::Config`,
//! and native-driver bootstrap helpers.

/// The process-wide allocator for every engine entry point.
///
/// Declared here rather than in each binary because `#[global_allocator]` is
/// resolved at link time across the whole graph, and `pie-worker` is the one
/// crate the CLI, the standalone worker and the pyo3 wheel all link.
///
/// This is a measured change, not a preference. The scheduler loop is a
/// single thread whose pass time is the critical path for every process hop
/// at a cohort boundary (CONTENTION_FOLLOWUP.md §20.23), and retiring a wave
/// frees 512 `LaunchPlan`s — thirty-odd `Vec`s each. Under glibc malloc those
/// frees cost 2.93 ms per pass; under mimalloc, 0.77 ms.
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod config;
pub mod config_layout;
pub mod config_schema;
pub mod driver_ffi;
pub mod embedded_driver;
pub mod paths;
pub mod state;
pub mod translate;
pub mod weights;

mod client_server;
pub mod engine;
mod executor;
mod lifecycle;
mod link;
mod preflight;

// Frozen crate-root public API (Seam 1) — these stay stable through the internal
// §8 `serve/*`→`link/` + `engine.rs` reorg, so `bin/worker` / `bin/pie` / the
// pyo3 wheel code against the top-level paths and the reorg moves impls
// underneath without reworking them.
pub use config::Config;
pub use engine::{WorkerHandle, run, run_with};
pub use pie_controller_rpc::Role;
// The control-plane seam `run_with` is generic over — re-exported so the
// composition root (`bin/pie`) can impl it for its `EmbeddedControl` adapter.
pub use link::control::ControlLink;
