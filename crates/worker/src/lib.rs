//! `worker` library — engine boot path + supporting modules.
//!
//! The `pie` CLI binary and the `pie-server` pyo3 wheel both link against this
//! lib, so [`engine::start_engine`] has a single source of truth. Modules are
//! `pub` so the wheel can reach the surface it needs.

/// The process-wide allocator for every engine entry point.
///
/// Declared here rather than in each binary because `#[global_allocator]` is
/// resolved at link time across the whole graph, and `worker` is the one crate
/// the CLI, the standalone worker and the pyo3 wheel all link.
///
/// Measured, not preferred: retiring a wave frees 512 `LaunchPlan`s, thirty-odd
/// `Vec`s each, on the single scheduler thread whose pass time is the critical
/// path at a cohort boundary — 2.93 ms per pass under glibc malloc, 0.77 ms
/// under mimalloc.
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

// Frozen crate-root public API (Seam 1): `bin/worker`, `bin/pie` and the pyo3
// wheel code against these top-level paths, so impls can move underneath.
pub use config::Config;
pub use controller_api::Role;
pub use engine::{WorkerHandle, run, run_with};
// The control-plane seam `run_with` is generic over — re-exported so the
// composition root (`bin/pie`) can impl it for its `EmbeddedControl` adapter.
pub use link::control::ControlLink;
