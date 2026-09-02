//! `worker` library — runtime boot path + supporting modules. The `pie` CLI
//! binary and the `pie-server` pyo3 wheel both link against this lib, so
//! [`serve::start_runtime`] has a single source of truth.

/// The process-wide allocator for every engine entry point. Declared here
/// (not per-binary) since `#[global_allocator]` resolves at link time and
/// `worker` is the crate all three binaries link. Measured: mimalloc drops
/// scheduler-thread wave-retirement pass time from 2.93ms to 0.77ms vs glibc.
#[global_allocator]
static GLOBAL_ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod backend;
pub mod config;
pub mod disk;
pub mod serve;
pub mod translate;
pub mod weights;

mod executor;
mod link;

// Frozen crate-root public API: `bin/worker`, `bin/pie` and the pyo3 wheel
// code against these top-level paths, so impls can move underneath.
pub use config::Config;
pub use controller_api::Role;
pub use serve::{WorkerHandle, run, run_with};
// The control-plane trait `run_with` is generic over, re-exported so
// `bin/pie` can impl it for its `EmbeddedControl` adapter.
pub use link::control::ControlLink;
