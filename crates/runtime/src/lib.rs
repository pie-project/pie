//! Pie - Programmable Inference Engine

// ── The native execution shells ──────────────────────────────────────
//
// This crate's `engine::backend::{cuda,metal}` declare the `extern "C"`
// entry points; the archives that define them are built by these crates. A
// crate rustc never loads contributes no native-library records, so without
// `extern crate ... as _` the declarations link against nothing -- which is
// exactly what `cargo test -p pie-engine --features engine-cuda` did before
// the engines were crates: undefined `pie_cuda_create`, `pie_cuda_launch`,
// and eight more. `as _` because there is nothing to call.
// THE CUTOVER, and it is one line because it has to be: the Rust shell
// exports the same thirteen `pie_cuda_*` symbols, so which crate supplies
// them is a link question. The boot reader (`engine_cuda::boot`, and
// `backend/cuda.rs` when this was written) never learns which one it reached.
#[cfg(feature = "_engine-cuda")]
extern crate engine_cuda as _;

pub mod bootstrap;
pub mod engine;
pub mod inferlet;
/// The served model: the global cache the runtime binds once at bootstrap,
/// and the serving table it is built from.
///
/// Moved here from `model`, which defines itself as backend-blind family
/// knowledge — and a process-global `OnceLock` holding whatever this runtime
/// booted is neither a family fact nor knowledge. `ModelMetadata` and the
/// `(layers, vocab, arch)` rows FOLLOWED IT at M18, which deleted
/// `model::serve` outright: what an artifact carries and what a fleet calls a
/// model are facts about serving, and this is the serving fabric.
pub mod model;
pub(crate) mod pipeline;
pub mod offload;
pub mod planner;
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
pub(crate) mod telemetry;
