//! Pie - Programmable Inference Engine

// Forces linking engine-cuda even though nothing here calls it directly;
// without this, the `extern "C"` entry points it defines go unresolved.
#[cfg(feature = "cuda")]
extern crate engine_cuda as _;

pub mod bootstrap;
pub mod engine;
pub mod inferlet;
/// The served model: the global cache the runtime binds once at bootstrap,
/// and the serving table it is built from.
pub mod model;
pub mod offload;
pub(crate) mod pipeline;
pub mod planner;
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
pub(crate) mod telemetry;
