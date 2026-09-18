#[cfg(feature = "cuda")]
extern crate engine_cuda as _;

// A tab has one WASI host, `wasi-web`; it answers to `wasmtime_wasi` here so
// the host modules keep their imports. A root `extern crate` alias lands in
// the extern prelude, which is what makes `use wasmtime_wasi::..` resolve.
#[cfg(target_arch = "wasm32")]
extern crate wasi_web as wasmtime_wasi;

pub mod bootstrap;
pub mod codec;
pub mod engine;
pub mod inferlet;
pub mod model;
pub mod offload;
pub(crate) mod pipeline;
pub mod planner;
pub mod rt;
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod telemetry;
