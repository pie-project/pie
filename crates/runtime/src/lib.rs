#[cfg(feature = "cuda")]
extern crate engine_cuda as _;
#[cfg(feature = "rocm")]
extern crate engine_rocm as _;

// The native GPU engine this build carries. engine-rocm is engine-cuda's
// shape on HIP, so every caller reaches either through this one name.
#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("`cuda` and `rocm` are each the native GPU engine; a build carries one");
#[cfg(feature = "cuda")]
pub(crate) use engine_cuda as gpu_engine;
#[cfg(feature = "rocm")]
pub(crate) use engine_rocm as gpu_engine;

#[cfg(target_arch = "wasm32")]
extern crate wasmtime_web as wasmtime_wasi;

pub mod bootstrap;
pub mod catalog;
pub mod engine;
pub mod inferlet;
pub mod model;
pub mod offload;
pub(crate) mod pipeline;
pub mod planner;
pub mod policy;
pub mod rt;
pub mod scheduler;
pub mod server;
pub(crate) mod service;
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod telemetry;
