//! The CUDA execution shell.
//!
//! The shell itself is C++ under `csrc/` — dispatch, memory and arenas,
//! streams, graph capture and replay, TP communication. `build.rs` compiles it
//! into `libpie_driver_cuda_lib.a` and emits the link line; what crosses into
//! Rust does so through `pie_driver_abi.h`, not through this crate.
//!
//! So there is deliberately no Rust here yet. The crate exists to make the
//! driver a DEPENDENCY rather than a branch inside `worker`'s build
//! script: `--features driver-cuda` now turns on an edge in the graph, and
//! cargo does the rest — builds it, links it, and rebuilds it when `csrc/`
//! changes. The Rust facade the restructure calls for grows in this file.
