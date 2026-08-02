//! The Metal execution shell.
//!
//! The shell itself is C++ under `csrc/` — dispatch, memory and arenas,
//! command queues, heaps and residency, command encoding, and the compiled .metal shader library. `build.rs` compiles it
//! into `libpie_driver_metal_lib.a` and emits the link line; what crosses into
//! Rust does so through `pie_driver_abi.h`, not through this crate.
//!
//! So there is deliberately no Rust here yet. The crate exists to make the
//! driver a DEPENDENCY rather than a branch inside `worker`'s build
//! script: `--features driver-metal` now turns on an edge in the graph, and
//! cargo does the rest — builds it, links it, and rebuilds it when `csrc/`
//! changes. The Rust facade the restructure calls for grows in this file.
