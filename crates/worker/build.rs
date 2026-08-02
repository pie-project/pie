//! What is left of the link bridge once the drivers build themselves.
//!
//! This script used to invoke CMake for both native drivers, forward the
//! include handoffs, and emit every `cargo:rustc-link-*` directive for the
//! final binary — 450 lines of it, half CUDA-only and half macOS-only, in a
//! crate that is neither. The worker does not need to know where the C++
//! trees are or how to build them; it needs a driver. So `driver-cuda` and
//! `driver-metal` are crates now, each building its own `csrc/` and emitting
//! its own link line, and `--features driver-cuda` turns on a dependency
//! edge rather than a branch here.
//!
//! Cargo carries the rest: link directives from any build script in the graph
//! reach the final rustc invocation, so nothing had to be re-emitted here to
//! keep the archives linked.
//!
//! The Rust dummy driver is a normal dependency and always linked; the
//! resulting binary still dispatches at runtime via `[model].driver.type` in
//! the config TOML. CUDA and Metal expose distinct create/destroy/direct-
//! operation symbols, so their static archives can coexist in one binary
//! without symbol collisions.

fn main() {
    // Nothing to build. Kept as an explicit no-op rather than deleted: the
    // file's absence would read as "this crate never needed a build script",
    // and the reason it no longer does is worth finding here.
    println!("cargo:rerun-if-changed=build.rs");
}
