//! What is left of the link bridge once the engines build themselves.
//!
//! This script used to invoke CMake for both native engines, forward the
//! include handoffs, and emit every `cargo:rustc-link-*` directive for the
//! final binary — 450 lines of it, half CUDA-only and half macOS-only, in a
//! crate that is neither. The worker does not need to know where the C++
//! trees are or how to build them; it needs an engine. So `engine-cuda` and
//! `engine-metal` are crates now, each building its own `csrc/` and emitting
//! its own link line, and `--features engine-cuda-13` turns on a dependency
//! edge rather than a branch here.
//!
//! Cargo carries the rest: link directives from any build script in the graph
//! reach the final rustc invocation, so nothing had to be re-emitted here to
//! keep the archives linked.
//!
//! The Rust dummy engine is a normal dependency and always linked; the
//! resulting binary still dispatches at runtime via `[model].engine.type` in
//! the config TOML. CUDA and Metal expose distinct create/destroy/direct-
//! operation symbols, so their static archives can coexist in one binary
//! without symbol collisions.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // `-lnccl`, AND THIS IS THE CRATE THAT OWES IT.
    //
    // `embedded_engine.rs` declares `ncclGetUniqueId` and
    // `ncclGetErrorString` as `extern "C"` under `feature = "_engine-cuda"`
    // and calls both, to mint the unique id a tensor-parallel group
    // rendezvouses on. Those are the ONLY two NCCL symbols anything in this
    // workspace references; no collective is called anywhere, and `cudarc`'s
    // `nccl` feature is off, so no binding is generated either.
    //
    // The flag lived in `engine-cuda/build.rs` until now, under a comment
    // saying it was *"the custom all-reduce's -- `comm::all_reduce_bf16`'s,
    // one crate down"*. That was false: `comm::all_reduce_bf16` is the P2P
    // arm and calls no NCCL. Removing it on that reading was right about the
    // sentence and wrong about the flag -- the engine came out with two
    // undefined `nccl*` symbols and no `DT_NEEDED`, and would only load
    // under `LD_PRELOAD`.
    //
    // Here instead, because a link line belongs to the crate whose source
    // names the symbol, and because the `#[cfg]` and the flag now share one
    // condition: a build without the CUDA shell neither declares them nor
    // asks the linker for them.
    //
    // `cfg!` and not `env::var_os("CARGO_FEATURE_...")`. The env var spelling
    // is a STRING that no tool checks: when this feature was renamed from
    // `engine-cuda` to `_engine-cuda`, `CARGO_FEATURE_ENGINE_CUDA` went
    // quietly false and took `-lnccl` with it -- the same two undefined
    // `nccl*` symbols described above, reached a third way. A `cfg!` on a
    // feature this package does not declare is caught by `unexpected_cfgs`,
    // which is the difference between a warning and a binary that only loads
    // under `LD_PRELOAD`.
    if cfg!(feature = "_engine-cuda") {
        println!("cargo:rustc-link-lib=nccl");
    }
}
