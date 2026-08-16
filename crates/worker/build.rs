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
    // Almost nothing to build. Kept as an explicit no-op for most of its
    // history rather than deleted: the file's absence would read as "this
    // crate never needed a build script", and the reason it barely does is
    // worth finding here.
    println!("cargo:rerun-if-changed=build.rs");

    // `-lnccl`, AND THIS IS THE CRATE THAT OWES IT.
    //
    // `embedded_driver.rs` declares `ncclGetUniqueId` and
    // `ncclGetErrorString` as `extern "C"` under `feature = "driver-cuda"`
    // and calls both, to mint the unique id a tensor-parallel group
    // rendezvouses on. Those are the ONLY two NCCL symbols anything in this
    // workspace references; no collective is called anywhere, and `cudarc`'s
    // `nccl` feature is off, so no binding is generated either.
    //
    // The flag lived in `driver-cuda/build.rs` until now, under a comment
    // saying it was *"the custom all-reduce's -- `comm::all_reduce_bf16`'s,
    // one crate down"*. That was false: `comm::all_reduce_bf16` is the P2P
    // arm and calls no NCCL. Removing it on that reading was right about the
    // sentence and wrong about the flag -- the engine came out with two
    // undefined `nccl*` symbols and no `DT_NEEDED`, and would only load
    // under `LD_PRELOAD`.
    //
    // Here instead, because a link line belongs to the crate whose source
    // names the symbol, and because the `#[cfg]` and the flag now share one
    // condition: a build without `driver-cuda` neither declares them nor
    // asks the linker for them.
    if std::env::var_os("CARGO_FEATURE_DRIVER_CUDA").is_some() {
        println!("cargo:rustc-link-lib=nccl");
    }
}
