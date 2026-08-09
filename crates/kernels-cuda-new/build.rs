//! Generate `carried.rs`: the header set NVRTC compiles against, into `OUT_DIR`.
//!
//! # What this script used to be, and why the argument is kept
//!
//! It generated `api.rs` — one typed Rust function per row — and the whole
//! header below argued for that. North star §6 half A retired the emitter, so
//! the argument is history; it is kept rather than deleted because **the same
//! reasoning still decides `carried.rs`**, which is the half that survived,
//! and a reader who finds only the conclusion will re-open the question.
//!
//! The rows are Rust data — `static` arrays of `DeviceKernel`, in this crate's
//! own `src/families/*.rs` and `src/x/*.rs`. Nothing in the language reads one
//! of those at compile time: a `macro_rules!` cannot iterate a static, and a
//! proc-macro would be a third crate that has to depend on the tables to
//! expand against them, which the tables' own crate cannot then depend on.
//! A build script can just call the generator.
//!
//! A checked-in output with a test that regenerates and diffs it was the other
//! candidate and it loses on one property: **a stale file still COMPILES.**
//! Generated into `OUT_DIR`, it cannot be stale by construction. That property
//! is what `carried.rs` needs most — it names the `.cuh` files the JIT carries,
//! and a stale list is a `NoLoweredName` at first fire on a machine with a GPU,
//! which is the run-time failure §12.4 says generation exists to convert into a
//! build-time one.
//!
//! # Why the emitter's argument did NOT survive with it
//!
//! `emit.rs` walked `unit::rows()` — the DEVICE row list — and not
//! `table::ROW_TABLES`. Every fn-world family contributes to it, because a
//! unit's rows carry the instantiation strings NVRTC lowers, so the generator
//! was never going to run dry: **the list it read is the list the JIT needs.**
//! What blocked it was callers, and there was one, in `tests/fire.rs`. The
//! direct-call surface is written by hand in `src/x/` now, where a wrong
//! operand is a type error at the call site rather than a row the generator
//! silently skipped with a comment.
//!
//! # Why the modules are included by `#[path]`
//!
//! A build script cannot depend on the crate it builds, so `src/device.rs`,
//! `src/unit.rs`, `src/families/mod.rs` and `src/x/mod.rs` are pulled in as
//! modules rather than reached through `kernels_cuda_new::`.
//! `kernels-cuda/build.rs` records the same constraint about `abi.rs` and
//! draws the consequence in one sentence: *"a module here that reached for a
//! sibling, or for `crate::`, would have to be pulled in with it."* It then
//! pulls in fourteen of them, because that is what reading the rows costs.
//!
//! This script pulls in four, for the same reason and after the same defect.
//! The emitter used to read `kernels_cuda::norm_device`'s two statics — the
//! only rows that existed when the pilot shipped — and once the families
//! began declaring their own it covered **ten of the hundred and thirty-five
//! the crate then declared**. Every other row had no typed entry point at
//! all, reachable only through `runtime::fire` with a string, which is
//! precisely the surface §12.4 says the generated façade exists to replace.
//! The rows live in `src/families/*.rs`, so this script reads
//! `src/families/*.rs`.
//!
//! `crate::` inside an included module resolves against THIS file, since a
//! build script is its own crate root — so the module tree below has to be
//! spelled with the library's own names, and is. `families` reaches for
//! `crate::device` and `crate::unit`; `unit` reaches for `crate::device`,
//! `crate::source` and `crate::families`; the closure stops there.
//! `#[allow(dead_code, unused_imports)]` on each is
//! `kernels-cuda/build.rs`'s trick with one lint added: this script calls two
//! of `emit`'s items and none of anything else's, and `device.rs`'s
//! `pub use` re-export is unused HERE and load-bearing in the library.
//! Silencing both here is what keeps a genuine unused-code warning in the
//! library readable.
//!
//! The cost is worth stating because it is met the first time a table is
//! edited: the tables are now compiled TWICE, once into this script and once
//! into the library, so a broken row is two errors rather than one and the
//! script's copy is reported first. That is the same bargain
//! `kernels-cuda/build.rs` makes with fourteen modules, and the alternative
//! to paying it is not reading the rows.
//!
//! # The one module that is a stub, and why it cannot be the file
//!
//! `crate::source` below is not `src/source.rs`. That file carries the header
//! set through `include!(concat!(env!("OUT_DIR"), "/carried.rs"))` — a file
//! THIS script writes, at the bottom of `main`. Including it would ask the
//! compiler for a file the script has not run yet: a clean build fails at the
//! include, and an incremental one succeeds off the previous run's output,
//! which is the worse half of the two orders because it makes the failure
//! look intermittent.
//!
//! What the tables take from that module is small and none of it is a source:
//! `Unit`'s `root` field, `norm`'s two roots, and the three items
//! `Unit::cache_key` folds. So the stub carries no text at all rather than a
//! second `include_str!` of the same two files — a copy drifts, an empty
//! string cannot, and nothing in this script hands a `Unit` to NVRTC. The
//! two functions `panic!` rather than answer: a cache key computed over an
//! empty header set is a WRONG key, and a wrong key is the one bug
//! `program/cache.rs` documents as unfindable.
//!
//! # What this script does NOT need
//!
//! A CUDA toolkit. `kernels-cuda/build.rs` needs nvcc, CMake and
//! `cuda_runtime.h` because it compiles an archive; this one writes a text
//! file, so the crate builds on a machine that has never seen a GPU — which
//! is the whole claim layers 1 and 2 make in `src/lib.rs`, and it would be an
//! odd claim to make while the build script quietly broke it. The modules
//! pulled in below are layer-1 and layer-2 data for the same reason: a row is
//! a `KernelSig` and a unit is a `&'static str`, and neither has ever needed
//! `cudarc`.

mod carried;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The rows themselves. Cargo would rebuild this script anyway -- every
    // file below is one of its compile inputs, and rustc reports them -- but
    // the list is stated because it is also the ANSWER to "what does the
    // generated API depend on", and a reader asking that should not have to
    // reconstruct it from a `#[path]` attribute.
    println!("cargo:rerun-if-changed=src/device.rs");
    println!("cargo:rerun-if-changed=src/unit.rs");
    println!("cargo:rerun-if-changed=src/families");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    // `emit::emit_rust_api(&emit::all_rows())` wrote `api.rs` here until north
    // star §6 half A. What it walked was `unit::rows()` and NOT
    // `table::ROW_TABLES` — the device row list, which every fn-world family
    // still contributes to, because a unit's rows carry the instantiation
    // strings NVRTC lowers. So the generator was never going to run dry on
    // its own: the row list it read is the one the JIT needs.
    //
    // The three `rerun-if-changed` lines above stay. They are `carried`'s
    // now — `device.rs`, `unit.rs` and `families/` are what decides the
    // header set, and that is the half of this script that survives.

    carried::generate(&out);
}
