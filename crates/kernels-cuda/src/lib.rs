//! CUDA's ahead-of-time archive: `csrc/` compiled by CMake, the `extern "C"`
//! shim that gives its launchers callable symbols, and the emitters that
//! generate both.
//!
//! # Where the table went, and why this crate still re-exports it
//!
//! The signature table — one row per launcher symbol in `csrc/` — was
//! authored here, beside the `.cu` files it describes. It is
//! `kernels_cuda_new::table`'s now, and everything below that looks like a
//! table is a `use` of it.
//!
//! The reason is not that the archive is finished. **109 of the 198 rows have
//! no JIT twin**, so `csrc/` and its `pie_k_*` symbols are load-bearing and
//! will stay so. The reason is what a table IS to each crate: the JIT crate
//! is the table's only remaining author — a row there states a `LaunchRule`
//! and a template instantiation, which is the whole kernel — while this crate
//! merely READS rows to emit C. A consumer that needs CMake, nvcc and a Linux
//! target must not be the crate `model-compiler` depends on to look up a
//! symbol's operand list, and it is exactly that dependency
//! (`default-features = false`, non-optional, read on every trace) that fixed
//! the direction.
//!
//! The two crates could not swap in two steps: while `kernels-cuda-new`
//! depended on this crate for its rows, adding the edge back would have been
//! a cycle. So the move and the inversion are one change, and the
//! re-exports below are what let it land without touching a consumer —
//! `driver-cuda` still writes `kernels_cuda::attn::KERNELS`, `model-compiler`
//! still writes `kernels_cuda::KERNELS`, and neither was edited.
//!
//! # What is still authored here
//!
//! **Nothing but the build.** [`abi`] was the last Rust in this crate that
//! was not a `use`, and it is `kernels_cuda_new::abi`'s now. The emitters
//! generate the ahead-of-time artefacts — the `extern "C" pie_k_*` shim that
//! DEFINES those symbols, the portable Rust declarations in [`ffi`], the
//! device typecheck, and the dispatch arms `driver-cuda`'s build script
//! writes — so they LOOKED like this crate's, and their output is. Their
//! input never was: each is a pure function from
//! `kernels_cuda_new::table`'s rows to a `String`, opening no `.cu` and
//! calling no nvcc, so a build script that wanted one had to depend on a
//! CMake project to get it. `driver-cuda`'s did, and that edge is gone.
//!
//! `kernels_cuda_new::emit` is the JIT's generator and reads the same rows to
//! a different end; the two are siblings there and deliberately not merged,
//! because a generator that served both would have to be told which build it
//! was serving on every call.
//!
//! So what is left is `csrc/`, `build.rs`, and the `use`s: the crate is
//! exactly the ahead-of-time archive now, and the day it can be deleted is a
//! question purely about kernels — 96 rows with no JIT unit, and `gemm.cpp`'s
//! 17 cuBLASLt/Marlin host algorithms that have no `<<<>>>` to write at any
//! price.
//!
//! ## Reading this without a GPU
//!
//! The table is still the crate's `default-features = false` surface, and
//! that is still deliberate: `model-compiler` reads it on every trace, and a
//! compiler dev loop must not pay nvcc to look up a symbol's contract.
//! Turning on `native` adds the CMake build of `csrc/` and the shim, and
//! nothing to the rows.
//!
//! The table is kept honest from the other end: `model-compiler`'s
//! `kernels::check_plan` refuses any `OpKind::Launch` symbol no row declares,
//! so a kernel cannot be stated by a model text without its contract.

// NOTHING IS RE-EXPORTED FROM HERE ANY MORE, and the way that was
// discovered is worth as much as the deletion.
//
// This file carried 66 lines of `pub use` — `kernels::{Cap, KernelSig,
// Prepare}`, `kernels_cuda_new::{abi, record}`, the whole of
// `kernels_cuda_new::table`, `norm_device`, and a `#[cfg(feature =
// "native")] pub mod ffi` — each with a paragraph explaining which consumer
// would break without it. The paragraph on `abi` said `driver-cuda`'s build
// script reaches the emitters "through THIS path" and that its `launch_abi`
// suite names `abi::Record`; the one on `record` said the same suite spells
// `kernels_cuda::record!` "in five places", and called the re-export "the
// difference between a move and a flag day".
//
// Measured: `launch_abi.rs` names `kernels_cuda_new::` **49 times** and
// `kernels_cuda::` **zero**. Across the whole workspace there is not one
// `use kernels_cuda`, not one `kernels_cuda::` outside a doc comment, and
// `kernels_cuda::ffi` has no caller — `driver-cuda` generates its own `ffi`
// module against its own `#[repr(C)]` mirrors.
//
// `build.rs` in this very crate already said so, in a doc comment written
// two refactors ago: *"no crate in the workspace names `kernels_cuda:: any
// more"*. Both statements were in the tree at once. The re-exports were kept
// alive by their own justification — a reason that is written down, is
// checkable, and is false is exactly the shape `new-horizon.md` §§34, 38 and
// 41 found at three other levels, and this is the fourth: a wall in front of
// a door nobody opens, spelled as a `pub use`.
//
// So the crate is now what its header already claimed it was: `csrc/` and
// `build.rs`. Its Rust is a lib target Cargo requires and a `links` key that
// carries `DEP_PIE_KERNELS_CUDA_LAUNCH_SHIM` to `driver-cuda`, which is the
// only handoff with a reader.
