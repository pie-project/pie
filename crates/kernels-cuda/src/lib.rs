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

pub use kernels::{Cap, KernelSig, Prepare};

/// The emitters, re-exported from where they are now authored.
///
/// `pub use` rather than `pub mod`: the file moved to `kernels-cuda-new` to
/// sit beside the rows every one of its emitters reads. `driver-cuda`'s build
/// script calls [`abi::emit_rust_bindings`] and [`abi::emit_rust_dispatch`]
/// through THIS path and its `launch_abi` suite names [`abi::Record`], so the
/// line below is the difference between a move and a flag day.
pub use kernels_cuda_new::abi;

pub mod norm_device;

/// The layout-record builder, beside the [`abi::Record`] it builds.
///
/// A `#[macro_export]` macro lives at its crate's root, so moving the
/// emitters moved this to `kernels_cuda_new::record!` — and `driver-cuda`'s
/// `launch_abi` suite spells `kernels_cuda::record!` in five places. Re-exported
/// rather than left to be renamed, for the reason every `use` above exists:
/// the move is not supposed to be visible from here. The expansion names
/// `kernels_cuda_new::abi::Record` and resolves at each call site, because a
/// crate that reads this crate's table already depends on the crate that owns
/// it.
pub use kernels_cuda_new::record;

/// Every kernel a lowered declaration may state, and the per-family tables it
/// concatenates.
///
/// `kernels_cuda_new::table`'s, re-exported by name rather than with a glob:
/// a family that appeared silently would be in [`KERNELS`] and absent from
/// every reader that walks the modules, and there are three of those —
/// `driver-cuda`'s build script, its `bind` module, and this crate's own
/// shim generator.
pub use kernels_cuda_new::table::{
    KERNELS, adapter, attn, driver_internal, gemm, layout, mlp, moe, norm, quant, rope, sample,
    ssm,
};

/// The `pie_k_*` entry points, for the rows any caller can state.
///
/// `native` builds `libpie_launch_shim.a`, which DEFINES these; this is the
/// matching declaration, generated from the same rows in the same process, so
/// a signature cannot drift from what the shim proves against the header.
///
/// Restricted to portable rows — see
/// [`abi::emit_rust_bindings_portable`]. A row taking `KvCacheLayerView` or a
/// FlashInfer plan is absent, because its declaration would name a
/// `#[repr(C)]` mirror this crate does not hold. Those belong to the shell,
/// which generates the full set against its own mirrors; nothing stops two
/// crates from declaring one symbol, because a declaration is not a
/// definition.
#[cfg(feature = "native")]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}
