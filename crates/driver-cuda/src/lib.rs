//! The CUDA execution shell, in Rust. Subsystems land here one at a time,
//! each keeping its C++ original in `driver-cuda/csrc` as the differential
//! oracle until the Rust side is proven byte-identical.
//!
//! It builds without CUDA: `cudarc` is pinned with `fallback-dynamic-loading`,
//! so nothing is linked and every symbol resolves through `dlopen` on first
//! call — the host logic compiles and tests on a machine with no CUDA.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

// No `compile_error!` for a featureless build: nothing links `cudarc`, so no
// segfault is reachable, and a consumer that forgets a feature is caught by an
// unresolved `driver_cuda::serve::*` path.
#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "driver-cuda needs exactly ONE of `cuda-12` / `cuda-13`, not both. \
     Cargo unifies features across a build graph, so this usually means two \
     dependents each picked a different one."
);

/// The exact `cudarc` build this shell speaks CUDA through, re-exported
/// because the API hands out raw `CUdeviceptr`s and nothing is linked.
#[cfg(feature = "_cuda")]
pub use cudarc;

/// Ungated, unlike its two CUDA variants: `Error` is the layout layer's
/// return type, so gating `cudarc` here would gate all of it.
mod error;

pub mod dtype;
pub mod tensor;

/// How big, where, how many — none of it needs a card. [`pools`]'s
/// `kv_cache` allocates the pages this shapes.
pub mod layout;

// Everything that names a CUDA symbol is gated on `_cuda` here and nowhere
// else, so a module that forgets its own `#[cfg]` sits unreachable rather than
// breaking a featureless build.

/// The only place vendor words are correct: stream, event, heap, allocator,
/// graph. Above it the crate uses one word per concept, shared with Metal.
#[cfg(feature = "_cuda")]
pub mod device;

/// What [`layout`] planned, allocated: KV, recurrent, swap.
#[cfg(feature = "_cuda")]
pub mod pools;

/// The checkpoint onto the device; the plan half is [`layout`]'s. Split at
/// `abi` inside the module, because `plan` and `stage` name loader types.
#[cfg(feature = "_cuda")]
pub mod weights;

/// A lowered launch onto a kernel entry, its arguments and its grid.
#[cfg(feature = "_cuda")]
pub mod bind;

/// One forward pass: its scratch, tables, recordings and retirement.
#[cfg(feature = "_cuda")]
pub mod fire;

// THE FIRE PATH: a `model_compiler::program::Program` per lane, built at
// load, fired by `fire::launch`. Not "beside" anything and behind no knob —
// R2 deleted the legacy lowering, dispatch and walk, and a checkpoint whose
// Program will not build is REFUSED at `load_model` rather than served by a
// second path.
//
// `pub(crate)` and not `pub`, unlike its neighbours: nothing outside this
// crate reaches it, and the surface it WOULD publish (`Baked`, the point
// shim, the staging table) is the surface `#[claims]` is going to generate.
// Publishing it now would be publishing a shape that is about to change.
//
// Gated on `_cuda` and not on `abi`, though only the `abi` shell has a
// `Shell` to hang it off: the load, the resolve and the shim name CUDA
// symbols and nothing else, so a `--features cuda-13` build compiles and
// unit-tests them without the ABI door.
#[cfg(feature = "_cuda")]
pub(crate) mod baker;

/// User programs: compile, cache, channel, run.
#[cfg(feature = "_cuda")]
pub mod program;

/// The door. create / load / launch / transfer / close.
#[cfg(all(feature = "_cuda", feature = "abi"))]
pub mod serve;

/// Every boot knob, parsed once.
pub mod boot;

pub use dtype::DType;
pub use error::{Error, Result};
