//! The CUDA execution shell, in Rust.
//!
//! This crate grows beside `driver-cuda` rather than inside it, on the terms
//! [`driver-metal-new`](../driver_metal_new/index.html) already established: the
//! C++ shell keeps running and keeps its tests, and nothing here is on the
//! serving path until a module here has an equivalent that passes them. A
//! rewrite that has to keep the old one working is a rewrite that can be
//! abandoned halfway without a revert.
//!
//! # What is here, and why it is shaped this way
//!
//! `driver-cuda/csrc` is ~123k lines, and the striking thing about it is how
//! little of it is device code. `store/` and `loader/` contain no `__global__`
//! at all; `pipeline/dispatch.cu` is 9.7k lines with eight mentions of one.
//! The overwhelming majority is host logic -- arenas, page arithmetic, stream
//! and event choreography, plan interpretation -- that is in C++ only because
//! it was written next to the part that is not.
//!
//! So this crate starts at the bottom, with [`cuda`]: the layer everything
//! above it needs, and the layer where the two design arguments for doing this
//! in Rust at all are actually cashed in (see below). Subsystems land on top
//! of it one at a time, each keeping its C++ original alive as the
//! differential oracle until the Rust side is proven byte-identical -- the
//! protocol `.wiki/plan/model-in-rust.md` §8 established and §9 rows 13-15
//! executed.
//!
//! # It builds without CUDA, and that is load-bearing
//!
//! `cudarc` is pinned to a named toolkit version with
//! `fallback-dynamic-loading`, so this crate emits no link line, probes for no
//! `nvcc`, and resolves every symbol through `dlopen` on first call. It
//! therefore compiles -- and its host logic tests -- on a machine with no CUDA
//! installed. `driver-cuda` cannot: its CMake opens with
//! `find_package(CUDAToolkit REQUIRED)`, so today the arena arithmetic and the
//! page accounting are only reachable from a box with a GPU on it. Keeping
//! them reachable from an ordinary `cargo test` is what stops them drifting
//! back into the untestable half.
//!
//! # Why `sys`, and why that is not a compromise
//!
//! Almost nothing here uses `cudarc`'s safe wrappers. That reads like a
//! concession and is the opposite of one.
//!
//! The shell's CUDA vocabulary is the exotic end of the API -- virtual memory
//! management, conditional graph nodes, IPC, stream-ordered pools -- and
//! cudarc's safe tier covers approximately none of it while its `sys` tier,
//! being mechanical bindgen output, covers all of it. What cudarc is worth
//! here is the version-gated bindings and the dynamic loader, not the
//! wrappers.
//!
//! And for the two places it matters most, the wrappers are actively wrong for
//! this shell. cudarc issue #590: `Drop for CudaSlice` bypasses the capture
//! bookkeeping, so a buffer dropped inside a stream capture poisons the
//! context and the error surfaces later, on an unrelated call. cudarc issue
//! #589: there is no non-owning wrapper for a foreign `CUstream`, which a
//! staged migration needs on day one because the C++ shell and this one have
//! to share streams while both exist. [`cuda::Allocator`] and
//! [`cuda::StreamRef`] are the answers to exactly those two, and neither could
//! be built by wrapping the safe tier -- they are what the safe tier would
//! have to have been.
//!
//! # The two things Rust is actually buying
//!
//! Stated plainly, because "rewrite it in Rust" is not a reason:
//!
//! 1. **Allocating during a graph capture stops being representable.**
//!    [`cuda::Allocator::begin_capture`] takes `&mut self`, so the allocator
//!    is exclusively borrowed for as long as the capture scope is alive and
//!    an allocation inside one is a borrow-check error rather than a runtime
//!    fault. In the C++ shell this is a rule in a comment.
//! 2. **Freeing during a capture stops being able to corrupt anything.**
//!    `Drop` is implicit and no type system can forbid it, so instead
//!    [`cuda::DeviceBuffer`]'s drop hands the pointer to a deferred queue
//!    whenever a capture is open, and the queue is drained when the last one
//!    closes. This is the #590 failure mode, made harmless rather than
//!    documented.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

#[cfg(not(any(feature = "cuda-12", feature = "cuda-13")))]
compile_error!(
    "driver-cuda-new needs exactly one of the `cuda-12` / `cuda-13` features, \
     e.g. `cargo build -p driver-cuda-new --features cuda-13`. It selects the \
     CUDA runtime ABI this build targets, and it must match the `libcudart.so` \
     that will be loaded at run time. There is no default on purpose: choosing \
     wrong is a segfault inside the driver, not a build error, so this crate \
     would rather not build than guess."
);
#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "driver-cuda-new needs exactly ONE of `cuda-12` / `cuda-13`, not both. \
     Cargo unifies features across a build graph, so this usually means two \
     dependents each picked a different one."
);

/// The exact `cudarc` build this shell speaks CUDA through.
///
/// Re-exported because the API here deliberately hands out raw
/// `CUdeviceptr`s -- an [`cuda::Arena`]'s mapped range is ordinary device
/// memory, and the layers above write to it through kernels rather than
/// through a byte-copy method on this crate. Anyone holding one of those
/// pointers needs a way to call CUDA, and they need *this* CUDA: with
/// `fallback-dynamic-loading` nothing is linked, so an independent
/// `extern "C"` declaration would be an undefined symbol, and an independent
/// `cudarc` dependency could resolve to a different version feature than the
/// one selected here.
#[cfg(feature = "_cuda")]
pub use cudarc;

// Everything below is gated on a feature being selected, so that a build with
// none of them reports the `compile_error!` above and nothing else. Left
// ungated, the same build would bury that one actionable message under a few
// dozen "cannot find crate `cudarc`" errors from every module that uses it.
#[cfg(feature = "_cuda")]
mod error;

#[cfg(feature = "_cuda")]
pub mod cuda;
#[cfg(feature = "_cuda")]
pub mod dtype;
// The kernel-facing records. Gated like its neighbours only because it names
// `dtype::DType`, which is: the mirrors themselves reference no CUDA symbol,
// and the layout proof over them needs no driver.
#[cfg(feature = "_cuda")]
pub mod launch;

/// The thirteen `pie_cuda_*` exports — the cutover's door. See the
/// module doc for the one-provider-per-binary rule.
#[cfg(feature = "abi")]
pub mod abi_shell;

/// The checkpoint's bytes onto the device, through `model-loader`'s plan.
#[cfg(all(feature = "abi", feature = "_cuda"))]
pub mod loader;

pub mod model;
/// PTIR on CUDA: NVRTC, cubins, and the compile cache.
///
/// The channel plane itself is [`driver_pipeline`] and is shared with the
/// Metal shell; this is the part that names a CUDA symbol. Gated on `_cuda`
/// because every file in it calls `cudarc`.
#[cfg(feature = "_cuda")]
pub mod ptir;
#[cfg(feature = "_cuda")]
pub mod store;

pub mod tensor;

#[cfg(feature = "_cuda")]
pub use dtype::DType;
#[cfg(feature = "_cuda")]
pub use error::{Error, Result};
