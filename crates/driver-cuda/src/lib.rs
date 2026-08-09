//! The CUDA execution shell, in Rust.
//!
//! This crate grows beside `driver-cuda` rather than inside it, on the terms
//! [`driver-metal`](../driver_metal/index.html) already established: the
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

// NO `compile_error!` FOR "NO FEATURE", and that is a reversal.
//
// It used to refuse to build at all, reasoning that "choosing wrong is a
// segfault inside the driver, not a build error, so this crate would
// rather not build than guess". The hazard is real and the conclusion did
// not follow: choosing wrong means picking cuda-12 on a box that loads
// cuda-13. It is not the hazard of picking NEITHER — with no feature
// there is no `cudarc` linked, no `dlopen`, and no segfault reachable.
//
// What the error cost was the portable half: ten thousand lines of
// geometry, budgets and plans that need no card could not be built or
// tested without naming a CUDA runtime they would never load.
//
// A dependent that forgets to pick one is still caught, and in a better
// place — `gpu::serve` is gated, so `driver_cuda::gpu::serve::pie_cuda_create`
// is an unresolved path IN THE CONSUMER, at compile time, naming the
// symbol it wanted.
//
// This also retires the `portable` feature, which existed only to opt out
// of the error below. Cargo features must be additive, and a feature
// whose job is to SUBTRACT is exactly the shape that breaks under feature
// unification: one dependent asking for `portable` and another for
// `cuda-13` would have got both.
#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "driver-cuda needs exactly ONE of `cuda-12` / `cuda-13`, not both. \
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

/// UNGATED, and its two CUDA variants are gated instead.
///
/// `Error` is the return type of the whole layout layer, so naming
/// `cudarc` here put every geometry, planner and cache module behind
/// `_cuda` — see `tests/portable_half.rs`.
mod error;

pub mod dtype;
pub mod tensor;

/// How big, where, how many — and none of it needs a card.
///
/// Geometry, memory budgets, swap plans, the load plan, the profile
/// cache. The line between this and [`gpu::pools`] is the one the
/// `Cargo.toml` already drew: `kv_geometry` says what shape the pages
/// are and `kv_cache` allocates them, and only the second needs a card.
pub mod layout;

/// THE ONE `#[cfg]`.
///
/// Everything that names a CUDA symbol lives under here, and nothing
/// above here does. That is the whole point of the directory: the gate
/// used to run per-file through `store/`, `model/` and `tensor.rs`, and
/// a per-file gate is discipline rather than structure — it holds until
/// someone adds a file.
///
/// Gated on `_cuda` rather than on a version, so a build with no feature
/// selected compiles the portable half instead of reporting an error
/// about a runtime it was not asked to pick.
#[cfg(feature = "_cuda")]
pub mod gpu;

// `facts.rs`, `config.rs` and `descriptor.rs` are GONE — 1,782 lines
// into `crates/model` as `deployment_cuda`, `descriptor` and a deleted
// re-export (§8 row 6). The table of 33 `model_type` rows was the
// question, and §4's rule is that a driver reads the answer.
//
// What comes back is `model::deployment::Deployment`: a value with no
// family name in it, derived once at load rather than boxed per fire.

/// Every boot knob, parsed once. See the module doc for the three
/// things ten scattered `env::var` reads cost.
pub mod boot;

pub use dtype::DType;
pub use error::{Error, Result};
