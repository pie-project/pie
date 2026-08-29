//! Pure CUDA kernel definitions — jit unit names, argument marshalling, and
//! launch geometry over a stream; no IR types, no execution state. An engine
//! `Run` resolves plan ids to handles and calls these entry functions (design §8).

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

pub mod attn;
pub mod channel;
pub mod collective;
pub mod custom;
pub mod elemwise;
pub mod jit;
pub mod layout;
pub mod linear;
pub mod source;
pub mod tensor;

pub use jit::{Arg, ArgValue, Ctx, Fire, Launch, Pad, Slabs};
pub use kernels::KernelError;

/// **THE RUNTIME THIS BINARY ALREADY LOADED, RE-EXPORTED FOR THIS CRATE'S OWN
/// GPU TEST TARGETS.**
///
/// A `tests/` target is a separate crate, so a device test that has to
/// allocate, copy and open a stream needs `cudarc` by some path. A
/// dev-dependency would be the wrong one twice: it puts a SECOND copy of the
/// runtime-version decision into the graph — the one thing this crate's
/// feature comment forbids, since Cargo unifies features and a `cudarc` chosen
/// here would silently decide for everybody — and, being non-optional, it
/// would drag `cudarc` into a plain `cargo check --workspace` on a box that
/// selected no version at all.
///
/// Re-exporting the dependency the `_cuda` feature already resolved has
/// neither problem: there is exactly one `cudarc` in the graph, and it is the
/// one the kernels fire through.
#[cfg(feature = "_cuda")]
#[doc(hidden)]
pub use cudarc;

pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
