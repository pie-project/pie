#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda-new's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda-new: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

/// The words a row is written in.
pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Operand, Prepare, Source, Ty};

/// The AHEAD-OF-TIME generator, over the same rows [`emit`] reads.
pub mod abi;
pub mod device;
/// Which of three things a stated symbol is, and who executes it.
pub mod execution;
/// FA2's launch arithmetic: `decode.cuh` and `prefill.cuh`'s host prologues.
pub mod fa2;
/// The kernel families, each owning the units it compiles.
pub mod families;
/// The attention scheduler: `flashinfer/attention/scheduler.cuh` as host Rust.
pub mod plan;
pub mod source;
pub mod table;
pub mod unit;
/// **kernel-x** — the floor a kernel stands on when it is written as a
pub mod x;

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub mod runtime;

/// The launch path, at the top level because it is the one thing every
#[cfg(feature = "_cuda")]
pub use runtime::{ArgValue, Dims, Error, Stream, fire, hosts};
