//! The thin layer beneath `kernels-cuda`: the stream everything is enqueued
//! on, and the bytes no kernel entry allocates.
//!
//! **WHY THIS EXISTS AT ALL, GIVEN `Ctx`.** `kernels_cuda::Ctx` is three raw
//! pointers and a launcher — it is *lent* a stream, it never makes one — and
//! the only allocation in that crate is its own process-global scratch, which
//! is deliberate: an entry that allocated per fire could not be captured
//! (`Ctx::scratch`'s contract). So everything with a lifetime — the stream,
//! the cuBLAS handle, the arena, the weight store, the pool pages, the
//! resident fire inputs — belongs to the shell, and this module is where the
//! shell touches the runtime. Nothing above it names `cudarc`.
//!
//! **It compiles without a runtime.** Every entry here is present in a build
//! that selected neither `cuda-12` nor `cuda-13`, and answers
//! [`Fault::Runtimeless`](crate::Fault::Runtimeless). That is what lets the
//! call-order code in [`serve`](crate::serve) — the part worth reading — be
//! type-checked by a plain workspace sweep on a machine with no GPU.
//!
//! [`graph`] is the step-5 arrival: `cudaStreamBeginCapture` and its three
//! companions, and nothing about when to use them. Policy — which fires
//! capture, what a graph is keyed by, when one is evicted — is
//! [`record`](crate::record)'s and [`serve`](crate::serve)'s, the same way
//! the eager plane's is.

pub mod alloc;
pub mod ctx;
pub mod graph;

pub use alloc::Buffer;
pub use ctx::{Context, present};
pub use graph::{Graph, GraphExec};
