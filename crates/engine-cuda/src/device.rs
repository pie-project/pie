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
//! capture, where a composition is CUT, which key a body answers to, when one
//! is evicted — is [`record`](crate::record)'s and [`serve`](crate::serve)'s,
//! the same way the eager plane's is.

pub mod alloc;
pub mod conditional;
pub mod ctx;
/// **The elastic supply** (alto design §8, wave C): a budgeted pool of
/// physical pages, and virtual ranges whose backing grows and trims under a
/// fixed address. The one module in this crate that maps memory rather than
/// allocating it.
pub mod elastic;
pub mod graph;
/// The coordinate system one capture publishes (`palo cuda-abi` §7, step 3),
/// and the diff between two of them. A MEASUREMENT surface since the tier-2
/// campaign deleted the fold that read it on the fire path.
pub mod map;
/// Captured-graph introspection: the walk [`map`] is built on, and the
/// write-side probes, which are the only place this shell prices a rebind.
pub mod nodes;

pub use alloc::{
    Buffer, Pinned, copy_any, copy_d2d, copy_d2h, free_bytes, write_raw, zero_span,
    zero_span_on,
};
pub use elastic::{Arena, PhysicalPool};
pub use ctx::{Context, present};
pub use graph::{Graph, GraphExec};
