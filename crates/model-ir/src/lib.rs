//! THE TRACED FORM — what a forward pass IS, once the declaration that
//! states it has run.
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──lower──▶  driver executes
//!  `model-dsl`             THIS CRATE              `model-compiler`
//! ```
//!
//! A model family's forward pass is a **declaration**: ordinary Rust that runs
//! at *model-load time*, with the checkpoint's config facts in hand, and
//! records what one pass computes. Static control flow — layer kinds, rope
//! variant, qk-norm, whether the deployment bound a fused QKV — executes
//! during tracing and leaves no trace. What remains is what lives here: the
//! operation sequence a driver executes, with shapes symbolic in the fire's
//! extents and weights referenced by declaration name.
//!
//! * [`trace`] — the op vocabulary, the [`ForwardPlan`] container, and the
//!   [`TraceBuilder`] a declaration records onto.
//! * [`seam`] — the named extension points a declaration states.
//! * [`kernels`] — the per-backend signature tables, and the load-time check
//!   that a plan only names symbols they cover.
//! * [`facts`] — the two words more than one family is written in.
//!
//! # Why this is its own crate
//!
//! It was three modules of `model-compiler` while that crate was the whole
//! toolchain, and the split is not about size. It is that **no consumer needed
//! all of it**: `model` writes declarations and never lowers one, every driver
//! lowers and never writes one, and the two halves met only here. Holding them
//! together meant `driver-metal`, `driver-vulkan` and `driver-wgpu` each
//! compiled 4,469 lines of `dsl::cuda` — an authoring surface for a backend
//! they are not — to reach `lower`.
//!
//! So the arrows now run in one direction and meet at this crate:
//!
//! ```text
//!   model ──▶ model-dsl ──┐
//!                         ├──▶ model-ir
//!   driver-* ──▶ model-compiler ──┘
//! ```
//!
//! Nothing here names a family, a `Val`, a weight handle, or a device. That is
//! the property that makes the layering hold rather than merely describe it.

pub mod facts;
pub mod kernels;
pub mod seam;
pub mod trace;

pub use facts::{NormPlacement, QkNorm};
pub use trace::{
    DType, Dim, DynAxis, FireClass, ForwardPlan, HookStage, Op, OpKind, Shape, StateRef,
    StateStore, TraceBuilder, ValueId,
};
