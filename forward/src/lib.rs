//! Forward-pass declarations.
//!
//! A model family's forward pass is a **declaration**: ordinary Rust that
//! runs at *model-load time*, with the checkpoint's config facts in hand,
//! and records what one pass computes. Static control flow — layer kinds,
//! rope variant, qk-norm, whether the deployment bound a fused QKV — executes
//! during tracing and leaves no trace. What remains is the **traced form**:
//! the operation sequence a driver executes, with shapes symbolic in the
//! fire's extents and weights referenced by declaration name.
//!
//! The shape mirrors `loader/`:
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──(C ABI, later)──▶  driver executes
//! (what a pass    (the ops to run,          (committed header,
//!  computes)       in what order)            generated)
//! ```
//!
//! Two rules carried from `pie-application-plan.md` §5:
//!
//! * **The declaration says what varies. It never says how to lower it.**
//!   Ops name operations (`rmsnorm`, `attention`), never kernels. Fusion —
//!   the hand-written passes' fused QKV+rope+KV-write, fused norm+rope —
//!   is an emitter decision made where the backend can see both the
//!   adjacency and the divergence, because a fused edge cannot be a merge
//!   point.
//! * **Syntax is required exactly where cost is incurred.** A declaration
//!   with no structural divergence is an ordinary forward pass; the first
//!   family here (`llama_like`) has none, so nothing in it is `dyn`.

pub mod facts;
pub mod family;
pub mod trace;

pub use facts::LlamaLikeFacts;
pub use trace::{Dim, DType, ForwardPlan, Op, OpKind, Shape, TraceBuilder, ValueId};
