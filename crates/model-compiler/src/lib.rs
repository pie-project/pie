//! The forward-pass TOOLCHAIN: an authoring eDSL, the traced form it
//! produces, and the lowering that costs one.
//!
//! A model family's forward pass is a **declaration**: ordinary Rust that runs
//! at *model-load time*, with the checkpoint's config facts in hand, and
//! records what one pass computes. Static control flow — layer kinds, rope
//! variant, qk-norm, whether the deployment bound a fused QKV — executes
//! during tracing and leaves no trace. What remains is the **traced form**:
//! the operation sequence a driver executes, with shapes symbolic in the
//! fire's extents and weights referenced by declaration name.
//!
//! ```text
//! declaration  ──trace──▶  forward plan  ──(C ABI)──▶  driver executes
//! (what a pass    (the ops to run,        (model::ffi, and the
//!  computes)       in what order)          committed header)
//! ```
//!
//! ## The declarations are not here
//!
//! They are in `crates/model`, one per generation, beside that model's chat
//! template and its load contract — `.wiki/tart-todo.md` item 1, and the shape
//! `.wiki/tart/dsl.md` ③ always described ("the model file is
//! `families/<family>/<backend>.rs`"). What is here is what a declaration is
//! WRITTEN IN, and it names no family:
//!
//! * [`dsl`] — the authoring surface. [`dsl::M`] offers the dense-transformer
//!   weight namespace, parameterized by a [`dsl::ModelShape`] each family
//!   projects into; nothing about one family's rope or norm placement reaches
//!   it.
//! * [`trace`] — the traced form, and the vocabulary of ops it is made of.
//! * [`lower`] — a traced form to a lowered one, with its costs.
//! * [`kernels`] — the compiler's end of the per-backend signature tables
//!   (the tables themselves are `kernels-cuda` / `kernels-metal`).
//! * [`facts`] — the two words more than one family is written in.
//!
//! The edge runs one way, and that is the whole reason for the split: a model
//! names the toolchain, the toolchain names no model.

pub mod dsl;
pub mod facts;
pub mod kernels;
pub mod lower;
pub mod trace;

pub use facts::{NormPlacement, QkNorm};
pub use trace::{
    DType, Dim, DynAxis, FireClass, ForwardPlan, HookStage, Op, OpKind, Shape, StateRef,
    StateStore, TraceBuilder, ValueId,
};

/// The tracer's fingerprint: an FNV-1a content hash of this crate's `src/`,
/// computed by `build.rs`.
///
/// The traced form is a pure function of (declaration code, facts), so this
/// number plus the facts identifies a plan exactly. `model`'s FFI stamps it
/// into every plan header so a consumer can key a cache or a golden on
/// `PieForwardPlan::compiler_version` and have it invalidate itself when the
/// tracer changes.
///
/// It is a function rather than a constant because `env!` only reads the
/// environment of the crate being compiled, and the crate that needs the
/// number is no longer this one.
pub fn compiler_version() -> u64 {
    env!("PIE_FORWARD_COMPILER_HASH").parse::<u64>().unwrap_or(0)
}
