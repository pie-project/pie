//! # `pie-dsl` — the PTIR embedded DSL (Thrust 3, Rust SDK)
//!
//! Author *programmable dataflows* as Rust closures that trace **once** into a
//! canonical PTIR trace container. A program is a closure whose effects are
//! channel `put`/`take`s; `if`/`for` resolve at trace time; a different branch
//! is a different program (batch-by-program).
//!
//! This crate is the boundary-agnostic authoring core: the `Tensor`/`Channel`
//! eDSL, the trace-recording session, the SDK span lints, and the neutral
//! [`Builder`] that lowers stage closures + descriptor-port
//! bindings into the IR's canonical
//! [`TraceContainer`](pie_ir::container::TraceContainer). Tracing is its
//! *implementation strategy*, not its identity — hence `pie-dsl`.
//!
//! It does **not** bind (the guest does not bind; `forward-pass.program` is the
//! authoritative gate) and knows nothing of WIT. The author-facing lifetime
//! objects (`ForwardPass`, `Pipeline`, `WorkingSet`, host `Channel` transport)
//! live in `inferlet`, which wraps the WIT resources and drives this builder.
//!
//! ```
//! use pie_dsl::prelude::*;
//! use pie_dsl::builder::Builder;
//! use pie_dsl::Port;
//!
//! let tok = Channel::new([1], dtype::i32);
//! let indptr = Channel::from([0u32, 1]);
//! let out = Channel::new([1], dtype::i32);
//! let rng = Channel::from([7u32, 0]);
//! tok.put([1i32]); // seed BOS
//!
//! let mut b = Builder::new(/* vocab */ 32, /* page_size */ 4);
//! b.bind_port(Port::EmbedTokens, &tok);
//! b.bind_port(Port::EmbedIndptr, &indptr);
//! b.stage(Stage::Epilogue, || {
//!     let logits = intrinsics::logits();
//!     let r = rng.take();
//!     let g = gumbel(&r, [intrinsics::vocab()]);
//!     let t = reduce_argmax(logits + g);
//!     rng.put(&r + Tensor::constant([0u32, 1]));
//!     tok.put(&t);
//!     out.put(t);
//! });
//!
//! let traced = b.build().expect("valid trace");
//! assert_ne!(traced.identity_hash(), 0);
//! ```
//!
//! ## Deviations from the spec (Rust limitations; flagged, manager-approved)
//! - Model constants are functions (`intrinsics::vocab()`), not bare paths.
//! - Bare integer-literal operands (`x + 1`) resolve to `i32`, but a scalar
//!   operand adopts the dtype of the tensor it meets, so the suffix only
//!   matters between two scalars.
//! - Values reused as op operands take `&` (a taken value used at multiple sites).

extern crate alloc;

pub mod builder;
pub mod channel;
mod context;
pub mod dtype;
pub mod error;
pub mod intrinsics;
mod lint;
pub mod model;
pub mod value;

pub use builder::{Builder, PortInput, Traced};
pub use channel::{Channel, IntoPut, Put};
pub use error::{Endpoint, Span, TraceError, TraceErrors};
/// The eDSL op surface. Glob-re-exported rather than listed: an op is public
/// exactly when it is `pub` in [`value`], so adding one is a single edit and
/// cannot be half-done (present at the root, missing from [`prelude`]).
pub use value::*;

/// The canonical PTIR contract (op-table, container, validator, interpreter) —
/// re-exported for tests and downstream carriers.
pub use pie_ir as ptir;
pub use pie_ir::registry::{Port, Stage};
pub use pie_ir::types::{DType, Shape, ValueType};

/// Glob-import surface for the DSL eDSL op/value names.
/// The author-facing `ForwardPass`/`Pipeline`/`WorkingSet` surface lives in
/// `inferlet::ptir::attention::prelude`, which re-exports this plus those wrapper types.
pub mod prelude {
    pub use crate::builder::{Builder, PortInput};
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::intrinsics;
    pub use crate::value::*;
    pub use pie_ir::registry::{Port, Stage};
}
