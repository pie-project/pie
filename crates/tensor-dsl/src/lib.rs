//! # `tensor-dsl` — the PTIR embedded DSL (Rust SDK)
//!
//! Author *programmable dataflows* as Rust closures that trace **once** into a
//! canonical PTIR trace container. A program is a closure whose effects are
//! channel `put`/`take`s; `if`/`for` resolve at trace time, so a different
//! branch is a different program (batch-by-program).
//!
//! This crate is the boundary-agnostic authoring core: the `Tensor`/`Channel`
//! eDSL, the trace-recording session, the SDK span lints, and the neutral
//! [`Builder`] that lowers stage closures and descriptor-port bindings into
//! [`TraceContainer`](tensor_ir::container::TraceContainer). It does **not**
//! bind and knows nothing of WIT; the author-facing lifetime objects live in
//! `inferlet`, which wraps the WIT resources and drives this builder.
//!
//! Two Rust-imposed deviations from the spec: model constants are functions
//! (`intrinsics::vocab()`) rather than bare paths, and a value reused as an op
//! operand takes `&`. A bare integer literal resolves to `i32`, but a scalar
//! adopts the dtype of the tensor it meets, so the suffix only matters between
//! two scalars.

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
/// exactly when it is `pub` in [`value`], so adding one cannot be half-done.
pub use value::*;

/// The canonical PTIR contract, re-exported for tests and downstream carriers.
pub use tensor_ir as ptir;
pub use tensor_ir::registry::{Port, Stage};
pub use tensor_ir::types::{DType, Shape, ValueType};

/// Glob-import surface for the DSL's op/value names.
pub mod prelude {
    pub use crate::builder::{Builder, PortInput};
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::intrinsics;
    pub use crate::value::*;
    pub use tensor_ir::registry::{Port, Stage};
}
