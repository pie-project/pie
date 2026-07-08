//! # `ptir-dsl` — the PTIR embedded DSL (Thrust 3, Rust SDK)
//!
//! Author *programmable dataflows* as Rust closures that trace **once** into a
//! canonical PTIR trace container. A program is a closure whose effects are
//! channel `put`/`take`s; `if`/`for` resolve at trace time; a different branch
//! is a different program (batch-by-program).
//!
//! This crate is the boundary-agnostic authoring core: the `Tensor`/`Channel`
//! eDSL, the trace-recording session, the SDK span lints, and the neutral
//! [`Builder`](builder::Builder) that lowers stage closures + descriptor-port
//! bindings into echo's canonical
//! [`TraceContainer`](pie_ptir::container::TraceContainer). Tracing is its
//! *implementation strategy*, not its identity — hence `ptir-dsl`.
//!
//! It does **not** bind (D6: the guest does not bind; `forward-pass.new` is the
//! authoritative gate) and knows nothing of WIT. The author-facing lifetime
//! objects (`ForwardPass`, `Pipeline`, `WorkingSet`, host `Channel` transport)
//! live in `inferlet`, which wraps the WIT resources and drives this builder.
//!
//! ```
//! use ptir_dsl::prelude::*;
//! use ptir_dsl::builder::{Builder, PortInput};
//! use ptir_dsl::Port;
//!
//! ptir_dsl::model::configure(/* vocab */ 32, /* page_size */ 4, /* num_layers */ 2);
//!
//! let tok = Channel::new([1], dtype::i32);
//! let out = Channel::new([1], dtype::i32);
//! let rng = Channel::from([7u32, 0]);
//! tok.put([1i32]); // seed BOS
//!
//! let mut b = Builder::new();
//! b.bind_port(Port::EmbedTokens, &tok);
//! b.bind_port(Port::EmbedIndptr, PortInput::constant(Tensor::constant([0u32, 1])));
//! b.stage(Stage::Epilogue, || {
//!     let logits = intrinsics::logits();
//!     let r = rng.take();
//!     let g = gumbel(&r, [intrinsics::vocab()]);
//!     let t = reduce_argmax(add(logits, g));
//!     rng.put(add(&r, Tensor::constant([0u32, 1])));
//!     tok.put(&t);
//!     out.put(t);
//! });
//!
//! let traced = b.build().expect("valid trace");
//! assert_ne!(traced.identity_hash(), 0);
//! ```
//!
//! ## Deviations from the overview (Rust limitations; flagged, manager-approved)
//! - Model constants are functions (`intrinsics::vocab()`), not bare paths.
//! - Bare integer-literal operands (`add(x, 1)`) resolve to `u32`; explicit
//!   `i32` constants use `Tensor::constant(-1i32)`.
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
pub use channel::{Channel, HostError, IntoPut, Put, Taken};
pub use error::{Endpoint, Span, TraceError, TraceErrors};
pub use value::{
    add, and, broadcast, cast, cumprod, cumsum, div, eq, exp, gather, gather_row, ge, gt, gumbel,
    iota, l2norm, le, log, log_softmax, lt, mask_apply, matmul, max_elem, min_elem, mul, ne, neg,
    not, or, pivot_threshold, cummass_le, prob_ge, rank_le, reduce_argmax, reduce_max, reduce_min,
    reduce_sum, rem, reshape, rng, scatter_add, scatter_set, select, softmax, sub, top_k, transpose,
    AsTensor, IntoConst, IntoShape, PredicateArg, Tensor,
};

/// The canonical PTIR contract (op-table, container, validator, interpreter) —
/// re-exported for tests and downstream carriers.
pub use pie_ptir as ptir;
pub use pie_ptir::registry::{Port, Stage};
pub use pie_ptir::types::{DType, Shape, ValueType};

/// Glob-import surface for the DSL eDSL: the verbatim overview op/value names.
/// The author-facing `ForwardPass`/`Pipeline`/`WorkingSet` surface lives in
/// `inferlet::ptir::prelude`, which re-exports this plus those wrapper types.
pub mod prelude {
    pub use crate::builder::{Builder, PortInput};
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::intrinsics;
    pub use crate::value::{
        add, and, broadcast, cast, cumprod, cumsum, cummass_le, div, eq, exp, gather, gather_row,
        ge, gt, gumbel, iota, l2norm, le, log, log_softmax, lt, mask_apply, matmul, max_elem,
        min_elem, mul, ne, neg, not, or, pivot_threshold, prob_ge, rank_le, reduce_argmax,
        reduce_max, reduce_min, reduce_sum, rem, reshape, rng, scatter_add, scatter_set, select,
        softmax, sub, top_k, transpose, Tensor,
    };
    pub use pie_ptir::registry::{Port, Stage};
}
