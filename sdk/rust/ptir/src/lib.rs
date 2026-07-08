//! # `ptir` — the PTIR tracing frontend (Thrust 3, Rust SDK)
//!
//! Author *programmable dataflows* as Rust closures that trace **once** into a
//! validated PTIR trace container. The surface matches the
//! `docs/ptir/overview.md` §3/§6 examples verbatim: a program is a closure whose
//! effects are channel `put`/`take`s; `if`/`for` resolve at trace time; a
//! different branch is a different program (batch-by-program).
//!
//! The trace lowers to echo's canonical
//! [`TraceContainer`](pie_ptir::container::TraceContainer) and is
//! validated by echo's [`bind`](pie_ptir::validate::bind) — the
//! shared P0 contract (`interface/sampling-ir/src/ptir`). This crate owns the
//! authoring surface + trace-once memoization + the SDK span lints.
//!
//! ```
//! use ptir::prelude::*;
//!
//! ptir::model::configure(/* vocab */ 32, /* page_size */ 4, /* num_layers */ 2);
//!
//! let tok = Channel::new([1], dtype::i32);
//! let out = Channel::new([1], dtype::i32);
//! let rng = Channel::from([7u32, 0]);
//!
//! let fwd = ForwardPass::new();
//! let lane_1 = Tensor::constant([0u32, 1]);
//! fwd.embed(&tok, lane_1);
//! fwd.epilogue(|| {
//!     let logits = intrinsics::logits();
//!     let r = rng.take();
//!     let g = gumbel(&r, [intrinsics::vocab()]);
//!     let t = reduce_argmax(add(logits, g));
//!     rng.put(add(&r, Tensor::constant([0u32, 1])));
//!     tok.put(&t);
//!     out.put(t);
//! });
//!
//! let traced = fwd.trace().expect("valid trace");
//! assert_ne!(traced.identity_hash(), 0);
//! ```
//!
//! ## Deviations from the overview (Rust limitations; flagged, manager-approved)
//! - Model constants are functions (`intrinsics::vocab()`), not bare paths.
//! - Bare integer-literal operands (`add(x, 1)`) resolve to `u32`; explicit
//!   `i32` constants use `Tensor::constant(-1i32)`.
//! - `attn_working_set`'s multi-arg arities are passed as tuples; the sugar
//!   `attn_working_set(&ws, &len)` stays verbatim.
//! - Values reused as op operands take `&` (a taken value used at multiple sites).

extern crate alloc;

pub mod channel;
mod context;
pub mod dtype;
pub mod error;
pub mod forward;
pub mod intrinsics;
mod lint;
pub mod model;
pub mod pipeline;
pub mod value;

pub use channel::{Channel, HostError, IntoPut, Put, Taken};
pub use error::{Endpoint, Span, TraceError, TraceErrors};
pub use forward::{AttnWsArgs, ForwardPass, Indptr, Remap, SlotGrant, TracedForward, WorkingSet};
pub use pipeline::Pipeline;
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
pub use pie_ptir::registry::Stage;
pub use pie_ptir::types::{DType, Shape, ValueType};

/// Glob-import surface for inferlet authors: the verbatim overview names.
pub mod prelude {
    pub use crate::channel::Channel;
    pub use crate::dtype;
    pub use crate::forward::{ForwardPass, WorkingSet};
    pub use crate::intrinsics;
    pub use crate::pipeline::Pipeline;
    pub use crate::value::{
        add, and, broadcast, cast, cumprod, cumsum, cummass_le, div, eq, exp, gather, gather_row,
        ge, gt, gumbel, iota, l2norm, le, log, log_softmax, lt, mask_apply, matmul, max_elem,
        min_elem, mul, ne, neg, not, or, pivot_threshold, prob_ge, rank_le, reduce_argmax,
        reduce_max, reduce_min, reduce_sum, rem, reshape, rng, scatter_add, scatter_set, select,
        softmax, sub, top_k, transpose, Tensor,
    };
}
