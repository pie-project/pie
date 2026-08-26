//! DeepSeek V4 — the MLA flagship on the menlo stack: hyper-connection
//! residual streams around every block, attention over a shared compressed
//! kv plane whose windowed main path merges a pooled long-range path by lse,
//! and a sqrt-softplus-routed MoE above the first dense layer. The
//! declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred until the new loader lands —
//! imports keep reading the old crate meanwhile.

pub mod forward;
pub mod model;

use model::Model;
use new_model_ir::Repr;

/// The representation the shipped checkpoints store their weights in — the
/// runtime successor of the old `ShippedW1` phantom type (design §5). The kv
/// dtype the old `ShippedKv` named is gone with it: a cache row's element
/// layout is load-time business, not a model parameter.
const SHIPPED_W1: Repr = Repr::Bf16;

pub const CATALOG: &[(&str, new_model_dsl::TraceFn)] = new_model_dsl::catalog![
    (
        "dsv4-base-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::base(SHIPPED_W1, 1),
    ),
    (
        "dsv4-base-bf16-kv-bf16-tp2",
        new_model_dsl::trace_hybrid,
        Model::base(SHIPPED_W1, 2),
    ),
];
