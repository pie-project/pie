//! Gemma 4 — a dense decoder interleaving sliding-window and full-attention
//! layers, with a kv-sharing tail that re-reads earlier layers' caches and,
//! on the e4b SKU, per-layer embeddings relayed into every layer. The
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
        "gemma4-e4b-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::e4b(SHIPPED_W1, 1),
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::b31(SHIPPED_W1, 1),
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        new_model_dsl::trace_hybrid,
        Model::b31(SHIPPED_W1, 2),
    ),
];
