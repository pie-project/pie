//! Qwen 3.5 — the first model on the menlo stack: a hybrid decoder
//! interleaving gated-delta-net layers with full attention, dense or routed
//! mlps by SKU. The declaration lives in [`model`], the forward pass in
//! [`forward`]; `import.rs` (checkpoint mapping) is deferred until the new
//! loader lands — imports keep reading the old crate meanwhile.

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
        "qwen35-a3b-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::a3b(SHIPPED_W1, 1),
    ),
    (
        "qwen35-d3b-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::d3b(SHIPPED_W1, 1),
    ),
    (
        "qwen35-d0.8b-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::d0_8b(SHIPPED_W1, 1),
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        new_model_dsl::trace_hybrid,
        Model::a3b(SHIPPED_W1, 2),
    ),
];
