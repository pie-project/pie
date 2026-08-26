//! Gemma 4 — a dense decoder interleaving sliding-window and full-attention
//! layers, with a kv-sharing tail that re-reads earlier layers' caches and,
//! on the e4b SKU, per-layer embeddings relayed into every layer. The
//! declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "gemma4-e4b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::e4b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
