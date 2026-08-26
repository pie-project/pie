//! Qwen 3.5 — the first model on the menlo stack: a hybrid decoder
//! interleaving gated-delta-net layers with full attention, dense or routed
//! mlps by SKU. The declaration lives in [`model`], the forward pass in
//! [`forward`]; `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "qwen35-a3b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d3b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::d3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-d0.8b-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::d0_8b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::a3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
