//! DeepSeek V4 — the MLA flagship on the menlo stack: hyper-connection
//! residual streams around every block, attention over a shared compressed
//! kv plane whose windowed main path merges a pooled long-range path by lse,
//! and a sqrt-softplus-routed MoE above the first dense layer. The
//! declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "dsv4-base-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "dsv4-base-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
