//! gpt-oss — a mixture-of-experts decoder alternating sliding-window and
//! full-attention layers, with learned attention sinks and mxfp4 expert
//! banks. The declaration lives in [`model`], the forward pass in
//! [`forward`]; `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "gptoss-20b-bf16-mxfp4-kv-bf16",
        model_dsl::trace_hybrid,
        Model::b20(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16",
        model_dsl::trace_hybrid,
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
