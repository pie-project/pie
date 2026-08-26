//! Kimi K3 on the menlo stack: a hybrid decoder interleaving KDA
//! (delta-attention) layers with full MLA layers, residual blending every
//! few layers, and situ-activated dense or routed mlps. The
//! declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "kimik3-bf16-mxfp4-kv-bf16",
        model_dsl::trace_hybrid,
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "kimik3-bf16-mxfp4-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
