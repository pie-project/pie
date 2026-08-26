//! GLM 5 on the menlo stack: multi-head latent attention with a sparse
//! top-k indexer on every layer, dense early layers giving way to a routed
//! mlp. The declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred to the loader port.

pub mod forward;
pub mod model;

use model::Model;
use model_dsl::Dtype;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "glm5-a12b-bf16-bf16-kv-bf16",
        model_dsl::trace_hybrid,
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
    ),
    (
        "glm5-a12b-bf16-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid,
        Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
    ),
];
