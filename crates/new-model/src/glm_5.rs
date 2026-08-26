//! GLM 5 on the menlo stack: multi-head latent attention with a sparse
//! top-k indexer on every layer, dense early layers giving way to a routed
//! mlp. The declaration lives in [`model`], the forward pass in [`forward`];
//! `import.rs` (checkpoint mapping) is deferred until the new loader lands —
//! imports keep reading the old crate meanwhile.

pub mod forward;
pub mod model;

use model::Model;
use new_model_ir::Repr;

/// The representations the shipped checkpoints store their weights in — the
/// runtime successors of the old `ShippedW1`/`ShippedW2` phantom types
/// (design §5). The kv dtype the old `ShippedKv` named is gone with them: a
/// cache row's element layout is load-time business, not a model parameter.
const SHIPPED_W1: Repr = Repr::Bf16;
const SHIPPED_W2: Repr = Repr::Bf16;

pub const CATALOG: &[(&str, new_model_dsl::TraceFn)] = new_model_dsl::catalog![
    (
        "glm5-a12b-bf16-bf16-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::a12b(SHIPPED_W1, SHIPPED_W2, 1),
    ),
    (
        "glm5-a12b-bf16-bf16-kv-bf16-tp2",
        new_model_dsl::trace_hybrid,
        Model::a12b(SHIPPED_W1, SHIPPED_W2, 2),
    ),
];
