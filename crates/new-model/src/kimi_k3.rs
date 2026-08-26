//! Kimi K3 on the menlo stack: a hybrid decoder interleaving KDA
//! (delta-attention) layers with full MLA layers, residual blending every
//! few layers, and situ-activated dense or routed mlps. The
//! declaration lives in [`model`], the forward pass in [`forward`];
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
const SHIPPED_W2: Repr = Repr::Mxfp4;

pub const CATALOG: &[(&str, new_model_dsl::TraceFn)] = new_model_dsl::catalog![
    (
        "kimik3-bf16-mxfp4-kv-bf16",
        new_model_dsl::trace_hybrid,
        Model::k3(SHIPPED_W1, SHIPPED_W2, 1),
    ),
    (
        "kimik3-bf16-mxfp4-kv-bf16-tp2",
        new_model_dsl::trace_hybrid,
        Model::k3(SHIPPED_W1, SHIPPED_W2, 2),
    ),
];
