//! The author registry's own shape.
//!
//! "Which `model_type` does pie support?" used to be asked in three places on
//! two sides of a C ABI, and this file was a SOURCE GREP holding the answers
//! together: [`HF_ROWS`] and [`MLX_ROWS`] here, against
//! `crates/driver-cuda/csrc/src/model/registry.cpp` and
//! `crates/driver-metal/csrc/src/model/facts.hpp` there. The property was
//! about what another language declared, and no Rust-side assertion could
//! reach it.
//!
//! **Both C++ trees are deleted and both drivers are Rust.** Each driver's
//! answer is now compared against these rows directly, in the crate that owns
//! it and can depend on both:
//!
//! * `driver-cuda/tests/facts_registry.rs` — `FACTS_ROWS` vs `HF_ROWS`.
//! * `driver-metal/tests/family_registry.rs` — `ModelFamily` vs `MLX_ROWS`.
//!
//! Both are strictly stronger than the greps they replace: a `match` arm is
//! not a literal list, and a table that stopped being greppable used to fail
//! the "did we find anything at all" guard rather than the check itself.
//!
//! What is left here is the one claim that is about these tables alone and
//! belongs beside them.

use std::collections::BTreeSet;

use model::contract::{HF_ROWS, MLX_ROWS};

/// A row appearing twice would shadow the second silently — the lookup takes
/// the first match — so the table's own shape is worth one assertion.
#[test]
fn no_model_type_is_declared_twice() {
    for (what, rows) in [("HF_ROWS", HF_ROWS), ("MLX_ROWS", MLX_ROWS)] {
        let unique: BTreeSet<_> = rows.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            unique.len(),
            rows.len(),
            "{what}: a model_type is listed more than once; the later row is dead"
        );
    }
}
