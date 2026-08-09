//! One row that describes no real checkpoint, for tests that need bytes.
//!
//! # Why a closed set needs this
//!
//! Identity is a manifest match: a checkpoint is a known model or it is
//! refused. That is the refactor's whole point and it is worth the cost it
//! names — but one of the costs is that **no test can afford a real row**.
//! The smallest catalog row is `llama-3.2-1b`, whose embedding alone is
//! 128 256 × 2048 × 2 bytes; a test that writes real safetensors and runs
//! them through the executor and the writer would move half a gigabyte to
//! check the shape of a fused bank.
//!
//! The tests that need real bytes are exactly the ones worth having. So
//! there is one row here, llama-shaped and small enough to write, and it is
//! behind a feature so that it is absent from anything shipped.
//!
//! # Why this is not a hole in the check
//!
//! A test row weakens the closed set only if a real checkpoint could match
//! it. This one cannot: a 64-wide model with a 128-token vocabulary is not
//! a thing anyone trained, and any checkpoint that *did* match those extents
//! is one this build could not serve for other reasons. The manifest is
//! still the whole check for it — a snapshot that gets one extent wrong is
//! refused here exactly as a real one would be, which is what makes it a
//! useful fixture rather than a bypass.
//!
//! # Why it is a `Llama3` and not its own type
//!
//! Because the point is to exercise the real path. A bespoke `Variant` impl
//! would be a second implementation of the thing under test, and the fixture
//! would stop being evidence about the code that runs in production. This is
//! a row of the same type, in the same table, authored by the same pass.

use crate::catalog::Variant;
use crate::llama_3::Llama3;
use crate::shared::llama_like::spec::LlamaLikeFacts;
use model_compiler::facts::{NormPlacement, QkNorm};
use model_compiler::trace::{NormVariant, RopeKind};

/// The id a test names with `--as`, and the one it will see reported back.
///
/// The `test-` prefix is load-bearing: `a_shipped_catalog_has_no_test_rows`
/// asserts that no row without this feature carries it, so the prefix is
/// how a test row that escaped into a real table would be caught.
pub const TINY_LLAMA: &str = "test-tiny-llama";

/// The row. Its numbers are `tests/model_artifact.rs`'s snapshot.
///
/// Untied embeddings on purpose — a separate `lm_head` is one more tensor
/// for the contract to place, and a fixture that skipped it would leave the
/// tied path as the only one any cheap test covered.
pub const VARIANTS: &[Llama3] = &[Llama3 {
    id: TINY_LLAMA,
    shape: LlamaLikeFacts {
        hidden: 64,
        layers: 2,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 16,
        n_experts: 0,
        experts_per_token: 0,
        moe_intermediate: 0,
        shared_intermediate: 0,
        intermediate: 96,
        vocab: 128,
        rope: RopeKind::Standard,
        norm_variant: NormVariant::Plain,
        norm_placement: NormPlacement::Pre,
        qk_norm: QkNorm::Off,
        fused_qkv: true,
        tied_embeddings: false,
        qkv_bias: false,
        o_bias: false,
        router_bias: false,
    },
    rope_theta: 10_000.0,
    norm_eps: 1e-6,
    window: -1,
    // No rescaling: this row states the plain ladder, so a test reading a
    // rope table is reading arithmetic it can do in its head.
    rope_factor: 1.0,
}];

/// This module's contribution to [`crate::catalog::catalog`].
#[must_use]
pub fn rows() -> &'static [&'static dyn Variant] {
    static ROWS: std::sync::OnceLock<Vec<&'static dyn Variant>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| VARIANTS.iter().map(|v| v as &'static dyn Variant).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture identifies itself, which is the only thing it must do.
    ///
    /// If this fails, every test that writes the snapshot fails with a
    /// manifest diff instead of the thing it was checking, so it is worth
    /// one test here to say which end is wrong.
    #[test]
    fn the_test_row_is_in_the_catalog_under_its_own_id() {
        let row = crate::catalog::find(TINY_LLAMA).expect("the test row is not in the catalog");
        assert_eq!(row.id(), TINY_LLAMA);
    }

    /// And it is the ONLY row with the prefix.
    ///
    /// The prefix is what `a_shipped_catalog_has_no_test_rows` looks for, so
    /// a second test row that forgot it would make that check pass while
    /// meaning less.
    #[test]
    fn every_test_row_says_so_in_its_id() {
        let prefixed: Vec<&str> = crate::catalog::ids()
            .into_iter()
            .filter(|id| id.starts_with("test-"))
            .collect();
        assert_eq!(
            prefixed,
            vec![TINY_LLAMA],
            "a test row without the `test-` prefix is invisible to the check \
             that keeps test rows out of a shipped catalog"
        );
    }
}
