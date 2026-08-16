//! What a trace that states one of `rope`'s symbols binds to.
//!
//! Three pairs of facts read alike and are not alike; getting one wrong binds
//! a plausible number rather than faulting. `keys::HeadDim` is the width
//! attention computes at, `keys::KvHeadDim` the width the cache was allocated
//! at; `keys::Theta` is the per-layer table, `keys::RopeTheta` the fire-wide
//! base; `keys::Rows` is `rows.count`, `keys::RowsTotal` the fire's whole row
//! space.

use super::Bound;

/// Every symbol this family binds. A note on an uncrossed row names what the
/// derived column refuses first.
pub static ARMS: &[Bound] = &[
    Bound::derived("rope::rope_standard_table"),
    Bound::derived("rope::rope_bf16"),
    // `keys::Theta`, the per-layer table -- not the fire-wide base.
    Bound::derived("rope::qk_rmsnorm_rope_bf16"),
    // `keys::RowsTotal`: the grid spans the fire's whole row space, so
    // `rows.count` under-launches a peel and leaves its tail rows unrotated.
    Bound::derived("rope::qk_rmsnorm_rope_bf16_devwin"),
    Bound::derived("rope::rope_partial_last_bf16"),
    Bound::derived("rope::rope_partial_bf16"),
    // `k_weight`'s `Or(&Weight(1), &Lit(Null))` is live: weights are checked as
    // a band, so a fused QK norm loads with one weight or two.
    Bound::derived("rope::qk_rmsnorm_rope_bf16_rounded"),
    // ── D2's three splits: the Q-alone form of the symbol above each ─────
    // The old spelling let a statement's operand count decide whether K rotated.
    Bound::derived("rope::rope_partial_last_q_bf16"),
    Bound::derived("rope::rope_partial_q_bf16"),
    // `weight(0)` is `?` and not `unwrap_or`: a Q-only norm has one weight.
    Bound::derived("rope::q_rmsnorm_rope_bf16_rounded"),
    Bound::derived("rope::rope_yarn_original_bf16"),
    Bound {
        symbol: "rope::rope_yarn_bf16",
        arm: None,
        unbound: Some(
            "llama-3's `low_freq_factor`/`high_freq_factor`, which no statement or context carries",
        ),
    },
    Bound {
        symbol: "rope::qk_rmsnorm_mrope_bf16",
        arm: None,
        unbound: Some(
            "the `(t, h, w)` section split, a vision checkpoint property no statement carries",
        ),
    },
    // No launcher of this name is declared, so a mark could never reach it.
    Bound {
        symbol: "rope::rope_partial_bf16_position_delta",
        arm: None,
        unbound: Some(
            "the position offset, a fact about a draft/verify pairing no statement carries",
        ),
    },
    // Not a missing mark: signature and statement disagree on how many operands
    // exist -- four `*mut` against one declared result -- and no vocabulary
    // settles that. `hnd_layout` keeps its `#[source(..)]` because
    // `keys::KvHndLayout` is declared over `i32` and the parameter is a `bool`.
    Bound {
        symbol: "rope::rope_write_kv_bf16",
        arm: None,
        unbound: Some(
            "which addresses `q` and `k` rotate at: the contract states no `in_place` pair",
        ),
    },
];
