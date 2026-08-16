//! What a trace that states one of `layout`'s symbols binds to.
//!
//! Every row derives. A zero row width refuses in the launcher, not here.

use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    // Binds, then refuses at run time: `keys::PerHeadDim` reads
    // `LaunchSpec::per_head_dim`, whose one writer fires on
    // `OpKind::RmsnormPerHead` and never on `OpKind::SplitQGate`.
    Bound::derived("layout::split_q_gate_bf16"),
    Bound::derived("layout::split_bf16_rows"),
    Bound::derived("layout::split_qwen_gdn_ba_bf16"),
    Bound::derived("layout::embed_bf16"),
    Bound::derived("layout::gather_bf16_rows"),
    Bound::derived("layout::transpose_bf16_nld_to_lnd"),
];
