use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// Recurrent-state mixers: causal conv, gated delta nets, KDA. `state` is the
/// recurrent cache — storage only, updated in place by the kernel.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Ssm {
    CausalConv1d {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        #[out]
        y: ValueId,
    },
    /// Prefill form: walks the fire's ambient request boundaries.
    CausalConv1dChunked {
        x: ValueId,
        weight: ValueId,
        state: ValueId,
        conv_width: u32,
        #[out]
        y: ValueId,
    },
    /// Folds `ba` with dt bias and A-log into per-head decay gates.
    GdnPrep {
        ba: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        #[out]
        gates: ValueId,
    },
    GatedDelta {
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        #[out]
        y: ValueId,
    },
    GatedDeltaChunked {
        qkv: ValueId,
        z: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        #[out]
        y: ValueId,
    },
    KdaStep {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        #[out]
        y: ValueId,
    },
    KdaChunked {
        mixed: ValueId,
        f: ValueId,
        b: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        state: ValueId,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        #[out]
        y: ValueId,
    },
}
