use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Moe {
    TopkSoftmax {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        #[out]
        routes: ValueId,
        #[out]
        weights: ValueId,
    },
    TopkSigmoid {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        #[out]
        routes: ValueId,
        #[out]
        weights: ValueId,
    },
    /// Sigmoid routing with a per-expert bias; weights pass through sqrt-softplus.
    TopkSqrtSoftplus {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        #[out]
        routes: ValueId,
        #[out]
        weights: ValueId,
    },
    /// Grouped matmul: each routed row multiplies the expert `routes` selects from `bank`.
    MatmulSelect {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        #[out]
        y: ValueId,
    },
    MatmulSelectBias {
        x: ValueId,
        bank: ValueId,
        bias: ValueId,
        routes: ValueId,
        #[out]
        y: ValueId,
    },
    /// Folds the top_k routed rows back to one row per token.
    WeightedSum {
        routed: ValueId,
        weights: ValueId,
        #[out]
        y: ValueId,
    },
    SigmoidGateAdd {
        routed: ValueId,
        shared: ValueId,
        gate: ValueId,
        #[out]
        y: ValueId,
    },
}
