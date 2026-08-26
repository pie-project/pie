use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Norm {
    Rmsnorm {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        #[out]
        y: ValueId,
    },
    RmsnormPerHead {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        #[out]
        y: ValueId,
    },
    /// Scales by `weight + 1` (Gemma-style).
    RmsnormPlusOne {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        #[out]
        y: ValueId,
    },
    RmsnormPerHeadPlusOne {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        #[out]
        y: ValueId,
    },
    RmsnormNoScale {
        x: ValueId,
        head_dim: u32,
        eps: f32,
        #[out]
        y: ValueId,
    },
    /// `x` is f32; the norm is gated by `gate`, per group of `head_dim`.
    RmsnormGated {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        #[out]
        y: ValueId,
    },
    /// Like `RmsnormGated`, but grouped by head count instead of head width.
    RmsnormGatedBy {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        heads: u32,
        eps: f32,
        #[out]
        y: ValueId,
    },
    ResidualAdd {
        x: ValueId,
        y: ValueId,
        #[out(alias = y)]
        y_out: ValueId,
    },
    AddBias {
        bias: ValueId,
        out: ValueId,
        #[out(alias = out)]
        out_out: ValueId,
    },
    MulScalar {
        s: f32,
        x: ValueId,
        #[out(alias = x)]
        x_out: ValueId,
    },
    Scale {
        s: ValueId,
        x: ValueId,
        #[out(alias = x)]
        x_out: ValueId,
    },
    /// Norms the summed blocks against the prefix, then projects the blend.
    ResBlend {
        prefix: ValueId,
        blocks: Vec<ValueId>,
        weight: ValueId,
        eps: f32,
        proj: ValueId,
        #[out]
        y: ValueId,
    },
}
