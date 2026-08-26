use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// Hyper-connections: residual streams expanded, mixed by learned gates, and
/// collapsed back to one stream.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Hc {
    /// Tiles `x` across `streams` residual streams.
    Expand {
        x: ValueId,
        streams: u32,
        #[out]
        y: ValueId,
    },
    RmsnormF32 {
        streams: ValueId,
        eps: f32,
        #[out]
        y: ValueId,
    },
    /// Computes the layer input `x` plus the post/comb mixing matrices.
    Gates {
        normed: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        #[out]
        x: ValueId,
        #[out]
        post_mix: ValueId,
        #[out]
        comb_mix: ValueId,
    },
    /// Mixes the layer output back into the streams under the gate matrices.
    Fold {
        x: ValueId,
        streams: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
        #[out]
        y: ValueId,
    },
    Collapse {
        streams: ValueId,
        head_scale: ValueId,
        head_base: ValueId,
        stream_count: u32,
        gate_eps: f32,
        #[out]
        y: ValueId,
    },
}
