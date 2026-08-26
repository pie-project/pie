use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// CUDA-only fused points (the old tier-2 set). Pure data like every family —
/// only the dispatch impl is gated. Emitting one is a model-source decision,
/// under the standing rule that each has a canonical unfused equivalent.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cuda {
    /// Splits packed qkv, head-norms q and k, ropes them, norms v, and appends
    /// k/v to the cache in one pass; `q` is the only tensor left over.
    /// `positions` doubles as the rope input and the write geometry.
    QkvFusedQknormRopeVnormWrite {
        packed: ValueId,
        positions: ValueId,
        q_norm_weight: ValueId,
        q_norm_eps: f32,
        k_norm_weight: ValueId,
        k_norm_eps: f32,
        cache: ValueId,
        kv_indices: ValueId,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        #[out]
        q: ValueId,
    },
}
