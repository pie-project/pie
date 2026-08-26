use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// The sparse-attention indexer: a small key cache (`keys`) scored against
/// queries to select which pages the main attention will read.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Index {
    LayernormRope {
        k: ValueId,
        positions: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        rope_dim: u32,
        theta: f32,
        #[out(alias = k)]
        k_out: ValueId,
    },
    Rope {
        q: ValueId,
        positions: ValueId,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
        #[out(alias = q)]
        q_out: ValueId,
    },
    /// Scores `q` against the cached keys; `selection` is the top-k page ids.
    Topk {
        q: ValueId,
        weights: ValueId,
        keys: ValueId,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        #[out]
        selection: ValueId,
    },
    KvAppend {
        k: ValueId,
        keys: ValueId,
        kv_indices: ValueId,
        positions: ValueId,
    },
}
