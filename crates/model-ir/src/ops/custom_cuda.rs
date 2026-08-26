use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// The deliberate backend escape hatch: fusions in cuda's vocabulary, refused by
/// every other plane; `CustomMetal`/`CustomVulkan` join only when those planes
/// grow fusions. Pure data like every family — only the dispatch impl is gated.
/// Emitting one is a model-source decision, under the standing rule that each
/// has a canonical unfused equivalent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CustomCuda {
    /// Splits packed qkv, head-norms q and k, ropes them, norms v, and appends
    /// k/v to the cache in one pass; `q` is the only tensor left over.
    /// `positions` feeds the rope math; `write_page`/`write_offset` address
    /// the cache rows the append lands in.
    QkvFusedQknormRopeVnormWrite {
        packed: ValueId,
        positions: ValueId,
        q_norm_weight: ValueId,
        q_norm_eps: f32,
        k_norm_weight: ValueId,
        k_norm_eps: f32,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        q: ValueId,
    },
}

impl Operands for CustomCuda {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                k_norm_weight,
                cache,
                write_page,
                write_offset,
                ..
            } => {
                sink.extend([
                    *packed,
                    *positions,
                    *q_norm_weight,
                    *k_norm_weight,
                    *cache,
                    *write_page,
                    *write_offset,
                ]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::QkvFusedQknormRopeVnormWrite { q, .. } => sink.push(*q),
        }
    }
    fn aliases(&self, _sink: &mut Vec<(ValueId, ValueId)>) {}
    fn name(&self) -> &'static str {
        match self {
            Self::QkvFusedQknormRopeVnormWrite { .. } => {
                "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"
            }
        }
    }
}
