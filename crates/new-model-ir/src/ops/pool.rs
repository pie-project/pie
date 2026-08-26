use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// Pooled (compressed) attention: every `ratio` tokens close a boundary whose
/// pooled entry lands in its own cache. Boundary outputs are token-shaped —
/// over-allocated with a sentinel in the non-boundary rows — so no counted
/// dim exists and the shapes stay trace-time facts.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pool {
    BoundaryDecode {
        positions: ValueId,
        ratio: u32,
        #[out]
        boundary_pos: ValueId,
        #[out]
        boundary_req: ValueId,
    },
    BoundaryPrefill {
        positions: ValueId,
        ratio: u32,
        #[out]
        boundary_pos: ValueId,
        #[out]
        boundary_req: ValueId,
    },
    /// Pools the closing window out of the kv cache into per-boundary entries.
    Gather {
        boundary_pos: ValueId,
        boundary_req: ValueId,
        pages: ValueId,
        head_dim: u32,
        ratio: u32,
        #[out]
        entries: ValueId,
    },
    KvAppend {
        entries: ValueId,
        boundary_pos: ValueId,
        boundary_req: ValueId,
        pool: ValueId,
        kv_indices: ValueId,
    },
    AttentionLse {
        q: ValueId,
        positions: ValueId,
        entries: ValueId,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
        #[out]
        lse: ValueId,
    },
}
