use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Layout {
    Embed {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        #[out]
        y: ValueId,
    },
    SplitQkv {
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
        #[out]
        q: ValueId,
        #[out]
        k: ValueId,
        #[out]
        v: ValueId,
    },
    /// Deinterleaves per-head (q, gate) pairs from the packed projection.
    SplitQGate {
        packed: ValueId,
        head_dim: u32,
        #[out]
        q: ValueId,
        #[out]
        gate: ValueId,
    },
    /// Splits each row at column `width`.
    SplitRows {
        x: ValueId,
        width: u32,
        #[out]
        left: ValueId,
        #[out]
        right: ValueId,
    },
    /// Views layer `layer`'s `width`-wide slice of a stacked table.
    Select {
        table: ValueId,
        layer: u32,
        width: u32,
        #[out]
        y: ValueId,
    },
}
