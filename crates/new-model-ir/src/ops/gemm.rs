use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Gemm {
    Matmul {
        act: ValueId,
        w: ValueId,
        #[out]
        y: ValueId,
    },
    LmHead {
        act: ValueId,
        w: ValueId,
        #[out]
        y: ValueId,
    },
    AttentionLanding {
        act: ValueId,
        w: ValueId,
        layer: u32,
        #[out]
        y: ValueId,
    },
}
