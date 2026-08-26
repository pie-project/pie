use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Gate {
    SigmoidMul {
        x: ValueId,
        gate: ValueId,
        #[out(alias = x)]
        x_out: ValueId,
    },
}
