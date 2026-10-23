use serde::{Deserialize, Serialize};

use crate::Op;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "collective")]
pub enum Collective {
    AllReduce {
        #[input]
        buf: ValueId,
        #[output(alias = buf)]
        buf_out: ValueId,
    },
    AllGather {
        #[input]
        x: ValueId,
        #[output]
        y: ValueId,
    },
    ReduceScatter {
        #[input]
        x: ValueId,
        #[output]
        y: ValueId,
    },
}
