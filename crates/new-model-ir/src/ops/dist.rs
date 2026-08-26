use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Dist {
    AllReduce {
        buf: ValueId,
        #[out(alias = buf)]
        buf_out: ValueId,
    },
    /// Concatenates each rank's shard into the full tensor on every rank.
    AllGather {
        x: ValueId,
        #[out]
        y: ValueId,
    },
    /// Sums across ranks, leaving each rank its shard of the result.
    ReduceScatter {
        x: ValueId,
        #[out]
        y: ValueId,
    },
}
