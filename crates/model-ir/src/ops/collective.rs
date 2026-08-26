use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Crosses devices — the SPMD tensor-parallel collectives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Collective {
    AllReduce {
        buf: ValueId,
        buf_out: ValueId,
    },
    /// Concatenates each rank's shard into the full tensor on every rank.
    AllGather {
        x: ValueId,
        y: ValueId,
    },
    /// Sums across ranks, leaving each rank its shard of the result.
    ReduceScatter {
        x: ValueId,
        y: ValueId,
    },
}

impl Operands for Collective {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::AllReduce { buf, .. } => sink.push(*buf),
            Self::AllGather { x, .. } => sink.push(*x),
            Self::ReduceScatter { x, .. } => sink.push(*x),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::AllReduce { buf_out, .. } => sink.push(*buf_out),
            Self::AllGather { y, .. } => sink.push(*y),
            Self::ReduceScatter { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::AllReduce { buf_out, buf } => sink.push((*buf_out, *buf)),
            Self::AllGather { .. } => {}
            Self::ReduceScatter { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::AllReduce { .. } => "collective.all_reduce",
            Self::AllGather { .. } => "collective.all_gather",
            Self::ReduceScatter { .. } => "collective.reduce_scatter",
        }
    }
}
