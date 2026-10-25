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
    /// A rank's tokens sent to the ranks that hold their routed experts,
    /// across an expert-parallel group of `ep` ranks: each token once to
    /// each rank holding one of its `top_k` routes. `routes` are logical
    /// experts (of `experts`); the engine attaches the placement of MoE layer
    /// `layer` (logical expert → physical slots, `slots` on each rank, a
    /// replica picked per route as a pure function of the source rank, token
    /// and route) from its expert tier. This rank receives `rows`
    /// (`ep · tokens` rows, ordered by source rank and token, padding past
    /// the received ones), and in `local` the local slot each of a row's
    /// `top_k` routes takes here, -1 for a route another rank serves and on
    /// padding; `row_weights` carries `weights` in the same layout, where
    /// the experts read them. `origin` is each row's source rank times
    /// `tokens` plus its token, -1 on padding; `counts` the routes each local
    /// slot received (`slots` rows), the expert tier's load. Only the
    /// number of live received rows varies at run time.
    Dispatch {
        #[input]
        x: ValueId,
        #[input(I32)]
        routes: ValueId,
        #[input(F32)]
        weights: Option<ValueId>,
        ep: u32,
        experts: u32,
        slots: u32,
        top_k: u32,
        layer: u32,
        #[output]
        rows: ValueId,
        #[output(I32)]
        local: ValueId,
        #[output(F32)]
        row_weights: Option<ValueId>,
        #[output(I32)]
        origin: ValueId,
        #[output(I32)]
        counts: ValueId,
    },
    /// The expert outputs `y` (`top_k` rows per received row, a route's row
    /// read only where its `local` is a slot and its `origin` a token) sent
    /// back to the ranks they came from: `routed` holds each of this rank's
    /// tokens' `top_k` rows in its routes' order, unweighted, as a routed
    /// select over every expert on one rank returns them.
    Combine {
        #[input]
        y: ValueId,
        #[input(I32)]
        origin: ValueId,
        #[input(I32)]
        local: ValueId,
        ep: u32,
        top_k: u32,
        #[output]
        routed: ValueId,
    },
}
