use serde::{Deserialize, Serialize};

use crate::Op;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "sampling")]
pub enum Sampling {
    /// Each row's argmax over the columns of `xs` laid end to end.
    Argmax {
        #[input]
        xs: Vec<ValueId>,
        #[output]
        y: ValueId,
    },
    /// Each row's `k` largest values and their columns.
    TopK {
        #[input]
        x: ValueId,
        k: u32,
        #[output(F32)]
        values: ValueId,
        #[output(I32)]
        indices: ValueId,
    },
    /// DSpark's Markov head over a drafted block's readouts: the one at depth
    /// `i` (its position past `ends - block + 1`, `ends` its lane's last key)
    /// adds the bias `head . embed[t]` of the token `t` picked at depth `i - 1`
    /// (at depth 0, its own in `tokens`) and picks the argmax. `biased` holds
    /// the biased logits.
    MarkovWalk {
        #[input]
        logits: ValueId,
        #[input(I32)]
        tokens: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input(I32)]
        ends: ValueId,
        #[input]
        embed: ValueId,
        #[input]
        head: ValueId,
        /// The most readouts a lane's run holds (the drafted block's rows).
        block: u32,
        #[output]
        biased: ValueId,
        /// Each readout's pick, its biased row's argmax.
        #[output(I32)]
        picks: ValueId,
    },
    SelectorWalk {
        #[input(I32)]
        cand: ValueId,
        #[input(F32)]
        unary: ValueId,
        #[input]
        hp: Option<ValueId>,
        #[input]
        tokens: ValueId,
        #[input]
        pred: ValueId,
        #[input]
        succ: ValueId,
        first: u32,
        #[output(I32)]
        picks: ValueId,
    },
}
