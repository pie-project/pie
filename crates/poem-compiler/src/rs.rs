//! Which causal convs the rs buffer covers: the one statement the chunk
//! plan's rs rows and every backend's rs layout read.
//!
//! A buffered fire (`RsVerb::Buffer`) leaves a recurrent state as it was and
//! lands the rows it would have folded in the lane's rs buffer; a later fold
//! replays them. Every state a fire advances must take part, or a partial
//! fold (a speculative verify keeping some of its drafts, a rewind) leaves it
//! advanced past rows the fold never took. Qwen4's PLE conv was left out
//! because the rule read "a conv that feeds a recurrence": it feeds a
//! residual, so its state advanced on every buffered fire.

use std::collections::HashMap;

use poem_ir::{Def, Operation, Ssm, Trace, ValueId};

/// The bytes a token of a gated-delta row's scalars plane takes
/// (`ChunkProfile::rs_corrected`): two f32 runs of `v_heads`, each value
/// head's log-gate `g = -exp(A_log) * softplus(a + dt_bias)`, then its key
/// norm's inverse (its key head's, repeated over the value heads that share
/// it), as the writing engine's scan norms a key: an engine reads only the
/// rows it wrote. The step that lands the row writes them
/// (it holds both); a reader of a buffered row takes them instead of
/// recomputing them from the row's `[b | a]` and keys. A backend whose
/// kernels recompute them leaves the plane unwritten and unread: a fork
/// copies it with the rest, and nothing reads what it holds.
#[must_use]
pub const fn gdn_scalars_bytes(v_heads: u32) -> u64 {
    2 * v_heads as u64 * 4
}

/// What a buffered conv's output feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feeds {
    /// A gated-delta scan's `qkv`.
    GatedDelta,
    /// A KDA scan's `mixed`.
    Kda,
    /// A Mamba-2 scan's `xbc`, buffered only where the backend replays
    /// Mamba-2 (`rs_mamba2`).
    Mamba2,
    /// No recurrence: its own state is the only one it advances (Qwen4's
    /// PLE conv).
    Alone,
}

/// A causal conv (`Ssm::CausalConv1d` or `CausalConv1dChunked`, any
/// dilation) whose state the rs buffer defers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferedConv {
    /// Its node's index in the trace.
    pub node: usize,
    /// The conv's input: the rows the rs buffer holds and a fold replays.
    pub x: ValueId,
    /// Its state, a cache row.
    pub state: ValueId,
    /// Its output.
    pub y: ValueId,
    /// The prompt form (`CausalConv1dChunked`), whose `x` the rs rows size.
    pub chunked: bool,
    pub feeds: Feeds,
}

/// The causal convs of `trace` the rs buffer covers, in node order: every
/// one whose state is a cache row, except one that feeds a recurrence
/// committing as it runs (a Mamba-2 scan where the backend does not replay
/// Mamba-2, `rs_mamba2` false), whose conv then commits with it.
pub fn buffered_convs(trace: &Trace, rs_mamba2: bool) -> impl Iterator<Item = BufferedConv> + '_ {
    let mut feeds: HashMap<u32, Feeds> = HashMap::new();
    for node in &trace.nodes {
        if let Operation::Ssm(op) = &node.op {
            match op {
                Ssm::GatedDelta { qkv, .. } | Ssm::GatedDeltaChunked { qkv, .. } => {
                    feeds.insert(qkv.0, Feeds::GatedDelta);
                }
                Ssm::Kda { mixed, .. } | Ssm::KdaChunked { mixed, .. } => {
                    feeds.insert(mixed.0, Feeds::Kda);
                }
                Ssm::Mamba2 { xbc, .. } | Ssm::Mamba2Chunked { xbc, .. } => {
                    feeds.insert(xbc.0, Feeds::Mamba2);
                }
                _ => {}
            }
        }
    }
    let cached = |id: ValueId| {
        matches!(
            trace.values.get(id.0 as usize).map(|value| &value.def),
            Some(Def::Cache(_))
        )
    };
    trace.nodes.iter().enumerate().filter_map(move |(node, n)| {
        let (x, state, y, chunked) = match &n.op {
            Operation::Ssm(Ssm::CausalConv1d { x, state, y, .. }) => (*x, *state, *y, false),
            Operation::Ssm(Ssm::CausalConv1dChunked { x, state, y, .. }) => (*x, *state, *y, true),
            _ => return None,
        };
        if !cached(state) {
            return None;
        }
        let feeds = feeds.get(&y.0).copied().unwrap_or(Feeds::Alone);
        if feeds == Feeds::Mamba2 && !rs_mamba2 {
            return None;
        }
        Some(BufferedConv {
            node,
            x,
            state,
            y,
            chunked,
            feeds,
        })
    })
}
