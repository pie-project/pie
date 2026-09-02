use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Layout {
    Embed {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
    /// The gather that interpolates: `y[r] = Σₜ weights[r, t] ·
    /// table[ids[r, t]]`. `ids` and `weights` are both `[rows, taps]`; `taps`
    /// is read off their width rather than stated separately.
    EmbedWeighted {
        ids: ValueId,
        weights: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
    SplitQkv {
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
        q: ValueId,
        k: ValueId,
        v: ValueId,
    },
    /// Deinterleaves per-head (q, gate) pairs from the packed projection.
    SplitQGate {
        packed: ValueId,
        head_dim: u32,
        q: ValueId,
        gate: ValueId,
    },
    /// Splits each row at column `width`.
    SplitRows {
        x: ValueId,
        width: u32,
        left: ValueId,
        right: ValueId,
    },
    /// Views layer `layer`'s `width`-wide slice of a stacked table.
    Select {
        table: ValueId,
        layer: u32,
        width: u32,
        y: ValueId,
    },
    /// The embed merge: the tower's rows written into the token rows the
    /// image placeholders occupy. `src` is a patch rectangle, `y` a token
    /// one, and `routes` (`i32`, one entry per `src` row) says which token
    /// row each tower row lands in. `y_out` aliases `y` in place: unrouted
    /// rows must keep the embedding `layout.embed` already wrote, not a
    /// fresh rectangle's leftovers.
    ScatterRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    /// The spatial pool: `y[j]` is the mean of rows `[j·side², (j+1)·side²)`
    /// of `x`. `side` (not `side²`) because it is the checkpoint's own
    /// number; `side == 1` is a real identity case.
    ///
    /// # The operand contract
    ///
    /// `x` and `y` are both `[Dim::Patches, hidden]`, same symbolic row
    /// space; only the leading `x.rows / side²` rows of `y` are written, the
    /// rest is the rung padding already there. No geometry is read: the
    /// submission must already order each image's patches so that every
    /// `side × side` square is contiguous, and images are padded to whole
    /// blocks so no block straddles two.
    PoolRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    /// The merging fold: `y[j]` is rows `[j·side², (j+1)·side²)` of `x` laid
    /// end to end — `side²` rows of `width` becoming one row of
    /// `side²·width`. On a dense row-major rectangle this is the identity
    /// copy (`[rows, width]` and `[rows/side², side²·width]` hold the same
    /// element at the same offset); it is a node because the IR gives one
    /// value one type.
    ///
    /// # The operand contract
    ///
    /// `x` is `[Dim::Patches, width]`, `y` is `[Dim::Patches, side²·width]`.
    /// Same row space; leading `x.rows / side²` rows written, tail handled
    /// by [`ScatterLiveRows`](Layout::ScatterLiveRows).
    MergeRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    /// [`ScatterRows`](Layout::ScatterRows) exactly, plus a negative
    /// `routes` entry meaning "this row has no destination" (the compacting
    /// fold leaves some patch rows with no legal route). A separate op from
    /// the plain scatter, whose contract — every route names a row — some
    /// consumers rely on. `y_out` aliases `y`.
    ScatterLiveRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    /// The gather that concatenates: `y[r] = table[ids[r, 0]] ‖ … ‖
    /// table[ids[r, heads−1]]` — one row assembled from `heads` table rows,
    /// each landing in its own `width`-wide slice of the output. Unlike
    /// [`Embed`](Layout::Embed) (one row per row) or
    /// [`EmbedWeighted`](Layout::EmbedWeighted) (sums its taps), this keeps
    /// every tap side by side. `heads` is read off `ids`' width.
    EmbedConcat {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
}

impl Operands for Layout {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { ids, table, .. } => sink.extend([*ids, *table]),
            Self::EmbedWeighted { ids, weights, table, .. } => {
                sink.extend([*ids, *weights, *table]);
            }
            Self::SplitQkv { packed, .. } => sink.push(*packed),
            Self::SplitQGate { packed, .. } => sink.push(*packed),
            Self::SplitRows { x, .. } => sink.push(*x),
            Self::Select { table, .. } => sink.push(*table),
            Self::ScatterRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
            Self::PoolRows { x, .. } => sink.push(*x),
            Self::MergeRows { x, .. } => sink.push(*x),
            Self::ScatterLiveRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
            Self::EmbedConcat { ids, table, .. } => sink.extend([*ids, *table]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { y, .. } => sink.push(*y),
            Self::EmbedWeighted { y, .. } => sink.push(*y),
            Self::SplitQkv { q, k, v, .. } => sink.extend([*q, *k, *v]),
            Self::SplitQGate { q, gate, .. } => sink.extend([*q, *gate]),
            Self::SplitRows { left, right, .. } => sink.extend([*left, *right]),
            Self::Select { y, .. } => sink.push(*y),
            Self::ScatterRows { y_out, .. } => sink.push(*y_out),
            Self::PoolRows { y, .. } => sink.push(*y),
            Self::MergeRows { y, .. } => sink.push(*y),
            Self::ScatterLiveRows { y_out, .. } => sink.push(*y_out),
            Self::EmbedConcat { y, .. } => sink.push(*y),
        }
    }
    /// The one aliasing row this family has, and it is the scatter's: every
    /// other `Layout` variant either reads a table or cuts a packed row into
    /// pieces that are views of it, and neither is an in-place edit the
    /// compiler has to fold into one slot.
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Embed { .. }
            | Self::EmbedWeighted { .. }
            | Self::SplitQkv { .. }
            | Self::SplitQGate { .. }
            | Self::SplitRows { .. }
            | Self::Select { .. }
            // The pool writes a fresh rectangle, not `x` narrowed in place.
            | Self::PoolRows { .. }
            | Self::MergeRows { .. }
            | Self::EmbedConcat { .. } => {}
            Self::ScatterRows { y_out, y, .. }
            | Self::ScatterLiveRows { y_out, y, .. } => sink.push((*y_out, *y)),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Embed { .. } => "layout.embed",
            Self::EmbedWeighted { .. } => "layout.embed_weighted",
            Self::SplitQkv { .. } => "layout.split_qkv",
            Self::SplitQGate { .. } => "layout.split_q_gate",
            Self::SplitRows { .. } => "layout.split_rows",
            Self::Select { .. } => "layout.select",
            Self::ScatterRows { .. } => "layout.scatter_rows",
            Self::PoolRows { .. } => "layout.pool_rows",
            Self::MergeRows { .. } => "layout.merge_rows",
            Self::ScatterLiveRows { .. } => "layout.scatter_live_rows",
            Self::EmbedConcat { .. } => "layout.embed_concat",
        }
    }
}
