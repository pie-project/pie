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
    SplitQGate {
        packed: ValueId,
        head_dim: u32,
        q: ValueId,
        gate: ValueId,
    },
    SplitRows {
        x: ValueId,
        width: u32,
        left: ValueId,
        right: ValueId,
    },
    GatherRows {
        x: ValueId,
        rows: ValueId,
        y: ValueId,
    },
    Select {
        table: ValueId,
        layer: u32,
        width: u32,
        y: ValueId,
    },
    ScatterRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    PoolRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    MergeRows {
        x: ValueId,
        side: u32,
        y: ValueId,
    },
    ScatterLiveRows {
        src: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    EmbedConcat {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        y: ValueId,
    },
    Argmax {
        xs: Vec<ValueId>,
        y: ValueId,
    },
    TopK {
        x: ValueId,
        k: u32,
        values: ValueId,
        indices: ValueId,
    },
    PackRows {
        x: ValueId,
        perm: ValueId,
        y: ValueId,
    },
    UnpackRows {
        x: ValueId,
        perm: ValueId,
        y: ValueId,
    },
}

impl Operands for Layout {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { ids, table, .. } => sink.extend([*ids, *table]),
            Self::EmbedWeighted {
                ids,
                weights,
                table,
                ..
            } => {
                sink.extend([*ids, *weights, *table]);
            }
            Self::SplitQkv { packed, .. } => sink.push(*packed),
            Self::SplitQGate { packed, .. } => sink.push(*packed),
            Self::SplitRows { x, .. } => sink.push(*x),
            Self::GatherRows { x, rows, .. } => sink.extend([*x, *rows]),
            Self::Select { table, .. } => sink.push(*table),
            Self::ScatterRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
            Self::PoolRows { x, .. } => sink.push(*x),
            Self::MergeRows { x, .. } => sink.push(*x),
            Self::ScatterLiveRows { src, routes, y, .. } => sink.extend([*src, *routes, *y]),
            Self::EmbedConcat { ids, table, .. } => sink.extend([*ids, *table]),
            Self::Argmax { xs, .. } => sink.extend(xs.iter().copied()),
            Self::TopK { x, .. } => sink.push(*x),
            Self::PackRows { x, perm, .. } => sink.extend([*x, *perm]),
            Self::UnpackRows { x, perm, .. } => sink.extend([*x, *perm]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { y, .. } => sink.push(*y),
            Self::EmbedWeighted { y, .. } => sink.push(*y),
            Self::SplitQkv { q, k, v, .. } => sink.extend([*q, *k, *v]),
            Self::SplitQGate { q, gate, .. } => sink.extend([*q, *gate]),
            Self::SplitRows { left, right, .. } => sink.extend([*left, *right]),
            Self::GatherRows { y, .. } => sink.push(*y),
            Self::Select { y, .. } => sink.push(*y),
            Self::ScatterRows { y_out, .. } => sink.push(*y_out),
            Self::PoolRows { y, .. } => sink.push(*y),
            Self::MergeRows { y, .. } => sink.push(*y),
            Self::ScatterLiveRows { y_out, .. } => sink.push(*y_out),
            Self::EmbedConcat { y, .. } => sink.push(*y),
            Self::Argmax { y, .. } => sink.push(*y),
            Self::TopK {
                values, indices, ..
            } => sink.extend([*values, *indices]),
            Self::PackRows { y, .. } => sink.push(*y),
            Self::UnpackRows { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Embed { .. }
            | Self::EmbedWeighted { .. }
            | Self::SplitQkv { .. }
            | Self::SplitQGate { .. }
            | Self::SplitRows { .. }
            | Self::GatherRows { .. }
            | Self::Select { .. }
            | Self::PoolRows { .. }
            | Self::MergeRows { .. }
            | Self::EmbedConcat { .. }
            | Self::Argmax { .. }
            | Self::TopK { .. }
            | Self::PackRows { .. }
            | Self::UnpackRows { .. } => {}
            Self::ScatterRows { y_out, y, .. } | Self::ScatterLiveRows { y_out, y, .. } => {
                sink.push((*y_out, *y))
            }
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Embed { .. } => "layout.embed",
            Self::EmbedWeighted { .. } => "layout.embed_weighted",
            Self::SplitQkv { .. } => "layout.split_qkv",
            Self::SplitQGate { .. } => "layout.split_q_gate",
            Self::SplitRows { .. } => "layout.split_rows",
            Self::GatherRows { .. } => "layout.gather_rows",
            Self::Select { .. } => "layout.select",
            Self::ScatterRows { .. } => "layout.scatter_rows",
            Self::PoolRows { .. } => "layout.pool_rows",
            Self::MergeRows { .. } => "layout.merge_rows",
            Self::ScatterLiveRows { .. } => "layout.scatter_live_rows",
            Self::EmbedConcat { .. } => "layout.embed_concat",
            Self::Argmax { .. } => "layout.argmax",
            Self::TopK { .. } => "layout.topk",
            Self::PackRows { .. } => "layout.pack_rows",
            Self::UnpackRows { .. } => "layout.unpack_rows",
        }
    }
}
