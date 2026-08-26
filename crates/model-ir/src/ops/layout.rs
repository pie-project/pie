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
}

impl Operands for Layout {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { ids, table, .. } => sink.extend([*ids, *table]),
            Self::SplitQkv { packed, .. } => sink.push(*packed),
            Self::SplitQGate { packed, .. } => sink.push(*packed),
            Self::SplitRows { x, .. } => sink.push(*x),
            Self::Select { table, .. } => sink.push(*table),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Embed { y, .. } => sink.push(*y),
            Self::SplitQkv { q, k, v, .. } => sink.extend([*q, *k, *v]),
            Self::SplitQGate { q, gate, .. } => sink.extend([*q, *gate]),
            Self::SplitRows { left, right, .. } => sink.extend([*left, *right]),
            Self::Select { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, _sink: &mut Vec<(ValueId, ValueId)>) {}
    fn name(&self) -> &'static str {
        match self {
            Self::Embed { .. } => "layout.embed",
            Self::SplitQkv { .. } => "layout.split_qkv",
            Self::SplitQGate { .. } => "layout.split_q_gate",
            Self::SplitRows { .. } => "layout.split_rows",
            Self::Select { .. } => "layout.select",
        }
    }
}
