//! A decode step's time in ms: the plan's partition ([`Features`]) at per-unit rates, plus each
//! layer's own cost. The rates are the chip's ([`crate::tuning`]): weight and head elements from
//! `TSTAMP:1` phase stamps, the reductions' chains by their critical path
//! ([`super::chain_cycles`]) at a fitted rate, KV elements and row span fitted. A layer's own cost
//! is its kinds'.

use super::Features;
use crate::compile::plan::{Plan, Shape};
use crate::tuning;

/// One PE cycle ([`super::CLOCK_HZ`]), in ms.
const CYCLE: f64 = 1e3 / super::CLOCK_HZ;

/// The partition's part of a step, in ms.
#[must_use]
pub fn partition(f: &Features) -> f64 {
    let t = tuning::get();
    t.decode_matvec * CYCLE * f.matvec
        + t.decode_kv * f.kv
        + t.decode_chain * f.chain
        + t.decode_span_row * f.span_rows
        + t.decode_span_col * f.span_cols
        + t.decode_head * CYCLE * f.head
        + t.decode_loop_reduce * CYCLE * f.loop_reduce
}

/// A layer's own cost by its mixer's kind (`layer_*`), over `shape`'s layers, a layer's mean.
#[must_use]
pub fn kinds_layer(shape: &Shape) -> f64 {
    use crate::compile::plan::LayerKind;
    let t = tuning::get();
    let n = shape.layers.len().max(1) as f64;
    shape
        .layers
        .iter()
        .map(|l| match l {
            LayerKind::Gdn => t.layer_gdn,
            LayerKind::Global => t.layer_global,
            LayerKind::Sliding { .. } => t.layer_sliding,
            LayerKind::SharedKv { .. } => t.layer_shared_kv,
        })
        .sum::<f64>()
        / n
}

/// A step of `plan`, in ms, with each layer costing `layer`.
#[must_use]
pub fn step(plan: &Plan, layer: f64) -> f64 {
    let f = Features::of(plan);
    partition(&f) + layer * f.layers
}
