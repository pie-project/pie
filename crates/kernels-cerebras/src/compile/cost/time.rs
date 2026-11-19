//! A decode step's time in ms: the plan's partition ([`Features`]) at per-unit rates, plus each
//! layer's own cost. The rates are the chip's: weight and head elements from `TSTAMP:1` phase
//! stamps, the reductions' chains by their critical path ([`super::chain_cycles`]) at a fitted
//! rate, KV elements and row span fitted. A layer's own cost is what the chip measured past those
//! terms ([`super::measured`]), else its kinds' ([`KIND_LAYER`]).

use super::Features;
use crate::compile::plan::{Plan, Shape};

/// One PE cycle ([`super::CLOCK_HZ`]), in ms.
const CYCLE: f64 = 1e3 / super::CLOCK_HZ;
/// ms a weight element: a weight PE's stamped matvec phases.
pub const MATVEC: f64 = 1524.0 / 3328.0 * CYCLE;
/// ms a KV element a row PE reads (fitted: the scores' and the values' passes on the CY column).
pub const KV: f64 = 6.436e-6;
/// ms a cycle of the reductions' chains ([`Features::chain`], fitted with KV and the span held):
/// below a PE cycle, as the simulator's hop cycles overprice the chip's chains.
pub const CHAIN: f64 = 1.093e-6;
/// ms a row a layer's collectives cross down the columns (fitted).
pub const SPAN_ROW: f64 = 1.267e-5;
/// ms a column they cross along the rows: fitted to 0 (none the chip resolves).
pub const SPAN_COL: f64 = 0.0;
/// ms a head element: the head PE's stamped matvec phase (0.537 cycles an element).
pub const HEAD: f64 = 0.537 * CYCLE;
/// ms a cycle of the token loop's max-reduce ([`Features::loop_reduce`]): the chip's cycle.
pub const LOOP_REDUCE: f64 = CYCLE;

/// The partition's part of a step, in ms.
#[must_use]
pub fn partition(f: &Features) -> f64 {
    MATVEC * f.matvec
        + KV * f.kv
        + CHAIN * f.chain
        + SPAN_ROW * f.span_rows
        + SPAN_COL * f.span_cols
        + HEAD * f.head
        + LOOP_REDUCE * f.loop_reduce
}

/// A layer's own cost, in ms: read off the measured step of the model's measured `reference`
/// plan, else its layers' kinds' ([`kinds_layer`]).
#[must_use]
pub fn layer(shape: &Shape, reference: Option<&Plan>) -> f64 {
    let sig = shape.signature();
    match (super::measured::on(sig).map(|m| m.ms), reference) {
        (Some(ms), Some(r)) => {
            let f = Features::of(r);
            (ms - partition(&f)) / f.layers
        }
        _ => kinds_layer(shape),
    }
}

/// A layer's own cost by its mixer's kind (ms: gated delta net, full attention, sliding-window
/// attention, attention over another layer's KV), least squares over the chip's measured programs
/// (`the_layer_kinds_follow_the_measurements` refits it). It leaves out per-model phases, so a
/// model the chip ran takes its own ([`layer`]); no partition choice reads it.
pub const KIND_LAYER: [f64; 4] = [8.018e-3, 5.707e-3, 10.653e-3, 19.700e-3];

/// [`KIND_LAYER`] over `shape`'s layers, a layer's mean.
#[must_use]
pub fn kinds_layer(shape: &Shape) -> f64 {
    use crate::compile::plan::LayerKind;
    let n = shape.layers.len().max(1) as f64;
    shape
        .layers
        .iter()
        .map(|l| match l {
            LayerKind::Gdn => KIND_LAYER[0],
            LayerKind::Global => KIND_LAYER[1],
            LayerKind::Sliding { .. } => KIND_LAYER[2],
            LayerKind::SharedKv { .. } => KIND_LAYER[3],
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
