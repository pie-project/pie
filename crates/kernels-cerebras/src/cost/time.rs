//! A decode step's time: what a plan's partition costs ([`Features`]) at per-unit rates, plus
//! each layer's own cost (its mixer's fixed phases), in ms.
//!
//! The rates: a weight element at the PE's f16 FMAC rate (4 a cycle at 875 MHz) and a KV element
//! the same; the reduction hops, the collectives' span and the head's elements fitted to the
//! chip (`fit`, below). A layer's own cost is what the chip measured of the model's shipping
//! program past those terms (`MEASURED`); a model the chip has not run takes the median.

use super::Features;
use crate::plan::{Plan, Shape};

/// ms a weight element: 4 f16 FMACs a cycle at 875 MHz.
pub const MATVEC: f64 = 1e3 / (4.0 * 875e6);
/// ms a KV element read (the scores' and the values' FMACs).
pub const KV: f64 = MATVEC;
/// ms a reduction hop within a group (fitted).
pub const GROUP_HOP: f64 = 1.069e-5;
/// ms a hop between group roots (fitted).
pub const ROOT_HOP: f64 = 8.345e-6;
/// ms a PE of distance a collective crosses (fitted).
pub const SPAN: f64 = 1.444e-5;
/// ms a head element (fitted: the head's f32-accumulated tiles).
pub const HEAD: f64 = 1.368e-5;

/// The chip's decode ms a step of each shipping model (its shape's signature) under the program
/// its preset builds: Pie decode, runs 1101-1131. A layer's own cost is read off these.
pub const MEASURED: &[([u32; 3], f64)] = &[
    ([4096, 32, 12288], 0.5238), // Qwen3.5 9B
    ([2560, 32, 9216], 0.4650),  // Qwen3.5 4B
    ([1024, 24, 3584], 0.3040),  // Qwen3.5 0.8B
    ([4096, 36, 12288], 0.5551), // Qwen3-8B
    ([4096, 32, 14336], 0.4964), // Llama 3.1 8B
    ([1536, 35, 6144], 0.6961),  // Gemma 4 E2B
    ([2560, 42, 10240], 0.8950), // Gemma 4 E4B
    ([3840, 48, 15360], 0.9527), // Gemma 4 12B
    ([2880, 24, 2880], 0.6534),  // gpt-oss-20b
];

/// The partition's part of a step, in ms.
#[must_use]
pub fn partition(f: &Features) -> f64 {
    MATVEC * f.matvec
        + KV * f.kv
        + GROUP_HOP * f.group_hops
        + ROOT_HOP * f.root_hops
        + SPAN * f.span
        + HEAD * f.head
}

/// A layer's own cost, in ms: read off the measured step of the model's shipping `reference` plan,
/// else the median of the measured models'.
#[must_use]
pub fn layer(shape: &Shape, reference: Option<&Plan>) -> f64 {
    let sig = shape.signature();
    match (MEASURED.iter().find(|(s, _)| *s == sig), reference) {
        (Some((_, ms)), Some(r)) => {
            let f = Features::of(r);
            (ms - partition(&f)) / f.layers
        }
        _ => MEDIAN_LAYER,
    }
}

/// The median layer cost of the measured models (their `layer`s), ms.
pub const MEDIAN_LAYER: f64 = 0.0045;

/// A step of `plan`, in ms, with each layer costing `layer`.
#[must_use]
pub fn step(plan: &Plan, layer: f64) -> f64 {
    let f = Features::of(plan);
    partition(&f) + layer * f.layers
}
