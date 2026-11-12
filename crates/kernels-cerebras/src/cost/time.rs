//! A decode step's time: what a plan's partition costs ([`Features`]) at per-unit rates, plus
//! each layer's own cost (its mixer's fixed phases), in ms.
//!
//! The rates are the chip's. The weight and head elements' are read off the phase stamps of a
//! `TSTAMP:1` run (requests 1179a/c/d): a weight PE's matvec phases (Llama 3.1 8B: 1,524 cycles a
//! layer for 3,328 elements) and the head PE's (0.537 cycles an element, Llama, E4B and Gemma 12B
//! alike). The KV elements', the hops' and the spans' are fitted to what moving the keys did to
//! the bare step (`ARCHITECTURE.md`, "Cost": requests 1151-1177c). A layer's own cost is what the
//! chip measured of the model's shipping program past those terms (`MEASURED`); a model the chip
//! has not run takes the median.

use super::Features;
use crate::plan::{Plan, Shape};

/// One cycle at 875 MHz, in ms.
const CYCLE: f64 = 1e3 / 875e6;
/// ms a weight element: a weight PE's stamped matvec phases (1,524 cycles for 3,328 elements).
pub const MATVEC: f64 = 1524.0 / 3328.0 * CYCLE;
/// ms a KV element a row PE reads (fitted: the scores' and the values' passes on the CY column).
pub const KV: f64 = 6.145e-6;
/// ms a reduction hop within a group (fitted).
pub const GROUP_HOP: f64 = 4.897e-6;
/// ms a hop between group roots (fitted).
pub const ROOT_HOP: f64 = 3.179e-6;
/// ms a row a layer's collectives cross down the columns (fitted).
pub const SPAN_ROW: f64 = 1.307e-5;
/// ms a column they cross along the rows: none the chip resolves (fitted to 0; halving the
/// 0.8B's columns cost what its doubled matvec and KV elements do).
pub const SPAN_COL: f64 = 0.0;
/// ms a head element: the head PE's stamped matvec phase (0.537 cycles an element).
pub const HEAD: f64 = 0.537 * CYCLE;

/// The chip's decode ms a step of each shipping model (its shape's signature) under the program
/// its preset builds: the bare step (wafer-bench, runs 1151-1175) where the chip ran it, else Pie
/// decode (runs 1101-1131). A layer's own cost is read off these.
pub const MEASURED: &[([u32; 3], f64)] = &[
    ([4096, 32, 12288], 0.5238), // Qwen3.5 9B (Pie)
    ([2560, 32, 9216], 0.4935),  // Qwen3.5 4B
    ([1024, 24, 3584], 0.2960),  // Qwen3.5 0.8B
    ([4096, 36, 12288], 0.5551), // Qwen3-8B (Pie)
    ([4096, 32, 14336], 0.4880), // Llama 3.1 8B
    ([1536, 35, 6144], 0.6961),  // Gemma 4 E2B (Pie)
    ([2560, 42, 10240], 0.8950), // Gemma 4 E4B (Pie)
    ([3840, 48, 15360], 0.9500), // Gemma 4 12B
    ([2880, 24, 2880], 0.6525),  // gpt-oss-20b
];

/// The partition's part of a step, in ms.
#[must_use]
pub fn partition(f: &Features) -> f64 {
    MATVEC * f.matvec
        + KV * f.kv
        + GROUP_HOP * f.group_hops
        + ROOT_HOP * f.root_hops
        + SPAN_ROW * f.span_rows
        + SPAN_COL * f.span_cols
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
pub const MEDIAN_LAYER: f64 = 0.0098;

/// A step of `plan`, in ms, with each layer costing `layer`.
#[must_use]
pub fn step(plan: &Plan, layer: f64) -> f64 {
    let f = Features::of(plan);
    partition(&f) + layer * f.layers
}

#[cfg(test)]
mod tests {
    use crate::target::{Ask, plan};

    /// What moving a preset's keys did to the bare step on the chip (requests 1151-1177c), and
    /// what the model predicts it does: within 3.5 µs of each (the stamps' and the fit's error).
    #[test]
    fn the_model_follows_the_chip_moves() {
        let moves = [
            ("llama31-8b-bf16-kv-bf16", "l3-8b", "VR=112", 0.488, 0.485),
            (
                "qwen35-d0.8b-bf16-kv-bf16",
                "0.8b",
                "PX=128,G2=10,G2Y=10",
                0.296,
                0.321,
            ),
            (
                "gemma4-12b-bf16-kv-bf16",
                "g12",
                "G2=15,GA=4,PY=282",
                0.950,
                0.9505,
            ),
            (
                "qwen35-d4b-bf16-kv-bf16",
                "4b",
                "PY=192,NE=86,G2Y=12",
                0.4935,
                0.4845,
            ),
            (
                "gptoss-20b-bf16-mxfp4-kv-bf16",
                "full",
                "G2=8",
                0.6525,
                0.650,
            ),
            (
                "gptoss-20b-bf16-mxfp4-kv-bf16",
                "full",
                "G2=24",
                0.6525,
                0.6555,
            ),
        ];
        for (model, preset, set, from, to) in moves {
            let t = poem_compiler::catalog::repository()
                .parse(model)
                .unwrap()
                .trace(poem_ir::Platform::Cerebras);
            let step = |set: &str| {
                let ask = Ask {
                    preset: Some(preset.into()),
                    set: set.into(),
                    ..Ask::default()
                };
                super::step(&plan(&t, &ask).unwrap().plan, 0.0)
            };
            let (want, got) = (to - from, step(set) - step(""));
            assert!(
                (got - want).abs() < 3.5e-3,
                "{model} {set}: {got} ms vs {want} ms"
            );
        }
    }
}
