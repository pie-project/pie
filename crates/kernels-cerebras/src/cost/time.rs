//! A decode step's time: what a plan's partition costs ([`Features`]) at per-unit rates, plus
//! each layer's own cost (its mixer's fixed phases), in ms.
//!
//! The rates are the chip's. The weight and head elements' are read off the phase stamps of a
//! `TSTAMP:1` run (requests 1179a/c/d): a weight PE's matvec phases (Llama 3.1 8B: 1,524 cycles a
//! layer for 3,328 elements) and the head PE's (0.537 cycles an element, Llama, E4B and Gemma 12B
//! alike). The KV elements', the group hops' and the rows' spans are fitted to what moving the
//! keys did to the bare step (`ARCHITECTURE.md`, "Cost": requests 1151-1195). A layer's own cost is what the
//! chip measured of the model's shipping program past those terms (`MEASURED`); a model the chip
//! has not run takes the median.

use super::Features;
use crate::plan::{Plan, Shape};

/// One cycle at 875 MHz, in ms.
const CYCLE: f64 = 1e3 / 875e6;
/// ms a weight element: a weight PE's stamped matvec phases (1,524 cycles for 3,328 elements).
pub const MATVEC: f64 = 1524.0 / 3328.0 * CYCLE;
/// ms a KV element a row PE reads (fitted: the scores' and the values' passes on the CY column).
pub const KV: f64 = 6.436e-6;
/// ms a reduction hop within a group (fitted).
pub const GROUP_HOP: f64 = 9.613e-6;
/// ms a hop between group roots: held at the six-move fit's, which the G2 sweeps leave
/// unresolved (any rate from 0 to 4e-6 fits them alike); 0 would send the search below the
/// smallest group the chip ran (G2=8).
pub const ROOT_HOP: f64 = 3.179e-6;
/// ms a row a layer's collectives cross down the columns (fitted).
pub const SPAN_ROW: f64 = 1.267e-5;
/// ms a column they cross along the rows: none the chip resolves (fitted to 0; halving the
/// 0.8B's columns cost what its doubled matvec and KV elements do).
pub const SPAN_COL: f64 = 0.0;
/// ms a head element: the head PE's stamped matvec phase (0.537 cycles an element).
pub const HEAD: f64 = 0.537 * CYCLE;

/// The chip's decode ms a step of each shipping model (its shape's signature) under the program
/// its pinned plan builds: the bare step (wafer-bench, runs 1151-1279) where the chip ran it, else
/// Pie decode (runs 1101-1131). A layer's own cost is read off these.
pub const MEASURED: &[([u32; 3], f64)] = &[
    ([4096, 32, 12288], 0.5100), // Qwen3.5 9B
    ([2560, 32, 9216], 0.4800),  // Qwen3.5 4B
    ([1024, 24, 3584], 0.2920),  // Qwen3.5 0.8B
    ([4096, 36, 12288], 0.5505), // Qwen3-8B
    ([4096, 32, 14336], 0.4850), // Llama 3.1 8B
    ([1536, 35, 6144], 0.6830),  // Gemma 4 E2B
    ([2560, 42, 10240], 0.8765), // Gemma 4 E4B
    ([3840, 48, 15360], 0.9495), // Gemma 4 12B
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
pub const MEDIAN_LAYER: f64 = 0.0093;

/// A step of `plan`, in ms, with each layer costing `layer`.
#[must_use]
pub fn step(plan: &Plan, layer: f64) -> f64 {
    let f = Features::of(plan);
    partition(&f) + layer * f.layers
}

#[cfg(test)]
mod tests {
    use crate::target::{Ask, plan};

    /// What moving a pinned plan's keys did to the bare step on the chip (requests 1151-1279),
    /// from the keys the chip ran first, and what the model predicts it does: within 6 µs of each,
    /// and the same way where the chip moved past its noise (5 µs: Llama's VR=216 and VR=180 ran
    /// in different sessions), except the G2 moves the hops' count does not explain, held only in
    /// size (15-18 µs): E4B's G2=12 gains 2.3% where the hops give 0.4%, and G2 below 12 costs the
    /// Qwen3-8B and the 12B 1% where the hops give about nothing (the chip's G2 has an optimum the
    /// hops miss: 12 on the Qwen3.5s, E4B and Qwen3-8B, 16 on the 12B, 8 on E2B).
    #[test]
    fn the_model_follows_the_chip_moves() {
        let l = "llama31-8b-bf16-kv-bf16";
        let q9 = "qwen35-d9b-bf16-kv-bf16";
        let q4 = "qwen35-d4b-bf16-kv-bf16";
        let q8 = "qwen3-8b-bf16-kv-bf16";
        let q08 = "qwen35-d0.8b-bf16-kv-bf16";
        let g12 = "gemma4-12b-bf16-kv-bf16";
        let e4 = "gemma4-e4b-bf16-kv-bf16";
        let e2 = "gemma4-e2b-bf16-kv-bf16";
        let o = "gptoss-20b-bf16-mxfp4-kv-bf16";
        #[rustfmt::skip]
        let moves = [
            (l, "l3-8b", "VR=216", "VR=112", 0.488, 0.485, 6.0),
            (l, "l3-8b", "VR=216", "VR=144", 0.488, 0.4865, 6.0),
            (l, "l3-8b", "VR=216", "VR=180", 0.488, 0.4915, 6.0),
            (l, "l3-8b", "", "G2=8", 0.485, 0.486, 6.0),
            (q08, "0.8b", "G2=16", "PX=128,G2=10,G2Y=10", 0.296, 0.321, 6.0),
            (q08, "0.8b", "G2=16", "G2=8", 0.297, 0.2965, 6.0),
            (q08, "0.8b", "G2=16", "", 0.297, 0.292, 6.0),
            (g12, "g12", "", "G2=15,GA=4,PY=282", 0.950, 0.9505, 6.0),
            (q4, "4b", "PY=256,NE=64,G2=16,G2Y=16", "G2=16", 0.4935, 0.4855, 6.0),
            (q4, "4b", "G2=16", "G2=8", 0.4855, 0.4845, 6.0),
            (q4, "4b", "G2=16", "", 0.4855, 0.480, 6.0),
            (q8, "q3-8b", "", "G2=8", 0.5505, 0.559, 15.0),
            (q8, "q3-8b", "", "G2=16", 0.5505, 0.560, 15.0),
            (g12, "g12", "", "G2=8", 0.9495, 0.960, 15.0),
            (g12, "g12", "", "G2=12", 0.9495, 0.9515, 6.0),
            (o, "full", "", "G2=8", 0.6525, 0.650, 6.0),
            (o, "full", "", "G2=24", 0.6525, 0.6555, 6.0),
            (q9, "9b", "G2=16", "G2=8", 0.5185, 0.5175, 6.0),
            (q9, "9b", "G2=16", "G2=12", 0.5185, 0.510, 6.5),
            (q9, "9b", "G2=16", "G2=24", 0.5185, 0.5225, 6.0),
            (e4, "e4b", "G2=16", "G2=8", 0.8975, 0.890, 6.0),
            (e4, "e4b", "G2=16", "G2=12", 0.8975, 0.8765, 18.0),
            (e4, "e4b", "G2=16", "G2=24", 0.8975, 0.907, 6.0),
            (e2, "e2b", "G2=16", "G2=8", 0.6915, 0.683, 6.0),
            (e2, "e2b", "G2=16", "G2=24", 0.6915, 0.695, 6.0),
        ];
        for (model, preset, from_set, to_set, from, to, tol_us) in moves {
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
            let (want, got) = (to - from, step(to_set) - step(from_set));
            assert!(
                (got - want).abs() < tol_us * 1e-3,
                "{model} {to_set}: {got} ms vs {want} ms"
            );
            assert!(
                want.abs() < 5e-3 || tol_us > 6.5 || got.signum() == want.signum(),
                "{model} {to_set}: {got} ms the other way from {want} ms"
            );
        }
    }
}
