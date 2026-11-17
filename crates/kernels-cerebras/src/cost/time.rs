//! A decode step's time: what a plan's partition costs ([`Features`]) at per-unit rates, plus
//! each layer's own cost (its mixer's fixed phases), in ms.
//!
//! The rates are the chip's. The weight and head elements' are read off the phase stamps of a
//! `TSTAMP:1` run (requests 1179a/c/d): a weight PE's matvec phases (Llama 3.1 8B: 1,524 cycles a
//! layer for 3,328 elements) and the head PE's (0.537 cycles an element, Llama, E4B and Gemma 12B
//! alike). The reductions' chains are timed by their critical path ([`super::chain_cycles`], whose
//! per-hop cycles are the fabric simulator's), at a rate a cycle fitted to what moving the keys
//! did to the bare step (requests 1151-1306); the KV elements' and the rows' span are the earlier
//! fits' (requests 1151-1195), which these moves cannot tell apart from the chains' rate
//! (`ARCHITECTURE.md`, "Cost"). A layer's own cost is what the
//! chip measured of the model's program past those terms ([`super::measured`]: the bare step
//! where the chip ran it, else Pie decode); a model the chip has not run takes its layers' kinds'
//! ([`KIND_LAYER`]).

use super::Features;
use crate::plan::{Plan, Shape};

/// One PE cycle ([`super::CLOCK_HZ`]), in ms.
const CYCLE: f64 = 1e3 / super::CLOCK_HZ;
/// ms a weight element: a weight PE's stamped matvec phases (1,524 cycles for 3,328 elements).
pub const MATVEC: f64 = 1524.0 / 3328.0 * CYCLE;
/// ms a KV element a row PE reads (fitted: the scores' and the values' passes on the CY column).
pub const KV: f64 = 6.436e-6;
/// ms a cycle of the reductions' chains ([`Features::chain`], fitted with KV and the span held):
/// 0.82 of a PE cycle (fitted when the clock was taken for 875 MHz: 0.96 of a cycle there), so the
/// simulator's hop cycles overprice the chip's chains by a fifth. Fitted with
/// KV free it takes 2.2 cycles and the G2 gains' full size, but KV then costs 15 cycles an element
/// and gpt-oss's layer less than nothing: the 4B's PY / NE move is the only one that moves both.
pub const CHAIN: f64 = 1.093e-6;
/// ms a row a layer's collectives cross down the columns (fitted).
pub const SPAN_ROW: f64 = 1.267e-5;
/// ms a column they cross along the rows: none the chip resolves (fitted to 0; halving the
/// 0.8B's columns cost what its doubled matvec and KV elements do).
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

/// A layer's own cost by its mixer's kind (ms: Qwen3.5's gated delta net, full attention,
/// sliding-window attention, attention over another layer's KV), least squares over the chip's
/// nine measured programs ([`super::measured`]: each model's layer cost past its partition's
/// terms against its layers' kinds; `the_layer_kinds_follow_the_measurements` refits it). Held
/// out a model at a time, a step it predicts is off by 11.8% RMS (2.3, -3.1, 1.2% on the
/// Qwen3.5s; 11.6, -8.1% on Gemma E2B, E4B; 13.3% gpt-oss; -22.9% Gemma 12B; 5.2% Llama; -17.8%
/// Qwen3-8B): what the kinds leave out (a q / k norm, a model's kernels' own phases) is a
/// model's, so a model the chip ran takes its own ([`layer`]); a partition's choice never
/// reads it (every partition of a model has its layers).
pub const KIND_LAYER: [f64; 4] = [8.018e-3, 5.707e-3, 10.653e-3, 19.700e-3];

/// [`KIND_LAYER`] over `shape`'s layers, a layer's mean.
#[must_use]
pub fn kinds_layer(shape: &Shape) -> f64 {
    use crate::plan::LayerKind;
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

#[cfg(test)]
mod tests {
    use crate::target::{Ask, plan};

    /// The chain model against the fabric simulator: a row chain of 128 members a half (a unit
    /// 256 columns wide and 8 rows, 16 dims a row: `scratch/genG2`) took 4,374 cycles a layer at
    /// G2=8 and at G2=16, 4,304 at 12 and 4,398 at 24, two row chains a layer.
    #[test]
    fn the_chain_follows_the_simulator() {
        let t = |g| super::super::chain_cycles(128, g) - super::super::chain_cycles(128, 12);
        for (g, sim) in [(8, 35.0), (16, 35.0), (24, 47.0)] {
            assert!(
                (t(g) - sim).abs() <= 2.0,
                "G2={g}: {} cycles vs {sim}",
                t(g)
            );
        }
    }

    /// The token loop's max-reduce: the three-level tree shortens the shipping control lines'
    /// critical path (0.8B's 1,153 head rows, the mini's 748 band columns at G2W 24) by more
    /// than a third, and on a line too short for it (a tiny head of 10 rows) costs no more than
    /// a hop's merge.
    #[test]
    fn the_three_level_max_reduce_is_shorter() {
        use super::super::loop_reduce_cycles as lr;
        for n in [1153, 748] {
            let (two, three) = (lr(n, 24, 0), lr(n, 8, 6));
            assert!(three < 0.65 * two, "{n}: {three} vs {two}");
        }
        assert!(lr(10, 2, 2) <= lr(10, 2, 0) + 80.0);
    }

    /// What moving a pinned plan's keys did to the bare step on the chip (requests 1151-1306 and
    /// 1456), from the keys the chip ran first, and what the model predicts it does: the same way
    /// where the chip moved past its noise (5 µs: Llama's VR=216 and VR=180 ran in different
    /// sessions), and within 6 µs of each but where the chains' cycles alone come short. The
    /// chains' critical path gives each swept model the chip's best G2 (12 on the Qwen3.5s, E4B
    /// and Qwen3-8B, 16 on the 12B, 8 of E2B's 8 / 16 / 24) but gpt-oss's (its row chains' halves
    /// are 64 attention and 288 MoE columns, and the long half makes G2=8 cost 2.6 µs where the
    /// chip gained 2.5). It gives the G2 moves about half their size on the chip (E4B's G2=12:
    /// 7 µs of 21, held at 15; Llama's G2=11, its pin since: 5 µs of 15, held at 11), and the
    /// 0.8B's PX move and the 4B's PY move, which moved the chains with the KV and the span, 12
    /// and 7 µs short.
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
            (l, "l3-8b", "G2=16,HR=1153,VR=216", "G2=16,HR=1153", 0.488, 0.485, 6.0),
            (l, "l3-8b", "G2=16,HR=1153,VR=216", "G2=16,HR=1153,VR=144", 0.488, 0.4865, 6.0),
            (l, "l3-8b", "G2=16,HR=1153,VR=216", "G2=16,HR=1153,VR=180", 0.488, 0.4915, 6.0),
            (l, "l3-8b", "G2=16,HR=1153", "G2=8,HR=1153", 0.485, 0.486, 6.0),
            (l, "l3-8b", "G2=16,HR=1153", "", 0.485, 0.470, 11.0),
            (q08, "0.8b", "G2=16", "PX=128,G2=10,G2Y=10", 0.296, 0.321, 12.5),
            (q08, "0.8b", "G2=16", "G2=8", 0.297, 0.2965, 6.0),
            (q08, "0.8b", "G2=16", "", 0.297, 0.292, 6.0),
            (g12, "g12", "", "G2=15,GA=4,PY=282", 0.950, 0.9505, 6.0),
            (q4, "4b", "PY=256,NE=64,G2=16,G2Y=16", "G2=16", 0.4935, 0.4855, 8.0),
            (q4, "4b", "G2=16", "G2=8", 0.4855, 0.4845, 6.0),
            (q4, "4b", "G2=16", "", 0.4855, 0.480, 6.0),
            (q8, "q3-8b", "", "G2=8", 0.5505, 0.559, 6.0),
            (q8, "q3-8b", "", "G2=16", 0.5505, 0.560, 6.0),
            (g12, "g12", "", "G2=8", 0.9495, 0.960, 7.5),
            (g12, "g12", "", "G2=12", 0.9495, 0.9515, 6.0),
            (o, "full", "", "G2=8", 0.6525, 0.650, 6.0),
            (o, "full", "", "G2=24", 0.6525, 0.6555, 6.0),
            (q9, "9b", "G2=16", "G2=8", 0.5185, 0.5175, 6.0),
            (q9, "9b", "G2=16", "G2=12", 0.5185, 0.510, 6.0),
            (q9, "9b", "G2=16", "G2=24", 0.5185, 0.5225, 6.0),
            (e4, "e4b", "G2=16", "G2=8", 0.8975, 0.890, 6.5),
            (e4, "e4b", "G2=16", "G2=12", 0.8975, 0.8765, 15.0),
            (e4, "e4b", "G2=16", "G2=24", 0.8975, 0.907, 7.5),
            (e2, "e2b", "G2=16", "G2=8", 0.6915, 0.683, 7.0),
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
                want.abs() < 5e-3 || got.signum() == want.signum(),
                "{model} {to_set}: {got} ms the other way from {want} ms"
            );
        }
    }

    /// [`super::KIND_LAYER`] is the least-squares fit of the nine measured programs' layer costs
    /// to their layers' kinds, and each model held out is predicted within what the doc states.
    #[test]
    fn the_layer_kinds_follow_the_measurements() {
        let names = [
            "qwen35-d0.8b-bf16-kv-bf16",
            "qwen35-d4b-bf16-kv-bf16",
            "qwen35-d9b-bf16-kv-bf16",
            "llama31-8b-bf16-kv-bf16",
            "qwen3-8b-bf16-kv-bf16",
            "gemma4-e2b-bf16-kv-bf16",
            "gemma4-e4b-bf16-kv-bf16",
            "gemma4-12b-bf16-kv-bf16",
            "gptoss-20b-bf16-mxfp4-kv-bf16",
        ];
        // (kinds' fractions, the layer cost, the partition's ms, layers, the measured ms)
        let mut rows: Vec<([f64; 4], f64, f64, f64, f64)> = Vec::new();
        for n in names {
            let t = poem_compiler::catalog::repository()
                .parse(n)
                .unwrap()
                .trace(poem_ir::Platform::Cerebras);
            let s = crate::target::shape(&t).unwrap();
            let m = super::super::measured::on(s.signature()).expect("measured");
            let plan = crate::floorplan::plan(&s, &m.preset).unwrap();
            let f = super::super::Features::of(&plan);
            let part = super::partition(&f);
            let mut x = [0.0; 4];
            for l in &s.layers {
                x[match l {
                    crate::plan::LayerKind::Gdn => 0,
                    crate::plan::LayerKind::Global => 1,
                    crate::plan::LayerKind::Sliding { .. } => 2,
                    crate::plan::LayerKind::SharedKv { .. } => 3,
                }] += 1.0 / f.layers;
            }
            rows.push((x, (m.ms - part) / f.layers, part, f.layers, m.ms));
        }
        // least squares by the normal equations (4 unknowns)
        let fit = |use_row: &dyn Fn(usize) -> bool| -> [f64; 4] {
            let mut a = [[0.0f64; 5]; 4];
            for (i, (x, y, ..)) in rows.iter().enumerate() {
                if !use_row(i) {
                    continue;
                }
                for r in 0..4 {
                    for c in 0..4 {
                        a[r][c] += x[r] * x[c];
                    }
                    a[r][4] += x[r] * y;
                }
            }
            for c in 0..4 {
                let p = (c..4)
                    .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                    .unwrap();
                a.swap(c, p);
                let pivot = a[c];
                for (r, row) in a.iter_mut().enumerate() {
                    if r != c {
                        let k = row[c] / pivot[c];
                        for (x, y) in row.iter_mut().zip(pivot).skip(c) {
                            *x -= k * y;
                        }
                    }
                }
            }
            [
                a[0][4] / a[0][0],
                a[1][4] / a[1][1],
                a[2][4] / a[2][2],
                a[3][4] / a[3][3],
            ]
        };
        let all = fit(&|_| true);
        for (k, (got, want)) in all.iter().zip(super::KIND_LAYER).enumerate() {
            assert!(
                (got - want).abs() < 2e-5,
                "kind {k}: fitted {got}, stated {want}"
            );
        }
        let mut sq = 0.0;
        for (i, n) in names.iter().enumerate() {
            let c = fit(&|j| j != i);
            let (x, _, part, layers, ms) = rows[i];
            let pred = part + layers * (0..4).map(|k| x[k] * c[k]).sum::<f64>();
            let err = (pred - ms) / ms;
            sq += err * err;
            eprintln!("{n}: held out {:+.1}%", err * 100.0);
            assert!(err.abs() < 0.25, "{n}: {err}");
        }
        let rms = (sq / names.len() as f64).sqrt();
        assert!((rms - 0.118).abs() < 0.01, "held-out RMS {rms}");
    }
}
