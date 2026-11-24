//! The grid regime's op-derived time: a grid program's passes, each a sweep over its rows. A row
//! costs its slowest PE's work (weight PEs' matvec tiles at `stream_p1`, roots' region work,
//! gathers, stores and queries) plus `grid_row`, its fixed latency; each pass adds its norms'
//! `grid_final`. `grid_row` is fitted to flux2-mini's decode on the simulator.

use serde::Serialize;

use super::stream::{RegionFeatures, query_cycles};
use crate::compile::stream::grid::{GridProgram, Step};
use crate::tuning;

/// One pass, priced.
#[derive(Clone, Debug, Serialize)]
pub struct PassCost {
    pub pass: usize,
    pub rows: u32,
    pub head_rows: u32,
    /// Cycles a tail row of the weight PEs and of the roots, and a head row's.
    pub matvec_row: f64,
    pub root_row: f64,
    pub head_row: f64,
    pub cycles: f64,
}

/// A grid program's passes and its cycles.
#[derive(Clone, Debug, Serialize)]
pub struct GridCost {
    pub passes: Vec<PassCost>,
    pub cycles: f64,
}

/// The weight elements a weight PE multiplies for matvec `m` a row (its busiest tile).
fn tile(g: &GridProgram, m: usize) -> f64 {
    let p = g.p;
    let kc = (0..p).map(|k| g.piece(m, k).len()).max().unwrap_or(0);
    let nc = (0..p).map(|k| g.slice(m, k).len()).max().unwrap_or(0);
    (kc * nc) as f64
}

/// A root's work a row for `steps` (a head's or a tail's), its gather's taps copied.
fn roots(g: &GridProgram, steps: &[Step], gathered: f64) -> f64 {
    let mut c = gathered * tuning::get().stream_dsd_elem;
    for s in steps {
        c += match s {
            Step::Region(r) => {
                let reg = &g.regs[*r];
                let chunk = g.chunk(reg.outs[0]);
                reg.printed
                    .as_ref()
                    .map_or(0.0, |pr| RegionFeatures::of(&pr.csl, chunk).cycles(chunk))
            }
            Step::Fetch(v) | Step::Hold(v) | Step::Out(v) | Step::Stats(v) | Step::Ring(v) => {
                f64::from(g.chunk(*v)) * tuning::get().stream_dsd_elem
            }
            Step::Store { .. } => tuning::get().stream_store,
            Step::Attend(_) => g
                .attn
                .as_ref()
                .map_or(0.0, |a| query_cycles(a.rows, a.pya, 1)),
            Step::Matvec(_) | Step::Gather => 0.0,
        };
    }
    c
}

/// `g`'s time: each pass's rows at their slowest PE's work and the row's fixed cost.
#[must_use]
pub fn time(g: &GridProgram) -> GridCost {
    let mut passes = Vec::new();
    let mut cycles = 0.0;
    for (pi, pass) in g.passes.iter().enumerate() {
        let matvec_row: f64 = pass
            .tail
            .iter()
            .chain(&pass.head)
            .filter_map(|s| match s {
                Step::Matvec(m) => Some(tile(g, *m) * tuning::get().stream_p1),
                _ => None,
            })
            .sum();
        let gathered = pass
            .gather
            .as_ref()
            .map_or(0.0, |gt| f64::from(gt.taps()) * f64::from(g.chunk(gt.src)));
        let root_row = roots(g, &pass.tail, gathered);
        let head_row = if pass.head.is_empty() {
            0.0
        } else {
            roots(g, &pass.head, 0.0) + tuning::get().grid_row
        };
        let rows = pass.rows();
        let head_rows = pass.src_rows();
        let finals: f64 = pass
            .finals
            .iter()
            .map(|&n| f64::from(g.gns[n].groups) / f64::from(g.p) * tuning::get().grid_final)
            .sum();
        let c = f64::from(rows) * (matvec_row.max(root_row) + tuning::get().grid_row)
            + f64::from(head_rows) * head_row
            + finals;
        cycles += c;
        passes.push(PassCost {
            pass: pi,
            rows,
            head_rows,
            matvec_row,
            root_row,
            head_row,
            cycles: c,
        });
    }
    GridCost { passes, cycles }
}
