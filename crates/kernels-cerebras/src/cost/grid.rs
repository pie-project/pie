//! The grid regime's op-derived time (GENERIC.md M2c): a grid program's passes, each a sweep
//! over its rows, priced by what its PEs do a row.
//!
//! A pass's rows go through the unit one at a time: a head row (its gather's source: a held
//! value fetched, a norm applied, into the line buffer) and a tail row (the gather, the matvecs,
//! the regions after them, the row kept or sent on). A row costs its slowest PE's work (the
//! roots' and the weight PEs' overlap a row) plus [`ROW`], the row's fixed cost: the fetch's
//! round trip up a chain, the gather's index math, the chains' latency, which a row of a single
//! unit does not hide. The pieces:
//!
//! - the weight PEs: each tail matvec's tile elements a row x [`P1_STREAM`] (as a stream unit's);
//! - the roots: each region's `row()` at its chunk ([`RegionFeatures`]), a gather's taps
//!   copied ([`DSD_ELEM`] a word), a fetched or kept row's chunk, a stored row and a query
//!   (P7: [`STORE`], [`query_cycles`]);
//! - a pass's finals: each norm's scale and shift ([`FINAL`]).
//!
//! [`ROW`] is fitted to flux2-mini's decode on the simulator (M1i: the 2 x 2 latent 4.72M
//! cycles, the 8 x 8 latent 73.5M; priced 4.51M and 75.2M). The weight PEs' term is the stream
//! regime's P1 rate (a mini's tiles are too small to show it).

use serde::Serialize;

use super::stream::{DSD_ELEM, P1_STREAM, RegionFeatures, STORE, query_cycles};
use crate::stream::grid::{GridProgram, Step};

/// A row's fixed cycles: the fetch's round trip, the gather's index math, the chains' latency.
pub const ROW: f64 = 1_150.0;
/// A norm's scale and shift a group at the end of its pass.
pub const FINAL: f64 = 400.0;

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
    let mut c = gathered * DSD_ELEM;
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
                f64::from(g.chunk(*v)) * DSD_ELEM
            }
            Step::Store { .. } => STORE,
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
                Step::Matvec(m) => Some(tile(g, *m) * P1_STREAM),
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
            roots(g, &pass.head, 0.0) + ROW
        };
        let rows = pass.rows();
        let head_rows = pass.src_rows();
        let finals: f64 = pass
            .finals
            .iter()
            .map(|&n| f64::from(g.gns[n].groups) / f64::from(g.p) * FINAL)
            .sum();
        let c = f64::from(rows) * (matvec_row.max(root_row) + ROW)
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
