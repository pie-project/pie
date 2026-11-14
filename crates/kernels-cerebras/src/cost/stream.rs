//! The stream regime's op-derived step time (GENERIC.md §2.4): each unit's rows priced by what
//! its PEs do a row, the units' phases in the order the row stream and the barriers allow.
//!
//! - P1-stream: the busiest weight PE's tile elements a row x [`P1_STREAM`] (M1c: 0.545 - 0.572
//!   cycles an element, f32 accumulation).
//! - P4 regions: their ops over the chunk at [`REGION_OP`] an element, [`ALL_REDUCE`] a segment
//!   all-reduce, [`ROTATION`] a rotation's P5 digit products (v0's scalar loops, M1c's measured
//!   region cycles).
//! - P7: [`STORE`] a stored row, [`QUERY`] a query (M1c, 16 entries a PE).
//!
//! A unit's phase costs its rows x its slowest PE a row (the PEs of a unit pipeline the rows);
//! the step is the units' phases along the stream: a unit's pre-barrier phase starts when its
//! rows reach it, its post-barrier phase when every unit of its block has stored.

use serde::Serialize;

use crate::stream::build::Program;
use crate::stream::place;
use crate::stream::unit::{Side, Step};

/// P1-stream cycles an element a row.
pub const P1_STREAM: f64 = 0.55;
/// A region op's cycles an element of the chunk (v0's DSD ops and copies).
pub const REGION_OP: f64 = 2.0;
/// A segment all-reduce over a unit's CX roots.
pub const ALL_REDUCE: f64 = 1_200.0;
/// A rotation's cos / sin from P5 tables a row (64 pairs, 4 levels of scalar math).
pub const ROTATION: f64 = 5_000.0;
/// P7: a stored row, a query.
pub const STORE: f64 = 4_613.0;
pub const QUERY: f64 = 3_521.0;

/// One unit's priced phases.
#[derive(Clone, Debug, Serialize)]
pub struct UnitCost {
    pub name: String,
    pub rows: u32,
    /// Cycles a row before / after the barrier (the slowest PE's).
    pub early_row: f64,
    pub late_row: f64,
    /// The weight PEs' share of them.
    pub matvec_row: f64,
}

/// The step's priced units and the step's cycles.
#[derive(Clone, Debug, Serialize)]
pub struct StreamCost {
    pub units: Vec<UnitCost>,
    pub cycles: f64,
}

fn region_cycles(p: &Program, u: usize, step: usize) -> f64 {
    let b = &p.units[u];
    let Some((at, printed, _)) = b.regions.get(&step) else {
        return 0.0;
    };
    let ops = match &b.unit.steps[step] {
        Step::Region { chain, .. } => chain.region.ops.len() as f64,
        _ => 0.0,
    };
    let rot = if printed.csl.contains("rp.tables") {
        ROTATION
    } else {
        0.0
    };
    ops * f64::from(at.chunk) * REGION_OP + f64::from(printed.all_reduces) * ALL_REDUCE + rot
}

/// The step time of `p`.
#[must_use]
pub fn time(p: &Program) -> StreamCost {
    let mut units = Vec::new();
    for (ui, b) in p.units.iter().enumerate() {
        let l = &b.layout;
        let (mut e, mut lt, mut mv) = (0.0f64, 0.0f64, 0.0f64);
        // the weight PEs: the busiest column a row, by phase
        for (phase, acc) in [(false, &mut e), (true, &mut lt)] {
            let busiest = (0..l.px())
                .map(|c| {
                    b.unit
                        .steps
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i >= b.unit.split) == phase)
                        .filter_map(|(_, s)| place::tile_dims(s, l, c))
                        .map(|(a, bb)| f64::from(a * bb) * P1_STREAM)
                        .sum::<f64>()
                })
                .fold(0.0, f64::max);
            mv = mv.max(busiest);
            // the roots: their regions a row, on the slowest side
            let mut row_side = 0.0;
            let mut col_side = 0.0;
            for (i, s) in b.unit.steps.iter().enumerate() {
                if (i >= b.unit.split) != phase {
                    continue;
                }
                match s {
                    Step::Region {
                        side: Side::Row, ..
                    } => row_side += region_cycles(p, ui, i),
                    Step::Region {
                        side: Side::Col, ..
                    } => col_side += region_cycles(p, ui, i),
                    Step::Store { .. } => col_side += STORE,
                    Step::Attend { .. } => col_side += QUERY,
                    _ => {}
                }
            }
            *acc = busiest.max(row_side).max(col_side);
        }
        units.push(UnitCost {
            name: b.unit.name.clone(),
            rows: b.unit.rows,
            early_row: e,
            late_row: lt,
            matvec_row: mv,
        });
    }
    // along the stream: a unit starts its rows once the unit before it has started its own
    // (rows pipeline), its late phase once its block has stored every row
    let mut t = 0.0f64;
    let mut done = 0.0f64;
    for u in &units {
        let early = f64::from(u.rows) * u.early_row;
        let late = f64::from(u.rows) * u.late_row;
        let start = t;
        let end_early = start + early;
        let end = end_early + late;
        done = done.max(end);
        // the next unit's rows arrive as this one's leave: after its barrier where it has one
        t = if late > 0.0 {
            end_early
        } else {
            start + u.early_row
        };
    }
    StreamCost {
        cycles: done,
        units,
    }
}
