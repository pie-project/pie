//! Units whose weight rows come in runs of different dims and roles (the plan's `row_dims`:
//! Gemma 4 12B's attention rows `0 .. RA` and plain rows past them). The units' floor and chains
//! are the standard ones ([`super::units`]); the scores segments run along the attention runs'
//! rows only; a weight / CX PE's `Hc` is its run's dims and a weight PE's `AROW` its run's role.
//! Copies each unit run by run.

use crate::abi::{Overrides, Param};
use crate::floor::{Coord, Floor, FloorBuilder, Rect, Unit, UnitGrid, skip};
use crate::plan::{Plan, RowRun};
use crate::program::Program;
use crate::role::Place;
use crate::weights::{Copies, From, Slice, chunks};
use crate::wiring::{Class, Wiring, segments};

use super::Block;
use super::segments::{SCORE_CAST, SCORE_L1, SEGMENT};
use super::units::{self, unit_chains};

type R<T> = Result<T, String>;

/// A weight / CX PE's dims a row, a weight PE's row role.
pub const HC: Param = Param("Hc");
pub const AROW: Param = Param("AROW");
/// [`AROW`]'s codes: an attention row, a plain row.
pub const ATT_ROW: u32 = 1;
pub const PLAIN_ROW: u32 = 2;

/// The units with row runs.
pub struct RowRuns {
    pub grid: UnitGrid,
    pub copies: RunCopies,
}

/// The run holding weight row `r` (`None` with a single run: nothing to override).
fn run_of(plan: &Plan, r: u32) -> Option<&RowRun> {
    let runs = &plan.partition.row_dims;
    (runs.len() > 1)
        .then(|| runs.iter().find(|x| x.rows.0 <= r && r < x.rows.1))
        .flatten()
}

/// The runs cut at the CY row, in placement order (the rows above it, then below): (first row,
/// end row, the run).
pub fn blocks(plan: &Plan) -> Vec<(u32, u32, &RowRun)> {
    let py = plan.partition.rows;
    let yc = py / 2;
    let mut out = Vec::new();
    for (a, b) in [(0, yc), (yc, py)] {
        for run in &plan.partition.row_dims {
            let (lo, hi) = (a.max(run.rows.0), b.min(run.rows.1));
            if lo < hi {
                out.push((lo, hi, run));
            }
        }
    }
    out
}

/// Weight row `r`'s dims.
pub fn dims_of(plan: &Plan, r: u32) -> u32 {
    plan.partition
        .row_dims
        .iter()
        .find(|x| x.rows.0 <= r && r < x.rows.1)
        .map_or(0, |x| x.dims)
}

impl Block for RowRuns {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        b.add_units(&self.grid, (0, 0))
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        unit_chains(plan, floor, w)?;
        let width = plan.partition.segments.scores;
        for u in &floor.units {
            for run in plan.partition.row_dims.iter().filter(|r| r.attention) {
                for r in run.rows.0..run.rows.1 {
                    let pts: Vec<Coord> = (0..u.cols())
                        .map(|c| Coord::new(u.col_x(c), u.row_y(r)))
                        .collect();
                    segments(&pts, width, true, (SCORE_L1, SCORE_CAST, SEGMENT), w)?;
                }
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        [units::COLOURS, super::segments::COLOURS].concat()
    }

    fn overrides(&self, prog: &Program, o: &mut Overrides) -> R<()> {
        for (at, role) in prog.floor.roles.iter() {
            match role.place {
                Place::Weight { row, .. } => {
                    if let Some(run) = run_of(&prog.plan, row) {
                        o.set(at, HC, run.dims);
                        o.set(at, AROW, if run.attention { ATT_ROW } else { PLAIN_ROW });
                    }
                }
                Place::Cx { row, .. } => {
                    if let Some(run) = run_of(&prog.plan, row) {
                        o.set(at, HC, run.dims);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        for unit in &prog.floor.units {
            self.copies.put(&prog.plan, out, unit);
        }
        Ok(())
    }
}

/// A unit's copies, run block by run block ([`blocks`]): each half of the columns' rcfg and
/// weights (`w{u}a` for an attention run, `w{u}w` else, its rows from the run's first) 16 rows a
/// copy, then the block's CX rcfg and gammas (`gama` / `gamw`); then each half of the CY row:
/// rcfg and each of `cy` (symbol, whether the table is per unit). Placement order is part of
/// the host's device protocol.
pub struct RunCopies {
    pub w_from: From,
    pub cy: &'static [(&'static str, bool)],
}

impl RunCopies {
    pub fn put(&self, plan: &Plan, out: &mut Copies, unit: &Unit) {
        let (px, py) = (plan.partition.cols, plan.partition.rows);
        let (xc, yc) = (px / 2, py / 2);
        let (x0u, y0u, u) = (unit.rect.x, unit.rect.y, unit.index);
        for (r0, r1, run) in blocks(plan) {
            let (rb0, tag) = (run.rows.0, if run.attention { 'a' } else { 'w' });
            for (c0, c1) in [(0, xc), (xc, px)] {
                let x0 = x0u + skip(c0, xc);
                out.rcfg(Rect::new(x0, y0u + skip(r0, yc), c1 - c0, r1 - r0));
                for (rb, re) in chunks(r0, r1, 16) {
                    out.slice(
                        "w",
                        Rect::new(x0, y0u + skip(rb, yc), c1 - c0, re - rb),
                        Slice::of(
                            &format!("w{u}{tag}"),
                            None,
                            vec![(rb - rb0, re - rb0), (c0, c1)],
                        ),
                        true,
                        self.w_from.clone(),
                    );
                }
            }
            let cx = Rect::new(x0u + xc, y0u + skip(r0, yc), 1, r1 - r0);
            out.rcfg(cx);
            let gam = Slice::of(
                &format!("gam{tag}"),
                None,
                vec![(u, u + 1), (r0 - rb0, r1 - rb0)],
            );
            out.slice("gam", cx, gam, false, Vec::new());
        }
        for (c0, c1) in [(0, xc), (xc, px)] {
            let r = Rect::new(x0u + skip(c0, xc), y0u + yc, c1 - c0, 1);
            out.rcfg(r);
            for &(sym, per_unit) in self.cy {
                let dims = if per_unit {
                    vec![(u, u + 1), (c0, c1)]
                } else {
                    vec![(c0, c1)]
                };
                out.slice(sym, r, Slice::of(sym, None, dims), false, Vec::new());
            }
        }
    }
}
