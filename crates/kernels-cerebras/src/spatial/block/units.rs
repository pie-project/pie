//! The units: `rows` x `cols` weight PEs around a CX column and a CY row, running consecutive
//! layers, visited in order (a snake of unit slots with gap slots, or explicit origins). Lays
//! each unit row's chain into its CX and each unit column's into its CY (two-level by the
//! plan's reduce groups), every root's loop-back, and the x handoff from each unit to the next;
//! gap slots relay the vertical handoff. Copies each unit's weights, CX gammas and CY tables.

use crate::spatial::floor::{Coord, Floor, FloorBuilder, Rect, Unit, UnitGrid, skip};
use crate::spatial::plan::Plan;
use crate::spatial::weights::{Copies, From, Slice, chunks};
use crate::spatial::wiring::{Chain, Class, LOOPBACK, Path, Slot, Wiring, level2, loopback, steps};

use super::Block;

type R<T> = Result<T, String>;

/// Unit rows' reductions into CX: level-1 halves, level 2, multicast.
pub const ROW_L1: [Class; 2] = [Class::new("XA"), Class::new("XB")];
pub const ROW_L2: [Class; 2] = Class::pair("X2");
pub const ROW_CAST: Class = Class::new("MX");
/// Unit columns' reductions into CY.
pub const COL_L1: [Class; 2] = [Class::new("YA"), Class::new("YB")];
pub const COL_L2: [Class; 2] = Class::pair("Y2");
pub const COL_CAST: Class = Class::new("MY");
/// The x handoff: along the row out of CX, vertical (by the leaving unit's parity: even,
/// odd), into the next unit's CX, horizontal between unit columns.
pub const HANDOFF_OUT: Class = Class::new("HA");
pub const HANDOFF_VERT: [Class; 2] = Class::pair("HV");
pub const HANDOFF_IN: Class = Class::new("HB");
pub const HANDOFF_ACROSS: Class = Class::new("HX");

/// The unit-row / unit-column reduction a PE is a member or root of.
pub const ROW_REDUCE: Slot = Slot::reduce("row", true);
pub const COL_REDUCE: Slot = Slot::reduce("col", true);
/// CX's x in / out.
pub const X_IN: Slot = Slot::io("x_in");
pub const X_OUT: Slot = Slot::io("x_out");
/// A weight PE's handoff turn before / after it (arriving at / leaving its unit).
pub const RELAY_PRE: Slot = Slot::io("relay_pre");
pub const RELAY_POST: Slot = Slot::io("relay_post");

/// The kernels' colours of these classes.
pub const COLOURS: &[(Class, u8)] = &[
    (ROW_L1[0], 1),
    (ROW_L1[1], 2),
    (ROW_CAST, 3),
    (COL_L1[0], 4),
    (COL_L1[1], 5),
    (COL_CAST, 6),
    (LOOPBACK, 7),
    (HANDOFF_OUT, 0),
    (HANDOFF_IN, 11),
    (HANDOFF_ACROSS, 12),
    (HANDOFF_VERT[0], 15),
    (HANDOFF_VERT[1], 14),
    (ROW_L2[0], 18),
    (ROW_L2[1], 19),
    (COL_L2[0], 20),
    (COL_L2[1], 16),
];

/// The units block.
pub struct Units {
    pub grid: UnitGrid,
    /// What each unit's PEs hold.
    pub copies: UnitCopies,
}

impl Block for Units {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        b.add_units(&self.grid, (0, 0))
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        unit_chains(plan, floor, w)
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        COLOURS.to_vec()
    }

    fn bindings(&self, prog: &crate::spatial::program::Program, out: &mut Copies) -> R<()> {
        for unit in &prog.floor.units {
            self.copies.put(out, unit);
        }
        for r in &prog.floor.relays {
            out.rcfg(*r);
        }
        Ok(())
    }
}

/// Per unit, the row chains into CX (two-level by `reduce.row`; [`ROW_REDUCE`]), the column
/// chains into CY (`reduce.col`; [`COL_REDUCE`]), every root's loop-back to its own ramp; then
/// the unit-to-unit x handoffs along the visit order ([`handoff`]).
pub fn unit_chains(plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
    let red = &plan.partition.reduce;
    for u in &floor.units {
        let o = u.origin();
        for r in 0..u.rows() {
            let y = u.row_y(r);
            w.chain(Chain {
                pts: (0..u.rect.w).map(|lx| Coord::new(o.x + lx, y)).collect(),
                root: u.cx as usize,
                l1: ROW_L1,
                l2: level2(ROW_L2, red.row),
                cast: ROW_CAST,
                loopback: false,
                slot: ROW_REDUCE,
            });
            w.turn(loopback(Coord::new(u.cx_x(), y)));
        }
        for c in 0..u.cols() {
            let x = u.col_x(c);
            w.chain(Chain {
                pts: (0..u.rect.h).map(|ly| Coord::new(x, o.y + ly)).collect(),
                root: u.cy as usize,
                l1: COL_L1,
                l2: level2(COL_L2, red.col),
                cast: COL_CAST,
                loopback: false,
                slot: COL_REDUCE,
            });
            w.turn(loopback(Coord::new(x, u.cy_y())));
        }
    }
    for i in 1..floor.units.len() {
        handoff(floor, i - 1, i, w)?;
    }
    Ok(())
}

/// The x handoff between consecutive units `a` -> `b`, one path a weight row `r`. Units in one
/// unit column (equal rect x): along row r from `a`'s CX to the weight PE at column r
/// ([`HANDOFF_OUT`]), down / up that column to `b`'s row r ([`HANDOFF_VERT`] by `a`'s parity; a
/// gap slot between them only relays), back along the row into `b`'s CX ([`HANDOFF_IN`]); the
/// turning weight PEs relay ([`RELAY_POST`] leaving `a`, [`RELAY_PRE`] arriving at `b`). Units
/// in different columns must share rows: one path along the row ([`HANDOFF_ACROSS`]). CX ends
/// bind [`X_OUT`] / [`X_IN`].
pub fn handoff(floor: &Floor, a: usize, b: usize, w: &mut Wiring) -> R<()> {
    let (ua, ub): (&Unit, &Unit) = match (floor.units.get(a), floor.units.get(b)) {
        (Some(ua), Some(ub)) => (ua, ub),
        _ => return Err(format!("wiring: handoff {a} -> {b} past the units")),
    };
    let (a0, b0) = (ua.origin(), ub.origin());
    let vertical = a0.x == b0.x;
    if !vertical && a0.y != b0.y {
        return Err(format!(
            "units {a} / {b}: a horizontal handoff joins units on the same rows"
        ));
    }
    let vert = HANDOFF_VERT[a % 2];
    for r in 0..ua.rows() {
        let (xa, ya) = (ua.cx_x(), ua.row_y(r));
        if vertical {
            let xcc = ua.col_x(r);
            let st: i64 = if xcc > xa { 1 } else { -1 };
            let yb = ub.row_y(r);
            let vs: i64 = if yb > ya { 1 } else { -1 };
            w.path(Path {
                pts: steps(xa as i64, xcc as i64 + st, st)
                    .map(|x| Coord::new(x, ya))
                    .collect(),
                class: HANDOFF_OUT,
                send: X_OUT,
                recv: RELAY_POST,
            });
            w.path(Path {
                pts: steps(ya as i64, yb as i64 + vs, vs)
                    .map(|y| Coord::new(xcc, y))
                    .collect(),
                class: vert,
                send: RELAY_POST,
                recv: RELAY_PRE,
            });
            w.path(Path {
                pts: steps(xcc as i64, ub.cx_x() as i64 - st, -st)
                    .map(|x| Coord::new(x, yb))
                    .collect(),
                class: HANDOFF_IN,
                send: RELAY_PRE,
                recv: X_IN,
            });
        } else {
            let xb = ub.cx_x();
            let st: i64 = if xb > xa { 1 } else { -1 };
            w.path(Path {
                pts: steps(xa as i64, xb as i64 + st, st)
                    .map(|x| Coord::new(x, ya))
                    .collect(),
                class: HANDOFF_ACROSS,
                send: X_OUT,
                recv: X_IN,
            });
        }
    }
    Ok(())
}

/// A unit's copies, quadrant by quadrant (rows, then columns): its rcfg and `w{u}` in blocks
/// of 16 rows; then each half of its CX column (rcfg, `gam`); then each half of its CY row
/// (rcfg, each of `cy`: (symbol, whether the table is per unit), optional when `cy_optional`).
/// Placement order is part of the host's device protocol.
pub struct UnitCopies {
    pub cols: u32,
    pub rows: u32,
    pub w_from: From,
    pub gam_b16: bool,
    pub gam_from: From,
    pub cy: &'static [(&'static str, bool)],
    pub cy_optional: bool,
}

impl UnitCopies {
    pub fn put(&self, out: &mut Copies, unit: &Unit) {
        let (px, py) = (self.cols, self.rows);
        let (xc, yc) = (px / 2, py / 2);
        let (x0u, y0u, u) = (unit.rect.x, unit.rect.y, unit.index);
        for (r0, r1) in [(0, yc), (yc, py)] {
            for (c0, c1) in [(0, xc), (xc, px)] {
                let x0 = x0u + skip(c0, xc);
                out.rcfg(Rect::new(x0, y0u + skip(r0, yc), c1 - c0, r1 - r0));
                for (rb, re) in chunks(r0, r1, 16) {
                    out.slice(
                        "w",
                        Rect::new(x0, y0u + skip(rb, yc), c1 - c0, re - rb),
                        Slice::of(&format!("w{u}"), None, vec![(rb, re), (c0, c1)]),
                        true,
                        self.w_from.clone(),
                    );
                }
            }
        }
        for (r0, r1) in [(0, yc), (yc, py)] {
            let rect = Rect::new(x0u + xc, y0u + skip(r0, yc), 1, r1 - r0);
            out.rcfg(rect);
            let gam = Slice::of("gam", Some(u), vec![(r0, r1)]);
            out.slice("gam", rect, gam, self.gam_b16, self.gam_from.clone());
        }
        for (c0, c1) in [(0, xc), (xc, px)] {
            let rect = Rect::new(x0u + skip(c0, xc), y0u + yc, c1 - c0, 1);
            out.rcfg(rect);
            for &(sym, per_unit) in self.cy {
                let s = Slice::of(sym, per_unit.then_some(u), vec![(c0, c1)]);
                let s = if self.cy_optional { s.optional() } else { s };
                out.slice(sym, rect, s, false, Vec::new());
            }
        }
    }
}
