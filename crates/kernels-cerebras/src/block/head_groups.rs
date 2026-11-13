//! The head east of the units, fed by groups of CX rows (the plan's `head.groups`: Gemma 4
//! 12B). Head column `c` takes the dims of its group's rows: on the last unit each group's rows
//! chain their x north into the group's first row ([`HEAD_GROUP`]), whose CX sends it east to
//! the head column, which multicasts it down the column. The token loop's winner comes back the
//! same way: up each head column, down to its group's first row, west into unit 0's CX there,
//! and down the group's chain. The head rows reduce as [`super::head`]'s; copies as its.

use crate::floor::{Coord, Floor, FloorBuilder};
use crate::plan::Plan;
use crate::program::Program;
use crate::weights::Copies;
use crate::wiring::{Class, Multicast, NoteKey, Path, Slot, Wiring};

use super::Block;
use super::head::{self, HEAD_COL_CAST, HEAD_IN, HeadCopies, ROLE, ROLE_SOURCE};
use super::row_runs::dims_of;
use super::token_loop::{self, EMB_DOWN, EMB_IN, EMB_UP, EMB_WEST, TURN, TURN_EVEN, TURN_TOP};
use super::units::{HANDOFF_ACROSS, X_OUT};

type R<T> = Result<T, String>;

/// A group's chain between adjacent CX rows, alternating by a member's place in its group.
pub const HEAD_GROUP: [Class; 2] = Class::pair("CH");

/// A CX's group chain in / out; the head column multicast; the embedding's turns at the head.
pub const CHAIN_IN: Slot = Slot::io("head_groups.chain_in");
pub const CHAIN_OUT: Slot = Slot::io("head_groups.chain_out");
pub const CAST: Slot = Slot::io("head_groups.cast");
const UP_OUT: Slot = Slot::io("head_groups.up_out");
const UP_IN: Slot = Slot::io("head_groups.up_in");
const DOWN_OUT: Slot = Slot::io("head_groups.down_out");
const DOWN_IN: Slot = Slot::io("head_groups.down_in");
const WEST_OUT: Slot = Slot::io("head_groups.west_out");

/// The dims a group member's CX forwards along its chain (its group's rows past it), on the
/// last unit (x north) and unit 0 (the embedding south).
pub const FORWARD: NoteKey = NoteKey("head_groups.forward");

pub const COLOURS: &[(Class, u8)] = &[(HEAD_GROUP[0], 4), (HEAD_GROUP[1], 5)];

/// The grouped head, its token loop always on.
pub struct HeadGroups {
    pub cols: u32,
    pub rows: u32,
    pub copies: HeadCopies,
}

/// The plan's head groups.
fn groups(plan: &Plan) -> R<&[Vec<u32>]> {
    plan.partition
        .head
        .groups
        .as_deref()
        .ok_or_else(|| "head groups: the plan has none".to_string())
}

/// Head column `c`'s dims: its group's rows'.
pub fn group_dims(plan: &Plan, c: usize) -> u32 {
    groups(plan).map_or(0, |g| g[c].iter().map(|&r| dims_of(plan, r)).sum())
}

/// One group a head column; each group consecutive rows on one side of the CY row (the plan
/// checks they cover every row once).
fn check(plan: &Plan, gs: &[Vec<u32>]) -> R<()> {
    let p = &plan.partition;
    if gs.len() as u32 != p.head.cols {
        return Err(format!(
            "head groups: {} groups for {} head columns",
            gs.len(),
            p.head.cols
        ));
    }
    let yc = p.rows / 2;
    for g in gs {
        let (a, b) = (g[0], g[g.len() - 1]);
        if g.iter().enumerate().any(|(i, &r)| r != a + i as u32) || (a < yc && yc <= b) {
            return Err(format!("head group {g:?} not consecutive on one side"));
        }
    }
    Ok(())
}

impl Block for HeadGroups {
    fn floor(&self, _plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        let x0 = b.w;
        b.head_east(x0, self.cols, self.rows, true);
        Ok(())
    }

    /// The head rows; per group the x in and the embedding's return; the max-reduce down the
    /// control column.
    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        let gs = groups(plan)?;
        check(plan, gs)?;
        head::head_rows(plan, floor, w)?;
        let hb = &floor.head;
        let hcol = hb.ctl_x().ok_or("head groups: no control column")?;
        let (y0, hr) = (hb.rect.y, hb.rect.h);
        w.link(token_loop::loop_reduce(
            plan,
            (y0..y0 + hr).map(|y| Coord::new(hcol, y)).collect(),
            (hr / 2) as usize,
        ));
        let (u0, ul) = match (floor.units.first(), floor.units.last()) {
            (Some(a), Some(b)) if a.rect.y == b.rect.y => (a, b),
            _ => return Err("head groups: unit 0 and the last unit share no rows".into()),
        };
        let forward = |g: &[u32], k: usize| g[k + 1..].iter().map(|&r| dims_of(plan, r)).sum();
        for (c, g) in gs.iter().enumerate() {
            let hx = hb.col_x(c as u32);
            // x in: the group's chain north on the last unit, its root east, the column cast
            let (xa, y) = (ul.cx_x(), ul.row_y(g[0]));
            w.path(Path {
                pts: (xa..=hx).map(|x| Coord::new(x, y)).collect(),
                class: HANDOFF_ACROSS,
                send: X_OUT,
                recv: HEAD_IN,
            });
            w.note(Coord::new(hx, y), ROLE, ROLE_SOURCE);
            w.multicast(Multicast {
                pts: (y0..y0 + hr).map(|yy| Coord::new(hx, yy)).collect(),
                src: (y - y0) as usize,
                class: HEAD_COL_CAST,
                slot: CAST,
            });
            // the embedding back: up the column, down to the group's first row, west into unit
            // 0's CX, down the chain
            let te = u0.row_y(g[0]);
            w.path(Path {
                pts: (y0..y0 + hr).rev().map(|yy| Coord::new(hx, yy)).collect(),
                class: EMB_UP,
                send: UP_OUT,
                recv: UP_IN,
            });
            if te > y0 {
                w.path(Path {
                    pts: (y0..=te).map(|yy| Coord::new(hx, yy)).collect(),
                    class: EMB_DOWN,
                    send: DOWN_OUT,
                    recv: DOWN_IN,
                });
            }
            w.path(Path {
                pts: (u0.cx_x()..=hx).rev().map(|x| Coord::new(x, te)).collect(),
                class: EMB_WEST,
                send: WEST_OUT,
                recv: EMB_IN,
            });
            w.note(Coord::new(hx, y0), TURN, TURN_TOP);
            w.note(Coord::new(hx, te), TURN, TURN_EVEN);
            for k in 0..g.len() {
                let (a, b) = (Coord::new(ul.cx_x(), ul.row_y(g[k])), u0.cx_x());
                let fw = forward(g, k);
                if fw > 0 {
                    w.note(a, FORWARD, fw);
                    w.note(Coord::new(b, u0.row_y(g[k])), FORWARD, fw);
                }
                if k == 0 {
                    continue;
                }
                let cls = HEAD_GROUP[k % 2];
                w.path(Path {
                    pts: vec![a, Coord::new(ul.cx_x(), ul.row_y(g[k - 1]))],
                    class: cls,
                    send: CHAIN_OUT,
                    recv: CHAIN_IN,
                });
                w.path(Path {
                    pts: vec![
                        Coord::new(b, u0.row_y(g[k - 1])),
                        Coord::new(b, u0.row_y(g[k])),
                    ],
                    class: cls,
                    send: CHAIN_OUT,
                    recv: CHAIN_IN,
                });
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        [head::COLOURS, token_loop::COLOURS, COLOURS].concat()
    }

    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        self.copies.put(out, prog.floor.head.rect.x);
        Ok(())
    }
}
