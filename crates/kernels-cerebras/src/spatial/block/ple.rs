//! Per-layer-input storage (Gemma 4's PLE): a storage block directly above each unit column's
//! top unit, every storage PE holding the PLE rows of `tokens` consecutive tokens (all the
//! column's layers'); the block's CX column routes only. The head control column sends each
//! token west along the storage rows ([`TOKEN`]); every storage PE of a column sends its row down
//! its weight column ([`bus`]), through the column's units, into each unit's CY PE. Non-unit PEs
//! west of the head that any link crosses run the store kernel as route-only PEs.

use std::collections::BTreeMap;

use crate::spatial::abi::{Overrides, Param};
use crate::spatial::floor::{
    Coord, CtlLine, Dir, Floor, FloorBuilder, Rect, StoreBlock, Unit, skip,
};
use crate::spatial::plan::Plan;
use crate::spatial::program::Program;
use crate::spatial::role::{Kind, PeRole, Place};
use crate::spatial::weights::{Copies, Slice, chunks, from};
use crate::spatial::wiring::{Class, Link, Port, Turn, Wiring};

use super::Block;

type R<T> = Result<T, String>;

/// The store kernel's PEs: storage PEs and the route-only PEs.
pub const STORE: Kind = Kind::new("ple", "ple");

/// The store kernel's words a PE (a route-only PE outside the storage blocks takes one).
pub const NTW: Param = Param("NTW");

/// The PLE token word along each storage row.
pub const TOKEN: Class = Class::new("TB");

/// The PLE bus down storage column `i`'s unit column.
pub fn bus(i: usize) -> Class {
    Class::at("PB", i as u8)
}

/// A route-only PE of the store kernel (a storage block's CX column, a pass PE).
pub const PASS_PE: Place = Place::Store {
    first: 0,
    count: 0,
    level: 0,
    top: false,
};

/// The largest staged PLE file (`prep_gemma.py`'s `MAXB`).
const FILE_MAX: u64 = 1900 * 1024 * 1024;

/// One unit column's storage: its units top to bottom, the storage rows above the top one, the
/// tokens a storage PE holds, its bus's colour.
pub struct PleColumn {
    pub units: Vec<u32>,
    pub rows: u32,
    pub tokens: u32,
    pub bus: u8,
}

/// The PLE storage of every unit column.
pub struct Ple {
    pub columns: Vec<PleColumn>,
}

/// The bus words of a column of `units` units (every layer's PLE row).
pub fn levels(plan: &Plan, units: usize) -> u32 {
    plan.partition.layers_per_unit * units as u32
}

/// The store kernel's words a PE (`NTW`): the largest column's tokens times its bus words.
pub fn words(plan: &Plan, floor: &Floor) -> u32 {
    floor
        .stores
        .iter()
        .map(|st| st.tokens_per_pe * levels(plan, st.units.len()))
        .max()
        .unwrap_or(0)
}

/// The storage block a PE belongs to (its index is its bus).
pub fn store_of(floor: &Floor, c: Coord) -> Option<usize> {
    floor.stores.iter().position(|s| s.rect.contains(c))
}

/// The storage column serving unit `u`: its index, the unit's place top to bottom, bus words.
pub fn column_of(floor: &Floor, plan: &Plan, u: u32) -> Option<(usize, u32, u32)> {
    floor.stores.iter().enumerate().find_map(|(i, s)| {
        s.units
            .iter()
            .position(|&v| v == u)
            .map(|ui| (i, ui as u32, levels(plan, s.units.len())))
    })
}

impl Block for Ple {
    fn floor(&self, plan: &Plan, b: &mut FloorBuilder) -> R<()> {
        for col in &self.columns {
            let top = *col.units.first().ok_or("ple: a column with no units")?;
            let u = b
                .units
                .iter()
                .find(|u| u.index == top)
                .ok_or_else(|| format!("ple: unit {top} not on the floor"))?;
            let (x, uy, uw, xc) = (u.rect.x, u.rect.y, u.rect.w, u.cx);
            let y0 = uy
                .checked_sub(col.rows)
                .ok_or("ple: the storage runs off the wafer's top")?;
            let lv = levels(plan, col.units.len());
            for y in y0..uy {
                for lx in 0..uw {
                    let place = if lx == xc {
                        PASS_PE
                    } else {
                        Place::Store {
                            first: (y - y0) * col.tokens,
                            count: col.tokens,
                            level: lv,
                            top: y == y0,
                        }
                    };
                    b.set_role(Coord::new(x + lx, y), PeRole::new(STORE, place));
                }
            }
            b.stores.push(StoreBlock {
                rect: Rect::new(x, y0, uw, col.rows),
                units: col.units.clone(),
                tokens_per_pe: col.tokens,
            });
        }
        Ok(())
    }

    fn wire(&self, plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
        token(floor, w)?;
        buses(plan, floor, w)
    }

    /// The route-only PEs: idle PEs west of the head outside every unit and storage block that
    /// a link crosses.
    fn settle(&self, _plan: &Plan, w: &Wiring, floor: &mut Floor) -> R<()> {
        let hx0 = floor.head.rect.x;
        for l in &w.links {
            let pts: &[Coord] = match l {
                Link::Chain(c) => &c.pts,
                Link::Multicast(m) => &m.pts,
                Link::Path(p) => &p.pts,
                Link::Turn(t) => std::slice::from_ref(&t.at),
            };
            for &c in pts {
                if c.x < hx0
                    && floor.roles.get(c).is_idle()
                    && floor.unit_at(c).is_none()
                    && store_of(floor, c).is_none()
                {
                    *floor.roles.get_mut(c) = PeRole::new(STORE, PASS_PE);
                }
            }
        }
        Ok(())
    }

    fn colours(&self) -> Vec<(Class, u8)> {
        let mut out = vec![(TOKEN, 0)];
        out.extend(
            self.columns
                .iter()
                .enumerate()
                .map(|(i, c)| (bus(i), c.bus)),
        );
        out
    }

    /// The route-only PEs outside the storage blocks take one word (`NTW 1`).
    fn overrides(&self, prog: &Program, o: &mut Overrides) -> R<()> {
        for (at, role) in prog.floor.roles.iter() {
            if role.kind == STORE && role.place == PASS_PE && store_of(&prog.floor, at).is_none() {
                o.set(at, NTW, 1u32);
            }
        }
        Ok(())
    }

    /// Per storage block and staged file `pt<col>_<n>`: each half's rcfg and tiles (padded to
    /// `NTW` words) in blocks of 16 rows; then the route-only PEs' rcfg by row runs.
    fn bindings(&self, prog: &Program, out: &mut Copies) -> R<()> {
        let (plan, floor) = (&prog.plan, &prog.floor);
        let px = plan.partition.cols;
        let ntw = words(plan, floor);
        for (ci, st) in floor.stores.iter().enumerate() {
            let xc = px / 2;
            let lv = levels(plan, st.units.len());
            let per = (FILE_MAX / (u64::from(px) * u64::from(st.tokens_per_pe) * u64::from(lv) * 2))
                .max(1) as u32;
            for (n, fr0) in (0..st.rect.h).step_by(per as usize).enumerate() {
                let fr1 = (fr0 + per).min(st.rect.h);
                for (c0, c1) in [(0, xc), (xc, px)] {
                    for (rb, re) in chunks(fr0, fr1, 16) {
                        let rect =
                            Rect::new(st.rect.x + skip(c0, xc), st.rect.y + rb, c1 - c0, re - rb);
                        out.rcfg(rect);
                        let dims = vec![(rb - fr0, re - fr0), (c0, c1)];
                        out.slice(
                            "ptab",
                            rect,
                            Slice::of(&format!("pt{ci}_{n}"), None, dims).pad_to(ntw),
                            true,
                            from("embed_tokens_per_layer, per_layer_model_projection"),
                        );
                    }
                }
            }
        }
        let mut rows: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (c, r) in floor.roles.iter() {
            if r.kind == STORE && matches!(r.place, Place::Store { count: 0, .. }) {
                rows.entry(c.y).or_default().push(c.x);
            }
        }
        for (y, mut xs) in rows {
            xs.sort_unstable();
            for run in xs.chunk_by(|a, b| b == &(a + 1)) {
                out.rcfg(Rect::new(run[0], y, run.len() as u32, 1));
            }
        }
        Ok(())
    }
}

fn turn(w: &mut Wiring, at: Coord, class: Class, from: Port, to: Vec<Port>) {
    w.turn(Turn {
        at,
        class,
        from,
        to,
    });
}

/// [`TOKEN`]: one stream a storage row, west from the head control column to the westmost
/// storage block on the row; every storage PE (not a block's CX column) takes a copy.
fn token(floor: &Floor, w: &mut Wiring) -> R<()> {
    let CtlLine::Column(ctl) = floor.head.ctl else {
        return Err("ple: no head control column".into());
    };
    let xc = floor.units.first().map_or(0, |u| u.cx);
    let mut rows: BTreeMap<u32, Vec<&StoreBlock>> = BTreeMap::new();
    for st in &floor.stores {
        for y in st.rect.y..st.rect.y + st.rect.h {
            rows.entry(y).or_default().push(st);
        }
    }
    for (y, blocks) in rows {
        let xw = blocks.iter().map(|b| b.rect.x).min().unwrap_or(0);
        let xs: Vec<u32> = (xw..=ctl).rev().collect();
        for (i, &x) in xs.iter().enumerate() {
            let from = if i == 0 {
                Port::Ramp
            } else {
                Port::Dir(Dir::E)
            };
            let mut to = Vec::new();
            if i + 1 < xs.len() {
                to.push(Port::Dir(Dir::W));
            }
            let takes = x < floor.head.rect.x
                && blocks
                    .iter()
                    .any(|b| b.rect.x <= x && x < b.rect.x + b.rect.w && x != b.rect.x + xc);
            if takes {
                to.push(Port::Ramp);
            }
            turn(w, Coord::new(x, y), TOKEN, from, to);
        }
    }
    Ok(())
}

/// [`bus`]: a bus a unit column, down each weight column from the storage block through the
/// column's units (and the gaps between them); each unit's CY PE takes a copy, the last unit's
/// ends it.
fn buses(plan: &Plan, floor: &Floor, w: &mut Wiring) -> R<()> {
    let down = || vec![Port::Dir(Dir::S)];
    for (i, st) in floor.stores.iter().enumerate() {
        let units: Vec<&Unit> = st
            .units
            .iter()
            .map(|&u| {
                floor
                    .units
                    .iter()
                    .find(|v| v.index == u)
                    .ok_or_else(|| format!("ple: column unit {u} not on the floor"))
            })
            .collect::<R<_>>()?;
        let (y0, y1) = (st.rect.y, st.rect.y + st.rect.h);
        if units.first().map(|u| u.rect.y) != Some(y1) {
            return Err("ple: the storage sits directly above the column's top unit".into());
        }
        for c in 0..plan.partition.cols {
            let x = st.rect.x + skip(c, units[0].cx);
            for y in y0..y1 {
                let from = if y == y0 {
                    Port::Ramp
                } else {
                    Port::Dir(Dir::N)
                };
                turn(w, Coord::new(x, y), bus(i), from, down());
            }
            let mut prev_end = y1;
            for (ui, unit) in units.iter().enumerate() {
                let uy = unit.rect.y;
                let last = ui + 1 == units.len();
                for y in prev_end..uy {
                    turn(w, Coord::new(x, y), bus(i), Port::Dir(Dir::N), down());
                }
                prev_end = uy + unit.rect.h;
                for ly in 0..unit.rect.h {
                    let at = Coord::new(x, uy + ly);
                    if ly == unit.cy {
                        let mut to = vec![Port::Ramp];
                        if !last {
                            to.push(Port::Dir(Dir::S));
                        }
                        turn(w, at, bus(i), Port::Dir(Dir::N), to);
                        if last {
                            break;
                        }
                    } else {
                        turn(w, at, bus(i), Port::Dir(Dir::N), down());
                    }
                }
            }
        }
    }
    Ok(())
}
